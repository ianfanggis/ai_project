import os
import re
import time
from getpass import getpass

import torch
import PyPDF2
import gradio

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# =========================
# 0) Auth + Cache
# =========================
hfapi_key = getpass("Enter your HuggingFace access token: ").strip()
if hfapi_key:
    os.environ["HF_TOKEN"] = hfapi_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key
    print("HF token set.")
else:
    print("HF token not set (public models only).")

set_llm_cache(InMemoryCache())

# =========================
# 1) Global config
# =========================
persist_directory = "docs/chroma/"
pdf_path = "InnoVait_TCM1.pdf"   # 改成你的 PDF 路徑

# =========================
# 2) Read PDF
# =========================
def get_documents() -> str:
    print("$$$$$ ENTER INTO get_documents $$$$$")
    parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
    full_text = "\n".join(parts)
    print("Chars:", len(full_text))
    print("@@@@@@ EXIT FROM get_documents @@@@@")
    return full_text

# =========================
# 3) Split text
# =========================
def getTextSplits():
    print("$$$$$ ENTER INTO getTextSplits $$$$$")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    texts = splitter.split_text(get_documents())
    print("Num chunks:", len(texts))
    print("@@@@@@ EXIT FROM getTextSplits @@@@@")
    return texts

# =========================
# 4) Embeddings
# =========================
def getEmbeddings():
    print("$$$$$ ENTER INTO getEmbeddings $$$$$")
    modelPath = "mixedbread-ai/mxbai-embed-large-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False},
    )
    print("@@@@@@ EXIT FROM getEmbeddings @@@@@")
    return embedding

# =========================
# 5) LLM (HF Inference)
# =========================
def getLLM():
    print("$$$$$ ENTER INTO getLLM $$$$$")
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,          # RAG 建議低一點，減少幻覺
        repetition_penalty=1.1,
        top_k=10,
    )
    print("@@@@@@ EXIT FROM getLLM @@@@@")
    return llm

# =========================
# 6) Chroma utils
# =========================
def is_chroma_db_present(directory: str):
    return os.path.exists(directory) and len(os.listdir(directory)) > 0

# =========================
# 7) Query classification (routing)
# =========================
def classify_query(query: str):
    q = query.lower()
    concept_patterns = [r"what is", r"define", r"explain", r"describe", r"concept of"]
    example_patterns = [r"give an example", r"demonstrate", r"illustrate"]
    code_patterns = [r"how to implement", r"python code", r"write a program"]

    for p in concept_patterns:
        if re.search(p, q):
            return "concept"
    for p in example_patterns:
        if re.search(p, q):
            return "example"
    for p in code_patterns:
        if re.search(p, q):
            return "code"
    return "general"

# =========================
# 8) Retriever (MMR / similarity)
# =========================
def getRetriever(query, metadata_filter=None):
    print("$$$$$ ENTER INTO getRetriever $$$$$")

    query_type = classify_query(query)
    print("Query type:", query_type)

    k_default = 2
    fetch_k_default = 5
    search_type_default = "mmr"

    if query_type == "concept":
        k_default = 5
        fetch_k_default = 10
        search_type_default = "mmr"
    elif query_type in ["example", "code"]:
        search_type_default = "similarity"

    # Load or build DB
    if is_chroma_db_present(persist_directory):
        print("Loading existing Chroma DB...")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=getEmbeddings(),
            collection_name="ai_tutor",
        )
    else:
        print("Building new Chroma DB from PDF...")
        vectordb = Chroma.from_texts(
            texts=getTextSplits(),
            embedding=getEmbeddings(),
            persist_directory=persist_directory,
            collection_name="ai_tutor",
        )

    # (Optional) metadata filter — 這裡保留接口，但若你沒存 metadata，請留空
    if metadata_filter:
        metadata_filter_dict = {"result": metadata_filter}
        if search_type_default == "similarity":
            return vectordb.as_retriever(
                search_type=search_type_default,
                search_kwargs={"k": k_default, "filter": metadata_filter_dict},
            )
        return vectordb.as_retriever(
            search_type=search_type_default,
            search_kwargs={"k": k_default, "fetch_k": fetch_k_default, "filter": metadata_filter_dict},
        )

    if search_type_default == "similarity":
        return vectordb.as_retriever(search_type=search_type_default, search_kwargs={"k": k_default})

    return vectordb.as_retriever(
        search_type=search_type_default, search_kwargs={"k": k_default, "fetch_k": fetch_k_default}
    )

# =========================
# 9) RAG pipeline (streaming)
# =========================
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def get_rag_response(query, metadata_filter=None):
    print("$$$$$ ENTER INTO get_rag_response $$$$$")

    retriever = getRetriever(query, metadata_filter)
    llm = getLLM()

    template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know. Do not make up an answer.

Context:
{context}

Question:
{question}

Helpful Answer:
"""
    prompt = PromptTemplate.from_template(template)

    def prepare_inputs(inputs):
        docs = retriever.invoke(inputs["question"])
        context = format_docs(docs)
        return {"context": context, "question": inputs["question"]}

    rag_chain = (
        RunnablePassthrough()
        | RunnableLambda(prepare_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )

    full_response = ""
    for chunk in rag_chain.stream({"question": query}):
        full_response += chunk
        time.sleep(0.03)
        yield full_response

# =========================
# 10) Gradio UI
# =========================
in_question = gradio.Textbox(
    lines=6,
    label="Ask a question",
    value="What are Artificial Intelligence and Machine Learning?",
)

in_metadata_filter = gradio.Textbox(
    lines=1,
    label="(Optional) Metadata filter",
)

out_response = gradio.Textbox(
    label="Response",
    interactive=False,
    show_copy_button=True,
)

iface = gradio.Interface(
    fn=get_rag_response,
    inputs=[in_question, in_metadata_filter],
    outputs=out_response,
    title="Your AI Tutor (RAG)",
    description="Ask questions about the PDF using Retrieval-Augmented Generation.",
    allow_flagging="never",
    stream_every=0.5,
)

iface.launch(share=True)