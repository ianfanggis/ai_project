# Starting Point for the Final Project of the "From Beginner to Advanced LLM Developer" Course

## Overview

This repository contains the code for the final project (Part 4: *Building Your Own Advanced LLM + RAG Project to Receive Certification*) of the **"From Beginner to Advanced LLM Developer"** course.

### Notebooks and Files

- **01_gpt2_InnoVait_TCM1_tutor_rag_v1.ipynb**  
  This notebook is only used to fine-tune the model and save it to Hugging Face.  
  It is **not required** to run in order to launch the application.

- **app.py**  
  This file contains the main RAG application logic and is used to launch the Gradio app.

## Features

This project includes the following features:

- Streaming responses for better user experience.
- Data collection and curation based on PDF documents (e.g., *InnoVait_TCM1.pdf*).
- Prompt caching using `InMemoryCache` and `set_llm_cache` from `langchain_core` to improve performance when repeating the same queries.  
  Reference: https://python.langchain.com/docs/how_to/llm_caching/
- Metadata-based filtering with a dynamically updated retriever.
- Query routing: the `search_type` is selected at runtime based on the query (e.g., `"mmr"` or `"similarity"`).  
  Try prompts like:
  - *"Demonstrate Artificial Intelligence and Machine Learning."*
  - *"What are Artificial Intelligence and Machine Learning?"*
- A query pipeline that includes function calling.

## Setup

1. Create a `.env` file and add your OpenAI API key (if required by your setup):

```bash
OPENAI_API_KEY="sk-..."

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate

3.Install the dependencies:
pip install -r requirements.txt

4.Launch the Gradio app:
python app.py