# neuro_RAG

A production grade Retrieval-Augmented Generation (RAG) system for querying Neuroscience research papers with hybrid search (semantic and keyword), multi-PDF ingestion, and a live RAGAS evaluation dashboard.


## Overview
NeuroRAG allows you to upload a library of neuroscience PDFs and ask natural-language questions across all of them simultaneously. Answers are grounded strictly in the retrieved content, with page-level citations. Every response can be scored automatically across three RAG quality metrics using a lightweight LLM-as-judge evaluation pipeline.

## Quickstart
1. Clone the repo

`git clone https://github.com/davina-k/neuro-rag.git`
`cd neuro-rag`

3. Install dependencies

`pip install -r requirements.txt`

4. Set your API key

`export ANTHROPIC_API_KEY=sk-ant-...`
Or enter it in the sidebar when the app launches.

5. Run the app

`streamlit run app.py`
