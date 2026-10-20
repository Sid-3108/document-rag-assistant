# Document RAG Assistant

A document-based question-answering application built with Retrieval-Augmented Generation (RAG). Upload a PDF, ask questions about its content, and receive answers generated using retrieved document context.

## Overview

This project demonstrates how RAG combines semantic search with a large language model to answer questions using information from a supplied document.

## Technology Stack

- **Python** — application logic
- **Streamlit** — interactive web interface
- **pypdf** — PDF text extraction
- **Sentence Transformers** — text embeddings using `all-MiniLM-L6-v2`
- **FAISS** — vector indexing and similarity search using `IndexFlatL2`
- **Anthropic Claude** — answer generation
- **python-dotenv** — environment variable management

## How It Works

1. Upload a PDF document.
2. Extract its text and divide it into chunks.
3. Generate embeddings for the text chunks.
4. Store the embeddings in a FAISS index.
5. Convert the user's question into an embedding and retrieve relevant chunks.
6. Pass the retrieved context to Claude to generate an answer.

## Getting Started

### Prerequisites

- Python installed
- An Anthropic API key

### Installation

Clone the repository:

```bash
git clone https://github.com/Sid-3108/document-rag-assistant.git
cd document-rag-assistant
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the project dependencies after creating a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Configure the API key

Create a `.env` file in the project root:

```text
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Replace the placeholder with your own API key. Never commit `.env` to GitHub.

### Run the application

```bash
streamlit run streamlit_app.py
```

## Project Status

This is a learning and demonstration project exploring PDF-based question answering with RAG.

## Future Improvements

- Display source passages alongside generated answers
- Add retrieval and answer quality evaluation
- Improve document processing and error handling
- Deploy the application for online access

## Security

Do not upload API keys, private documents, or confidential data to the public repository.


## Demo & Documentation

- [RAG Architecture Presentation](docs-ppt/RAG_Architecture.pdf)
- [Application Output Screenshot](screenshots/screencapture-localhost-8501-2026-10-20-19_28_50.pdf)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

