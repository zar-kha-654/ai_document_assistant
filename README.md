# Streamlit AI Document Assistant

A simple RAG (Retrieval-Augmented Generation) document assistant built with Streamlit, FAISS, Sentence Transformers, and Groq LLMs.

## Features
- **Multi-format Support**: Local upload for PDF, DOCX, TXT, and MD files.
- **Google Drive Integration**: Support for public Google Drive file/folder URLs via `gdown`.
- **Modular Extraction**: Isolated document parsing functions preserving file name and page metadata.
- **Overlapping Text Chunking**: Prepares manageable context blocks while retaining metadata.
- **Hybrid Search**: Combines FAISS vector search with keyword overlap scoring.
- **Groq Integration**: Answers questions strictly using retrieved document context.
- **Session State Caching**: Embeddings are calculated once per document ingestion and cached in memory.

---

## Setup & Configuration

### 1. Installation
Install dependencies:
```bash
pip install -r requirements.txt
