# AI Research & Knowledge Assistant

An enterprise-grade **AI Research & Knowledge Assistant** built with **FastAPI**, **Streamlit**, **TensorFlow**, **LangChain**, **FAISS**, and **SQLite**.

Designed for Sequelstring Solutions And Consultancy Pvt Ltd assignment specifications.

---

## 🌟 Key Features

1. **Document Ingestion & Metadata Tracking**: Upload and process multi-page PDFs with page-by-page text extraction (`PyMuPDF`/`pypdf`) and SQLite metadata persistence (`doc_id`, `file_name`, `total_pages`, `total_chunks`, `status`, `category`).
2. **TensorFlow Machine Learning Domain Classifier**: Custom TensorFlow/Keras deep learning model (`models/tf_classifier.h5`) to auto-categorize uploaded PDFs into tech domains (*Artificial Intelligence, Cyber Security, Cloud Computing, Robotics, Data Science*).
3. **Retrieval-Augmented Generation (RAG) with Page Citations**: Answers complex research questions with strict context grounding and precise page and document citations. Includes conversation memory.
4. **Multi-Document Comparison & Summarization Engine**:
   - Single-document multi-tier summaries (Executive, Technical, Takeaways).
   - Side-by-side multi-document comparative analysis (Methodologies, Pros/Cons, Similarities).
5. **System Analytics Dashboard**: Real-time metrics for total indexed documents, page/chunk counts, query history, and TensorFlow category breakdown.
6. **REST API & Swagger UI**: Auto-generated interactive OpenAPI docs available at `/docs`.

---

## 📁 Repository Directory Structure

```
ai-research-assistant/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic settings & environment configuration
│
├── data/
│   ├── raw_documents/           # Stored uploaded PDF files
│   ├── vector_db/               # Local persistence for FAISS index
│   └── metadata.db              # SQLite metadata database
│
├── models/
│   ├── tf_classifier.h5         # Saved TensorFlow classification model
│   └── tokenizer.pickle         # Model tokenizer / vectorizer artifact
│
├── src/
│   ├── database/
│   │   ├── base.py              # SQLAlchemy database session & engine
│   │   └── models.py            # SQLite ORM models (DocumentMetadata, QueryLog)
│   │
│   ├── document_processing/
│   │   ├── pdf_parser.py        # PDF text parser preserving page-level metadata
│   │   └── chunker.py           # Recursive chunker preserving page numbers & doc_ids
│   │
│   ├── ml/
│   │   ├── dataset_prep.py      # Dataset preparation script
│   │   ├── train_classifier.py  # TensorFlow model architecture & training pipeline
│   │   └── predictor.py         # Model loading & inference wrapper
│   │
│   ├── vector_store/
│   │   └── manager.py           # FAISS vector store indexing & hybrid search
│   │
│   ├── rag/
│   │   ├── qa_chain.py          # RAG pipeline with page citations & session memory
│   │   ├── summarizer.py        # Executive & Technical summarizer
│   │   └── comparator.py        # Multi-document comparison engine
│   │
│   └── analytics/
│       └── metrics.py           # Metrics calculation & query logging
│
├── routes/
│   ├── document_routes.py       # Endpoints: upload, list, get, delete
│   ├── search_routes.py         # Endpoints: semantic search, RAG QA with citation
│   ├── analysis_routes.py       # Endpoints: summarize, compare, classify
│   └── analytics_routes.py      # Endpoints: system stats & metrics
│
├── frontend/
│   └── app.py                   # Streamlit interactive UI dashboard
│
├── tests/
│   ├── test_parser.py           # Unit tests for parsing & chunking
│   ├── test_rag.py              # Unit tests for RAG QA
│   └── test_ml.py               # Unit tests for ML classifier
│
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Project dependencies
└── README.md                    # Primary repository documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python `3.10` or `3.11`
- An OpenAI API Key (optional for LLM response synthesis; fallback context mode available out-of-the-box).

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/Roshini311/Chatbot_RAG.git
cd Chatbot_RAG

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
VECTOR_DB_DIR=./data/vector_db
DATABASE_URL=sqlite:///./data/metadata.db
MODEL_PATH=./models/tf_classifier.h5
TOKENIZER_PATH=./models/tokenizer.pickle
```

### 4. Run the Backend API
```bash
python main.py
```
*The FastAPI backend will run at `http://localhost:8000`. You can inspect the Swagger UI documentation at `http://localhost:8000/docs`.*

### 5. Run the Streamlit Dashboard
In a separate terminal:
```bash
streamlit run frontend/app.py
```
*Access the interactive 4-tab UI at `http://localhost:8501`.*

---

## 🧪 Running Unit Tests

Execute `pytest` to run automated test suites:
```bash
pytest
```

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/documents/upload` | Upload PDF, parse pages, categorize with TensorFlow, and index chunks |
| `GET` | `/documents/list` | List all ingested documents with metadata & categories |
| `GET` | `/documents/{doc_id}` | Retrieve metadata for a specific document |
| `DELETE` | `/documents/{doc_id}` | Delete a document from vector index & metadata database |
| `POST` | `/search/semantic` | Execute dense vector similarity search |
| `POST` | `/search/qa` | Run RAG QA returning answer with exact page citations |
| `POST` | `/analysis/summarize` | Generate multi-tier document summary |
| `POST` | `/analysis/compare` | Compare research methodologies across multiple documents |
| `POST` | `/analysis/classify` | Predict domain category of input text snippet using ML model |
| `GET` | `/analytics/stats` | Retrieve system-wide usage stats & category distribution |
