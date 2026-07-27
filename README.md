# AI Research & Knowledge Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.1-1C3C3C.svg?style=flat)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8.0-00599C.svg?style=flat)](https://github.com/facebookresearch/faiss)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?style=flat&logo=python)](https://www.python.org/)

An enterprise-grade **AI Research & Knowledge Assistant** built to process, analyze, auto-categorize, and query large repositories of unstructured PDF documents. Developed for the **Sequelstring Solutions And Consultancy Pvt Ltd** technical assignment specifications.

---

## 📋 Table of Contents
- [1. Executive Summary & Capabilities](#1-executive-summary--capabilities)
- [2. System Architecture & Workflow](#2-system-architecture--workflow)
- [3. Technology Stack](#3-technology-stack)
- [4. Project Structure](#4-project-structure)
- [5. Step-by-Step Setup Guide](#5-step-by-step-setup-guide)
- [6. Environment Configuration](#6-environment-configuration)
- [7. Detailed API Documentation](#7-detailed-api-documentation)
- [8. Technical Justifications & Design Decisions](#8-technical-justifications--design-decisions)
  - [Chunking Strategy Justification](#chunking-strategy-justification)
  - [Search Modes & Retrieval Strategy](#search-modes--retrieval-strategy)
  - [Machine Learning Classifier Design](#machine-learning-classifier-design)
- [9. Limitations & Future Roadmap](#9-limitations--future-roadmap)
- [10. Deliverables Verification](#10-deliverables-verification)

---

## 1. Executive Summary & Capabilities

The **AI Research & Knowledge Assistant** provides a modular, scalable architecture combining **Retrieval-Augmented Generation (RAG)**, **TensorFlow Machine Learning Classification**, and **SQL-backed Metadata Tracking**.

### Key System Capabilities
1. **Document Ingestion & Metadata Tracking**: Upload multi-page PDFs with page-by-page text extraction (`PyMuPDF`/`pypdf`) and SQLite metadata persistence (`doc_id`, `file_name`, `upload_timestamp`, `total_pages`, `total_chunks`, `status`, `category`).
2. **TensorFlow ML Domain Classifier**: Deep learning model (`models/tf_classifier.h5`) to auto-categorize uploaded PDFs into tech domains (*Artificial Intelligence, Cyber Security, Cloud Computing, Robotics, Data Science*).
3. **Context-Grounded RAG Q&A with Citations**: Answers complex domain questions strictly grounded in document context, returning explicit **Page and Document Citations** (`[{"document": file_name, "page": page_number}]`) alongside conversation memory.
4. **Multi-Document Comparison & Summarization Engine**:
   - Single-document multi-tier summaries (*Executive, Technical, Bullet Breakdown, Key Takeaways*).
   - Side-by-side comparative analysis across multiple PDFs (*Methodologies, Pros/Cons, Similarities, Implementation*).
5. **System Analytics Dashboard**: Real-time tracking of total indexed documents, extracted pages, vector chunks, query counts, and TensorFlow category breakdown.
6. **Decoupled Architecture & REST APIs**: Production-ready FastAPI REST backend with auto-generated OpenAPI/Swagger documentation (`/docs`) and a 4-tab Streamlit dashboard.

---

## 2. System Architecture & Workflow

```
       ┌────────────────────────┐
       │   PDF Document Upload  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐      ┌───────────────────────────────┐
       │ PDF Parser & Page      │ ───► │ TensorFlow Domain Classifier  │
       │ Metadata Extractor     │      │ (.h5 / ML Model)              │
       └───────────┬────────────┘      └───────────────┬───────────────┘
                   │                                   │
                   ▼                                   ▼
       ┌────────────────────────┐      ┌───────────────────────────────┐
       │ Recursive Chunker      │      │ SQLite Metadata Storage       │
       │ (~1000 chars, 150 ovl) │      │ (DocumentMetadata & QueryLog) │
       └───────────┬────────────┘      └───────────────────────────────┘
                   │
                   ▼
       ┌────────────────────────┐      ┌───────────────────────────────┐
       │ Dense Feature Vector   │ ───► │ FAISS Vector Database Index   │
       │ Embedding Generator    │      │ (data/vector_db/index.faiss)  │
       └────────────────────────┘      └───────────────┬───────────────┘
                                                       │
                                                       ▼
       ┌────────────────────────┐      ┌───────────────────────────────┐
       │ User Query & Memory    │ ───► │ Hybrid Vector Search & RAG    │ ───► Answer + Citations
       │ Session History        │      │ Grounding Engine              │
       └────────────────────────┘      └───────────────────────────────┘
```

---

## 3. Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| **Backend API** | `FastAPI`, `Uvicorn` | RESTful endpoints, async request processing, OpenAPI/Swagger docs. |
| **Frontend UI** | `Streamlit` | Interactive 4-tab dashboard (Documents, RAG Q&A, Compare, Analytics). |
| **PDF Extraction** | `PyMuPDF` (`fitz`), `pypdf` | Page-by-page text parsing retaining exact page numbers. |
| **Text Chunking** | `LangChain`, `RecursiveCharacterTextSplitter` | Overlapping text splitting with page metadata preservation. |
| **Machine Learning** | `TensorFlow` / `Keras`, `scikit-learn` | Document technical domain auto-classification (`.h5` model). |
| **Vector DB** | `FAISS` | Persistent dense vector indexing and similarity retrieval. |
| **Embeddings** | `UltraFastEmbeddings` / `HuggingFace` | 384-dimensional dense feature hashing & semantic embeddings. |
| **LLM Synthesis** | `OpenAI GPT-3.5-Turbo` / `GPT-4o` | Citation-grounded QA, document comparison, and summarization. |
| **Database** | `SQLite`, `SQLAlchemy ORM` | Persistent storage for document metadata and query execution logs. |
| **Testing** | `pytest` | Automated unit and integration testing suites. |

---

## 4. Project Structure

```
ai-research-assistant/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic environment configuration & app settings
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
│   │   ├── dataset_prep.py      # Preprocessing & synthetic training dataset
│   │   ├── train_classifier.py  # TensorFlow model architecture & training script
│   │   └── predictor.py         # Model loading & inference wrapper
│   │
│   ├── vector_store/
│   │   └── manager.py           # FAISS vector store indexing & hybrid search
│   │
│   ├── rag/
│   │   ├── qa_chain.py          # RAG pipeline with page citations & session memory
│   │   ├── summarizer.py        # Executive & Technical summarizer
│   │   └── comparator.py        # Multi-document comparative analysis engine
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
├── Dockerfile                   # Single-container production deployment Dockerfile
├── openapi.json                 # Exported OpenAPI schema (Postman compatible)
├── .env.example                 # Environment variables template
├── requirements.txt             # Project dependencies
└── README.md                    # Repository documentation
```

---

## 5. Step-by-Step Setup Guide

### 1. Repository Setup
```bash
git clone https://github.com/Roshini311/Chatbot_RAG.git
cd Chatbot_RAG
```

### 2. Virtual Environment Creation
```bash
# Create environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Backend API
```bash
python main.py
```
*The FastAPI backend will start at `http://localhost:8000`. You can test endpoints via Swagger UI at `http://localhost:8000/docs`.*

### 5. Launch Streamlit UI
In a second terminal window:
```bash
streamlit run frontend/app.py
```
*Access the dashboard at `http://localhost:8501`.*

---

## 6. Environment Configuration

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
VECTOR_DB_DIR=./data/vector_db
DATABASE_URL=sqlite:///./data/metadata.db
MODEL_PATH=./models/tf_classifier.h5
TOKENIZER_PATH=./models/tokenizer.pickle
```

---

## 7. Detailed API Documentation

| Method | Endpoint | Description | Sample Payload |
| --- | --- | --- | --- |
| `POST` | `/documents/upload` | Upload PDF file, parse pages, run TF classifier, index into FAISS | `multipart/form-data` file |
| `GET` | `/documents/list` | List all uploaded documents with metadata & category | `N/A` |
| `GET` | `/documents/{doc_id}` | Retrieve metadata for a specific document ID | `N/A` |
| `DELETE` | `/documents/{doc_id}` | Delete document record and remove chunks | `N/A` |
| `POST` | `/search/semantic` | Execute dense similarity search | `{"query": "neural networks", "top_k": 4}` |
| `POST` | `/search/qa` | Run RAG QA with page & document citations | `{"query": "What are the key results?", "session_history": ""}` |
| `POST` | `/analysis/summarize` | Generate multi-tier summary | `{"doc_id": "uuid-here"}` |
| `POST` | `/analysis/compare` | Multi-document comparative analysis | `{"doc_ids": ["uuid-1", "uuid-2"]}` |
| `POST` | `/analysis/classify` | Predict domain category of input text | `{"text": "Cyber security vulnerability scanning..."}` |
| `GET` | `/analytics/stats` | Retrieve system analytics & category distribution | `N/A` |

---

## 8. Technical Justifications & Design Decisions

### Chunking Strategy Justification
- **Parameters**: `chunk_size = 1000` characters, `chunk_overlap = 150` characters.
- **Justification**:
  1. **Boundary Context Retention**: PDF page breaks often split sentences mid-thought. An overlap of 150 characters guarantees that key phrases extending across chunk or page boundaries are not truncated.
  2. **Optimal Embedding Density**: A size of 1000 characters corresponds to ~150–200 words, which fits well within dense vector representation limits without diluting specific facts.
  3. **Page Metadata Attachment**: Each chunk explicitly inherits `page_number`, `doc_id`, and `file_name`, ensuring 100% citation traceability.

### Search Modes & Retrieval Strategy
- **Keyword Search**: Performs exact character matching. Useful for searching exact document IDs, standard reference codes, or specific acronyms.
- **Semantic Search**: Computes dense vector similarity using cosine distance. Essential for conceptual queries where user phrasing differs from document wording.
- **Hybrid Search (Default)**: Combines dense vector similarity scores with BM25/word overlap re-ranking. Ensures both high semantic recall and exact keyword precision.

### Machine Learning Classifier Design
- **Architecture**: Sequential Neural Network (`TextVectorization` + `Embedding` + `GlobalAveragePooling1D` + `Dense(64, ReLU)` + `Dropout(0.3)` + `Dense(Softmax)`).
- **Categories**: *Artificial Intelligence, Machine Learning, Computer Vision, Natural Language Processing, Robotics, Cyber Security, Cloud Computing, Data Science*.
- **Optimization**: Includes lazy loading and fallback predictors so the server boots up in under 1 second without running out of memory.

---

## 9. Limitations & Future Roadmap

### Current Limitations
1. **Scanned PDF Parsing**: OCR (Optical Character Recognition) is required for image-only scanned PDFs without embedded text layers.
2. **SQLite Write Locks**: SQLite handles single-writer concurrency. For ultra-high concurrency enterprise deployments, PostgreSQL is recommended.

### Future Roadmap
- [ ] Integration of Tesseract OCR for scanned image PDFs.
- [ ] Redis caching for frequent RAG queries.
- [ ] Cloud vector database integration option (Qdrant / Pinecone).
- [ ] User authentication & role-based access control (JWT OAuth2).

---

## 10. Deliverables Verification

- [x] **Complete Source Code**: Pushed to GitHub repository.
- [x] **OpenAPI Schema / Postman Collection**: Exported `openapi.json` included in root folder.
- [x] **Sample Dataset**: Sample test files included in `data/`.
- [x] **Trained Model**: TensorFlow `.h5` / tokenizer model saved in `models/`.
- [x] **Unit & Integration Tests**: Verified passing via `pytest`.
