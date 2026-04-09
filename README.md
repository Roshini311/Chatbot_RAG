# AI-powered Knowledge Assistant (RAG)

A complete Retrieval-Augmented Generation chatbot built with FastAPI, Streamlit, FAISS, and LangChain.

## Prerequisites
- Python 3.9+
- An OpenAI API Key (for the LLM generation)

## Setup Instructions

1. **Navigate to the directory**:
   ```bash
   cd "C:\Users\sanja\OneDrive\Desktop\Chatbot_RAG"
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Start the Backend API**:
   Open a new terminal, activate the virtual environment, and run:
   ```bash
   cd "C:\Users\sanja\OneDrive\Desktop\Chatbot_RAG"
   uvicorn backend.main:app --reload
   ```

6. **Start the Frontend UI**:
   Open a new terminal, activate the virtual environment, and run:
   ```bash
   cd "C:\Users\sanja\OneDrive\Desktop\Chatbot_RAG"
   streamlit run frontend/app.py
   ```

## Folder Structure
- `backend/`: FastAPI backend and Langchain RAG implementation.
- `frontend/`: Streamlit Chat UI.
- `data/`: A place to store uploaded documents.
- `vectorstore/`: A folder automatically generated to save the FAISS local database.
