from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
import os
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel

from .embeddings import process_document
from .vector_store import add_documents_to_faiss, get_vector_store
from .rag_pipeline import generate_answer

# Load environment variables from .env if present
load_dotenv()

DATA_DIR = "data"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup runs before app starts receiving requests
    os.makedirs(DATA_DIR, exist_ok=True)
    yield
    # Cleanup code can go here

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """ Endpoint to upload a document, chunk it, and store embeddings in FAISS """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    file_path = os.path.join(DATA_DIR, file.filename)
    
    # Save file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Process and vectorize
        chunks = process_document(file_path, file.filename)
        add_documents_to_faiss(chunks)
        return {"message": f"Successfully processed '{file.filename}' into {len(chunks)} chunks and stored vector embeddings."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/query")
async def query_assistant(request: QueryRequest):
    """ Endpoint to query the RAG system """
    # Check if OPENAI_API_KEY is available
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="OPENAI_API_KEY environment variable is missing. Please set it in your .env file."
        )
        
    vector_store = get_vector_store()
    if not vector_store:
        raise HTTPException(
            status_code=400, 
            detail="No documents have been uploaded or processed yet. Please upload a document first."
        )
        
    try:
        result = generate_answer(request.query, vector_store)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
