from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.vector_store.manager import VectorStoreManager
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.ml.predictor import DocumentClassifierPredictor

router = APIRouter(prefix="/analysis", tags=["Summarization & Comparison"])

vector_manager = VectorStoreManager()
summarizer = DocumentSummarizer(vector_manager=vector_manager)
comparator = DocumentComparator(vector_manager=vector_manager)
predictor = DocumentClassifierPredictor()

class SummarizeRequest(BaseModel):
    doc_id: str
    file_name: Optional[str] = ""

class CompareRequest(BaseModel):
    doc_ids: List[str]
    file_names: Optional[List[str]] = None

class ClassifyRequest(BaseModel):
    text: str

@router.post("/summarize")
def summarize_document(request: SummarizeRequest):
    """
    Generates multi-tier executive and technical summary for a document.
    """
    result = summarizer.summarize_document(request.doc_id, file_name=request.file_name or "")
    return result

@router.post("/compare")
def compare_documents(request: CompareRequest):
    """
    Compares methodologies, advantages, and similarities across multiple documents.
    """
    result = comparator.compare_documents(request.doc_ids, file_names=request.file_names)
    return result

@router.post("/classify")
def classify_text(request: ClassifyRequest):
    """
    Classifies input text into domain categories using TensorFlow ML model.
    """
    category = predictor.predict_category(request.text)
    return {"text_snippet": request.text[:200], "predicted_category": category}
