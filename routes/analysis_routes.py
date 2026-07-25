from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.vector_store.manager import VectorStoreManager
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.ml.predictor import DocumentClassifierPredictor

router = APIRouter(prefix="/analysis", tags=["Summarization & Comparison"])

_vector_manager = None
_summarizer = None
_comparator = None
_predictor = None

def get_vector_manager():
    global _vector_manager
    if _vector_manager is None:
        _vector_manager = VectorStoreManager()
    return _vector_manager

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = DocumentSummarizer(vector_manager=get_vector_manager())
    return _summarizer

def get_comparator():
    global _comparator
    if _comparator is None:
        _comparator = DocumentComparator(vector_manager=get_vector_manager())
    return _comparator

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DocumentClassifierPredictor()
    return _predictor

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
    s = get_summarizer()
    result = s.summarize_document(request.doc_id, file_name=request.file_name or "")
    return result

@router.post("/compare")
def compare_documents(request: CompareRequest):
    """
    Compares methodologies, advantages, and similarities across multiple documents.
    """
    c = get_comparator()
    result = c.compare_documents(request.doc_ids, file_names=request.file_names)
    return result

@router.post("/classify")
def classify_text(request: ClassifyRequest):
    """
    Classifies input text into domain categories using ML model.
    """
    p = get_predictor()
    category = p.predict_category(request.text)
    return {"text_snippet": request.text[:200], "predicted_category": category}
