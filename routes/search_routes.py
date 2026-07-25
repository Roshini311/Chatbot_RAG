import time
import json
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import QueryLog
from src.vector_store.manager import VectorStoreManager
from src.rag.qa_chain import RAGQuestionAnswering

router = APIRouter(prefix="/search", tags=["Semantic Search & RAG QA"])

_vector_manager = None
_rag_chain = None

def get_vector_manager():
    global _vector_manager
    if _vector_manager is None:
        _vector_manager = VectorStoreManager()
    return _vector_manager

def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGQuestionAnswering(vector_manager=get_vector_manager())
    return _rag_chain

class SemanticSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
    doc_ids: Optional[List[str]] = None

class QARequest(BaseModel):
    query: str
    session_history: Optional[str] = ""
    doc_ids: Optional[List[str]] = None

@router.post("/semantic")
def semantic_search(request: SemanticSearchRequest):
    """
    Executes dense vector similarity search across ingested document chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    vm = get_vector_manager()
    results = vm.search_similarity(
        query=request.query, 
        k=request.top_k or 4, 
        doc_ids=request.doc_ids
    )

    formatted = [
        {
            "content": r.page_content,
            "file_name": r.metadata.get("file_name", "Unknown"),
            "page_number": r.metadata.get("page_number", "N/A"),
            "doc_id": r.metadata.get("doc_id", "")
        }
        for r in results
    ]

    return {"query": request.query, "results": formatted}

@router.post("/qa")
def question_answering(request: QARequest, db: Session = Depends(get_db)):
    """
    RAG QA endpoint returning answer strictly grounded in context with page/document citations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    rag = get_rag_chain()
    result = rag.answer_question(
        query=request.query,
        session_history=request.session_history or "",
        doc_ids=request.doc_ids
    )
    elapsed = time.time() - start_time

    # Log query execution for system analytics
    query_id = str(uuid.uuid4())
    log_entry = QueryLog(
        query_id=query_id,
        query_text=request.query,
        referenced_docs=json.dumps(result.get("citations", [])),
        response_time=elapsed
    )
    db.add(log_entry)
    db.commit()

    return {
        "query_id": query_id,
        "query": request.query,
        "answer": result["answer"],
        "citations": result["citations"],
        "retrieved_context": result["retrieved_context"],
        "response_time_seconds": round(elapsed, 3)
    }
