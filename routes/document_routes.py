import os
import uuid
import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from config.settings import settings
from src.database.base import get_db
from src.database.models import DocumentMetadata
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker
from src.ml.predictor import DocumentClassifierPredictor
from src.vector_store.manager import VectorStoreManager

router = APIRouter(prefix="/documents", tags=["Document Management"])

# Lazy singletons
_pdf_parser = None
_chunker = None
_predictor = None
_vector_manager = None

def get_pdf_parser():
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = PDFParser()
    return _pdf_parser

def get_chunker():
    global _chunker
    if _chunker is None:
        _chunker = DocumentChunker()
    return _chunker

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DocumentClassifierPredictor()
    return _predictor

def get_vector_manager():
    global _vector_manager
    if _vector_manager is None:
        _vector_manager = VectorStoreManager()
    return _vector_manager

os.makedirs(settings.RAW_DOCUMENTS_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a PDF document, extracts metadata, predicts domain category,
    chunks text with page numbers, and indexes into FAISS vector database.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(settings.RAW_DOCUMENTS_DIR, f"{doc_id}_{file.filename}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Create initial DB record
    db_doc = DocumentMetadata(
        doc_id=doc_id,
        file_name=file.filename,
        upload_timestamp=datetime.datetime.utcnow(),
        processing_status="PROCESSING",
        file_path=save_path
    )
    db.add(db_doc)
    db.commit()

    try:
        parser = get_pdf_parser()
        chunker = get_chunker()
        predictor = get_predictor()
        vector_mgr = get_vector_manager()

        # 1. Parse pages
        pages_data = parser.extract_text_with_metadata(save_path, doc_id, file.filename)
        total_pages = len(pages_data)

        full_text = "\n".join([p["text"] for p in pages_data])

        # 2. Predict domain category using ML classifier
        category = predictor.predict_category(full_text)

        # 3. Create chunks with page numbers
        chunks = chunker.create_chunks(pages_data)
        total_chunks = len(chunks)

        # 4. Store embeddings in vector store
        vector_mgr.add_chunks(chunks)

        # Update DB record
        db_doc.total_pages = total_pages
        db_doc.total_chunks = total_chunks
        db_doc.category = category
        db_doc.processing_status = "PROCESSED"
        db.commit()

        return {
            "message": "Document successfully uploaded, processed, categorized, and indexed.",
            "doc_id": doc_id,
            "file_name": file.filename,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "category": category,
            "status": "PROCESSED"
        }
    except Exception as e:
        db_doc.processing_status = "FAILED"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/list")
def list_documents(db: Session = Depends(get_db)):
    """
    Returns list of all ingested documents with metadata & categories.
    """
    docs = db.query(DocumentMetadata).all()
    return [
        {
            "doc_id": d.doc_id,
            "file_name": d.file_name,
            "upload_timestamp": d.upload_timestamp.isoformat() if d.upload_timestamp else None,
            "total_pages": d.total_pages,
            "total_chunks": d.total_chunks,
            "category": d.category,
            "processing_status": d.processing_status
        }
        for d in docs
    ]

@router.get("/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Retrieves metadata for a specific document.
    """
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "upload_timestamp": doc.upload_timestamp.isoformat() if doc.upload_timestamp else None,
        "total_pages": doc.total_pages,
        "total_chunks": doc.total_chunks,
        "category": doc.category,
        "processing_status": doc.processing_status,
        "file_path": doc.file_path
    }

@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Deletes a document record from metadata store.
    """
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception:
            pass

    db.delete(doc)
    db.commit()
    return {"message": f"Document {doc_id} deleted successfully."}
