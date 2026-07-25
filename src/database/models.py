import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from src.database.base import Base

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String, unique=True, index=True, nullable=False)
    file_name = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String, default="PENDING")  # PENDING, PROCESSED, FAILED
    category = Column(String, default="Uncategorized")
    file_path = Column(String, nullable=False)

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String, unique=True, index=True, nullable=False)
    query_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    referenced_docs = Column(Text, nullable=True)  # JSON formatted citations
    response_time = Column(Float, default=0.0)
