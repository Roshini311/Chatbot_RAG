from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.models import DocumentMetadata, QueryLog

class AnalyticsEngine:
    """
    Computes system usage analytics, dataset category distribution, and query metrics.
    """
    def __init__(self):
        pass

    def get_system_stats(self, db: Session) -> Dict[str, Any]:
        total_docs = db.query(DocumentMetadata).count()
        total_chunks = db.query(func.sum(DocumentMetadata.total_chunks)).scalar() or 0
        total_pages = db.query(func.sum(DocumentMetadata.total_pages)).scalar() or 0
        total_queries = db.query(QueryLog).count()

        # Category breakdown
        category_counts = db.query(
            DocumentMetadata.category, 
            func.count(DocumentMetadata.id)
        ).group_by(DocumentMetadata.category).all()

        category_distribution = {cat: count for cat, count in category_counts}

        # Status breakdown
        status_counts = db.query(
            DocumentMetadata.processing_status,
            func.count(DocumentMetadata.id)
        ).group_by(DocumentMetadata.processing_status).all()

        status_distribution = {st: count for st, count in status_counts}

        return {
            "total_documents": total_docs,
            "total_pages": int(total_pages),
            "total_chunks": int(total_chunks),
            "total_queries": total_queries,
            "category_distribution": category_distribution,
            "status_distribution": status_distribution
        }
