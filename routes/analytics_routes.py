from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database.base import get_db
from src.analytics.metrics import AnalyticsEngine

router = APIRouter(prefix="/analytics", tags=["System Analytics"])

analytics_engine = AnalyticsEngine()

@router.get("/stats")
def get_system_stats(db: Session = Depends(get_db)):
    """
    Returns analytics metrics: document counts, page/chunk totals, and category breakdown.
    """
    stats = analytics_engine.get_system_stats(db)
    return stats
