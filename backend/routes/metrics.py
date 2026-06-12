"""Metrics API – system resources and token analytics."""

from fastapi import APIRouter
from backend.utils.monitoring import get_system_metrics
from backend.database.db import get_token_analytics, get_all_topics
from backend.rag.vector_store import collection_info

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/system")
def system_metrics():
    return get_system_metrics()


@router.get("/tokens")
def token_metrics(period: str = "today"):
    return get_token_analytics(period)


@router.get("/storage")
def storage_metrics():
    """Aggregate vector stats across all topics."""
    topics = get_all_topics()
    total_vectors = 0
    topic_stats = []
    for t in topics:
        info = collection_info(t["id"])
        vecs = info.get("total_vectors", 0)
        total_vectors += vecs
        topic_stats.append({
            "id": t["id"],
            "name": t["name"],
            "doc_count": t.get("doc_count", 0),
            "total_chunks": t.get("total_chunks", 0),
            "total_vectors": vecs,
            "status": info.get("status", "unknown"),
        })
    return {"total_vectors": total_vectors, "topics": topic_stats}
