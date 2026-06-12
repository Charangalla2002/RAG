"""
VaultRAG – Vector Store

Manages per-topic ChromaDB collections and provides:
  - ingest_file()   : chunk → embed → upsert
  - query()         : semantic nearest-neighbour search
  - delete_file()   : remove chunks by source filename
  - collection_info(): stats for a topic
"""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config.settings import (
    APP_NAME, TOPICS_DIR, EMBEDDING_MODEL, EMBEDDING_DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP,
)
from .processor import extract_text

logger = logging.getLogger(APP_NAME)

# ─── Lazy-loaded globals ─────────────────────────────────────────────────────

_embedding_model: Optional[SentenceTransformer] = None
_splitter: Optional[RecursiveCharacterTextSplitter] = None


def _get_embedder() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")
    return _embedding_model


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    return _splitter


# ─── ChromaDB client ─────────────────────────────────────────────────────────

def _get_chroma_collection(topic_id: str):
    """Return (or create) a persistent Chroma collection for the topic."""
    chroma_path = TOPICS_DIR / topic_id / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=f"topic_{topic_id}",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ─── Ingest ───────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str, int, int], None]   # stage, current, total


def ingest_file(
    file_path: Path,
    topic_id: str,
    doc_id: str,
    progress_cb: Optional[ProgressCallback] = None,
) -> int:
    """
    Parse, chunk, embed, and store *file_path* into the topic vector store.

    Returns the number of chunks stored.
    Raises on fatal error so the caller can mark the document as failed.
    """

    def _progress(stage, cur, total):
        if progress_cb:
            progress_cb(stage, cur, total)

    # 1 – Extract text
    _progress("extracting", 0, 1)
    text = extract_text(file_path)
    if text is None:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    if not text.strip():
        logger.warning("No text extracted from %s", file_path.name)
        return 0
    _progress("extracting", 1, 1)

    # 2 – Chunk
    _progress("chunking", 0, 1)
    splitter = _get_splitter()
    chunks = splitter.split_text(text)
    if not chunks:
        return 0
    _progress("chunking", 1, 1)

    # 3 – Embed
    embedder = _get_embedder()
    total = len(chunks)
    embeddings = []
    batch_size = 64
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vecs = embedder.encode(batch, show_progress_bar=False).tolist()
        embeddings.extend(vecs)
        _progress("embedding", min(i + batch_size, total), total)

    # 4 – Upsert into ChromaDB
    collection = _get_chroma_collection(topic_id)

    # Remove any existing chunks for this doc (re-ingest idempotency)
    try:
        existing = collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids = [f"{doc_id}_{i}" for i in range(total)]
    metadatas = [
        {
            "doc_id": doc_id,
            "source": file_path.name,
            "chunk_index": i,
            "topic_id": topic_id,
        }
        for i in range(total)
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    _progress("storing", total, total)
    logger.info("Ingested %d chunks from %s into topic %s", total, file_path.name, topic_id)
    return total


# ─── Query ───────────────────────────────────────────────────────────────────

def query_topic(topic_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return the top-k most relevant chunks for *query_text*."""
    try:
        collection = _get_chroma_collection(topic_id)
        embedder = _get_embedder()
        query_vec = embedder.encode([query_text]).tolist()
        results = collection.query(
            query_embeddings=query_vec,
            n_results=min(top_k, collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "doc_id": meta.get("doc_id", ""),
                    "score": round(1 - dist, 4),   # cosine similarity
                }
            )
        return hits
    except Exception as e:
        logger.error("Query failed for topic %s: %s", topic_id, e)
        return []


# ─── Delete ───────────────────────────────────────────────────────────────────

def delete_document_vectors(topic_id: str, doc_id: str):
    """Remove all vectors for *doc_id* from the topic collection."""
    try:
        collection = _get_chroma_collection(topic_id)
        existing = collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info("Deleted %d vectors for doc %s", len(existing["ids"]), doc_id)
    except Exception as e:
        logger.error("Vector delete failed: %s", e)


# ─── Stats ───────────────────────────────────────────────────────────────────

def collection_info(topic_id: str) -> Dict[str, Any]:
    """Return basic stats for the topic's Chroma collection."""
    try:
        collection = _get_chroma_collection(topic_id)
        count = collection.count()
        return {"total_vectors": count, "status": "healthy"}
    except Exception as e:
        return {"total_vectors": 0, "status": f"error: {e}"}
