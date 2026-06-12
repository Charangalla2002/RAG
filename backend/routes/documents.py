"""
Documents API

POST /api/documents/upload              – buffer files
POST /api/documents/process/{topic_id} – ingest buffered files (SSE progress)
GET  /api/documents/{topic_id}          – list documents
DELETE /api/documents/{topic_id}/{doc_id} – delete document
POST /api/documents/reprocess/{doc_id} – reprocess a document
"""

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from config.settings import APP_NAME, UPLOAD_BUFFER_DIR, TOPICS_DIR
from backend.database.db import (
    get_documents, get_document, create_document,
    update_document_status, delete_document, get_topic,
)
from backend.rag.vector_store import ingest_file, delete_document_vectors
from backend.rag.processor import SUPPORTED_EXTENSIONS

logger = logging.getLogger(APP_NAME)

router = APIRouter(prefix="/api/documents", tags=["documents"])


# ─── Upload buffer ────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Save files to the upload buffer and return their temporary IDs."""
    results = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({"filename": f.filename, "status": "rejected", "reason": f"Unsupported type {ext}"})
            continue

        tmp_id = str(uuid.uuid4())
        tmp_path = UPLOAD_BUFFER_DIR / f"{tmp_id}{ext}"
        content = await f.read()
        tmp_path.write_bytes(content)
        results.append({
            "tmp_id": tmp_id,
            "filename": f.filename,
            "size": len(content),
            "ext": ext,
            "status": "buffered",
        })
    return results


# ─── Process (ingest) ────────────────────────────────────────────────────────

@router.post("/process/{topic_id}")
async def process_files(topic_id: str, body: dict):
    """
    Ingest a list of buffered files into *topic_id*.
    Body: { "files": [ { "tmp_id": "...", "filename": "original.pdf" }, ... ] }
    Streams SSE progress events.
    """
    if not get_topic(topic_id):
        raise HTTPException(404, "Topic not found")

    files = body.get("files", [])
    if not files:
        raise HTTPException(400, "No files specified")

    async def event_stream():
        topic_dir = TOPICS_DIR / topic_id / "files"
        topic_dir.mkdir(parents=True, exist_ok=True)

        for entry in files:
            tmp_id = entry.get("tmp_id")
            original_name = entry.get("filename", "unknown")
            ext = Path(original_name).suffix.lower()
            tmp_path = UPLOAD_BUFFER_DIR / f"{tmp_id}{ext}"

            if not tmp_path.exists():
                yield _sse("file_error", {"filename": original_name, "error": "Buffer file not found"})
                continue

            doc_id = str(uuid.uuid4())
            dest_path = topic_dir / f"{doc_id}{ext}"
            shutil.copy2(tmp_path, dest_path)
            tmp_path.unlink(missing_ok=True)

            create_document(doc_id, topic_id, dest_path.name, original_name,
                            dest_path.stat().st_size, ext)

            yield _sse("file_start", {"doc_id": doc_id, "filename": original_name})

            try:
                loop = asyncio.get_event_loop()

                progress_events = []

                def progress_cb(stage, current, total):
                    progress_events.append((stage, current, total))

                chunks = await loop.run_in_executor(
                    None,
                    lambda: ingest_file(dest_path, topic_id, doc_id, progress_cb),
                )

                # Flush progress events
                for stage, cur, tot in progress_events:
                    yield _sse("progress", {"stage": stage, "current": cur, "total": tot})

                update_document_status(doc_id, "completed", chunks)
                yield _sse("file_done", {
                    "doc_id": doc_id,
                    "filename": original_name,
                    "chunks": chunks,
                    "status": "completed",
                })

            except Exception as e:
                logger.error("Ingest failed for %s: %s", original_name, e)
                update_document_status(doc_id, "failed", error_msg=str(e))
                yield _sse("file_error", {
                    "doc_id": doc_id,
                    "filename": original_name,
                    "error": str(e),
                })

        yield _sse("batch_done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── List ─────────────────────────────────────────────────────────────────────

@router.get("/{topic_id}")
def list_documents(topic_id: str):
    if not get_topic(topic_id):
        raise HTTPException(404, "Topic not found")
    return get_documents(topic_id)


# ─── Delete ──────────────────────────────────────────────────────────────────

@router.delete("/{topic_id}/{doc_id}", status_code=204)
def remove_document(topic_id: str, doc_id: str):
    doc = get_document(doc_id)
    if not doc or doc["topic_id"] != topic_id:
        raise HTTPException(404, "Document not found")

    # Delete vectors
    delete_document_vectors(topic_id, doc_id)

    # Delete file on disk
    file_path = TOPICS_DIR / topic_id / "files" / doc["filename"]
    file_path.unlink(missing_ok=True)

    # Delete DB record
    delete_document(doc_id)


# ─── Reprocess ───────────────────────────────────────────────────────────────

@router.post("/reprocess/{doc_id}")
async def reprocess_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    file_path = TOPICS_DIR / doc["topic_id"] / "files" / doc["filename"]
    if not file_path.exists():
        raise HTTPException(410, "Source file no longer on disk")

    update_document_status(doc_id, "processing")

    async def event_stream():
        try:
            loop = asyncio.get_event_loop()
            progress_events = []

            def progress_cb(stage, current, total):
                progress_events.append((stage, current, total))

            delete_document_vectors(doc["topic_id"], doc_id)
            chunks = await loop.run_in_executor(
                None,
                lambda: ingest_file(file_path, doc["topic_id"], doc_id, progress_cb),
            )

            for stage, cur, tot in progress_events:
                yield _sse("progress", {"stage": stage, "current": cur, "total": tot})

            update_document_status(doc_id, "completed", chunks)
            yield _sse("done", {"doc_id": doc_id, "chunks": chunks})
        except Exception as e:
            update_document_status(doc_id, "failed", error_msg=str(e))
            yield _sse("error", {"error": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
