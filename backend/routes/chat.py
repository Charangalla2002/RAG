"""Chat API – sessions, streaming, model management."""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import APP_NAME, DEFAULT_MODEL
from backend.database.db import (
    get_all_sessions, get_session, create_session, update_session,
    delete_session, get_messages, get_token_analytics,
)
from backend.rag.pipeline import stream_chat, check_ollama, get_active_model, set_active_model

logger = logging.getLogger(APP_NAME)
router = APIRouter(prefix="/api/chat", tags=["chat"])


# ─── Pydantic models ──────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    topic_id: str
    model: str = DEFAULT_MODEL
    title: str = "New Chat"


class ChatRequest(BaseModel):
    session_id: str
    query: str
    model: Optional[str] = None   # None → use active model
    topic_id: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


class ModelSwitchRequest(BaseModel):
    model: str


# ─── Sessions ─────────────────────────────────────────────────────────────────

@router.get("/sessions")
def list_sessions():
    return get_all_sessions()


@router.post("/sessions", status_code=201)
def new_session(body: NewSessionRequest):
    session_id = str(uuid.uuid4())
    return create_session(session_id, body.topic_id, body.model, body.title)


@router.get("/sessions/{session_id}")
def get(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    s["messages"] = get_messages(session_id)
    return s


@router.patch("/sessions/{session_id}/rename")
def rename(session_id: str, body: RenameRequest):
    if not get_session(session_id):
        raise HTTPException(404, "Session not found")
    return update_session(session_id, title=body.title)


@router.delete("/sessions/{session_id}", status_code=204)
def remove_session(session_id: str):
    """
    Delete a chat session and all its messages.
    Returns 204 on success. Returns 204 even if already gone (idempotent).
    """
    session = get_session(session_id)
    if not session:
        # Already deleted — treat as success so UI doesn't show an error
        return
    delete_session(session_id)
    logger.info("Deleted session %s", session_id)


@router.get("/sessions/{session_id}/export")
def export_session(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    messages = get_messages(session_id)
    lines = [f"# {s['title']}\n", f"Model: {s['model']}  |  Topic: {s['topic_id']}\n\n"]
    for m in messages:
        role = "You" if m["role"] == "user" else "VaultRAG"
        lines.append(f"### {role}\n{m['content']}\n\n")
    return {"filename": f"{s['title'][:40]}.md", "content": "".join(lines)}


# ─── Streaming chat ───────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(body: ChatRequest):
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    topic_id = body.topic_id or session["topic_id"]
    # Resolve model: request body → session model → active model
    model = body.model or session.get("model") or get_active_model()
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in get_messages(body.session_id)
    ]

    async def gen():
        async for chunk in stream_chat(
            body.session_id, topic_id, model, body.query, history
        ):
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Ollama / model management ────────────────────────────────────────────────

@router.get("/ollama/status")
async def ollama_status():
    result = await check_ollama()
    result["default_model"] = DEFAULT_MODEL
    result["active_model"] = get_active_model()
    return result


@router.get("/models")
async def list_models():
    """Return all models installed in Ollama."""
    result = await check_ollama()
    return {
        "models": result.get("models", []),
        "active_model": get_active_model(),
        "running": result.get("running", False),
    }


@router.post("/models/switch")
async def switch_model(body: ModelSwitchRequest):
    """
    Switch the active model immediately — no restart needed.
    Validates the model exists in Ollama before switching.
    """
    result = await check_ollama()
    if not result.get("running"):
        raise HTTPException(503, "Ollama is not running")

    available = result.get("models", [])
    if available and body.model not in available:
        raise HTTPException(400, f"Model '{body.model}' is not installed. "
                                 f"Available: {', '.join(available)}")

    set_active_model(body.model)
    logger.info("Model switched to: %s", body.model)
    return {"active_model": get_active_model(), "switched": True}


# ─── Analytics ────────────────────────────────────────────────────────────────

@router.get("/analytics")
def analytics(period: str = "today"):
    return get_token_analytics(period)
