"""
VaultRAG – RAG Pipeline v2

Retrieves context from ChromaDB + MCP servers, builds a prompt,
streams the Ollama response via SSE, and records token usage.
"""

import json
import logging
import time
import uuid
from typing import AsyncGenerator, List, Dict, Any, Optional

import aiohttp

from config.settings import APP_NAME, OLLAMA_BASE_URL, TOP_K_RESULTS, DEFAULT_MODEL
from backend.database.db import (
    add_message, record_token_usage, update_session,
)
from .vector_store import query_topic
from .mcp_client import mcp_manager

logger = logging.getLogger(APP_NAME)

# ─── Active model state (in-process, no restart needed) ──────────────────────
_active_model: str = DEFAULT_MODEL


def get_active_model() -> str:
    return _active_model


def set_active_model(model: str):
    global _active_model
    _active_model = model
    logger.info("Active model switched to: %s", model)


# ─── Prompt builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are VaultRAG, a helpful AI assistant that answers questions \
using the provided context from documents and external tools.

Rules:
- Base your answers primarily on the provided context.
- If the context doesn't contain enough information, say so honestly.
- Always cite the sources you use with [Source: filename, chunk N].
- Be concise and accurate.
- Format responses in Markdown when appropriate."""


def _build_prompt(
    query: str,
    context_chunks: List[Dict],
    mcp_contexts: List[Dict],
) -> str:
    parts = []

    if context_chunks:
        doc_parts = []
        for i, chunk in enumerate(context_chunks):
            doc_parts.append(
                f"[Doc {i+1} | {chunk['source']}, Chunk {chunk['chunk_index']+1} | Score {chunk['score']}]\n"
                f"{chunk['text']}"
            )
        parts.append("## Document Context\n\n" + "\n\n---\n\n".join(doc_parts))

    if mcp_contexts:
        mcp_parts = []
        for m in mcp_contexts:
            mcp_parts.append(
                f"[MCP: {m['server']} / {m['tool']}]\n{m['text']}"
            )
        parts.append("## External Tool Context\n\n" + "\n\n---\n\n".join(mcp_parts))

    if not parts:
        return f"No relevant context found.\n\nUser question: {query}"

    return "\n\n".join(parts) + f"\n\n---\n\nUser question: {query}"


# ─── Ollama connectivity ──────────────────────────────────────────────────────

async def check_ollama() -> Dict[str, Any]:
    """Ping Ollama and return {running, url, models, active_model}."""
    url = OLLAMA_BASE_URL
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3)
        ) as session:
            async with session.get(f"{url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return {
                        "running": True,
                        "url": url,
                        "models": models,
                        "active_model": get_active_model(),
                    }
    except Exception as e:
        logger.debug("Ollama check failed: %s", e)
    return {
        "running": False,
        "url": url,
        "models": [],
        "active_model": get_active_model(),
    }


# ─── Streaming chat ───────────────────────────────────────────────────────────

async def stream_chat(
    session_id: str,
    topic_id: str,
    model: str,
    query: str,
    chat_history: Optional[List[Dict]] = None,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted strings.

    event types: status | chunk | citation | mcp | meta | error | done
    """

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # Always use the resolved active model (may differ from session default)
    resolved_model = model or get_active_model()
    start_time = time.monotonic()

    # 1 – ChromaDB retrieval
    yield sse("status", {"stage": "retrieving", "message": "Searching knowledge base…"})
    try:
        hits = query_topic(topic_id, query, top_k=TOP_K_RESULTS)
    except Exception as e:
        yield sse("error", {"error": f"Retrieval failed: {e}"})
        return

    citations = [
        {
            "source": h["source"],
            "chunk_index": h["chunk_index"],
            "score": h["score"],
            "text": h["text"][:300],
        }
        for h in hits
    ]

    # 2 – MCP context (non-blocking; failures are silently skipped)
    mcp_contexts: List[Dict] = []
    if mcp_manager.any_connected():
        yield sse("status", {"stage": "mcp", "message": "Fetching external tool context…"})
        try:
            mcp_contexts = await mcp_manager.get_context(query)
            if mcp_contexts:
                yield sse("mcp", {"sources": [
                    {"server": m["server"], "tool": m["tool"]}
                    for m in mcp_contexts
                ]})
        except Exception as e:
            logger.warning("MCP context fetch error: %s", e)

    # 3 – Build Ollama messages
    yield sse("status", {"stage": "thinking", "message": "Generating response…"})
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-10:])
    messages.append({"role": "user", "content": _build_prompt(query, hits, mcp_contexts)})

    # 4 – Stream from Ollama
    full_response = ""
    response_tokens = 0

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as session:
            payload = {
                "model": resolved_model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.1},
            }
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    yield sse("error", {"error": f"Ollama error {resp.status}: {body[:200]}"})
                    return

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        response_tokens += 1
                        yield sse("chunk", {"text": token})

                    if data.get("done"):
                        break

    except aiohttp.ClientConnectorError:
        yield sse("error", {"error": "Cannot connect to Ollama. Is it running?"})
        return
    except Exception as e:
        yield sse("error", {"error": str(e)})
        return

    # 5 – Persist
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    prompt_tokens = sum(len(m["content"].split()) for m in messages)

    user_msg_id = str(uuid.uuid4())
    asst_msg_id = str(uuid.uuid4())
    usage_id = str(uuid.uuid4())

    try:
        add_message(user_msg_id, session_id, "user", query)
        add_message(asst_msg_id, session_id, "assistant", full_response, citations)
        record_token_usage(
            usage_id, session_id, asst_msg_id,
            resolved_model, prompt_tokens, response_tokens,
        )
        update_session(session_id, title=query[:60] if query else "New Chat")
    except Exception as e:
        logger.error("Failed to persist chat: %s", e)

    yield sse("citation", {"citations": citations})
    yield sse("meta", {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "elapsed_ms": elapsed_ms,
        "model": resolved_model,
    })
    yield sse("done", {})
