"""
VaultRAG – SQLite database initialisation and helper queries.

Tables
------
topics          – knowledge domains
documents       – ingested file metadata
chat_sessions   – conversation threads
messages        – individual turns inside a session
token_usage     – per-message token accounting
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime

from config.settings import DB_PATH, APP_NAME

logger = logging.getLogger(APP_NAME)


# ─── Connection helpers ───────────────────────────────────────────────────────

@contextmanager
def get_db():
    """Yield a WAL-mode SQLite connection; always commit or rollback."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS topics (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    color       TEXT DEFAULT '#7c3aed',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    topic_id      TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    filename      TEXT NOT NULL,
    original_name TEXT NOT NULL,
    file_size     INTEGER DEFAULT 0,
    file_type     TEXT DEFAULT '',
    chunk_count   INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',   -- pending | processing | completed | failed
    error_msg     TEXT DEFAULT '',
    uploaded_at   TEXT NOT NULL,
    processed_at  TEXT
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id         TEXT PRIMARY KEY,
    topic_id   TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    title      TEXT NOT NULL DEFAULT 'New Chat',
    model      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,   -- user | assistant
    content     TEXT NOT NULL,
    citations   TEXT DEFAULT '[]',   -- JSON array
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS token_usage (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES chat_sessions(id) ON DELETE SET NULL,
    message_id      TEXT REFERENCES messages(id) ON DELETE SET NULL,
    model           TEXT NOT NULL DEFAULT '',
    prompt_tokens   INTEGER DEFAULT 0,
    response_tokens INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_topic   ON documents(topic_id);
CREATE INDEX IF NOT EXISTS idx_messages_session  ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_token_session     ON token_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_token_recorded    ON token_usage(recorded_at);
"""


def init_db():
    """Create tables and seed a default topic if the DB is brand-new."""
    logger.info("Initializing SQLite database...")
    with get_db() as conn:
        conn.executescript(DDL)

        # Seed "General" topic
        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            INSERT OR IGNORE INTO topics (id, name, description, color, created_at, updated_at)
            VALUES ('general', 'General', 'Default knowledge domain', '#7c3aed', ?, ?)
            """,
            (now, now),
        )
    logger.info("SQLite database initialized successfully.")


# ─── Topic helpers ────────────────────────────────────────────────────────────

def get_all_topics():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT t.*,
                   COUNT(DISTINCT d.id) AS doc_count,
                   COALESCE(SUM(d.chunk_count), 0) AS total_chunks
            FROM topics t
            LEFT JOIN documents d ON d.topic_id = t.id
            GROUP BY t.id
            ORDER BY t.created_at
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_topic(topic_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        return dict(row) if row else None


def create_topic(topic_id: str, name: str, description: str = "", color: str = "#7c3aed"):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO topics (id, name, description, color, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (topic_id, name, description, color, now, now),
        )
    return get_topic(topic_id)


def update_topic(topic_id: str, **kwargs):
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [topic_id]
    with get_db() as conn:
        conn.execute(f"UPDATE topics SET {sets} WHERE id = ?", vals)
    return get_topic(topic_id)


def delete_topic(topic_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))


# ─── Document helpers ─────────────────────────────────────────────────────────

def get_documents(topic_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE topic_id = ? ORDER BY uploaded_at DESC",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


def create_document(doc_id, topic_id, filename, original_name, file_size=0, file_type=""):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, topic_id, filename, original_name, file_size, file_type,
                                   chunk_count, status, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, 'pending', ?)
            """,
            (doc_id, topic_id, filename, original_name, file_size, file_type, now),
        )
    return get_document(doc_id)


def update_document_status(doc_id: str, status: str, chunk_count: int = 0, error_msg: str = ""):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE documents
            SET status = ?, chunk_count = ?, error_msg = ?, processed_at = ?
            WHERE id = ?
            """,
            (status, chunk_count, error_msg, now if status in ("completed", "failed") else None, doc_id),
        )


def delete_document(doc_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# ─── Chat helpers ─────────────────────────────────────────────────────────────

def get_sessions(topic_id: str):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT cs.*,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = cs.id) AS message_count,
                   (SELECT COALESCE(SUM(total_tokens),0) FROM token_usage tu WHERE tu.session_id = cs.id) AS total_tokens
            FROM chat_sessions cs
            WHERE cs.topic_id = ?
            ORDER BY cs.updated_at DESC
            """,
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_sessions():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT cs.*,
                   t.name AS topic_name,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = cs.id) AS message_count
            FROM chat_sessions cs
            LEFT JOIN topics t ON t.id = cs.topic_id
            ORDER BY cs.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_session(session_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def create_session(session_id, topic_id, model, title="New Chat"):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, topic_id, title, model, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (session_id, topic_id, title, model, now, now),
        )
    return get_session(session_id)


def update_session(session_id: str, **kwargs):
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    with get_db() as conn:
        conn.execute(f"UPDATE chat_sessions SET {sets} WHERE id = ?", vals)
    return get_session(session_id)


def delete_session(session_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


def get_messages(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_message(msg_id, session_id, role, content, citations=None):
    import json
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, citations, created_at) VALUES (?,?,?,?,?,?)",
            (msg_id, session_id, role, content, json.dumps(citations or []), now),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )


def record_token_usage(usage_id, session_id, message_id, model, prompt_tokens, response_tokens):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO token_usage (id, session_id, message_id, model,
                                     prompt_tokens, response_tokens, total_tokens, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (usage_id, session_id, message_id, model,
             prompt_tokens, response_tokens, prompt_tokens + response_tokens, now),
        )


# ─── Analytics ────────────────────────────────────────────────────────────────

def get_token_analytics(period: str = "today"):
    """Return aggregated token stats for 'today', 'week', or 'all'."""
    import datetime as dt
    now = dt.datetime.utcnow()
    if period == "today":
        since = now.replace(hour=0, minute=0, second=0).isoformat()
    elif period == "week":
        since = (now - dt.timedelta(days=7)).isoformat()
    else:
        since = "1970-01-01T00:00:00"

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(prompt_tokens),0)   AS prompt_tokens,
                   COALESCE(SUM(response_tokens),0) AS response_tokens,
                   COALESCE(SUM(total_tokens),0)    AS total_tokens,
                   COUNT(*)                         AS requests
            FROM token_usage
            WHERE recorded_at >= ?
            """,
            (since,),
        ).fetchone()
        return dict(row)
