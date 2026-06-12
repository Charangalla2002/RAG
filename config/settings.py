"""
VaultRAG - Configuration Settings
Centralizes all environment-specific settings, paths, and model defaults.
"""

import os
from pathlib import Path

# ─── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_BUFFER_DIR = DATA_DIR / "upload_buffer"
TOPICS_DIR = DATA_DIR / "topics"
DB_PATH = DATA_DIR / "vaultrag.db"

# Ensure directories exist
for _dir in [DATA_DIR, UPLOAD_BUFFER_DIR, TOPICS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")   # "cpu" | "cuda" | "mps"

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b")

# ─── RAG Retrieval ───────────────────────────────────────────────────────────
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# ─── OCR ─────────────────────────────────────────────────────────────────────
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")   # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
APP_NAME = "vaultrag"
