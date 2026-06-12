"""VaultRAG – FastAPI Application Entry Point v2"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config.settings import APP_NAME, LOG_LEVEL, BASE_DIR
from backend.database.db import init_db
from backend.routes import topics_router, documents_router, chat_router, metrics_router
from backend.routes.mcp import router as mcp_router
from backend.rag.mcp_client import mcp_manager

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(APP_NAME)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VaultRAG",
    description="Secure offline Retrieval-Augmented Generation server",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(topics_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(mcp_router)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "VaultRAG", "version": "2.0.0"}


# ─── Static / SPA ─────────────────────────────────────────────────────────────
_frontend = BASE_DIR / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")
    logger.info("Frontend mounted from: %s", _frontend)

    @app.get("/")
    def root():
        return FileResponse(str(_frontend / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        candidate = _frontend / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_frontend / "index.html"))


# ─── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Initializing SQLite database…")
    init_db()
    logger.info("Initializing MCP clients…")
    await mcp_manager.connect_all()
    logger.info("VaultRAG v2.0.0 ready.")


@app.on_event("shutdown")
async def shutdown():
    await mcp_manager.shutdown()
    logger.info("VaultRAG shutdown complete.")
