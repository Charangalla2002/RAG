"""MCP API – server registration, status, tool calling."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from config.settings import APP_NAME
from backend.rag.mcp_client import mcp_manager

logger = logging.getLogger(APP_NAME)
router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class RegisterRequest(BaseModel):
    name: str
    url: str
    enabled: bool = True


class ToolCallRequest(BaseModel):
    server: str
    tool: str
    arguments: Dict[str, Any] = {}


@router.get("/status")
def status():
    """Return connection status for all registered MCP servers."""
    return {
        "servers": mcp_manager.status(),
        "any_connected": mcp_manager.any_connected(),
        "total": len(mcp_manager._servers),
        "connected": sum(1 for s in mcp_manager._servers.values() if s.connected),
    }


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    """Register and immediately connect to a new MCP server."""
    mcp_manager.register(body.name, body.url, body.enabled)
    server = mcp_manager._servers[body.name]
    await mcp_manager._connect(server)
    return {"registered": True, "connected": server.connected, "error": server.last_error}


@router.post("/reconnect/{server_name}")
async def reconnect(server_name: str):
    """Force-reconnect a specific server."""
    server = mcp_manager._servers.get(server_name)
    if not server:
        raise HTTPException(404, f"Server '{server_name}' not registered")
    await mcp_manager._connect(server)
    return {"connected": server.connected, "error": server.last_error}


@router.post("/tools/call")
async def call_tool(body: ToolCallRequest):
    """Call a specific tool on a connected MCP server."""
    result = await mcp_manager.call_tool(body.server, body.tool, body.arguments)
    if result is None:
        raise HTTPException(502, "Tool call failed or server not connected")
    return {"result": result}


@router.delete("/servers/{server_name}", status_code=204)
def remove_server(server_name: str):
    """Unregister an MCP server."""
    if server_name not in mcp_manager._servers:
        raise HTTPException(404, f"Server '{server_name}' not found")
    del mcp_manager._servers[server_name]
