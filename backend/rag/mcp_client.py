"""
VaultRAG – MCP Client Manager

Manages connections to multiple MCP servers, exposes their tools,
and provides context retrieval for the RAG pipeline.

Architecture:
  - MCPServer      : config + connection state for one server
  - MCPClientManager : lifecycle (connect/disconnect/health) for all servers
  - get_mcp_context  : called by pipeline.py to fetch tool results for a query
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import APP_NAME

logger = logging.getLogger(APP_NAME)


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict = field(default_factory=dict)


@dataclass
class MCPServer:
    name: str
    url: str                          # e.g. "http://localhost:3000"
    enabled: bool = True
    connected: bool = False
    tools: List[MCPTool] = field(default_factory=list)
    last_ping_ms: Optional[float] = None
    last_error: str = ""
    connected_at: Optional[float] = None


# ─── Manager ─────────────────────────────────────────────────────────────────

class MCPClientManager:
    """
    Singleton that manages all MCP server connections.

    Usage:
        manager = MCPClientManager()
        await manager.connect_all()
        context = await manager.get_context("What is the traffic violation fine?")
        await manager.shutdown()
    """

    def __init__(self):
        self._servers: Dict[str, MCPServer] = {}
        self._lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None

    # ── Configuration ─────────────────────────────────────────────────────────

    def register(self, name: str, url: str, enabled: bool = True):
        """Register an MCP server. Call before connect_all()."""
        self._servers[name] = MCPServer(name=name, url=url.rstrip("/"), enabled=enabled)
        logger.info("MCP server registered: %s → %s", name, url)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect_all(self):
        """Attempt to connect to all registered, enabled servers."""
        tasks = [
            self._connect(s)
            for s in self._servers.values()
            if s.enabled
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Start background health-check loop
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info(
            "MCP init complete. Connected: %d / %d",
            sum(1 for s in self._servers.values() if s.connected),
            len(self._servers),
        )

    async def shutdown(self):
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("MCP manager shut down.")

    async def _connect(self, server: MCPServer):
        """Try to connect to one server and discover its tools."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                # MCP initialize handshake
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "VaultRAG", "version": "2.0"},
                    },
                }
                t0 = time.monotonic()
                async with session.post(
                    f"{server.url}/mcp", json=payload
                ) as resp:
                    if resp.status != 200:
                        raise ConnectionError(f"HTTP {resp.status}")
                    server.last_ping_ms = round((time.monotonic() - t0) * 1000, 1)

                # Discover tools
                tools_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                }
                async with session.post(
                    f"{server.url}/mcp", json=tools_payload
                ) as resp:
                    data = await resp.json()
                    raw_tools = data.get("result", {}).get("tools", [])
                    server.tools = [
                        MCPTool(
                            name=t.get("name", ""),
                            description=t.get("description", ""),
                            input_schema=t.get("inputSchema", {}),
                        )
                        for t in raw_tools
                    ]

            server.connected = True
            server.connected_at = time.time()
            server.last_error = ""
            logger.info(
                "MCP connected: %s (%d tools, %s ms)",
                server.name, len(server.tools), server.last_ping_ms,
            )
        except Exception as e:
            server.connected = False
            server.last_error = str(e)
            logger.warning("MCP connect failed [%s]: %s", server.name, e)

    async def _health_loop(self):
        """Ping all servers every 30 s; reconnect if disconnected."""
        while True:
            await asyncio.sleep(30)
            for server in list(self._servers.values()):
                if not server.enabled:
                    continue
                try:
                    await self._ping(server)
                except Exception:
                    pass
                if not server.connected:
                    # Try to reconnect silently
                    await self._connect(server)

    async def _ping(self, server: MCPServer):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as session:
                payload = {"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}}
                t0 = time.monotonic()
                async with session.post(f"{server.url}/mcp", json=payload) as resp:
                    server.last_ping_ms = round((time.monotonic() - t0) * 1000, 1)
                    if resp.status == 200:
                        server.connected = True
                        server.last_error = ""
        except Exception as e:
            server.connected = False
            server.last_error = str(e)

    # ── Tool calling ──────────────────────────────────────────────────────────

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict
    ) -> Optional[str]:
        """
        Call a specific tool on a named MCP server.
        Returns the text result or None on failure.
        """
        server = self._servers.get(server_name)
        if not server or not server.connected:
            return None
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }
                async with session.post(f"{server.url}/mcp", json=payload) as resp:
                    data = await resp.json()
                    result = data.get("result", {})
                    # MCP spec: result.content is a list of {type, text} blocks
                    content = result.get("content", [])
                    texts = [
                        c.get("text", "")
                        for c in content
                        if c.get("type") == "text"
                    ]
                    return "\n".join(texts) if texts else json.dumps(result)
        except Exception as e:
            logger.error("MCP tool call failed [%s/%s]: %s", server_name, tool_name, e)
            return None

    async def get_context(self, query: str) -> List[Dict]:
        """
        Ask all connected servers for context relevant to *query*.
        Uses each server's first available search/query tool.
        Returns a list of {server, tool, text} dicts.
        """
        results = []
        for server in self._servers.values():
            if not server.connected or not server.tools:
                continue
            # Find the best tool (prefer names containing search/query/retrieve)
            tool = _pick_search_tool(server.tools)
            if not tool:
                continue
            try:
                text = await self.call_tool(
                    server.name, tool.name, {"query": query}
                )
                if text:
                    results.append(
                        {"server": server.name, "tool": tool.name, "text": text}
                    )
            except Exception as e:
                logger.warning("MCP context fetch failed [%s]: %s", server.name, e)
        return results

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> List[Dict]:
        """Return serialisable status for all registered servers."""
        out = []
        for s in self._servers.values():
            out.append(
                {
                    "name": s.name,
                    "url": s.url,
                    "enabled": s.enabled,
                    "connected": s.connected,
                    "tools": [
                        {"name": t.name, "description": t.description}
                        for t in s.tools
                    ],
                    "tool_count": len(s.tools),
                    "last_ping_ms": s.last_ping_ms,
                    "last_error": s.last_error,
                    "connected_at": s.connected_at,
                }
            )
        return out

    def any_connected(self) -> bool:
        return any(s.connected for s in self._servers.values())


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pick_search_tool(tools: List[MCPTool]) -> Optional[MCPTool]:
    """Return the most relevant tool for context retrieval."""
    priority_keywords = ["search", "query", "retrieve", "find", "lookup", "get"]
    for kw in priority_keywords:
        for t in tools:
            if kw in t.name.lower() or kw in t.description.lower():
                return t
    return tools[0] if tools else None


# ─── Global singleton ─────────────────────────────────────────────────────────

mcp_manager = MCPClientManager()
