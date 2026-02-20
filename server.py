"""
agentmem.server — MCP server for agent memory.

6 tools:
1. recall(query) — hybrid search (keywords + semantics)
2. remember(content) — store a new memory
3. save_state(state) — emergency save before context compression
4. today() — what happened today
5. forget(memory_id) — archive a memory
6. stats() — memory statistics

Run: python -m agentmem.server [--db memory.db] [--port 8422]
"""
import json
import os
import sys
import argparse
from pathlib import Path

# MCP protocol via stdin/stdout (stdio transport)
# This is the simplest, most universal MCP transport

def _get_store():
    """Lazy singleton for MemoryStore."""
    global _store
    if "_store" not in globals() or _store is None:
        from .core import MemoryStore
        from .embeddings import get_embedding_model

        db_path = os.environ.get("AGENTMEM_DB", "memory.db")
        backend = os.environ.get("AGENTMEM_BACKEND", "auto")

        embed_model = get_embedding_model(backend)

        _store = MemoryStore(db_path=db_path, embedding_dim=embed_model.dim or 256)
        if embed_model.dim > 0:
            _store.set_embed_fn(embed_model)  # pass model object for batch support

    return _store


# ================================================================
# MCP Protocol Implementation (stdio JSON-RPC)
# ================================================================

TOOLS = [
    {
        "name": "recall",
        "description": (
            "Search agent memory using hybrid keyword + semantic search. "
            "Use this to find information from past sessions, stored facts, "
            "learned knowledge, or working state. "
            "Returns the most relevant memories ranked by relevance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Can be a question, keywords, or a topic.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                    "default": 5,
                },
                "tier": {
                    "type": "string",
                    "description": "Filter by tier: core, learned, episodic, working. Omit for all.",
                    "enum": ["core", "learned", "episodic", "working"],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Store a new memory. Use this to save important facts, learnings, "
            "decisions, or any information that should persist across sessions. "
            "Automatically deduplicates by content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store.",
                },
                "tier": {
                    "type": "string",
                    "description": (
                        "Memory tier: "
                        "core (permanent facts), "
                        "learned (discovered knowledge), "
                        "episodic (what happened), "
                        "working (current task state, auto-expires)."
                    ),
                    "enum": ["core", "learned", "episodic", "working"],
                    "default": "learned",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization.",
                },
                "source": {
                    "type": "string",
                    "description": "Where this memory came from (file, tool, conversation).",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "save_state",
        "description": (
            "Emergency save of current working context. "
            "Call this before context compression to preserve your current state. "
            "Replaces any previous working state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": (
                        "Current working state to save. Include: "
                        "what you're doing, key decisions made, "
                        "what's left to do, any blockers."
                    ),
                },
            },
            "required": ["state"],
        },
    },
    {
        "name": "today",
        "description": (
            "Get all memories created today. "
            "Use this to see what happened during the current day."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "forget",
        "description": (
            "Archive a memory (soft delete). "
            "The memory is hidden from search but not permanently deleted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "ID of the memory to archive.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "stats",
        "description": (
            "Get memory statistics: total count, breakdown by tier, "
            "database size, and latest activity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def handle_request(request: dict) -> dict:
    """Handle a single MCP JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "agentmem",
                    "version": "0.1.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None  # No response needed for notifications

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        return _call_tool(req_id, tool_name, args)

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def _call_tool(req_id, tool_name: str, args: dict) -> dict:
    """Execute a tool and return MCP response."""
    store = _get_store()

    try:
        if tool_name == "recall":
            result = store.recall(
                query=args["query"],
                limit=args.get("limit", 5),
                tier=args.get("tier"),
            )
            text = _format_recall_results(result)

        elif tool_name == "remember":
            result = store.remember(
                content=args["content"],
                tier=args.get("tier", "learned"),
                tags=args.get("tags", []),
                source=args.get("source", ""),
            )
            text = json.dumps(result, indent=2)

        elif tool_name == "save_state":
            result = store.save_state(state=args["state"])
            text = json.dumps(result, indent=2)

        elif tool_name == "today":
            result = store.today()
            text = _format_today_results(result)

        elif tool_name == "forget":
            result = store.forget(memory_id=args["memory_id"])
            text = json.dumps(result, indent=2)

        elif tool_name == "stats":
            result = store.stats()
            text = json.dumps(result, indent=2)

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text}],
            },
        }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            },
        }


def _format_recall_results(results: list[dict]) -> str:
    """Format recall results as readable text."""
    if not results:
        return "No memories found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] (score: {r['score']:.2f}, tier: {r['tier']}, method: {r['method']})")
        # Truncate long content
        content = r["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(content)
        if r["source"]:
            lines.append(f"  — source: {r['source']}")
        lines.append("")

    return "\n".join(lines)


def _format_today_results(results: list[dict]) -> str:
    """Format today's memories."""
    if not results:
        return "No memories from today."

    import datetime

    lines = [f"Today's memories ({len(results)} total):\n"]
    current_tier = None
    for r in results:
        if r["tier"] != current_tier:
            current_tier = r["tier"]
            lines.append(f"\n## {current_tier.title()}")

        ts = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%H:%M")
        content_preview = r["content"][:200]
        if len(r["content"]) > 200:
            content_preview += "..."
        lines.append(f"  [{ts}] {content_preview}")

    return "\n".join(lines)


def run_stdio():
    """Run MCP server on stdio (stdin/stdout JSON-RPC)."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="agentmem — Lightweight agent memory MCP server")
    parser.add_argument("--db", default="memory.db", help="Database file path")
    parser.add_argument("--backend", default="auto", choices=["auto", "model2vec", "null"],
                        help="Embedding backend")
    args = parser.parse_args()

    os.environ["AGENTMEM_DB"] = args.db
    os.environ["AGENTMEM_BACKEND"] = args.backend

    # Pre-initialize store (loads model)
    store = _get_store()
    stats = store.stats()
    print(f"agentmem ready: {stats['total_memories']} memories, {stats['db_size_human']}",
          file=sys.stderr)

    run_stdio()


if __name__ == "__main__":
    main()
