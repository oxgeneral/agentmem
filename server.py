"""
agentmem.server — MCP server for agent memory.

16 tools:
1. recall(query) — hybrid search (keywords + semantics)
2. remember(content) — store a new memory
3. save_state(state) — emergency save before context compression
4. today() — what happened today
5. forget(memory_id) — archive a memory
6. stats() — memory statistics
7. compact() — archive low-value memories to reduce noise
8. unarchive(memory_id) — restore an archived memory
9. update_memory(old_id, new_content) — replace a memory, preserving version chain
10. history(memory_id) — get version history of a memory
11. consolidate() — find and merge semantically similar memories
12. related(entity) — find memories connected to the same entity
13. entities() — list all known entities
14. get_procedures() — get all procedural memories formatted for system prompt
15. add_procedure(rule) — add a behavioral rule (procedural memory)
16. process_conversation(messages) — auto-extract facts/decisions/todos from chat history

Run: python -m agentmem.server [--db memory.db] [--port 8422]
"""
import json
import os
import sys
import argparse
from pathlib import Path

from .core import (
    MemoryNotFoundError, InvalidTierError, EmbeddingError, AgentMemError,
)

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
                    "description": "Filter by tier: core, learned, episodic, working, procedural. Omit for all.",
                    "enum": ["core", "learned", "episodic", "working", "procedural"],
                },
                "recency_weight": {
                    "type": "number",
                    "description": "How much recency affects scoring (0.0-1.0). Default 0.1.",
                    "default": 0.1,
                },
                "namespace": {
                    "type": "string",
                    "description": (
                        "Filter by namespace (prefix match). "
                        "E.g. 'agent' matches 'agent', 'agent/alice', 'agent/bob'. "
                        "Omit to search all namespaces."
                    ),
                },
                "current_only": {
                    "type": "boolean",
                    "description": (
                        "If true (default), exclude memories that have been superseded "
                        "by a newer version. Set to false to include all versions."
                    ),
                    "default": True,
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
                        "working (current task state, auto-expires), "
                        "procedural (behavioral rules for system prompt)."
                    ),
                    "enum": ["core", "learned", "episodic", "working", "procedural"],
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
                "namespace": {
                    "type": "string",
                    "description": "Namespace for memory isolation (e.g. 'agent/alice'). Default: global.",
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
                "namespace": {
                    "type": "string",
                    "description": "Namespace for state isolation. Default: global.",
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
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Filter by namespace (prefix match). Omit for all.",
                },
            },
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
                "namespace": {
                    "type": "string",
                    "description": "Safety guard: only forget if memory belongs to this namespace.",
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
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Filter stats by namespace (prefix match). Omit for all.",
                },
            },
        },
    },
    {
        "name": "compact",
        "description": (
            "Archive low-value memories to reduce noise in search results. "
            "Archives memories older than max_age_days with access_count <= min_access. "
            "Core and procedural tier memories are never auto-archived. "
            "Use dry_run=true to preview how many would be archived."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_age_days": {
                    "type": "integer",
                    "description": "Archive memories older than this many days (default 90).",
                    "default": 90,
                },
                "min_access": {
                    "type": "integer",
                    "description": "Archive memories accessed this many times or less (default 0 = never accessed).",
                    "default": 0,
                },
                "tier": {
                    "type": "string",
                    "description": "Only compact this tier. Omit for all tiers except core and procedural.",
                    "enum": ["learned", "episodic", "working"],
                },
                "namespace": {
                    "type": "string",
                    "description": "Only compact this namespace (prefix match).",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, return count without archiving (default false).",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "unarchive",
        "description": (
            "Restore an archived memory, making it active and searchable again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "ID of the archived memory to restore.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "update_memory",
        "description": (
            "Replace a memory with updated content, preserving the version chain. "
            "The old memory is archived and linked via supersedes. "
            "Use this when a fact changes (e.g. IP address, config value). "
            "Tier, tags, namespace default to the old memory's values if not specified."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "old_id": {
                    "type": "integer",
                    "description": "ID of the memory to replace.",
                },
                "new_content": {
                    "type": "string",
                    "description": "The updated content.",
                },
                "tier": {
                    "type": "string",
                    "description": "New tier (defaults to old memory's tier).",
                    "enum": ["core", "learned", "episodic", "working", "procedural"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New tags (defaults to old memory's tags).",
                },
                "namespace": {
                    "type": "string",
                    "description": "New namespace (defaults to old memory's namespace).",
                },
            },
            "required": ["old_id", "new_content"],
        },
    },
    {
        "name": "history",
        "description": (
            "Get the version history of a memory. "
            "Follows the supersedes chain to show all previous versions. "
            "Returns newest first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "ID of the memory to get history for.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "consolidate",
        "description": (
            "Find and merge semantically similar memories (near-duplicates). "
            "Groups memories with cosine similarity above the threshold, "
            "keeps the longest/newest in each group, and archives the rest. "
            "Requires an embedding function to be set. "
            "Use dry_run=true to preview groups without modifying anything."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "similarity_threshold": {
                    "type": "number",
                    "description": "Min cosine similarity to consider as duplicates (default 0.85, range 0.0-1.0).",
                    "default": 0.85,
                },
                "namespace": {
                    "type": "string",
                    "description": "Only consolidate within this namespace (prefix match).",
                },
                "tier": {
                    "type": "string",
                    "description": "Only consolidate this tier.",
                    "enum": ["core", "learned", "episodic", "working", "procedural"],
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, return groups but don't merge (default false).",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "related",
        "description": (
            "Find memories connected to the same entity. "
            "Entities are auto-extracted from memory content: "
            "@mentions, URLs, IPs, ports, file paths, env vars, money, etc. "
            "Use this to discover all memories referencing a specific entity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity name to search for (e.g. '@username', '10.0.0.1', 'OPENAI_API_KEY').",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Filter by entity type: mention, url, email, hashtag, ip, port, path, money, number_unit, env_var.",
                    "enum": ["mention", "url", "email", "hashtag", "ip", "port", "path", "money", "number_unit", "env_var"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10).",
                    "default": 10,
                },
                "namespace": {
                    "type": "string",
                    "description": "Filter by namespace (prefix match).",
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "entities",
        "description": (
            "List all known entities extracted from memories. "
            "Shows entity names, types, and how many memories reference each. "
            "Useful for discovering what entities are stored in memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "description": "Filter by entity type: mention, url, email, hashtag, ip, port, path, money, number_unit, env_var.",
                    "enum": ["mention", "url", "email", "hashtag", "ip", "port", "path", "money", "number_unit", "env_var"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 50).",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "get_procedures",
        "description": (
            "Get all active procedural memories (behavioral rules) formatted for system prompt injection. "
            "Returns a markdown-formatted string with all rules as a bullet list. "
            "If no procedural memories exist, returns empty string. "
            "Use this at the start of a session to load agent behavioral rules."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Filter by namespace (prefix match). Omit for all.",
                },
            },
        },
    },
    {
        "name": "add_procedure",
        "description": (
            "Add a behavioral rule (procedural memory). "
            "Procedural memories are never auto-expired or auto-compacted. "
            "They are meant to be injected into the agent's system prompt to modify behavior. "
            "Examples: 'Always respond in bullet points', 'Never expose API keys'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {
                    "type": "string",
                    "description": "The behavioral rule text.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace for rule isolation (e.g. 'agent/alice'). Default: global.",
                },
            },
            "required": ["rule"],
        },
    },
    {
        "name": "process_conversation",
        "description": (
            "Automatically extract and store memories from a conversation history. "
            "Scans messages for facts, decisions, preferences, TODOs, config values, "
            "learnings, and important notes using regex heuristics (no LLM). "
            "Stores extracted memories with appropriate tiers and tags. "
            "Accepts standard OpenAI/Anthropic message format."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Message role: user, assistant, or system.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Message text content.",
                            },
                        },
                        "required": ["role", "content"],
                    },
                    "description": "List of conversation messages in OpenAI/Anthropic format.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace to store extracted memories in. Default: global.",
                },
                "source": {
                    "type": "string",
                    "description": "Source label for extracted memories (default 'conversation').",
                },
            },
            "required": ["messages"],
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
                recency_weight=args.get("recency_weight"),
                namespace=args.get("namespace"),
                current_only=args.get("current_only", True),
            )
            text = _format_recall_results(result)

        elif tool_name == "remember":
            result = store.remember(
                content=args["content"],
                tier=args.get("tier", "learned"),
                tags=args.get("tags", []),
                source=args.get("source", ""),
                namespace=args.get("namespace", ""),
            )
            text = json.dumps(result, indent=2)

        elif tool_name == "save_state":
            result = store.save_state(
                state=args["state"],
                namespace=args.get("namespace", ""),
            )
            text = json.dumps(result, indent=2)

        elif tool_name == "today":
            result = store.today(namespace=args.get("namespace"))
            text = _format_today_results(result)

        elif tool_name == "forget":
            result = store.forget(
                memory_id=args["memory_id"],
                namespace=args.get("namespace"),
            )
            text = json.dumps(result, indent=2)

        elif tool_name == "stats":
            result = store.stats(namespace=args.get("namespace"))
            text = json.dumps(result, indent=2)

        elif tool_name == "compact":
            result = store.compact(
                max_age_days=args.get("max_age_days", 90),
                min_access=args.get("min_access", 0),
                tier=args.get("tier"),
                namespace=args.get("namespace"),
                dry_run=args.get("dry_run", False),
            )
            text = json.dumps(result, indent=2)

        elif tool_name == "unarchive":
            result = store.unarchive(memory_id=args["memory_id"])
            text = json.dumps(result, indent=2)

        elif tool_name == "update_memory":
            result = store.update_memory(
                old_id=args["old_id"],
                new_content=args["new_content"],
                tier=args.get("tier"),
                tags=args.get("tags"),
                namespace=args.get("namespace"),
            )
            text = json.dumps(result, indent=2)

        elif tool_name == "history":
            result = store.history(memory_id=args["memory_id"])
            text = _format_history_results(result)

        elif tool_name == "consolidate":
            result = store.consolidate(
                similarity_threshold=args.get("similarity_threshold", 0.85),
                namespace=args.get("namespace"),
                tier=args.get("tier"),
                dry_run=args.get("dry_run", False),
            )
            text = _format_consolidate_results(result)

        elif tool_name == "related":
            result = store.related(
                entity=args["entity"],
                entity_type=args.get("entity_type"),
                limit=args.get("limit", 10),
                namespace=args.get("namespace"),
            )
            text = _format_related_results(result, args["entity"])

        elif tool_name == "entities":
            result = store.entities(
                entity_type=args.get("entity_type"),
                limit=args.get("limit", 50),
            )
            text = _format_entities_results(result)

        elif tool_name == "get_procedures":
            text = store.get_procedures(
                namespace=args.get("namespace"),
            )
            if not text:
                text = "(no procedural memories)"

        elif tool_name == "add_procedure":
            result = store.add_procedure(
                rule=args["rule"],
                tags=args.get("tags"),
                namespace=args.get("namespace", ""),
            )
            text = json.dumps(result, indent=2)

        elif tool_name == "process_conversation":
            result = store.process_conversation(
                messages=args["messages"],
                namespace=args.get("namespace", ""),
                source=args.get("source", "conversation"),
            )
            text = _format_process_results(result)

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

    except MemoryNotFoundError as e:
        hint = "Use the recall tool to search for memory IDs."
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {e}\nHint: {hint}"}],
                "isError": True,
            },
        }
    except InvalidTierError as e:
        hint = "Valid tiers: core, learned, episodic, working, procedural."
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {e}\nHint: {hint}"}],
                "isError": True,
            },
        }
    except EmbeddingError as e:
        hint = "Check that the embedding backend is configured correctly."
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {e}\nHint: {hint}"}],
                "isError": True,
            },
        }
    except AgentMemError as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            },
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}],
                "isError": True,
            },
        }


def _format_recall_results(results: list[dict]) -> str:
    """Format recall results as readable text."""
    if not results:
        return "No memories found."

    lines = []
    for i, r in enumerate(results, 1):
        imp_str = f", importance: {r['importance']:.2f}" if "importance" in r else ""
        lines.append(f"[{i}] (score: {r['score']:.2f}, tier: {r['tier']}, method: {r['method']}{imp_str})")
        # Truncate long content
        content = r["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(content)
        if r["source"]:
            lines.append(f"  — source: {r['source']}")
        lines.append("")

    return "\n".join(lines)


def _format_history_results(results: list[dict]) -> str:
    """Format version history as readable text."""
    if not results:
        return "No history found."

    import datetime

    lines = [f"Version history ({len(results)} versions, newest first):\n"]
    for i, r in enumerate(results):
        ts = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        status = " [archived]" if r["archived"] else " [current]"
        lines.append(f"  [{i+1}] id={r['id']} {ts}{status}")
        content = r["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"      {content}")
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


def _format_consolidate_results(result: dict) -> str:
    """Format consolidation results as readable text."""
    if result["groups"] == 0:
        return "No duplicate groups found."

    prefix = "Dry run: " if result["dry_run"] else ""
    lines = [f"{prefix}{result['groups']} group(s) found, {result['archived']} memories {'would be ' if result['dry_run'] else ''}archived.\n"]

    for i, detail in enumerate(result["details"], 1):
        lines.append(f"Group {i}: kept id={detail['kept']}, archived {len(detail['archived_ids'])} memories")
        for preview in detail["contents_preview"]:
            lines.append(f"  - {preview}")
        lines.append("")

    return "\n".join(lines)


def _format_related_results(results: list[dict], entity: str) -> str:
    """Format related memories as readable text."""
    if not results:
        return f"No memories found related to '{entity}'."

    lines = [f"Memories related to '{entity}' ({len(results)} found):\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] id={r['id']} tier={r['tier']} entity={r['entity_name']} ({r['entity_type']})")
        content = r["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"  {content}")
        if r["source"]:
            lines.append(f"  — source: {r['source']}")
        lines.append("")

    return "\n".join(lines)


def _format_entities_results(results: list[dict]) -> str:
    """Format entities list as readable text."""
    if not results:
        return "No entities found."

    lines = [f"Known entities ({len(results)}):\n"]
    for r in results:
        lines.append(f"  {r['name']} ({r['type']}) — {r['memory_count']} memories")

    return "\n".join(lines)


def _format_process_results(result: dict) -> str:
    """Format process_conversation results as readable text."""
    if result["extracted"] == 0:
        return "No extractable patterns found in the conversation."

    lines = [f"Extracted {result['extracted']} memories from conversation:\n"]
    for etype, count in sorted(result["by_type"].items()):
        lines.append(f"  {etype}: {count}")
    if result["memories"]:
        lines.append(f"\nStored memory IDs: {result['memories']}")
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
