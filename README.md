# agentmem

mcp-name: io.github.oxgeneral/agentmem

Lightweight persistent memory for AI agents. One SQLite file. Hybrid search (keywords + semantics). Zero to 12MB install.

No PyTorch. No cloud. No server. Just memory.

## Why

Every AI agent session starts from zero. Context windows compress, conversations end, memory vanishes. `agentmem` gives agents persistent memory that survives across sessions — in a single SQLite file.

- **Hybrid search**: FTS5 full-text keywords + vector semantic search, fused with adaptive ranking
- **4 operational modes**: from zero dependencies (stdlib only) to best quality (12MB)
- **16 MCP tools**: recall, remember, save_state, compact, consolidate, entities, and more
- **HTTP REST API**: 14 endpoints, zero-dependency server, CORS-ready
- **5 memory tiers**: core, learned, episodic, working (auto-expires), procedural (behavioral rules)
- **Namespaces**: multi-user, multi-agent memory isolation
- **Temporal versioning**: fact evolution chains with supersedes tracking
- **Entity extraction**: auto-extracts @mentions, URLs, IPs, env vars, money amounts
- **Conversation extraction**: auto-extracts facts, decisions, TODOs from chat history
- **Importance scoring**: auto-scores memories by tier, length, specificity, structure
- **Memory consolidation**: finds and merges near-duplicate memories
- **Recency boost**: newer memories rank higher with configurable decay
- **Multilingual**: Russian keywords via FTS5, English semantics via embeddings
- **Fast**: <1ms/query hybrid search, <5ms cold start, <0.2ms/chunk import

## Install

```bash
# Best quality (sqlite-vec + model2vec, 12MB total)
pip install agentmem[all]

# Minimal (sqlite-vec + hash embeddings, 151KB)
pip install agentmem

# Zero dependencies (pure Python, stdlib only)
pip install agentmem --no-deps
```

## Quick Start

### Python API

```python
from agentmem import MemoryStore, get_embedding_model

# Auto-selects best available backend
embed = get_embedding_model()
store = MemoryStore("memory.db", embedding_dim=embed.dim)
store.set_embed_fn(embed)

# Store memories with namespaces
store.remember("Server costs $50/month", tier="core", namespace="infra")
store.remember("API returns 403 without auth", tier="learned", namespace="api")
store.remember("Deployed v2.1 at 15:30", tier="episodic")

# Search — hybrid keyword + semantic, with recency boost
results = store.recall("server costs", recency_weight=0.15)

# Namespace isolation
results = store.recall("server", namespace="infra")

# Save working state before context compression
store.save_state("Working on auth fix, step 3/5, blocked by CORS")

# Add behavioral rules (procedural memory)
store.add_procedure("Always use HTTPS in production")
store.add_procedure("Never expose debug endpoints")
rules = store.get_procedures()  # → formatted for system prompt

# Update facts with version chain
store.update_memory(old_id=1, new_content="Server costs $75/month")
history = store.history(memory_id=2)  # → trace fact evolution

# Find related memories by entity
related = store.related("10.0.0.1")  # → all memories mentioning this IP
entities = store.entities(entity_type="ip")  # → list all known IPs

# Auto-extract from conversations
messages = [
    {"role": "user", "content": "Set API_KEY to sk-abc123. Always validate input."},
    {"role": "assistant", "content": "Noted. I decided to use pydantic for validation."},
]
result = store.process_conversation(messages, namespace="project")
# → extracts config, preferences, decisions automatically

# Maintenance
store.compact(max_age_days=90)  # archive old low-value memories
store.consolidate(similarity_threshold=0.85)  # merge near-duplicates

# Import markdown files
store.import_markdown("MEMORY.md", tier="core")
```

### CLI

```bash
# Initialize database
agentmem init --db memory.db

# Import markdown files
agentmem import MEMORY.md --tier core -n my-agent
agentmem import-dir ./daily-logs/ --tier episodic

# Search with namespace filter
agentmem search "deployment process" --limit 5 -n infra

# Manage procedures
agentmem add-procedure "Always use markdown formatting"
agentmem procedures

# View entities and relations
agentmem entities --type ip
agentmem related 10.0.0.1

# Maintenance
agentmem compact --max-age-days 90 --dry-run
agentmem consolidate --threshold 0.85

# Process conversation
agentmem process chat.json -n project

# Stats and export
agentmem stats
agentmem export --tier core
```

### MCP Server (stdio)

```bash
python -m agentmem --db memory.db
```

Add to your MCP client config:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "agentmem", "--db", "/path/to/memory.db"]
    }
  }
}
```

### HTTP REST API

```bash
# Start HTTP server
agentmem serve-http --port 8422

# Or directly
agentmem-http --port 8422 --db memory.db
```

```bash
# Store a memory
curl -X POST http://localhost:8422/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Server IP is 10.0.0.1", "tier": "core", "namespace": "infra"}'

# Search
curl "http://localhost:8422/recall?query=server+IP&namespace=infra"

# Health check
curl http://localhost:8422/health
```

**16 MCP tools / 14 HTTP endpoints:**

| Tool | HTTP | Description |
|------|------|-------------|
| `recall` | `GET /recall` | Hybrid keyword + semantic search |
| `remember` | `POST /remember` | Store a new memory |
| `save_state` | `POST /save_state` | Emergency save before context compression |
| `today` | `GET /today` | Get all memories from today |
| `forget` | `POST /forget` | Archive a memory (soft delete) |
| `unarchive` | `POST /unarchive` | Restore an archived memory |
| `stats` | `GET /stats` | Memory statistics and health |
| `compact` | `POST /compact` | Archive low-value memories |
| `consolidate` | `POST /consolidate` | Merge near-duplicate memories |
| `update_memory` | `POST /update_memory` | Replace a memory with version chain |
| `history` | `GET /history` | Trace fact version history |
| `related` | `GET /related` | Find memories by entity |
| `entities` | `GET /entities` | List all extracted entities |
| `get_procedures` | — | Get behavioral rules for system prompt |
| `add_procedure` | — | Add a behavioral rule |
| `process_conversation` | — | Auto-extract from chat history |

## Memory Tiers

| Tier | Purpose | Auto-compacted | Example |
|------|---------|---------------|---------|
| `core` | Permanent facts | Never | "Server IP is 10.0.0.1" |
| `procedural` | Behavioral rules | Never | "Always use HTTPS" |
| `learned` | Discovered knowledge | After 90 days | "API returns 403 without auth" |
| `episodic` | Events | After 90 days | "Deployed v2.1 at 15:30" |
| `working` | Current task state | After 24 hours | "Working on step 3/5" |

## Namespaces

Isolate memories per user, agent, or project:

```python
# Store in namespaces
store.remember("Alice's API key", namespace="user/alice")
store.remember("Bob's config", namespace="user/bob")
store.remember("Shared fact", namespace="team")

# Search within namespace (prefix matching)
store.recall("API", namespace="user/alice")  # only Alice's memories
store.recall("API", namespace="user")  # Alice + Bob (prefix match)
store.recall("API")  # everything
```

## Temporal Versioning

Track how facts evolve over time:

```python
# Initial fact
r1 = store.remember("Server costs $50/month", tier="core")

# Fact changes — old version archived, linked via supersedes
r2 = store.update_memory(r1["id"], "Server costs $75/month")

# Trace the history
history = store.history(r2["id"])
# → [{"id": 2, "content": "...$75..."}, {"id": 1, "content": "...$50..."}]
```

## Entity Extraction

Automatic regex-based NER on every `remember()` call:

| Type | Pattern | Example |
|------|---------|---------|
| `mention` | `@username` | @alice |
| `url` | `https://...` | https://api.example.com |
| `ip` | `N.N.N.N` | 10.0.0.1 |
| `port` | `:NNNN` | :8080 |
| `email` | `user@domain` | admin@example.com |
| `env_var` | `ALL_CAPS` | OPENAI_API_KEY |
| `money` | `$NNN` | $50 |
| `path` | `/unix/path` | /etc/nginx/conf.d |
| `hashtag` | `#tag` | #deployment |

```python
# Find all memories mentioning an entity
store.related("10.0.0.1")
store.related("@alice", entity_type="mention")

# List all known entities
store.entities()  # sorted by memory count
store.entities(entity_type="ip")
```

## Conversation Auto-Extraction

Auto-extract memories from chat history (regex-only, no LLM):

```python
messages = [
    {"role": "user", "content": "Set DATABASE_URL to postgres://localhost/mydb"},
    {"role": "assistant", "content": "I decided to use connection pooling. Important: max 20 connections."},
    {"role": "user", "content": "Always validate input. TODO: add rate limiting."},
]
result = store.process_conversation(messages)
# Extracts: config→core, decisions→episodic, preferences→procedural, todos→working, important→core
```

## Operational Modes

agentmem automatically selects the best available mode:

| Mode | Install Size | Init Time | Query Time | Dependencies |
|------|-------------|-----------|------------|-------------|
| **sqlite-vec + model2vec** | 12 MB | ~5ms* | ~1ms | sqlite-vec, model2vec, numpy |
| **sqlite-vec + hash** | 151 KB | ~5ms | ~0.8ms | sqlite-vec |
| **pure Python + hash** | 0 KB | ~3ms | ~1.8ms | none (stdlib only) |
| **pure + int8 quantize** | 0 KB | ~3ms | ~3ms | none (stdlib only) |

*\*With lazy loading — model2vec loads on first query, not on init*

## Architecture

```
┌──────────────────────────────────────────────┐
│              MemoryStore                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  FTS5    │  │  Vector  │  │  Entity    │  │
│  │ keywords │  │  Index   │  │  Index     │  │
│  │ + BM25   │  │ cosine   │  │  regex NER │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       └──────┬───────┘              │         │
│    Adaptive Hybrid Scorer           │         │
│  (query classify + recency +        │         │
│   importance boost)                 │         │
│  ┌──────────────────────────────────┴───────┐ │
│  │            SQLite + WAL                  │ │
│  │  memories │ memories_fts │ vecs │ entities│ │
│  └──────────────────────────────────────────┘ │
│            One file: memory.db                │
└───────────────────────────────────────────────┘
```

## Comparison

| Feature | agentmem | ChromaDB | LanceDB | mem0 | Zep |
|---------|----------|----------|---------|------|-----|
| Install size | 0-12 MB | 400+ MB | 100+ MB | 500+ MB | Cloud |
| Cold start | 3-5 ms | seconds | seconds | seconds | N/A |
| PyTorch required | No | Yes | No | Yes | N/A |
| Cloud required | No | No | No | Yes | Yes |
| Zero-dep mode | Yes | No | No | No | No |
| Keyword search | FTS5 (BM25) | No | No | No | Yes |
| MCP server | 16 tools | No | No | Yes | No |
| HTTP API | Built-in | Yes | No | Yes | Yes |
| Single file DB | Yes | No | Yes | No | No |
| Namespaces | Yes | Yes | Yes | Yes | Yes |
| Temporal versioning | Yes | No | Yes | No | Yes |
| Entity extraction | Auto (regex) | No | No | No | No |
| Procedural memory | Yes | No | No | No | No |
| Importance scoring | Auto | No | No | No | No |
| Conversation extraction | Auto (regex) | No | No | Yes (LLM) | Yes (LLM) |
| Memory consolidation | Yes | No | No | Yes (LLM) | No |

## License

MIT
