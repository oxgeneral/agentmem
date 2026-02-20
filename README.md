# agentmem

Lightweight persistent memory for AI agents. One SQLite file. Hybrid search (keywords + semantics). Zero to 12MB install.

No PyTorch. No cloud. No server. Just memory.

## Why

Every AI agent session starts from zero. Context windows compress, conversations end, memory vanishes. `agentmem` gives agents persistent memory that survives across sessions — in a single SQLite file.

- **Hybrid search**: FTS5 full-text keywords + vector semantic search, fused with adaptive ranking
- **4 operational modes**: from zero dependencies (stdlib only) to best quality (12MB)
- **MCP server**: 6 tools via stdio JSON-RPC — plug into any MCP-compatible agent
- **Tiered storage**: core (permanent), learned (discovered), episodic (events), working (auto-expires)
- **Multilingual**: Russian keywords via FTS5, English semantics via embeddings, mixed queries handled
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

# Store memories
store.remember("Server costs $50/month, runs Ubuntu 24.04", tier="core")
store.remember("Discovered that API returns 403 without auth header", tier="learned")
store.remember("Deployed v2.1 to production at 15:30", tier="episodic")

# Search — hybrid keyword + semantic
results = store.recall("server costs")
# → finds the memory even with "how much does hosting cost?"

# Save working state before context compression
store.save_state("Working on auth fix, step 3/5, blocked by CORS")

# Import existing markdown files
store.import_markdown("MEMORY.md", tier="core")
store.import_markdown("2024-01-15.md", tier="episodic")
```

### CLI

```bash
# Initialize database
agentmem init --db memory.db

# Import markdown files
agentmem import MEMORY.md --tier core
agentmem import-dir ./daily-logs/ --tier episodic

# Search
agentmem search "deployment process" --limit 5

# Stats
agentmem stats

# Export
agentmem export --tier core
```

### MCP Server

```bash
# Start MCP server (stdio transport)
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

**6 MCP tools:**

| Tool | Description |
|------|-------------|
| `recall` | Hybrid keyword + semantic search |
| `remember` | Store a new memory with tier and tags |
| `save_state` | Emergency save before context compression |
| `today` | Get all memories created today |
| `forget` | Archive a memory (soft delete) |
| `stats` | Memory statistics and health |

## Operational Modes

agentmem automatically selects the best available mode:

| Mode | Install Size | Init Time | Query Time | Dependencies |
|------|-------------|-----------|------------|-------------|
| **sqlite-vec + model2vec** | 12 MB | ~5ms* | ~1ms | sqlite-vec, model2vec, numpy |
| **sqlite-vec + hash** | 151 KB | ~5ms | ~0.8ms | sqlite-vec |
| **pure Python + hash** | 0 KB | ~3ms | ~1.8ms | none (stdlib only) |
| **pure + int8 quantize** | 0 KB | ~3ms | ~3ms | none (stdlib only) |

*\*With lazy loading — model2vec loads on first query, not on init*

### HashEmbedding (zero dependencies)

Pure Python SimHash-style embeddings using MD5 random projection. No pip packages needed.

- 128-dimensional vectors from word unigrams + char n-grams
- ~0.05ms per embedding
- ~70-80% quality of transformer models for entity/keyword retrieval
- Handles any UTF-8 text (Russian, CJK, etc.)

### model2vec (best quality)

[model2vec](https://github.com/MinishLab/model2vec) potion-base-8M: static embeddings distilled from transformers.

- 256-dimensional vectors
- 8MB model, numpy-only (no PyTorch)
- 30,000+ sentences/sec on CPU
- Best semantic understanding

## Architecture

```
┌──────────────────────────────────────┐
│            MemoryStore               │
│  ┌──────────┐  ┌──────────────────┐  │
│  │  FTS5    │  │  Vector Index    │  │
│  │ keywords │  │ (sqlite-vec or   │  │
│  │  + BM25  │  │  pure Python)    │  │
│  └─────┬────┘  └────────┬─────────┘  │
│        └───────┬────────┘            │
│         Adaptive Hybrid Scorer       │
│     (classifies query → weights)     │
│  ┌──────────────────────────────────┐│
│  │         SQLite + WAL             ││
│  │  memories │ memories_fts │ vecs  ││
│  └──────────────────────────────────┘│
└──────────────────────────────────────┘
         One file: memory.db
```

**Hybrid search pipeline:**
1. Query classifier detects type: keyword-heavy, semantic-heavy, or mixed
2. FTS5 runs phrase match + prefix match + individual terms (with stop-word filtering)
3. Vector search runs cosine similarity (sqlite-vec KNN or pure Python brute-force)
4. Results merged with adaptive weights based on query type
5. Deduplicated and ranked by fused score

## Smart Features

### Adaptive Query Classification
```python
# Keyword query → FTS5 weighted higher
store.recall("API endpoint port 8420")

# Semantic query → vector search weighted higher
store.recall("how to restart after a crash")

# Mixed → balanced fusion
store.recall("Telegram bot growth strategies")
```

### Smart Markdown Chunking
Paragraph-level splitting with bullet-point awareness. A 200-line MEMORY.md becomes ~64 searchable chunks (avg ~250 chars) instead of ~10 coarse blocks.

### Deduplication
Content-hash based. Re-importing the same file is a no-op — only new/changed chunks are added.

### Int8 Quantization
Optional 4x storage reduction with ~98% recall quality:
```python
store = MemoryStore("memory.db", embedding_dim=128, quantize=True)
```

## Benchmarks

Tested on real agent memory data (MEMORY.md + daily logs, ~64 chunks):

```
Operation          Time
─────────────────────────
Cold start         3-5 ms
Import (per chunk) 0.15 ms
Hybrid query       0.8-1.8 ms
FTS5-only query    0.2 ms
Vector-only query  0.1-0.5 ms
```

## Comparison

| Feature | agentmem | ChromaDB | LanceDB | mem0 |
|---------|----------|----------|---------|------|
| Install size | 0-12 MB | 400+ MB | 100+ MB | 500+ MB |
| Cold start | 3-5 ms | seconds | seconds | seconds |
| PyTorch required | No | Yes | No | Yes |
| Cloud required | No | No | No | Yes (default) |
| Zero-dep mode | Yes | No | No | No |
| Keyword search | FTS5 (BM25) | No | No | No |
| MCP server | Built-in | No | No | No |
| Single file DB | Yes | No | Yes | No |

## License

MIT
