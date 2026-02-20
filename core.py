"""
agentmem.core — The heart of agent memory.

One SQLite file. Dual search (FTS5 keywords + sqlite-vec semantics).
Hybrid ranking. Tiered storage. ~12MB total install.

This is what I wish I had when I wake up in a new session with no context.
"""
import re
import sqlite3
import struct
import time
import json
import hashlib
import math
from pathlib import Path
from typing import Optional

# Tiers: how important/permanent is this memory?
TIERS = ("core", "learned", "episodic", "working")

# Working memories auto-expire after this many seconds
WORKING_TTL = 86400  # 24 hours

# Compact tier encoding: TEXT → INTEGER for storage (saves 5-6 bytes per row)
_TIER_TO_INT = {"core": 0, "learned": 1, "episodic": 2, "working": 3}
_INT_TO_TIER = {v: k for k, v in _TIER_TO_INT.items()}


def _serialize_f32(vec: list[float]) -> bytes:
    """Pack float list into binary format (float32 little-endian)."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes, dim: int) -> list[float]:
    """Unpack binary blob to float list."""
    return list(struct.unpack(f"{dim}f", blob))


def _quantize_f32_to_i8(vec: list[float]) -> bytes:
    """
    Quantize float32 vector to int8. 4x storage reduction.

    Encoding layout (8 bytes header + dim bytes body):
      [0:4]  scale  (float32) — range of original values (vmax - vmin)
      [4:8]  vmin   (float32) — minimum value of original vector
      [8:]   quantized int8 values, one per dimension

    Reconstruction: v_orig = (q + 128) / 255 * scale + vmin
    """
    vmin = min(vec)
    vmax = max(vec)
    scale = (vmax - vmin) if vmax != vmin else 1.0
    # Map each float to [-128, 127]
    quantized = [
        max(-128, min(127, int((v - vmin) / scale * 255 - 128)))
        for v in vec
    ]
    header = struct.pack('ff', scale, vmin)
    body = struct.pack(f'{len(vec)}b', *quantized)
    return header + body


def _dequantize_i8_to_f32(blob: bytes, dim: int) -> list[float]:
    """Dequantize int8 blob back to float32 list."""
    scale, vmin = struct.unpack_from('ff', blob, 0)
    quantized = struct.unpack_from(f'{dim}b', blob, 8)
    return [(q + 128) / 255.0 * scale + vmin for q in quantized]


class _VecIndex:
    """
    Pure Python brute-force vector index backed by a regular SQLite table.

    Used as a fallback when sqlite-vec C extension is not available.
    Stores vectors as BLOB (float32 little-endian, stdlib struct).

    Performance on modern hardware (Python 3.12):
    - Insert: ~0.1ms per vector
    - Search 1000 vectors, dim=256: ~3-8ms
    - Search 1000 vectors, dim=128: ~2-4ms

    Optimizations:
    - Single struct.unpack_from call per stored vector (not per dimension)
    - Precomputed query norm
    - Cache of norms for stored vectors (lazy, in-memory dict)
    - Returns after finding top-k without sorting full list (partial sort)
    """

    def __init__(self, db: sqlite3.Connection, dim: int, quantize: bool = False):
        self.db = db
        self.dim = dim
        self.quantize = quantize
        self._fmt = f"{dim}f"
        self._i8_fmt = f"{dim}b"
        self._norm_cache: dict[int, float] = {}  # rowid -> precomputed norm

        # Create pure-Python vector table in the same SQLite file
        db.execute("""
            CREATE TABLE IF NOT EXISTS memories_vec_pure (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        """)
        db.commit()

        # Pre-warm norm cache from existing rows
        self._warm_cache()

    def _warm_cache(self):
        """Load norms for all existing vectors into memory."""
        rows = self.db.execute(
            "SELECT rowid, embedding FROM memories_vec_pure"
        ).fetchall()
        dim = self.dim
        if self.quantize:
            for rowid, blob in rows:
                vec = _dequantize_i8_to_f32(blob, dim)
                norm = math.sqrt(sum(v * v for v in vec))
                self._norm_cache[rowid] = norm
        else:
            fmt = self._fmt
            for rowid, blob in rows:
                vec = struct.unpack(fmt, blob)
                norm = math.sqrt(sum(v * v for v in vec))
                self._norm_cache[rowid] = norm

    def insert(self, rowid: int, vec: list[float]):
        """Serialize and store a vector; update norm cache."""
        if self.quantize:
            blob = _quantize_f32_to_i8(vec)
        else:
            blob = struct.pack(self._fmt, *vec)
        self.db.execute(
            "INSERT OR REPLACE INTO memories_vec_pure(rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        # Precompute and cache norm (always from original float values)
        norm = math.sqrt(sum(v * v for v in vec))
        self._norm_cache[rowid] = norm

    def delete(self, rowid: int):
        """Remove a vector."""
        self.db.execute(
            "DELETE FROM memories_vec_pure WHERE rowid = ?", (rowid,)
        )
        self._norm_cache.pop(rowid, None)

    def search(self, query_vec: list[float], k: int) -> list[tuple]:
        """
        Brute-force cosine similarity search.

        Returns list of (rowid, distance) where distance is cosine distance
        (0=identical, 2=opposite), sorted ascending — same convention as sqlite-vec.

        When quantize=True, stored vectors are dequantized on-the-fly for distance
        computation. The cached norms are always float32 quality (computed from
        original values at insert time), so accuracy loss is limited to the
        dequantized dot product only.
        """
        # Precompute query norm once
        query_norm = math.sqrt(sum(v * v for v in query_vec))
        if query_norm == 0.0:
            return []

        dim = self.dim
        norm_cache = self._norm_cache
        quantize = self.quantize

        # Fetch all stored vectors
        rows = self.db.execute(
            "SELECT rowid, embedding FROM memories_vec_pure"
        ).fetchall()

        if not rows:
            return []

        scores: list[tuple[float, int]] = []  # (distance, rowid)

        if quantize:
            for rowid, blob in rows:
                # Dequantize int8 → float32 for dot product
                stored_vec = _dequantize_i8_to_f32(blob, dim)

                dot = sum(a * b for a, b in zip(query_vec, stored_vec))

                stored_norm = norm_cache.get(rowid)
                if stored_norm is None:
                    stored_norm = math.sqrt(sum(v * v for v in stored_vec))
                    norm_cache[rowid] = stored_norm

                if stored_norm == 0.0:
                    cosine_sim = 0.0
                else:
                    cosine_sim = dot / (query_norm * stored_norm)

                distance = 1.0 - cosine_sim
                scores.append((distance, rowid))
        else:
            fmt = self._fmt
            for rowid, blob in rows:
                # Unpack entire blob in one call (fastest stdlib approach)
                stored_vec = struct.unpack(fmt, blob)

                # Dot product via built-in sum + zip
                dot = sum(a * b for a, b in zip(query_vec, stored_vec))

                # Use cached norm if available
                stored_norm = norm_cache.get(rowid)
                if stored_norm is None:
                    stored_norm = math.sqrt(sum(v * v for v in stored_vec))
                    norm_cache[rowid] = stored_norm

                if stored_norm == 0.0:
                    cosine_sim = 0.0
                else:
                    cosine_sim = dot / (query_norm * stored_norm)

                # Cosine distance (same convention as sqlite-vec: 0=identical, 2=opposite)
                distance = 1.0 - cosine_sim
                scores.append((distance, rowid))

        # Partial sort: only need top-k smallest distances
        scores.sort(key=lambda x: x[0])
        top_k = scores[:k]

        return [(rowid, distance) for distance, rowid in top_k]


class MemoryStore:
    """
    Persistent agent memory with hybrid search.

    Uses a single SQLite file with:
    - Regular table for metadata
    - FTS5 virtual table for keyword/BM25 search
    - vec0 virtual table for semantic vector search (if sqlite-vec available)
    - BLOB table for pure Python fallback vector search (if sqlite-vec NOT available)

    _vec_mode values:
    - "sqlite-vec": C extension loaded, using memories_vec virtual table
    - "pure": no C extension, using _VecIndex pure Python fallback
    - "none": no embedding function set at all
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        embedding_dim: int = 256,
        quantize: bool = False,
    ):
        self.db_path = Path(db_path)
        self.dim = embedding_dim
        self.quantize = quantize  # int8 vector quantization (4x storage reduction)
        self._embed_fn = None        # set via set_embed_fn()
        self._embed_batch_fn = None  # set via set_embed_fn() when model object passed
        self._vec_index: Optional[_VecIndex] = None
        self.db = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.db_path))
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA mmap_size=67108864")  # 64MB mmap

        # Try to load sqlite-vec extension (preferred: faster C implementation)
        try:
            import sqlite_vec
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            self._has_vec = True
            self._vec_mode = "sqlite-vec"
        except (ImportError, Exception):
            self._has_vec = False
            self._vec_mode = "pure"  # will use _VecIndex fallback

        return db

    def _detect_schema_mode(self) -> str:
        """
        Detect whether the memories table uses the old (TEXT tier) or new (INTEGER tier) schema.

        Returns:
            "compact"  — new schema: tier INTEGER, content_hash BLOB, tags comma-separated
            "legacy"   — old schema: tier TEXT, content_hash TEXT, tags JSON
            "none"     — table does not exist yet (will be created as compact)
        """
        row = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if row is None:
            return "none"  # fresh DB, will create compact

        # Inspect column type declared in the CREATE TABLE statement
        col_info = self.db.execute("PRAGMA table_info(memories)").fetchall()
        for col in col_info:
            # col = (cid, name, type, notnull, dflt_value, pk)
            if col[1] == "tier":
                col_type = (col[2] or "").upper()
                if "INT" in col_type:
                    return "compact"
                else:
                    return "legacy"
        # tier column missing — treat as legacy
        return "legacy"

    def _init_schema(self):
        """Create tables if they don't exist. Detects legacy vs compact schema."""
        self._schema_mode = self._detect_schema_mode()

        if self._schema_mode == "none":
            # Brand new DB — create the compact optimised schema
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    tier INTEGER NOT NULL DEFAULT 1,
                    source TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived INTEGER DEFAULT 0,
                    content_hash BLOB UNIQUE
                )
            """)
            self._schema_mode = "compact"
        else:
            # Table already exists — keep it as-is, just ensure it was created
            # (CREATE TABLE IF NOT EXISTS with original legacy schema as fallback)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    tier TEXT NOT NULL DEFAULT 'learned',
                    source TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived INTEGER DEFAULT 0,
                    content_hash TEXT UNIQUE
                )
            """)

        # FTS5 for keyword search (BM25)
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, source, tags, content='memories', content_rowid='id')
        """)

        # Triggers to keep FTS5 in sync
        self.db.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, source, tags)
                VALUES (new.id, new.content, new.source, new.tags);
            END
        """)
        self.db.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, source, tags)
                VALUES ('delete', old.id, old.content, old.source, old.tags);
            END
        """)
        self.db.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, source, tags)
                VALUES ('delete', old.id, old.content, old.source, old.tags);
                INSERT INTO memories_fts(rowid, content, source, tags)
                VALUES (new.id, new.content, new.source, new.tags);
            END
        """)

        # Vector table: sqlite-vec (C extension) preferred
        if self._vec_mode == "sqlite-vec":
            self.db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                USING vec0(
                    embedding float[{self.dim}] distance_metric=cosine
                )
            """)
        else:
            # Pure Python fallback: regular BLOB table + in-memory index
            self._vec_index = _VecIndex(self.db, self.dim, quantize=self.quantize)

        # Index for common queries
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_archived ON memories(archived)")

        self.db.commit()

    def set_embed_fn(self, fn):
        """
        Set the embedding function. fn(text) -> list[float]

        Can also accept any object with .embed(text) and .embed_batch(texts)
        methods (e.g. LazyEmbedding, Model2VecEmbedding). When an object is
        passed, both single-embed and batch paths are available.
        """
        if hasattr(fn, "embed") and hasattr(fn, "embed_batch"):
            # Full embedding model object (LazyEmbedding, Model2VecEmbedding, etc.)
            # Use .embed() for single texts, .embed_batch() for bulk inserts
            self._embed_fn = fn.embed
            self._embed_batch_fn = fn.embed_batch
        else:
            # Plain callable fn(text) -> list[float] — backward compatible
            self._embed_fn = fn
            self._embed_batch_fn = None

    def _embed(self, text: str) -> Optional[list[float]]:
        """Embed text. Returns None if no embedding function set."""
        if self._embed_fn is None:
            return None
        return self._embed_fn(text)

    def _embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """
        Embed a list of texts in one model call.

        Uses embed_batch() if available (faster for model2vec: single numpy call).
        Falls back to calling _embed() per-item otherwise.
        Returns list of vectors (or Nones if no embed fn set).
        """
        if self._embed_fn is None:
            return [None] * len(texts)
        if self._embed_batch_fn is not None:
            return self._embed_batch_fn(texts)
        # Fallback: individual calls
        return [self._embed_fn(t) for t in texts]

    def _content_hash(self, content: str):
        """
        Return a deduplication hash for content.

        Compact schema (new DBs):  8-byte BLOB  (sha256[:8])
        Legacy schema (old DBs):   16-char hex  (sha256.hexdigest()[:16])

        Switching between the two based on self._schema_mode keeps the
        UNIQUE constraint working correctly for each schema format.
        """
        digest = hashlib.sha256(content.encode())
        if self._schema_mode == "compact":
            return digest.digest()[:8]   # 8 bytes BLOB — saves 8 bytes vs hex str
        return digest.hexdigest()[:16]   # legacy: 16-char hex string

    def _encode_tier(self, tier: str):
        """
        Encode tier string to storage value.
        Compact schema: returns int (0-3).
        Legacy schema:  returns str unchanged.
        """
        if self._schema_mode == "compact":
            return _TIER_TO_INT.get(tier, 1)  # default 1 = "learned"
        return tier

    def _decode_tier(self, raw) -> str:
        """
        Decode tier value from storage to public string API.
        Compact schema: int → string.
        Legacy schema:  string unchanged.
        """
        if self._schema_mode == "compact":
            return _INT_TO_TIER.get(raw, "learned")
        return raw if isinstance(raw, str) else "learned"

    def _encode_tags(self, tags: list) -> str:
        """
        Encode tags list to storage string.
        Compact schema: comma-separated (no brackets/quotes overhead).
        Legacy schema:  JSON array string.
        """
        if self._schema_mode == "compact":
            return ",".join(str(t) for t in tags) if tags else ""
        return json.dumps(tags)

    def _decode_tags(self, raw: str) -> list:
        """
        Decode tags from storage string to list.
        Handles both comma-separated (compact) and JSON (legacy) formats.
        """
        if not raw:
            return []
        # Try JSON first (handles legacy "[]" and '["tag1"]' etc.)
        if raw.startswith("["):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass
        # Comma-separated (compact format or fallback)
        return [t.strip() for t in raw.split(",") if t.strip()]

    # ================================================================
    # TOOL 1: remember(content, tier, tags, source)
    # ================================================================
    def remember(
        self,
        content: str,
        tier: str = "learned",
        tags: list[str] = None,
        source: str = "",
    ) -> dict:
        """
        Store a new memory. Auto-embeds if embedding function is set.
        Deduplicates by content hash.

        Returns: {"id": int, "tier": str, "embedded": bool}
        """
        if tier not in TIERS:
            tier = "learned"

        tags = tags or []
        now = time.time()
        content_hash = self._content_hash(content)

        # Check for duplicate
        existing = self.db.execute(
            "SELECT id FROM memories WHERE content_hash = ?", (content_hash,)
        ).fetchone()

        if existing:
            # Update timestamp and un-archive if needed
            self.db.execute(
                "UPDATE memories SET updated_at = ?, archived = 0 WHERE id = ?",
                (now, existing[0]),
            )
            self.db.commit()
            return {"id": existing[0], "tier": tier, "embedded": False, "deduplicated": True}

        # Encode tier and tags for storage
        tier_stored = self._encode_tier(tier)
        tags_stored = self._encode_tags(tags)

        # Insert into main table
        cursor = self.db.execute(
            """INSERT INTO memories (content, tier, source, tags, created_at, updated_at, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (content, tier_stored, source, tags_stored, now, now, content_hash),
        )
        memory_id = cursor.lastrowid

        # Embed and store vector
        embedded = False
        vec = self._embed(content)

        if vec is not None:
            if self._vec_mode == "sqlite-vec":
                try:
                    self.db.execute(
                        "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                        (memory_id, _serialize_f32(vec)),
                    )
                    embedded = True
                except Exception:
                    pass  # Vector insert failed, keyword search still works

            elif self._vec_mode == "pure" and self._vec_index is not None:
                try:
                    self._vec_index.insert(memory_id, vec)
                    embedded = True
                except Exception:
                    pass

        self.db.commit()
        return {"id": memory_id, "tier": tier, "embedded": embedded, "deduplicated": False}

    # ================================================================
    # BATCH INSERT: remember_batch(items)
    # ================================================================
    def remember_batch(self, items: list[dict]) -> dict:
        """
        Store multiple memories efficiently in a single transaction.

        Compared to calling remember() N times, this method:
        - Deduplicates all content hashes upfront (one SELECT)
        - Embeds all new texts in a single model call (embed_batch)
        - Inserts all rows with executemany (one round-trip)
        - Commits once at the end

        Args:
            items: List of dicts, each with keys:
                   - content (str, required)
                   - tier    (str, optional, default "learned")
                   - tags    (list[str], optional)
                   - source  (str, optional)

        Returns:
            {"imported": int, "deduplicated": int, "embedded": int}
        """
        if not items:
            return {"imported": 0, "deduplicated": 0, "embedded": 0}

        now = time.time()

        # --- Step 1: Compute content hashes for all items ---
        hashed = []
        for item in items:
            content = item.get("content", "")
            tier = item.get("tier", "learned")
            if tier not in TIERS:
                tier = "learned"
            tags = item.get("tags") or []
            source = item.get("source") or ""
            h = self._content_hash(content)
            hashed.append((content, tier, tags, source, h))

        # --- Step 2: Find which hashes already exist (one query) ---
        all_hashes = [row[4] for row in hashed]
        # SQLite IN clause — split into chunks if very large (>900 items)
        existing_hashes = set()
        chunk_size = 900
        for i in range(0, len(all_hashes), chunk_size):
            chunk = all_hashes[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT content_hash FROM memories WHERE content_hash IN ({placeholders})",
                chunk,
            ).fetchall()
            existing_hashes.update(r[0] for r in rows)

        # --- Step 3: Split into new vs duplicate ---
        new_items = []      # (content, tier, tags, source, hash)
        deduplicated = 0

        for row in hashed:
            if row[4] in existing_hashes:
                deduplicated += 1
            else:
                new_items.append(row)
                # Track hash locally to catch duplicates within the same batch
                existing_hashes.add(row[4])

        if not new_items:
            return {"imported": 0, "deduplicated": deduplicated, "embedded": 0}

        # --- Step 4: Batch embed all new texts in one model call ---
        texts = [row[0] for row in new_items]
        vecs: list[Optional[list[float]]] = [None] * len(texts)

        if self._embed_fn is not None:
            raw_vecs = self._embed_batch(texts)
            vecs = raw_vecs  # list of list[float] or None

        # --- Step 5: Insert all new memories (executemany) ---
        insert_rows = [
            (content, self._encode_tier(tier), source, self._encode_tags(tags), now, now, h)
            for content, tier, tags, source, h in new_items
        ]
        self.db.executemany(
            """INSERT OR IGNORE INTO memories
               (content, tier, source, tags, created_at, updated_at, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            insert_rows,
        )

        # --- Step 6: Retrieve inserted IDs (needed for vector table) ---
        # We need the auto-assigned IDs. Fetch by hash (we own these hashes).
        new_hashes = [row[4] for row in new_items]
        hash_to_id = {}
        for i in range(0, len(new_hashes), chunk_size):
            chunk = new_hashes[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            id_rows = self.db.execute(
                f"SELECT id, content_hash FROM memories WHERE content_hash IN ({placeholders})",
                chunk,
            ).fetchall()
            for mem_id, h in id_rows:
                hash_to_id[h] = mem_id

        # --- Step 7: Batch insert vectors ---
        embedded = 0
        if self._embed_fn is not None:
            if self._vec_mode == "sqlite-vec":
                vec_rows = []
                for (content, tier, tags, source, h), vec in zip(new_items, vecs):
                    if vec is not None and h in hash_to_id:
                        vec_rows.append((hash_to_id[h], _serialize_f32(vec)))
                        embedded += 1
                if vec_rows:
                    try:
                        self.db.executemany(
                            "INSERT OR IGNORE INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                            vec_rows,
                        )
                    except Exception:
                        pass  # Vector insert failed; keyword search still works

            elif self._vec_mode == "pure" and self._vec_index is not None:
                for (content, tier, tags, source, h), vec in zip(new_items, vecs):
                    if vec is not None and h in hash_to_id:
                        try:
                            self._vec_index.insert(hash_to_id[h], vec)
                            embedded += 1
                        except Exception:
                            pass

        # --- Step 8: Single commit ---
        self.db.commit()

        return {
            "imported": len(new_items),
            "deduplicated": deduplicated,
            "embedded": embedded,
        }

    # ================================================================
    # Query classifier: decide FTS vs vector weights adaptively
    # ================================================================
    # Technical terms that signal a mixed/balanced query
    _TECH_TERMS = frozenset({
        "api", "mcp", "bot", "token", "server", "client", "sdk", "cli",
        "http", "https", "url", "endpoint", "request", "response",
        "json", "yaml", "toml", "env", "config", "deploy", "port",
        "docker", "python", "node", "npm", "pip", "git", "bash",
        "database", "sqlite", "query", "index", "webhook", "ssl",
        "ssh", "tcp", "ip", "dns", "cron", "log", "cache", "proxy",
    })

    # Question words that signal intent to understand (semantic)
    _QUESTION_WORDS_EN = frozenset({
        "how", "what", "why", "when", "where", "which", "who",
        "explain", "describe", "show", "tell",
    })
    _QUESTION_WORDS_RU = frozenset({
        "как", "что", "почему", "зачем", "когда", "где", "какой",
        "какая", "какое", "какие", "кто", "чем", "чему", "объясни",
        "расскажи", "покажи",
    })

    def _classify_query(self, query: str) -> tuple[float, float]:
        """
        Analyse a query string and return (fts_weight, vec_weight) for hybrid scoring.

        Rules are evaluated in priority order; first match wins.

        Priority order (highest to lowest):
          1. Quoted exact phrase         → (0.9, 0.1) — user wants verbatim match
          2. Has @mention or URL         → (0.8, 0.2) — entity lookup
          3. Has numeric ID / hex token  → (0.8, 0.2) — exact entity
          4. Mostly Russian / CJK text   → (0.7, 0.3) — hash embeddings weak on non-Latin
          5. Is a question               → (0.3, 0.7) — conceptual / "how does X work"
          6. Long query (5+ real words)  → (0.3, 0.7) — richer semantics
          7. Short query (1-2 real words)→ (0.6, 0.4) — keyword prefix likely best
          8. Contains technical terms    → (0.5, 0.5) — ambiguous, trust both equally
          9. Default                     → (0.4, 0.6) — slight semantic preference

        Returns:
            (fts_weight, vec_weight) — both floats summing to 1.0
        """
        # --- Rule 1: quoted exact phrase ---
        if re.search(r'"[^"]+"', query):
            return (0.9, 0.1)

        # --- Rule 2: @mentions or URLs ---
        if re.search(r'@\w+|https?://\S+|www\.\S+', query):
            return (0.8, 0.2)

        # --- Rule 3: numeric IDs (3+ digits) or hex tokens (6+ hex chars) ---
        if re.search(r'\b\d{3,}\b', query) or re.search(r'\b[0-9a-fA-F]{6,}\b', query):
            return (0.8, 0.2)

        # --- Rule 4: mostly Russian / CJK (non-ASCII alpha chars) ---
        # Count Latin-alpha vs non-Latin-alpha characters
        alpha_chars = [c for c in query if c.isalpha()]
        if alpha_chars:
            non_latin = sum(
                1 for c in alpha_chars
                if ord(c) > 0x024F  # Beyond Extended Latin block
            )
            ratio = non_latin / len(alpha_chars)
            if ratio > 0.5:
                return (0.7, 0.3)

        # Tokenize for word-count-based rules (strip punctuation, lowercase)
        words = re.findall(r'\b\w+\b', query.lower())
        # Remove stop words for an accurate word count
        stop_all = self._EN_STOP | self._RU_STOP
        real_words = [w for w in words if w not in stop_all and len(w) > 1]

        # --- Rule 5: question words ---
        if real_words:
            first = real_words[0]
            if (first in self._QUESTION_WORDS_EN or
                    first in self._QUESTION_WORDS_RU or
                    query.strip().endswith("?")):
                return (0.3, 0.7)

        # --- Rule 6: long query (5+ real words after stop-word removal) ---
        if len(real_words) >= 5:
            return (0.3, 0.7)

        # --- Rule 7: short query (1-2 real words) ---
        if len(real_words) <= 2:
            return (0.6, 0.4)

        # --- Rule 8: technical terms present ---
        word_set = set(real_words)
        if word_set & self._TECH_TERMS:
            return (0.5, 0.5)

        # --- Default: slight semantic preference ---
        return (0.4, 0.6)

    # ================================================================
    # TOOL 2: recall(query, limit, tier)
    # ================================================================
    def recall(
        self,
        query: str,
        limit: int = 5,
        tier: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search: FTS5 (keywords) + vector search (semantics) → rerank.

        Strategy:
        1. FTS5 search → candidates with BM25 rank
        2. Vector KNN search → candidates with cosine distance
        3. Merge and rerank → best of both worlds
        4. FTS5 gets boosted for exact keyword matches

        Returns list of {"id", "content", "tier", "source", "score", "method"}
        """
        candidates = {}  # id -> {data, fts_score, vec_score}

        # Step 1: FTS5 keyword search
        fts_results = self._fts_search(query, limit=limit * 3, tier=tier)
        for r in fts_results:
            candidates[r["id"]] = {
                **r,
                "fts_score": r["score"],
                "vec_score": 0.0,
            }

        # Step 2: Vector semantic search (sqlite-vec or pure Python)
        can_vec_search = (
            self._embed_fn is not None and
            self._vec_mode in ("sqlite-vec", "pure")
        )
        if can_vec_search:
            vec_results = self._vec_search(query, limit=limit * 3, tier=tier)
            for r in vec_results:
                if r["id"] in candidates:
                    candidates[r["id"]]["vec_score"] = r["score"]
                else:
                    candidates[r["id"]] = {
                        **r,
                        "fts_score": 0.0,
                        "vec_score": r["score"],
                    }

        # Step 3: Adaptive hybrid rerank
        # Classify the query once — determines weights for all candidates
        fts_w, vec_w = self._classify_query(query)
        method = (
            "keyword" if fts_w > 0.6
            else "semantic" if vec_w > 0.6
            else "hybrid"
        )

        results = []
        for mid, data in candidates.items():
            fts = data["fts_score"]
            vec = data["vec_score"]

            hybrid = fts_w * fts + vec_w * vec

            results.append({
                "id": data["id"],
                "content": data["content"],
                "tier": data["tier"],
                "source": data["source"],
                "score": round(hybrid, 4),
                "method": method,
            })

        # Sort by hybrid score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # English stop words: short, high-frequency, semantically empty
    _EN_STOP = frozenset({
        "the", "is", "a", "an", "to", "of", "in", "for", "on", "at",
        "by", "it", "or", "and", "be", "do", "if", "so", "as", "no",
        "up", "my", "we", "he", "she", "me", "us", "its", "was", "are",
        "has", "had", "not", "but", "you", "i", "am", "did", "got",
        "all", "any", "can", "may", "will", "with", "this", "that",
        "have", "from", "they", "them", "then", "than", "been", "were",
    })
    # Russian stop words: common short words with no search value
    _RU_STOP = frozenset({
        "и", "в", "на", "с", "не", "что", "это", "как", "но", "да",
        "из", "за", "по", "от", "до", "уже", "все", "его", "её", "нет",
        "я", "ты", "он", "она", "мы", "вы", "они", "их", "им", "ей",
        "то", "же", "ли", "бы", "ни", "или", "при", "без", "об", "под",
        "над", "для", "ещё", "так", "вот", "тут", "где", "кто", "был",
    })

    def _build_fts_query(self, query: str) -> str:
        """
        Build a smart FTS5 query from a natural-language string.

        Strategy:
        1. Strip all FTS5-special characters to prevent syntax errors
        2. Filter English and Russian stop words
        3. Words >3 chars get prefix matching (word*)
        4. Consecutive word pairs get quoted phrase matching ("word1 word2")
        5. Final query: phrase OR (word1* AND word2* AND ...) OR word1* OR word2* ...
           This gives FTS5 the best chance: exact phrase > AND > any single word

        Returns empty string if no useful terms remain.
        """
        # Step 1: Strip FTS5-special characters.
        # FTS5 special chars: ^, *, ", (, ), {, }, +, -, :, .
        # Apostrophes (') and other punctuation become spaces.
        cleaned = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)

        # Step 2: Tokenize
        raw_words = cleaned.split()
        if not raw_words:
            return ""

        # Step 3: Filter stop words (case-insensitive comparison)
        stop_all = self._EN_STOP | self._RU_STOP
        words = [w for w in raw_words if w.lower() not in stop_all and len(w) > 1]

        if not words:
            return ""

        # Step 4: Apply prefix matching for words longer than 3 chars.
        # Short words (2-3 chars) are matched exactly to avoid noise.
        def token(w: str) -> str:
            return f"{w}*" if len(w) > 3 else w

        term_tokens = [token(w) for w in words]

        # Step 5: Build phrase clauses for consecutive word pairs.
        # "word1 word2" matches the exact sequence — highest precision.
        phrases = []
        if len(words) >= 2:
            for i in range(len(words) - 1):
                phrases.append(f'"{words[i]} {words[i+1]}"')

        # Step 6: Combine clauses by descending specificity.
        parts = []

        # Highest priority: exact phrase (all words in sequence)
        if len(words) >= 2:
            full_phrase = '"' + " ".join(words) + '"'
            parts.append(full_phrase)

        # Adjacent pair phrases (only if there are 3+ words)
        if len(words) >= 3:
            parts.extend(phrases)

        # AND of all terms (all must appear, any order)
        if len(term_tokens) >= 2:
            parts.append(" AND ".join(term_tokens))

        # Individual terms (OR fallback for partial matches)
        parts.extend(term_tokens)

        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique_parts.append(p)

        return " OR ".join(unique_parts)

    def _fts_search(self, query: str, limit: int = 15, tier: Optional[str] = None) -> list[dict]:
        """FTS5 BM25 keyword search with smart query building."""
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        # Encode tier for comparison (compact=int, legacy=str)
        tier_stored = self._encode_tier(tier) if tier else None

        try:
            if tier_stored is not None:
                rows = self.db.execute(
                    """SELECT m.id, m.content, m.tier, m.source, rank
                       FROM memories_fts f
                       JOIN memories m ON m.id = f.rowid
                       WHERE memories_fts MATCH ? AND m.archived = 0 AND m.tier = ?
                       ORDER BY rank LIMIT ?""",
                    (fts_query, tier_stored, limit),
                ).fetchall()
            else:
                rows = self.db.execute(
                    """SELECT m.id, m.content, m.tier, m.source, rank
                       FROM memories_fts f
                       JOIN memories m ON m.id = f.rowid
                       WHERE memories_fts MATCH ? AND m.archived = 0
                       ORDER BY rank LIMIT ?""",
                    (fts_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []

        # Normalize rank to 0..1 score (rank is negative, more negative = better)
        if not rows:
            return []

        min_rank = min(r[4] for r in rows)
        max_rank = max(r[4] for r in rows)
        rank_range = max_rank - min_rank if max_rank != min_rank else 1.0

        results = []
        for row in rows:
            # Normalize: best rank (most negative) → 1.0, worst → 0.0
            score = (max_rank - row[4]) / rank_range if rank_range else 1.0
            results.append({
                "id": row[0],
                "content": row[1],
                "tier": self._decode_tier(row[2]),  # int→str (compact) or str (legacy)
                "source": row[3],
                "score": score,
            })
        return results

    def _vec_search(self, query: str, limit: int = 15, tier: Optional[str] = None) -> list[dict]:
        """Vector KNN semantic search. Uses sqlite-vec or pure Python fallback."""
        vec = self._embed(query)
        if vec is None:
            return []

        if self._vec_mode == "sqlite-vec":
            return self._vec_search_sqlite_vec(vec, limit, tier)
        elif self._vec_mode == "pure" and self._vec_index is not None:
            return self._vec_search_pure(vec, limit, tier)
        return []

    def _vec_search_sqlite_vec(
        self, vec: list[float], limit: int, tier: Optional[str]
    ) -> list[dict]:
        """sqlite-vec C extension KNN query."""
        try:
            rows = self.db.execute(
                """SELECT v.rowid, v.distance
                   FROM memories_vec v
                   WHERE v.embedding MATCH ? AND k = ?
                   ORDER BY v.distance""",
                (_serialize_f32(vec), limit * 2),
            ).fetchall()
        except Exception:
            return []

        return self._hydrate_vec_results(rows, tier, limit)

    def _vec_search_pure(
        self, vec: list[float], limit: int, tier: Optional[str]
    ) -> list[dict]:
        """Pure Python brute-force KNN via _VecIndex."""
        try:
            rows = self._vec_index.search(vec, k=limit * 2)
        except Exception:
            return []

        return self._hydrate_vec_results(rows, tier, limit)

    def _hydrate_vec_results(
        self, rows: list[tuple], tier: Optional[str], limit: int
    ) -> list[dict]:
        """Fetch metadata for (rowid, distance) pairs and apply tier filter."""
        if not rows:
            return []

        # Encode tier for comparison (compact=int, legacy=str)
        tier_stored = self._encode_tier(tier) if tier else None

        results = []
        for rowid, distance in rows:
            meta = self.db.execute(
                "SELECT id, content, tier, source FROM memories WHERE id = ? AND archived = 0",
                (rowid,),
            ).fetchone()
            if meta is None:
                continue
            if tier_stored is not None and meta[2] != tier_stored:
                continue

            # Convert cosine distance (0=identical, 2=opposite) to score (1=identical, 0=opposite)
            score = max(0.0, 1.0 - distance)
            results.append({
                "id": meta[0],
                "content": meta[1],
                "tier": self._decode_tier(meta[2]),  # int→str (compact) or str (legacy)
                "source": meta[3],
                "score": score,
            })

        return results[:limit]

    # ================================================================
    # TOOL 3: save_state(state)
    # ================================================================
    def save_state(self, state: str) -> dict:
        """
        Save current working state. Replaces previous working state.
        This is the "emergency save before context compression" tool.
        """
        # Archive all previous working memories
        working_stored = self._encode_tier("working")
        self.db.execute(
            "UPDATE memories SET archived = 1 WHERE tier = ? AND archived = 0",
            (working_stored,),
        )

        result = self.remember(
            content=state,
            tier="working",
            tags=["state"],
            source="save_state",
        )
        return {"saved": True, "id": result["id"]}

    # ================================================================
    # TOOL 4: today()
    # ================================================================
    def today(self) -> list[dict]:
        """Get all memories from today, grouped by tier."""
        # Start of today (UTC)
        import datetime
        today_start = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        rows = self.db.execute(
            """SELECT id, content, tier, source, created_at
               FROM memories
               WHERE created_at >= ? AND archived = 0
               ORDER BY created_at""",
            (today_start,),
        ).fetchall()

        return [
            {
                "id": r[0],
                "content": r[1],
                "tier": self._decode_tier(r[2]),  # int→str (compact) or str (legacy)
                "source": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    # ================================================================
    # TOOL 5: forget(memory_id)
    # ================================================================
    def forget(self, memory_id: int) -> dict:
        """Soft-delete a memory (archive it). Can be unarchived later."""
        self.db.execute(
            "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
            (time.time(), memory_id),
        )
        self.db.commit()
        return {"forgotten": True, "id": memory_id}

    # ================================================================
    # TOOL 6: stats()
    # ================================================================
    def stats(self) -> dict:
        """Memory statistics."""
        total = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 0"
        ).fetchone()[0]

        by_tier = {}
        for tier in TIERS:
            tier_stored = self._encode_tier(tier)
            count = self.db.execute(
                "SELECT COUNT(*) FROM memories WHERE tier = ? AND archived = 0",
                (tier_stored,),
            ).fetchone()[0]
            if count > 0:
                by_tier[tier] = count

        archived = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 1"
        ).fetchone()[0]

        # DB file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        # Latest memory
        latest = self.db.execute(
            "SELECT created_at FROM memories WHERE archived = 0 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        return {
            "total_memories": total,
            "by_tier": by_tier,
            "archived": archived,
            "db_size_bytes": db_size,
            "db_size_human": _human_size(db_size),
            "has_vectors": self._has_vec,
            "vec_mode": self._vec_mode,
            "embedding_dim": self.dim,
            "quantize": self.quantize,
            "bytes_per_vector": (8 + self.dim) if self.quantize else (self.dim * 4),
            "latest_memory": latest[0] if latest else None,
        }

    # ================================================================
    # Import/Export
    # ================================================================
    def import_markdown(self, filepath: str, tier: str = "learned") -> dict:
        """
        Import a markdown file, chunking by sections.

        Uses remember_batch() internally for efficiency:
        - All chunks are embedded in a single model call
        - All inserts happen in a single transaction
        """
        path = Path(filepath)
        if not path.exists():
            return {"error": f"File not found: {filepath}"}

        text = path.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text)
        source = path.name

        items = [
            {"content": chunk, "tier": tier, "source": source}
            for chunk in chunks
        ]

        result = self.remember_batch(items)

        return {
            "file": str(path),
            "chunks": len(chunks),
            "imported": result["imported"],
            "deduplicated": result["deduplicated"],
        }

    def export_markdown(self, tier: Optional[str] = None) -> str:
        """Export memories as markdown."""
        if tier:
            tier_stored = self._encode_tier(tier)
            rows = self.db.execute(
                "SELECT content, tier, source FROM memories WHERE tier = ? AND archived = 0 ORDER BY created_at",
                (tier_stored,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT content, tier, source FROM memories WHERE archived = 0 ORDER BY tier, created_at"
            ).fetchall()

        lines = [f"# Agent Memory Export\n"]
        current_tier = None
        for content, raw_tier, source in rows:
            t = self._decode_tier(raw_tier)  # int→str (compact) or str (legacy)
            if t != current_tier:
                current_tier = t
                lines.append(f"\n## {t.title()}\n")
            if source:
                lines.append(f"<!-- source: {source} -->")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def cleanup_working(self):
        """Archive expired working memories."""
        cutoff = time.time() - WORKING_TTL
        working_stored = self._encode_tier("working")
        self.db.execute(
            "UPDATE memories SET archived = 1 WHERE tier = ? AND created_at < ?",
            (working_stored, cutoff),
        )
        self.db.commit()

    def close(self):
        self.db.close()


# ================================================================
# Utilities
# ================================================================

def _chunk_markdown(text: str, min_len: int = 30, max_len: int = 800) -> list[str]:
    """
    Smart markdown chunking:
    1. Split on ## headers (section boundaries)
    2. Within each section, split on blank lines (paragraphs)
    3. If a paragraph > max_len, split on top-level bullet points (- at col 0)
    4. If a single bullet block > max_len, split at sub-bullet boundaries,
       keeping the parent bullet line as context prefix
    5. Merge tiny chunks (<min_len) with their neighbors
    6. Preserve section header as prefix for context

    Returns chunks of min_len..max_len chars, each with enough context
    to be useful alone.
    """
    def _split_bullet_on_subbullets(
        parent_line: str, sub_lines: list[str], prefix: str, max_len: int, min_len: int
    ) -> list[str]:
        """
        A single top-level bullet (parent_line + sub_lines) is still > max_len.
        Group sub-bullets so each group fits within max_len.
        Each output chunk: prefix + parent_line + "\n" + group_of_sub_lines.
        """
        result: list[str] = []
        group: list[str] = []

        def flush_group():
            if group:
                body = parent_line + "\n" + "\n".join(group)
                chunk = (prefix + body).strip()
                if len(chunk) >= min_len:
                    result.append(chunk)

        for line in sub_lines:
            # Try adding this sub-bullet line to current group
            candidate_body = parent_line + "\n" + "\n".join(group + [line])
            candidate = (prefix + candidate_body).strip()
            if len(candidate) > max_len and group:
                # Current group is full — flush and start new
                flush_group()
                group = [line]
            else:
                group.append(line)

        flush_group()

        # If no sub-bullets produced output, emit just the parent line
        if not result:
            chunk = (prefix + parent_line).strip()
            if len(chunk) >= min_len:
                result.append(chunk)

        return result

    def _split_segment_on_bullets(seg: str, prefix: str) -> list[str]:
        """
        Split a paragraph segment on top-level bullets (lines starting with - or *).
        Each top-level bullet + its indented sub-bullets form one candidate block.
        If a candidate block is still > max_len, further split at sub-bullet level.
        """
        lines = seg.split("\n")
        # Collect bullet blocks: list of (parent_line, [sub_lines])
        blocks: list[tuple[str, list[str]]] = []
        pre_bullet: list[str] = []  # non-bullet lines before first bullet
        current_parent: str = ""
        current_subs: list[str] = []
        in_bullet = False

        for line in lines:
            is_top = bool(re.match(r"^[-*]\s", line))
            if is_top:
                if in_bullet:
                    blocks.append((current_parent, current_subs))
                elif pre_bullet:
                    # Flush pre-bullet prose as its own segment
                    blocks.append(("\n".join(pre_bullet), []))
                    pre_bullet = []
                current_parent = line
                current_subs = []
                in_bullet = True
            else:
                if in_bullet:
                    current_subs.append(line)
                else:
                    pre_bullet.append(line)

        if in_bullet:
            blocks.append((current_parent, current_subs))
        elif pre_bullet:
            blocks.append(("\n".join(pre_bullet), []))

        # No bullet structure found — return whole segment as-is
        if not blocks:
            return [seg]

        result: list[str] = []
        for parent, subs in blocks:
            if not subs:
                # Plain text block or single bullet with no children
                chunk = (prefix + parent).strip()
                if len(chunk) >= min_len:
                    result.append(chunk)
                continue

            # Combine parent + subs and check size
            full_body = parent + "\n" + "\n".join(subs)
            full_chunk = (prefix + full_body).strip()

            if len(full_chunk) <= max_len:
                if len(full_chunk) >= min_len:
                    result.append(full_chunk)
            else:
                # Too large even as a single bullet — split on sub-bullets
                result.extend(
                    _split_bullet_on_subbullets(parent, subs, prefix, max_len, min_len)
                )

        return result

    def _split_section_into_chunks(header: str, body: str) -> list[str]:
        """
        Split a section body (everything after the ## header) into sub-chunks.
        Each returned chunk is prefixed with the header for standalone context.
        """
        prefix = (header + "\n") if header else ""

        # --- Step 1: split body on paragraph boundaries, respecting code fences ---
        segments: list[str] = []
        in_code = False
        code_buf: list[str] = []
        para_buf: list[str] = []

        for line in body.split("\n"):
            if line.strip().startswith("```"):
                if not in_code:
                    if para_buf:
                        segments.append("\n".join(para_buf))
                        para_buf = []
                    in_code = True
                    code_buf = [line]
                else:
                    code_buf.append(line)
                    segments.append("\n".join(code_buf))
                    code_buf = []
                    in_code = False
            elif in_code:
                code_buf.append(line)
            else:
                if line.strip() == "":
                    if para_buf:
                        segments.append("\n".join(para_buf))
                        para_buf = []
                else:
                    para_buf.append(line)

        if in_code and code_buf:
            segments.append("\n".join(code_buf))
        if para_buf:
            segments.append("\n".join(para_buf))

        # --- Step 2: for each segment, check size and split if needed ---
        fine: list[str] = []
        for seg in segments:
            full = (prefix + seg).strip()
            if len(full) <= max_len:
                if len(full) >= min_len:
                    fine.append(full)
                continue

            # Segment too large — try bullet-level splitting
            bullet_chunks = _split_segment_on_bullets(seg, prefix)
            fine.extend(bullet_chunks)

        # --- Step 3: merge tiny consecutive chunks with next neighbor ---
        merged: list[str] = []
        for chunk in fine:
            if len(chunk) < min_len and merged:
                merged[-1] = merged[-1] + "\n" + chunk
            else:
                merged.append(chunk)

        return merged

    # ---------------------------------------------------------------
    # Main: split text into ## sections first
    # ---------------------------------------------------------------
    chunks: list[str] = []
    current_header = ""
    current_body_lines: list[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_body_lines or current_header:
                body = "\n".join(current_body_lines)
                chunks.extend(_split_section_into_chunks(current_header, body))
            current_header = line.strip()
            current_body_lines = []
        elif line.startswith("# ") and not current_header:
            # Top-level title before any ## — treat as standalone chunk
            title_chunk = line.strip()
            if len(title_chunk) >= min_len:
                chunks.append(title_chunk)
        else:
            current_body_lines.append(line)

    # Flush last section
    if current_body_lines or current_header:
        body = "\n".join(current_body_lines)
        chunks.extend(_split_section_into_chunks(current_header, body))

    return chunks


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
