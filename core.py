"""
agentmem.core — The heart of agent memory.

One SQLite file. Dual search (FTS5 keywords + sqlite-vec semantics).
Hybrid ranking. Tiered storage. ~12MB total install.

This is what I wish I had when I wake up in a new session with no context.
"""
from __future__ import annotations

import contextlib
import re
import sqlite3
import struct
import time
import json
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, TypedDict

# ---------------------------------------------------------------------------
# Exception hierarchy — actionable errors for users
# ---------------------------------------------------------------------------

class AgentMemError(Exception):
    """Base exception for all agentmem errors."""
    pass


class MemoryNotFoundError(AgentMemError):
    """Raised when an operation targets a memory ID that does not exist."""
    pass


class InvalidTierError(AgentMemError):
    """Raised when an invalid tier name is used."""
    pass


class EmbeddingError(AgentMemError):
    """Raised when embedding computation fails."""
    pass


# ---------------------------------------------------------------------------
# Typed dicts for public return types
# ---------------------------------------------------------------------------

class RememberResult(TypedDict):
    """Return type of MemoryStore.remember()."""
    id: int
    tier: str
    embedded: bool
    deduplicated: bool


class BatchResult(TypedDict):
    """Return type of MemoryStore.remember_batch()."""
    imported: int
    deduplicated: int
    embedded: int


class RecallResult(TypedDict, total=False):
    """Single item in the list returned by MemoryStore.recall()."""
    id: int
    content: str
    tier: str
    source: str
    score: float
    method: str
    importance: float


class SaveStateResult(TypedDict):
    """Return type of MemoryStore.save_state()."""
    saved: bool
    id: int


class TodayResult(TypedDict):
    """Single item in the list returned by MemoryStore.today()."""
    id: int
    content: str
    tier: str
    source: str
    created_at: float


class ForgetResult(TypedDict, total=False):
    """Return type of MemoryStore.forget()."""
    forgotten: bool
    id: int
    reason: str


class UnarchiveResult(TypedDict, total=False):
    """Return type of MemoryStore.unarchive()."""
    unarchived: bool
    id: int
    reason: str


class StatsResult(TypedDict):
    """Return type of MemoryStore.stats()."""
    total_memories: int
    by_tier: dict[str, int]
    archived: int
    db_size_bytes: int
    db_size_human: str
    has_vectors: bool
    vec_mode: str
    embedding_dim: int
    quantize: bool
    bytes_per_vector: int
    latest_memory: float | None
    avg_importance: float


class CompactResult(TypedDict):
    """Return type of MemoryStore.compact()."""
    archived: int
    dry_run: bool


class ConsolidateResult(TypedDict, total=False):
    """Return type of MemoryStore.consolidate()."""
    groups: int
    archived: int
    dry_run: bool
    details: list[dict[str, Any]]
    error: str


class UpdateResult(TypedDict):
    """Return type of MemoryStore.update_memory()."""
    id: int
    supersedes: int


class HistoryItem(TypedDict):
    """Single item in the list returned by MemoryStore.history()."""
    id: int
    content: str
    created_at: float
    archived: bool


class RelatedResult(TypedDict):
    """Single item in the list returned by MemoryStore.related()."""
    id: int
    content: str
    tier: str
    source: str
    entity_name: str
    entity_type: str


class EntityResult(TypedDict):
    """Single item in the list returned by MemoryStore.entities()."""
    name: str
    type: str
    memory_count: int


class ImportResult(TypedDict, total=False):
    """Return type of MemoryStore.import_markdown()."""
    file: str
    chunks: int
    imported: int
    deduplicated: int
    error: str


class ProcessConversationResult(TypedDict):
    """Return type of MemoryStore.process_conversation()."""
    extracted: int
    by_type: dict[str, int]
    memories: list[int]


# Tiers: how important/permanent is this memory?
TIERS = ("core", "learned", "episodic", "working", "procedural")

# Working memories auto-expire after this many seconds
WORKING_TTL = 86400  # 24 hours

# Schema version — increment when adding migrations
SCHEMA_VERSION = 3

# Compact tier encoding: TEXT → INTEGER for storage (saves 5-6 bytes per row)
_TIER_TO_INT = {"core": 0, "learned": 1, "episodic": 2, "working": 3, "procedural": 4}
_INT_TO_TIER = {v: k for k, v in _TIER_TO_INT.items()}


def _escape_like(s: str) -> str:
    """Escape LIKE wildcard characters (% and _) so they match literally."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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


class _LSHIndex:
    """
    Locality-Sensitive Hashing for approximate nearest neighbor pre-filtering.

    Uses SimHash (random hyperplane) signatures split into bands.
    Zero dependencies — uses only stdlib (random, hash builtins).

    At threshold=0.85 with default params (128 bits, 16 bands, 8 rows):
    - True positive rate: ~100% (virtually never misses real duplicates)
    - Candidate reduction: ~45-85% fewer pairs depending on data diversity

    This turns consolidate() from O(n^2) full cosine comparisons into
    O(n * avg_candidates) where avg_candidates << n for dissimilar memories.
    """

    def __init__(self, dim: int, num_bits: int = 128, bands: int = 16):
        # Generate random hyperplanes (deterministic seed for reproducibility)
        import random as _random
        rng = _random.Random(42)
        self._planes = [
            [rng.gauss(0, 1) for _ in range(dim)]
            for _ in range(num_bits)
        ]
        self._bands = bands
        self._rows = num_bits // bands
        self._buckets: dict[tuple, set] = {}  # (band_idx, hash) -> set of ids

    def _signature(self, vec: list[float]) -> list[int]:
        """Compute binary SimHash signature."""
        return [
            1 if sum(v * p for v, p in zip(vec, plane)) >= 0 else 0
            for plane in self._planes
        ]

    def _band_hashes(self, sig: list[int]) -> list[int]:
        """Split signature into bands and hash each."""
        hashes = []
        for b in range(self._bands):
            start = b * self._rows
            band_bits = tuple(sig[start:start + self._rows])
            hashes.append(hash(band_bits))
        return hashes

    def add(self, item_id: int, vec: list[float]):
        """Index a vector by its LSH bands."""
        sig = self._signature(vec)
        for band_idx, h in enumerate(self._band_hashes(sig)):
            key = (band_idx, h)
            if key not in self._buckets:
                self._buckets[key] = set()
            self._buckets[key].add(item_id)

    def candidates(self, item_id: int, vec: list[float]) -> set[int]:
        """Find candidate near-neighbors (may include false positives, almost no false negatives)."""
        sig = self._signature(vec)
        result = set()
        for band_idx, h in enumerate(self._band_hashes(sig)):
            key = (band_idx, h)
            if key in self._buckets:
                result.update(self._buckets[key])
        result.discard(item_id)  # don't match self
        return result


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
        recency_weight: float = 0.1,
        decay_rate: float = 0.01,
        checkpoint_interval: int = 1000,
    ) -> None:
        self.db_path: Path = Path(db_path)
        self.dim: int = embedding_dim
        self.quantize: bool = quantize  # int8 vector quantization (4x storage reduction)
        self.recency_weight: float = recency_weight  # default recency factor in hybrid scoring
        self.decay_rate: float = decay_rate          # exponential decay rate (per hour)
        self.checkpoint_interval: int = checkpoint_interval  # auto-checkpoint after N writes (0 = disabled)
        self._writes_since_checkpoint: int = 0
        self._closed: bool = False
        self._embed_fn: Callable[[str], list[float]] | None = None        # set via set_embed_fn()
        self._embed_batch_fn: Callable[[list[str]], list[list[float]]] | None = None  # set via set_embed_fn() when model object passed
        self._vec_index: _VecIndex | None = None
        self.db: sqlite3.Connection = self._connect()
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

    def _init_schema(self) -> None:
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
                    namespace TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived INTEGER DEFAULT 0,
                    content_hash BLOB,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    supersedes INTEGER DEFAULT NULL,
                    importance REAL DEFAULT 0.5,
                    UNIQUE(content_hash, namespace)
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

        # Apply versioned migrations (adds columns, creates tables, etc.)
        self._migrate()

        # Index for namespace queries
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_namespace ON memories(namespace)")

        # Index for supersedes lookups (temporal versioning)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_supersedes ON memories(supersedes)")

        # Entities table for lightweight entity extraction
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                memory_id INTEGER NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_entity_memory ON entities(memory_id)")

        self.db.commit()

    # ------------------------------------------------------------------
    # Versioned migration system
    # ------------------------------------------------------------------

    # Each migration is a callable(db: sqlite3.Connection).
    # Key = target version (applied when upgrading FROM key-1 TO key).
    _MIGRATIONS = {
        2: lambda db: [
            db.execute(
                f"ALTER TABLE memories ADD COLUMN {col}"
            )
            for col in (
                "access_count INTEGER DEFAULT 0",
                "last_accessed REAL",
                "namespace TEXT DEFAULT ''",
                "supersedes INTEGER DEFAULT NULL",
                "importance REAL DEFAULT 0.5",
            )
        ],
        3: lambda db: [
            # Drop global UNIQUE on content_hash, create composite UNIQUE on (content_hash, namespace).
            # SQLite cannot DROP constraints, so we create an index instead.
            # The old UNIQUE constraint may still exist on legacy DBs — that's OK,
            # the Python-level check in remember() handles namespace isolation regardless.
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_hash_namespace ON memories(content_hash, namespace)"
            ),
        ],
    }

    def _migrate(self):
        """
        Apply pending schema migrations using PRAGMA user_version.

        - Reads current version from user_version (0 for new/old DBs).
        - For new DBs (compact schema with all columns), just stamps the
          latest version — no ALTER TABLE needed.
        - For old DBs, applies each migration in order, wrapped in a
          transaction, then sets user_version after each step.
        """
        current = self.db.execute("PRAGMA user_version").fetchone()[0]

        if current >= SCHEMA_VERSION:
            return  # Already up to date

        # Check if this is a fresh DB that already has all columns
        # (compact schema created in _init_schema includes everything).
        if current == 0:
            cols = {
                c[1] for c in
                self.db.execute("PRAGMA table_info(memories)").fetchall()
            }
            # All v2 columns present means compact schema — just stamp
            v2_cols = {"access_count", "last_accessed", "namespace",
                       "supersedes", "importance"}
            if v2_cols.issubset(cols):
                self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                self.db.commit()
                return

        # Apply migrations sequentially, each step in its own transaction
        for version in range(current + 1, SCHEMA_VERSION + 1):
            migration_fn = self._MIGRATIONS.get(version)
            if migration_fn is None:
                continue  # No migration defined for this step
            try:
                with self.transaction() as conn:
                    migration_fn(conn)
                    conn.execute(f"PRAGMA user_version = {version}")
            except sqlite3.OperationalError:
                # Column already exists (partially migrated DB) — mark done
                self.db.execute(f"PRAGMA user_version = {version}")
                self.db.commit()

    def set_embed_fn(self, fn: Callable[[str], list[float]] | Any) -> None:
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

    # ------------------------------------------------------------------
    # WAL checkpoint management
    # ------------------------------------------------------------------

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        """
        Run a WAL checkpoint.

        Modes:
            PASSIVE  — checkpoint as much as possible without blocking (default)
            FULL     — blocks writers until checkpoint completes
            TRUNCATE — blocks writers + truncates WAL file to zero bytes

        Returns:
            (busy, log, checkpointed) — pages busy/total/checkpointed
        """
        mode = mode.upper()
        if mode not in ("PASSIVE", "FULL", "TRUNCATE"):
            raise ValueError(f"Invalid checkpoint mode '{mode}'. Use PASSIVE, FULL, or TRUNCATE.")
        row = self.db.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        self._writes_since_checkpoint = 0
        return tuple(row) if row else (0, 0, 0)

    def _track_write(self, count: int = 1) -> None:
        """Increment write counter and auto-checkpoint if threshold hit."""
        self._writes_since_checkpoint += count
        if (
            self.checkpoint_interval > 0
            and self._writes_since_checkpoint >= self.checkpoint_interval
        ):
            self.checkpoint("PASSIVE")

    # ------------------------------------------------------------------
    # Transaction safety
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self):
        """
        Context manager for atomic transactions.

        Usage:
            with store.transaction() as conn:
                conn.execute("INSERT ...")
                conn.execute("UPDATE ...")
            # auto-commits on success, rolls back on exception

        Yields the db connection. BEGIN is issued on entry;
        COMMIT on clean exit, ROLLBACK on exception.
        """
        self.db.execute("BEGIN")
        try:
            yield self.db
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _embed(self, text: str) -> list[float] | None:
        """Embed text. Returns None if no embedding function set."""
        if self._embed_fn is None:
            return None
        try:
            return self._embed_fn(text)
        except EmbeddingError:
            raise
        except Exception as e:
            preview = text[:50] + "..." if len(text) > 50 else text
            raise EmbeddingError(
                f"Embedding failed for text: {preview} — {e}"
            ) from e

    def _embed_batch(self, texts: list[str]) -> list[list[float] | None]:
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

    def _content_hash(self, content: str) -> bytes | str:
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

    def _encode_tier(self, tier: str) -> int | str:
        """
        Encode tier string to storage value.
        Compact schema: returns int (0-4).
        Legacy schema:  returns str unchanged.
        """
        if self._schema_mode == "compact":
            return _TIER_TO_INT.get(tier, 1)  # default 1 = "learned"
        return tier

    def _decode_tier(self, raw: int | str) -> str:
        """
        Decode tier value from storage to public string API.
        Compact schema: int → string.
        Legacy schema:  string unchanged.
        """
        if self._schema_mode == "compact":
            return _INT_TO_TIER.get(raw, "learned")
        return raw if isinstance(raw, str) else "learned"

    def _encode_tags(self, tags: list[str]) -> str:
        """
        Encode tags list to storage string.
        Compact schema: comma-separated (no brackets/quotes overhead).
        Legacy schema:  JSON array string.
        """
        if self._schema_mode == "compact":
            return ",".join(str(t) for t in tags) if tags else ""
        return json.dumps(tags)

    def _decode_tags(self, raw: str) -> list[str]:
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
    # Entity extraction (regex-based, no LLM)
    # ================================================================
    @staticmethod
    def _extract_entities(text: str) -> list[tuple[str, str]]:
        """
        Extract (name, type) entity pairs from text using regex patterns.

        Entity types detected:
        - mention: @username patterns
        - url: http/https URLs
        - email: email addresses
        - hashtag: #tag patterns
        - ip: IPv4 addresses
        - port: :PORT numbers
        - path: /unix/file/paths and ~/paths
        - money: $50, $1,000 etc.
        - number_unit: 100MB, 8080ms, 2.5GB etc.
        - env_var: ENV_VAR_NAMES (all-caps with underscores, 3+ chars)

        Returns list of (entity_name, entity_type) tuples, deduplicated.
        """
        results: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _add(name: str, etype: str):
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                results.append((name, etype))

        # Order matters: more specific patterns first to avoid overlapping matches

        # email (before url and mention to avoid partial matches)
        for m in re.finditer(r'\S+@\S+\.\S+', text):
            _add(m.group(), 'email')

        # url
        for m in re.finditer(r'https?://\S+', text):
            _add(m.group().rstrip('.,;)'), 'url')

        # mention (skip if part of email)
        for m in re.finditer(r'@\w+', text):
            full_context = text[max(0, m.start()-1):m.end()+5]
            # Skip if this @word is part of an email (char before @ is not space/start)
            if m.start() > 0 and text[m.start()-1] not in (' ', '\t', '\n', ',', ';', '(', '['):
                continue
            _add(m.group(), 'mention')

        # hashtag
        for m in re.finditer(r'#\w+', text):
            _add(m.group(), 'hashtag')

        # ip (before port to avoid overlap)
        for m in re.finditer(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text):
            _add(m.group(), 'ip')

        # port
        for m in re.finditer(r':(\d{4,5})\b', text):
            _add(':' + m.group(1), 'port')

        # path (unix paths starting with / or ~/)
        for m in re.finditer(r'(?:^|\s)([~/]\S+)', text, re.MULTILINE):
            path_str = m.group(1)
            # Must have at least one / separator to be a real path
            if '/' in path_str and len(path_str) > 2:
                _add(path_str, 'path')

        # money
        for m in re.finditer(r'\$\d[\d,]*', text):
            _add(m.group(), 'money')

        # number_unit
        for m in re.finditer(r'\b\d+(?:\.\d+)?(?:MB|GB|KB|TB|ms|s|min|hr|h)\b', text):
            _add(m.group(), 'number_unit')

        # env_var (3+ chars, all caps with underscores, must contain at least one underscore or be 4+ chars)
        for m in re.finditer(r'\b[A-Z][A-Z0-9_]{2,}\b', text):
            word = m.group()
            # Filter out common English words that happen to be all caps
            if '_' in word or len(word) >= 4:
                _add(word, 'env_var')

        return results

    def _store_entities(self, memory_id: int, content: str) -> None:
        """Extract entities from content and store them in the entities table."""
        entities = self._extract_entities(content)
        if entities:
            self.db.executemany(
                "INSERT INTO entities (name, type, memory_id) VALUES (?, ?, ?)",
                [(name, etype, memory_id) for name, etype in entities],
            )

    # ================================================================
    # Importance scoring (computed once on insert)
    # ================================================================
    @staticmethod
    def _compute_importance(content: str, tier: str, entities_count: int = 0) -> float:
        """
        Compute importance score (0.0 to 1.0) for a memory using heuristics.

        Factors (weighted):
        - Tier weight (0.3): core=1.0, procedural=0.9, learned=0.6, episodic=0.4, working=0.2
        - Content length (0.2): longer = more info, but capped (100-500 chars is ideal)
        - Specificity (0.2): named entities count (from _extract_entities)
        - Structure (0.15): has bullet points, numbered lists, key-value pairs
        - Actionability (0.15): contains action words, decisions, config values

        Returns float in [0.0, 1.0]
        """
        # --- Tier weight (30%) ---
        tier_weights = {
            "core": 1.0, "procedural": 0.9, "learned": 0.6,
            "episodic": 0.4, "working": 0.2,
        }
        tier_score = tier_weights.get(tier, 0.5)

        # --- Length score (20%) ---
        length = len(content)
        if length < 30:
            length_score = 0.2
        elif length < 100:
            # Linear scale from 0.2 to 1.0 over 30-100 chars
            length_score = 0.2 + 0.8 * (length - 30) / 70.0
        elif length <= 500:
            length_score = 1.0
        elif length <= 800:
            # Linear scale from 1.0 to 0.7 over 500-800 chars
            length_score = 1.0 - 0.3 * (length - 500) / 300.0
        else:
            length_score = 0.7

        # --- Specificity (20%) ---
        specificity_score = min(1.0, entities_count / 5.0)

        # --- Structure (15%) ---
        structure_score = 0.0
        # Bullet points
        if re.search(r'^- ', content, re.MULTILINE):
            structure_score += 0.2
        if re.search(r'^\* ', content, re.MULTILINE):
            structure_score += 0.2
        # Numbered lists
        if re.search(r'^\d+\. ', content, re.MULTILINE):
            structure_score += 0.2
        # Key-value pairs (key: value or key = value)
        if re.search(r'\w+:\s+\S', content):
            structure_score += 0.2
        if re.search(r'\w+=\S', content):
            structure_score += 0.2
        # Code blocks
        if '```' in content:
            structure_score += 0.2
        structure_score = min(1.0, structure_score)

        # --- Actionability (15%) ---
        actionability_score = 0.0
        content_lower = content.lower()
        action_words = [
            "always", "never", "must", "should", "important", "critical",
            "todo", "fixme", "password", "token", "key", "secret",
        ]
        for word in action_words:
            if word in content_lower:
                actionability_score += 0.2
        # Config patterns (KEY=VALUE with uppercase key)
        if re.search(r'[A-Z_]{3,}=\S', content):
            actionability_score += 0.2
        actionability_score = min(1.0, actionability_score)

        # --- Weighted sum ---
        importance = (
            0.30 * tier_score
            + 0.20 * length_score
            + 0.20 * specificity_score
            + 0.15 * structure_score
            + 0.15 * actionability_score
        )

        return round(max(0.0, min(1.0, importance)), 4)

    # ================================================================
    # TOOL 1: remember(content, tier, tags, source)
    # ================================================================
    def remember(
        self,
        content: str,
        tier: str = "learned",
        tags: list[str] | None = None,
        source: str = "",
        namespace: str = "",
        supersedes: int | None = None,
    ) -> RememberResult:
        """
        Store a new memory. Auto-embeds if embedding function is set.
        Deduplicates by content hash.

        Args:
            namespace: Optional isolation namespace (e.g. "agent/alice"). Default "" (global).
            supersedes: Optional ID of an older memory this one replaces. The old memory
                        is auto-archived to maintain a version chain.

        Returns: {"id": int, "tier": str, "embedded": bool}
        """
        if not content or not content.strip():
            raise AgentMemError("Content cannot be empty")

        if tier not in TIERS:
            raise InvalidTierError(
                f"Unknown tier '{tier}'. Valid tiers: {', '.join(TIERS)}"
            )

        tags = tags or []
        now = time.time()
        content_hash = self._content_hash(content)

        # Check for duplicate within the same namespace
        existing = self.db.execute(
            "SELECT id FROM memories WHERE content_hash = ? AND namespace = ? AND archived = 0",
            (content_hash, namespace),
        ).fetchone()

        if existing:
            # Update timestamp and un-archive if needed
            self.db.execute(
                "UPDATE memories SET updated_at = ?, archived = 0 WHERE id = ?",
                (now, existing[0]),
            )
            self.db.commit()
            self._track_write()
            return {"id": existing[0], "tier": tier, "embedded": False, "deduplicated": True}

        # Encode tier and tags for storage
        tier_stored = self._encode_tier(tier)
        tags_stored = self._encode_tags(tags)

        # Insert into main table.
        # On legacy DBs with global UNIQUE(content_hash), same content in a different
        # namespace would raise IntegrityError. We insert with NULL hash to bypass,
        # since the Python-level dedup check (above) already handles namespace-aware dedup.
        try:
            cursor = self.db.execute(
                """INSERT INTO memories (content, tier, source, tags, namespace, created_at, updated_at, content_hash, supersedes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (content, tier_stored, source, tags_stored, namespace, now, now, content_hash, supersedes),
            )
            memory_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            # Legacy DB: global UNIQUE on content_hash fired because same content
            # exists in another namespace. Insert with NULL hash to bypass the constraint.
            cursor = self.db.execute(
                """INSERT INTO memories (content, tier, source, tags, namespace, created_at, updated_at, content_hash, supersedes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (content, tier_stored, source, tags_stored, namespace, now, now, supersedes),
            )
            memory_id = cursor.lastrowid

        # Auto-archive the superseded memory
        if supersedes is not None:
            self.db.execute(
                "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                (now, supersedes),
            )

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

        # Extract and store entities
        self._store_entities(memory_id, content)

        # Compute and store importance score
        entity_count = len(self._extract_entities(content))
        importance = self._compute_importance(content, tier, entity_count)
        self.db.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (importance, memory_id),
        )

        self.db.commit()
        self._track_write()
        return {"id": memory_id, "tier": tier, "embedded": embedded, "deduplicated": False}

    # ================================================================
    # BATCH INSERT: remember_batch(items)
    # ================================================================
    def remember_batch(self, items: list[dict[str, Any]], namespace: str = "") -> BatchResult:
        """
        Store multiple memories efficiently in a single transaction.

        Compared to calling remember() N times, this method:
        - Deduplicates all content hashes upfront (one SELECT)
        - Embeds all new texts in a single model call (embed_batch)
        - Inserts all rows with executemany (one round-trip)
        - Commits once at the end

        Args:
            items: List of dicts, each with keys:
                   - content   (str, required)
                   - tier      (str, optional, default "learned")
                   - tags      (list[str], optional)
                   - source    (str, optional)
                   - namespace (str, optional, overrides batch-level namespace)
            namespace: Default namespace for all items. Each item can override
                       with its own "namespace" key.

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
                raise InvalidTierError(
                    f"Unknown tier '{tier}'. Valid tiers: {', '.join(TIERS)}"
                )
            tags = item.get("tags") or []
            source = item.get("source") or ""
            ns = item.get("namespace", namespace)  # per-item overrides batch default
            h = self._content_hash(content)
            hashed.append((content, tier, tags, source, ns, h))

        # --- Step 2: Find which (hash, namespace) pairs already exist ---
        # We need namespace-aware dedup: same content in different namespaces is NOT a duplicate
        existing_hash_ns = set()  # set of (content_hash_bytes, namespace)
        chunk_size = 900
        all_hashes = list({row[5] for row in hashed})  # unique hashes only
        for i in range(0, len(all_hashes), chunk_size):
            chunk = all_hashes[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT content_hash, namespace FROM memories WHERE content_hash IN ({placeholders}) AND archived = 0",
                chunk,
            ).fetchall()
            for r in rows:
                existing_hash_ns.add((r[0], r[1] or ""))

        # --- Step 3: Split into new vs duplicate ---
        new_items = []      # (content, tier, tags, source, namespace, hash)
        deduplicated = 0

        for row in hashed:
            key = (row[5], row[4] or "")  # (content_hash, namespace)
            if key in existing_hash_ns:
                deduplicated += 1
            else:
                new_items.append(row)
                # Track (hash, namespace) locally to catch duplicates within the same batch
                existing_hash_ns.add(key)

        if not new_items:
            return {"imported": 0, "deduplicated": deduplicated, "embedded": 0}

        # --- Step 4: Batch embed all new texts in one model call ---
        texts = [row[0] for row in new_items]
        vecs: list[list[float] | None] = [None] * len(texts)

        if self._embed_fn is not None:
            raw_vecs = self._embed_batch(texts)
            vecs = raw_vecs  # list of list[float] or None

        # --- Steps 5-9 wrapped in a transaction for atomicity ---
        embedded = 0
        with self.transaction() as conn:
            # --- Step 5: Insert all new memories (executemany) ---
            insert_rows = [
                (content, self._encode_tier(tier), source, self._encode_tags(tags), ns, now, now, h)
                for content, tier, tags, source, ns, h in new_items
            ]
            conn.executemany(
                """INSERT OR IGNORE INTO memories
                   (content, tier, source, tags, namespace, created_at, updated_at, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                insert_rows,
            )

            # --- Step 6: Retrieve inserted IDs (needed for vector table) ---
            new_hashes = [row[5] for row in new_items]
            hash_to_id = {}
            for i in range(0, len(new_hashes), chunk_size):
                chunk = new_hashes[i:i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                id_rows = conn.execute(
                    f"SELECT id, content_hash FROM memories WHERE content_hash IN ({placeholders})",
                    chunk,
                ).fetchall()
                for mem_id, h in id_rows:
                    hash_to_id[h] = mem_id

            # --- Step 7: Batch insert vectors ---
            if self._embed_fn is not None:
                if self._vec_mode == "sqlite-vec":
                    vec_rows = []
                    for (content, tier, tags, source, ns, h), vec in zip(new_items, vecs):
                        if vec is not None and h in hash_to_id:
                            vec_rows.append((hash_to_id[h], _serialize_f32(vec)))
                            embedded += 1
                    if vec_rows:
                        try:
                            conn.executemany(
                                "INSERT OR IGNORE INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                                vec_rows,
                            )
                        except Exception:
                            pass  # Vector insert failed; keyword search still works

                elif self._vec_mode == "pure" and self._vec_index is not None:
                    for (content, tier, tags, source, ns, h), vec in zip(new_items, vecs):
                        if vec is not None and h in hash_to_id:
                            try:
                                self._vec_index.insert(hash_to_id[h], vec)
                                embedded += 1
                            except Exception:
                                pass

            # --- Step 8: Extract and store entities for all new memories ---
            entity_rows = []
            entity_counts = {}  # hash -> entity count (for importance scoring)
            for content, tier, tags, source, ns, h in new_items:
                mem_id = hash_to_id.get(h)
                if mem_id is not None:
                    entities = self._extract_entities(content)
                    entity_counts[h] = len(entities)
                    for name, etype in entities:
                        entity_rows.append((name, etype, mem_id))
            if entity_rows:
                conn.executemany(
                    "INSERT INTO entities (name, type, memory_id) VALUES (?, ?, ?)",
                    entity_rows,
                )

            # --- Step 8b: Compute and store importance scores ---
            importance_rows = []
            for content, tier, tags, source, ns, h in new_items:
                mem_id = hash_to_id.get(h)
                if mem_id is not None:
                    ec = entity_counts.get(h, 0)
                    importance = self._compute_importance(content, tier, ec)
                    importance_rows.append((importance, mem_id))
            if importance_rows:
                conn.executemany(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    importance_rows,
                )

        # Track writes after successful commit
        self._track_write(len(new_items))

        return {
            "imported": len(new_items),
            "deduplicated": deduplicated,
            "embedded": embedded,
        }

    # ================================================================
    # TOOL 9: update_memory(old_id, new_content, ...)
    # ================================================================
    def update_memory(
        self,
        old_id: int,
        new_content: str,
        tier: str | None = None,
        tags: list[str] | None = None,
        namespace: str | None = None,
    ) -> UpdateResult:
        """
        Replace a memory with updated content, preserving the version chain.

        The old memory is archived and linked via supersedes.
        Tier, tags, namespace default to the old memory's values if not specified.

        Returns: {"id": new_id, "supersedes": old_id}
        """
        # Fetch old memory to inherit defaults
        row = self.db.execute(
            "SELECT content, tier, tags, namespace FROM memories WHERE id = ?",
            (old_id,),
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(
                f"Memory #{old_id} not found. Use recall() to search for the memory you want to update."
            )

        old_tier = self._decode_tier(row[1])
        old_tags = self._decode_tags(row[2])
        old_namespace = row[3] or ""

        # Use provided values or fall back to old memory's values
        use_tier = tier if tier is not None else old_tier
        use_tags = tags if tags is not None else old_tags
        use_namespace = namespace if namespace is not None else old_namespace

        result = self.remember(
            content=new_content,
            tier=use_tier,
            tags=use_tags,
            namespace=use_namespace,
            supersedes=old_id,
        )

        # If remember() returned a deduplicated result, the supersedes parameter
        # was ignored and the old memory was NOT archived. Fix that here.
        if result.get("deduplicated"):
            now = time.time()
            # Archive the old memory
            self.db.execute(
                "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                (now, old_id),
            )
            # Link the existing duplicate to the old memory via supersedes
            self.db.execute(
                "UPDATE memories SET supersedes = ? WHERE id = ?",
                (old_id, result["id"]),
            )
            self.db.commit()
            self._track_write()

        return {"id": result["id"], "supersedes": old_id}

    # ================================================================
    # TOOL 10: history(memory_id)
    # ================================================================
    def history(self, memory_id: int) -> list[HistoryItem]:
        """
        Get the version history of a memory (newest first).

        Follows the supersedes chain backward to find all previous versions.
        Returns list of {"id", "content", "created_at", "archived"}.
        """
        chain = []

        # Start from the given memory_id and walk backward
        current_id = memory_id
        visited = set()  # guard against cycles

        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            row = self.db.execute(
                "SELECT id, content, created_at, archived, supersedes FROM memories WHERE id = ?",
                (current_id,),
            ).fetchone()
            if row is None:
                break

            chain.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "archived": bool(row[3]),
            })
            current_id = row[4]  # follow supersedes chain backward

        # Also check if there are newer versions that supersede this memory
        # Walk forward from the original memory_id
        forward_id = memory_id
        forward_visited = set()
        forward_chain = []

        while True:
            forward_row = self.db.execute(
                "SELECT id, content, created_at, archived FROM memories WHERE supersedes = ?",
                (forward_id,),
            ).fetchone()
            if forward_row is None or forward_row[0] in forward_visited:
                break
            forward_visited.add(forward_row[0])
            forward_chain.append({
                "id": forward_row[0],
                "content": forward_row[1],
                "created_at": forward_row[2],
                "archived": bool(forward_row[3]),
            })
            forward_id = forward_row[0]

        # Combine: forward chain (newest first) + current chain (already newest first)
        # forward_chain is oldest-to-newest of items AFTER memory_id
        # chain starts at memory_id and goes backward
        # Result should be newest first
        forward_chain.reverse()  # now newest first
        # Remove duplicates: chain[0] is memory_id, which is not in forward_chain
        result = forward_chain + chain

        return result

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

    def _maybe_translate(self, query: str) -> str:
        """
        Detect non-ASCII-heavy queries and translate to English.

        Uses deep-translator (GoogleTranslator) if available.
        Falls back to original query if translation fails or package not installed.
        """
        # Quick heuristic: if >50% of alpha chars are non-ASCII, likely non-English
        alpha_chars = [c for c in query if c.isalpha()]
        if not alpha_chars:
            return query
        non_ascii = sum(1 for c in alpha_chars if ord(c) > 127)
        if non_ascii / len(alpha_chars) < 0.5:
            return query  # Looks English enough

        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='auto', target='en').translate(query)
            if translated:
                return translated
        except Exception:
            pass
        return query

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
        tier: str | None = None,
        recency_weight: float | None = None,
        decay_rate: float | None = None,
        namespace: str | None = None,
        current_only: bool = True,
        auto_translate: bool = False,
    ) -> list[RecallResult]:
        """
        Hybrid search: FTS5 (keywords) + vector search (semantics) → rerank.

        Strategy:
        1. FTS5 search → candidates with BM25 rank
        2. Vector KNN search → candidates with cosine distance
        3. Merge and rerank → best of both worlds
        4. FTS5 gets boosted for exact keyword matches
        5. Recency boost: newer memories score slightly higher

        Args:
            recency_weight: How much recency affects final score (0.0-1.0). Default from constructor.
            decay_rate: Exponential decay rate per hour. Default from constructor.
            namespace: Filter by namespace. Prefix match: "agent" matches "agent", "agent/alice", etc.
                       None means search all namespaces.
            current_only: If True (default), exclude memories that have been superseded
                          (i.e., memories whose id appears in another memory's supersedes column).
                          This ensures only the latest version of each fact is returned.
            auto_translate: If True, detect non-English queries and translate to English
                           before searching. Requires deep-translator package. Useful when
                           memories are stored in English but queries come in other languages.

        Returns list of {"id", "content", "tier", "source", "score", "method"}
        """
        # Auto-translate non-English queries if requested
        if auto_translate and query:
            query = self._maybe_translate(query)
        # Use instance defaults if not overridden per-call
        if recency_weight is None:
            recency_weight = self.recency_weight
        if decay_rate is None:
            decay_rate = self.decay_rate
        candidates = {}  # id -> {data, fts_score, vec_score}

        # Step 1: FTS5 keyword search
        fts_results = self._fts_search(query, limit=limit * 3, tier=tier, namespace=namespace)
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
            vec_results = self._vec_search(query, limit=limit * 3, tier=tier, namespace=namespace)
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

        # Fetch created_at and importance for all candidate IDs
        now = time.time()
        created_at_map: dict[int, float] = {}
        importance_map: dict[int, float] = {}
        if candidates:
            cand_ids = list(candidates.keys())
            # Fetch in chunks to stay within SQLite variable limits
            for i in range(0, len(cand_ids), 900):
                chunk = cand_ids[i:i + 900]
                placeholders = ",".join("?" * len(chunk))
                rows = self.db.execute(
                    f"SELECT id, created_at, importance FROM memories WHERE id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    created_at_map[row[0]] = row[1]
                    importance_map[row[0]] = row[2] if row[2] is not None else 0.5

        results = []
        for mid, data in candidates.items():
            fts = data["fts_score"]
            vec = data["vec_score"]

            hybrid = fts_w * fts + vec_w * vec

            # Recency boost: newer memories score slightly higher
            if recency_weight > 0 and mid in created_at_map:
                age_hours = max(0.0, (now - created_at_map[mid]) / 3600.0)
                recency = math.exp(-age_hours * decay_rate)
                hybrid = hybrid * (1.0 - recency_weight) + recency * recency_weight

            # Importance boost: multiplicative, subtle (0.8x to 1.2x)
            importance = importance_map.get(mid, 0.5)
            importance_boost = 0.8 + 0.4 * importance
            hybrid *= importance_boost

            results.append({
                "id": data["id"],
                "content": data["content"],
                "tier": data["tier"],
                "source": data["source"],
                "score": round(hybrid, 4),
                "method": method,
                "importance": round(importance, 4),
            })

        # Temporal versioning filter
        if current_only and results:
            # Filter out superseded memories (only keep latest versions)
            result_ids = [r["id"] for r in results]
            superseded_ids = set()
            for i in range(0, len(result_ids), 900):
                chunk = result_ids[i:i + 900]
                placeholders = ",".join("?" * len(chunk))
                rows = self.db.execute(
                    f"SELECT supersedes FROM memories WHERE supersedes IN ({placeholders}) AND supersedes IS NOT NULL",
                    chunk,
                ).fetchall()
                for row in rows:
                    superseded_ids.add(row[0])
            if superseded_ids:
                results = [r for r in results if r["id"] not in superseded_ids]
        elif not current_only:
            # Include archived superseded versions: for each result that has a
            # supersedes chain, pull in the older archived versions too.
            existing_ids = {r["id"] for r in results}
            # Find all archived memories that are part of supersedes chains
            # and match any of the current result IDs
            chain_ids_to_check = list(existing_ids)
            visited = set(existing_ids)
            while chain_ids_to_check:
                batch = chain_ids_to_check[:900]
                chain_ids_to_check = chain_ids_to_check[900:]
                placeholders = ",".join("?" * len(batch))
                # Walk backward: find what these memories supersede
                rows = self.db.execute(
                    f"SELECT id, supersedes FROM memories WHERE id IN ({placeholders}) AND supersedes IS NOT NULL",
                    batch,
                ).fetchall()
                for _, sup_id in rows:
                    if sup_id not in visited:
                        visited.add(sup_id)
                        chain_ids_to_check.append(sup_id)
            # Also walk forward: find memories that supersede any of our results
            forward_check = list(existing_ids)
            while forward_check:
                batch = forward_check[:900]
                forward_check = forward_check[900:]
                placeholders = ",".join("?" * len(batch))
                rows = self.db.execute(
                    f"SELECT id FROM memories WHERE supersedes IN ({placeholders})",
                    batch,
                ).fetchall()
                for (fwd_id,) in rows:
                    if fwd_id not in visited:
                        visited.add(fwd_id)
                        forward_check.append(fwd_id)
            # Fetch and add any chain members not already in results
            new_ids = visited - existing_ids
            if new_ids:
                for nid in new_ids:
                    row = self.db.execute(
                        "SELECT id, content, tier, source FROM memories WHERE id = ?",
                        (nid,),
                    ).fetchone()
                    if row:
                        results.append({
                            "id": row[0],
                            "content": row[1],
                            "tier": self._decode_tier(row[2]),
                            "source": row[3],
                            "score": 0.0,  # no search score for chain members
                            "method": method,
                        })

        # Sort by hybrid score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        final = results[:limit]

        # Update access tracking for returned results
        if final:
            returned_ids = [r["id"] for r in final]
            placeholders = ",".join("?" * len(returned_ids))
            self.db.execute(
                f"UPDATE memories SET access_count = access_count + 1, last_accessed = ? "
                f"WHERE id IN ({placeholders})",
                [now] + returned_ids,
            )
            self.db.commit()

        return final

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

    def _fts_search(
        self, query: str, limit: int = 15, tier: str | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS5 BM25 keyword search with smart query building."""
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        # Encode tier for comparison (compact=int, legacy=str)
        tier_stored = self._encode_tier(tier) if tier else None

        # Build dynamic WHERE clause for tier + namespace
        where_parts = ["memories_fts MATCH ?", "m.archived = 0"]
        params: list = [fts_query]

        if tier_stored is not None:
            where_parts.append("m.tier = ?")
            params.append(tier_stored)

        if namespace is not None:
            where_parts.append("(m.namespace = ? OR m.namespace LIKE ? ESCAPE '\\')")
            params.extend([namespace, _escape_like(namespace) + "/%"])

        where_clause = " AND ".join(where_parts)
        params.append(limit)

        try:
            rows = self.db.execute(
                f"""SELECT m.id, m.content, m.tier, m.source, rank
                    FROM memories_fts f
                    JOIN memories m ON m.id = f.rowid
                    WHERE {where_clause}
                    ORDER BY rank LIMIT ?""",
                params,
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

    def _vec_search(
        self, query: str, limit: int = 15, tier: str | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector KNN semantic search. Uses sqlite-vec or pure Python fallback."""
        vec = self._embed(query)
        if vec is None:
            return []

        if self._vec_mode == "sqlite-vec":
            return self._vec_search_sqlite_vec(vec, limit, tier, namespace)
        elif self._vec_mode == "pure" and self._vec_index is not None:
            return self._vec_search_pure(vec, limit, tier, namespace)
        return []

    def _vec_search_sqlite_vec(
        self, vec: list[float], limit: int, tier: str | None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
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

        return self._hydrate_vec_results(rows, tier, limit, namespace)

    def _vec_search_pure(
        self, vec: list[float], limit: int, tier: str | None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure Python brute-force KNN via _VecIndex."""
        try:
            rows = self._vec_index.search(vec, k=limit * 2)
        except Exception:
            return []

        return self._hydrate_vec_results(rows, tier, limit, namespace)

    def _hydrate_vec_results(
        self, rows: list[tuple[int, float]], tier: str | None, limit: int,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch metadata for (rowid, distance) pairs and apply tier/namespace filter."""
        if not rows:
            return []

        # Encode tier for comparison (compact=int, legacy=str)
        tier_stored = self._encode_tier(tier) if tier else None

        # Single batch query instead of N+1 per-ID queries
        rowids = [rowid for rowid, _ in rows]
        distance_map = {rowid: distance for rowid, distance in rows}

        # Fetch all metadata in one query (chunk if needed for SQLite limits)
        meta_map: dict[int, tuple] = {}
        chunk_size = 900
        for i in range(0, len(rowids), chunk_size):
            chunk = rowids[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            meta_rows = self.db.execute(
                f"SELECT id, content, tier, source, namespace FROM memories "
                f"WHERE id IN ({placeholders}) AND archived = 0",
                chunk,
            ).fetchall()
            for meta in meta_rows:
                meta_map[meta[0]] = meta

        results = []
        for rowid in rowids:
            meta = meta_map.get(rowid)
            if meta is None:
                continue
            if tier_stored is not None and meta[2] != tier_stored:
                continue
            # Namespace prefix filter
            if namespace is not None:
                ns = meta[4] or ""
                if ns != namespace and not ns.startswith(namespace + "/"):
                    continue

            # Convert cosine distance (0=identical, 2=opposite) to score (1=identical, 0=opposite)
            distance = distance_map.get(rowid)
            if distance is None:
                continue
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
    def save_state(self, state: str, namespace: str = "") -> SaveStateResult:
        """
        Save current working state. Replaces previous working state.
        This is the "emergency save before context compression" tool.

        Args:
            namespace: Isolate state per namespace. Default "" (global).
        """
        # Archive all previous working memories (only in this namespace)
        working_stored = self._encode_tier("working")
        if namespace:
            self.db.execute(
                "UPDATE memories SET archived = 1 WHERE tier = ? AND archived = 0 AND namespace = ?",
                (working_stored, namespace),
            )
        else:
            self.db.execute(
                "UPDATE memories SET archived = 1 WHERE tier = ? AND archived = 0",
                (working_stored,),
            )

        result = self.remember(
            content=state,
            tier="working",
            tags=["state"],
            source="save_state",
            namespace=namespace,
        )
        return {"saved": True, "id": result["id"]}

    # ================================================================
    # TOOL 4: today()
    # ================================================================
    def today(self, namespace: str | None = None) -> list[TodayResult]:
        """Get all memories from today, grouped by tier.

        Args:
            namespace: Filter by namespace prefix. None means all namespaces.
        """
        # Start of today (UTC)
        import datetime
        today_start = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        if namespace is not None:
            rows = self.db.execute(
                """SELECT id, content, tier, source, created_at
                   FROM memories
                   WHERE created_at >= ? AND archived = 0
                     AND (namespace = ? OR namespace LIKE ? ESCAPE '\\')
                   ORDER BY created_at""",
                (today_start, namespace, _escape_like(namespace) + "/%"),
            ).fetchall()
        else:
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
    def forget(self, memory_id: int, namespace: str | None = None) -> ForgetResult:
        """Soft-delete a memory (archive it). Can be unarchived later.

        Args:
            namespace: If set, only forget if memory belongs to this namespace (safety guard).

        Raises:
            MemoryNotFoundError: If memory_id does not exist.
        """
        # Check existence first
        row = self.db.execute(
            "SELECT namespace FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(
                f"Memory #{memory_id} not found. Use recall() to search for memories."
            )

        if namespace is not None:
            mem_ns = row[0] or ""
            if mem_ns != namespace and not mem_ns.startswith(namespace + "/"):
                return {"forgotten": False, "id": memory_id, "reason": "namespace mismatch"}

        self.db.execute(
            "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
            (time.time(), memory_id),
        )
        self.db.commit()
        self._track_write()
        return {"forgotten": True, "id": memory_id}

    # ================================================================
    # TOOL 6: stats()
    # ================================================================
    def stats(self, namespace: str | None = None) -> StatsResult:
        """Memory statistics.

        Args:
            namespace: If set, only count memories in this namespace (prefix match).
        """
        ns_filter = ""
        ns_params: list = []
        if namespace is not None:
            ns_filter = " AND (namespace = ? OR namespace LIKE ? ESCAPE '\\')"
            ns_params = [namespace, _escape_like(namespace) + "/%"]

        total = self.db.execute(
            f"SELECT COUNT(*) FROM memories WHERE archived = 0{ns_filter}",
            ns_params,
        ).fetchone()[0]

        by_tier = {}
        for tier in TIERS:
            tier_stored = self._encode_tier(tier)
            count = self.db.execute(
                f"SELECT COUNT(*) FROM memories WHERE tier = ? AND archived = 0{ns_filter}",
                [tier_stored] + ns_params,
            ).fetchone()[0]
            if count > 0:
                by_tier[tier] = count

        archived = self.db.execute(
            f"SELECT COUNT(*) FROM memories WHERE archived = 1{ns_filter}",
            ns_params,
        ).fetchone()[0]

        # DB file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        # Latest memory
        latest = self.db.execute(
            f"SELECT created_at FROM memories WHERE archived = 0{ns_filter} ORDER BY created_at DESC LIMIT 1",
            ns_params,
        ).fetchone()

        # Average importance
        avg_imp_row = self.db.execute(
            f"SELECT AVG(importance) FROM memories WHERE archived = 0{ns_filter}",
            ns_params,
        ).fetchone()
        avg_importance = round(avg_imp_row[0], 4) if avg_imp_row and avg_imp_row[0] is not None else 0.5

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
            "avg_importance": avg_importance,
        }

    # ================================================================
    # Import/Export
    # ================================================================
    def import_markdown(self, filepath: str, tier: str = "learned", namespace: str = "") -> ImportResult:
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

        result = self.remember_batch(items, namespace=namespace)

        return {
            "file": str(path),
            "chunks": len(chunks),
            "imported": result["imported"],
            "deduplicated": result["deduplicated"],
        }

    def export_markdown(self, tier: str | None = None) -> str:
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

    # ================================================================
    # TOOL 7: compact()
    # ================================================================
    def compact(
        self,
        max_age_days: int = 90,
        min_access: int = 0,
        tier: str | None = None,
        namespace: str | None = None,
        dry_run: bool = False,
    ) -> CompactResult:
        """
        Archive low-value memories to reduce noise in search results.

        Archives memories that match ALL of these criteria:
        - Older than max_age_days
        - access_count <= min_access
        - Not in 'core' or 'procedural' tier (these are never auto-archived)
        - Optionally filtered by tier and namespace

        Args:
            max_age_days: Archive memories older than this (default 90)
            min_access: Archive memories accessed this many times or less (default 0 = never accessed)
            tier: Only compact this tier (default: all except core and procedural)
            namespace: Only compact this namespace
            dry_run: If True, return count but don't archive

        Returns:
            {"archived": int, "dry_run": bool}
        """
        cutoff = time.time() - max_age_days * 86400
        core_stored = self._encode_tier("core")
        proc_stored = self._encode_tier("procedural")

        where_parts = [
            "archived = 0",
            "tier != ?",
            "tier != ?",
            "created_at < ?",
            "access_count <= ?",
        ]
        params: list = [core_stored, proc_stored, cutoff, min_access]

        if tier is not None:
            tier_stored = self._encode_tier(tier)
            where_parts.append("tier = ?")
            params.append(tier_stored)

        if namespace is not None:
            where_parts.append("(namespace = ? OR namespace LIKE ? ESCAPE '\\')")
            params.extend([namespace, _escape_like(namespace) + "/%"])

        where_clause = " AND ".join(where_parts)

        if dry_run:
            count = self.db.execute(
                f"SELECT COUNT(*) FROM memories WHERE {where_clause}",
                params,
            ).fetchone()[0]
            return {"archived": count, "dry_run": True}

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE memories SET archived = 1 WHERE {where_clause}",
                params,
            )
            archived_count = cursor.rowcount

        self._track_write(archived_count)
        return {"archived": archived_count, "dry_run": False}

    # ================================================================
    # TOOL 8: unarchive(memory_id)
    # ================================================================
    def unarchive(self, memory_id: int) -> UnarchiveResult:
        """Restore an archived memory."""
        cursor = self.db.execute(
            "UPDATE memories SET archived = 0 WHERE id = ? AND archived = 1",
            (memory_id,),
        )
        self.db.commit()
        if cursor.rowcount == 0:
            return {"unarchived": False, "id": memory_id, "reason": "not found or not archived"}
        return {"unarchived": True, "id": memory_id}

    # ================================================================
    # TOOL 12: related(entity)
    # ================================================================
    def related(
        self,
        entity: str,
        entity_type: str | None = None,
        limit: int = 10,
        namespace: str | None = None,
    ) -> list[RelatedResult]:
        """
        Find memories related to a specific entity.

        Searches the entities table for matching entity names (case-insensitive),
        then returns the associated memories.

        Args:
            entity: Entity name to search for (e.g. "@username", "10.0.0.1")
            entity_type: Optional filter by type (mention, url, ip, etc.)
            limit: Max results
            namespace: Optional namespace filter

        Returns list of {"id", "content", "tier", "source", "entity_name", "entity_type"}
        """
        where_parts = ["LOWER(e.name) = LOWER(?)"]
        params: list = [entity]

        if entity_type is not None:
            where_parts.append("e.type = ?")
            params.append(entity_type)

        where_clause = " AND ".join(where_parts)

        # Join with memories to get content and apply namespace/archived filters
        ns_filter = ""
        if namespace is not None:
            ns_filter = " AND (m.namespace = ? OR m.namespace LIKE ? ESCAPE '\\')"
            params.extend([namespace, _escape_like(namespace) + "/%"])

        params.append(limit)

        rows = self.db.execute(
            f"""SELECT DISTINCT m.id, m.content, m.tier, m.source, e.name, e.type
                FROM entities e
                JOIN memories m ON m.id = e.memory_id
                WHERE {where_clause} AND m.archived = 0{ns_filter}
                ORDER BY m.created_at DESC
                LIMIT ?""",
            params,
        ).fetchall()

        return [
            {
                "id": r[0],
                "content": r[1],
                "tier": self._decode_tier(r[2]),
                "source": r[3],
                "entity_name": r[4],
                "entity_type": r[5],
            }
            for r in rows
        ]

    # ================================================================
    # TOOL 13: entities()
    # ================================================================
    def entities(self, entity_type: str | None = None, limit: int = 50) -> list[EntityResult]:
        """
        List all known entities, optionally filtered by type.
        Returns list of {"name", "type", "memory_count"} sorted by memory_count desc.
        """
        if entity_type is not None:
            rows = self.db.execute(
                """SELECT e.name, e.type, COUNT(DISTINCT e.memory_id) as cnt
                   FROM entities e
                   JOIN memories m ON m.id = e.memory_id
                   WHERE e.type = ? AND m.archived = 0
                   GROUP BY LOWER(e.name), e.type
                   ORDER BY cnt DESC
                   LIMIT ?""",
                (entity_type, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT e.name, e.type, COUNT(DISTINCT e.memory_id) as cnt
                   FROM entities e
                   JOIN memories m ON m.id = e.memory_id
                   WHERE m.archived = 0
                   GROUP BY LOWER(e.name), e.type
                   ORDER BY cnt DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            {"name": r[0], "type": r[1], "memory_count": r[2]}
            for r in rows
        ]

    # ================================================================
    # TOOL 11: consolidate()
    # ================================================================
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors. Returns 0.0-1.0."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def consolidate(
        self,
        similarity_threshold: float = 0.85,
        namespace: str | None = None,
        tier: str | None = None,
        merge_fn: Callable[[list[dict[str, Any]]], str] | None = None,
        dry_run: bool = False,
    ) -> ConsolidateResult:
        """
        Find and merge semantically similar memories.

        Algorithm:
        1. Load all active (non-archived) memories (optionally filtered by namespace/tier)
        2. Embed all of them
        3. For each pair with cosine similarity > threshold, group them
        4. For each group: keep the longest/newest memory, archive the rest
           OR use merge_fn(memories) -> merged_content if provided

        Args:
            similarity_threshold: Min cosine similarity to consider as duplicates (0.85 default)
            namespace: Only consolidate within this namespace
            tier: Only consolidate this tier
            merge_fn: Optional callback fn(list[dict]) -> str that merges multiple memories
                      into one. Each dict has {id, content, tier, created_at}.
                      If None, keeps the longest content and archives the rest.
            dry_run: If True, return groups but don't modify

        Returns:
            {"groups": int, "archived": int, "dry_run": bool, "details": list[dict]}
            where details is list of {"kept": id, "archived_ids": [int], "contents_preview": [str]}
        """
        if self._embed_fn is None:
            raise EmbeddingError(
                "consolidate() requires embeddings. Call set_embed_fn() first."
            )

        # --- Step 1: Load all active memories matching filters ---
        where_parts = ["archived = 0"]
        params: list = []

        if tier is not None:
            tier_stored = self._encode_tier(tier)
            where_parts.append("tier = ?")
            params.append(tier_stored)

        if namespace is not None:
            where_parts.append("(namespace = ? OR namespace LIKE ? ESCAPE '\\')")
            params.extend([namespace, _escape_like(namespace) + "/%"])

        where_clause = " AND ".join(where_parts)

        rows = self.db.execute(
            f"SELECT id, content, tier, created_at FROM memories WHERE {where_clause} ORDER BY id",
            params,
        ).fetchall()

        if len(rows) < 2:
            return {"groups": 0, "archived": 0, "dry_run": dry_run, "details": []}

        # --- Step 2: Embed all memories ---
        texts = [row[1] for row in rows]
        vecs = self._embed_batch(texts)

        # Build list of (id, content, tier_str, created_at, vec) for memories with valid embeddings
        memories = []
        for row, vec in zip(rows, vecs):
            if vec is not None:
                memories.append({
                    "id": row[0],
                    "content": row[1],
                    "tier": self._decode_tier(row[2]),
                    "created_at": row[3],
                    "vec": vec,
                })

        if len(memories) < 2:
            return {"groups": 0, "archived": 0, "dry_run": dry_run, "details": []}

        # --- Step 3: Union-Find grouping by cosine similarity ---
        n = len(memories)
        # parent[i] = parent index in memories list
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Compare pairs — use LSH pre-filtering for large datasets
        if n > 100:
            # Build LSH index for candidate pre-filtering.
            # With 128-bit SimHash, 16 bands of 8 rows: ~100% recall at threshold=0.85,
            # typically 2-4x fewer pairs to check on diverse data.
            dim = len(memories[0]["vec"])
            lsh = _LSHIndex(dim=dim)
            for i, m in enumerate(memories):
                lsh.add(i, m["vec"])

            # Collect all unique candidate pairs via LSH
            candidate_pairs: set[tuple[int, int]] = set()
            for i in range(n):
                for j in lsh.candidates(i, memories[i]["vec"]):
                    if j > i:
                        candidate_pairs.add((i, j))
                    elif j < i:
                        candidate_pairs.add((j, i))

            # Only do expensive cosine similarity on candidate pairs
            for i, j in candidate_pairs:
                sim = self._cosine_similarity(memories[i]["vec"], memories[j]["vec"])
                if sim >= similarity_threshold:
                    union(i, j)
        else:
            # Small dataset — brute force O(n^2) is fine
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._cosine_similarity(memories[i]["vec"], memories[j]["vec"])
                    if sim >= similarity_threshold:
                        union(i, j)

        # Collect groups (only groups with 2+ members)
        from collections import defaultdict
        groups_map: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups_map[root].append(i)

        groups = [indices for indices in groups_map.values() if len(indices) >= 2]

        if not groups:
            return {"groups": 0, "archived": 0, "dry_run": dry_run, "details": []}

        # --- Step 4: For each group, decide what to keep and what to archive ---
        total_archived = 0
        details = []

        now = time.time()

        if not dry_run and merge_fn is None:
            # Wrap all archiving in a single transaction (atomic)
            with self.transaction() as conn:
                for group_indices in groups:
                    group_memories = [memories[i] for i in group_indices]
                    group_memories.sort(
                        key=lambda m: (len(m["content"]), m["created_at"]),
                        reverse=True,
                    )
                    kept = group_memories[0]
                    to_archive = group_memories[1:]
                    archived_ids = [m["id"] for m in to_archive]

                    for mid in archived_ids:
                        conn.execute(
                            "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                            (now, mid),
                        )

                    existing_supersedes = conn.execute(
                        "SELECT supersedes FROM memories WHERE id = ?",
                        (kept["id"],),
                    ).fetchone()
                    if existing_supersedes and existing_supersedes[0] is None:
                        conn.execute(
                            "UPDATE memories SET supersedes = ? WHERE id = ?",
                            (archived_ids[0], kept["id"]),
                        )

                    kept_id = kept["id"]
                    total_archived += len(to_archive)

                    details.append({
                        "kept": kept_id,
                        "archived_ids": [m["id"] for m in to_archive],
                        "contents_preview": [m["content"][:80] for m in group_memories],
                    })

            self._track_write(total_archived)
        else:
            # dry_run or merge_fn path
            for group_indices in groups:
                group_memories = [memories[i] for i in group_indices]
                group_memories.sort(
                    key=lambda m: (len(m["content"]), m["created_at"]),
                    reverse=True,
                )
                kept = group_memories[0]
                to_archive = group_memories[1:]
                archived_ids = [m["id"] for m in to_archive]

                if merge_fn is not None:
                    merge_input = [
                        {"id": m["id"], "content": m["content"], "tier": m["tier"], "created_at": m["created_at"]}
                        for m in group_memories
                    ]
                    merged_content = merge_fn(merge_input)

                    if not dry_run:
                        all_ids = [m["id"] for m in group_memories]
                        for mid in all_ids:
                            self.db.execute(
                                "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                                (now, mid),
                            )

                        result = self.remember(
                            content=merged_content,
                            tier=kept["tier"],
                            supersedes=kept["id"],
                        )
                        kept_id = result["id"]
                        archived_ids = all_ids
                    else:
                        kept_id = kept["id"]
                        archived_ids = [m["id"] for m in group_memories]
                else:
                    kept_id = kept["id"]

                total_archived += len(archived_ids) if merge_fn is not None and not dry_run else len(to_archive)

                details.append({
                    "kept": kept_id,
                    "archived_ids": archived_ids if merge_fn is not None else [m["id"] for m in to_archive],
                    "contents_preview": [m["content"][:80] for m in group_memories],
                })

            if not dry_run:
                self.db.commit()
                self._track_write(total_archived)

        return {
            "groups": len(groups),
            "archived": total_archived,
            "dry_run": dry_run,
            "details": details,
        }

    def cleanup_working(self) -> None:
        """Archive expired working memories."""
        cutoff = time.time() - WORKING_TTL
        working_stored = self._encode_tier("working")
        self.db.execute(
            "UPDATE memories SET archived = 1 WHERE tier = ? AND created_at < ?",
            (working_stored, cutoff),
        )
        self.db.commit()

    # ================================================================
    # TOOL 14: get_procedures(namespace)
    # ================================================================
    def get_procedures(self, namespace: str | None = None) -> str:
        """
        Get all active procedural memories formatted for system prompt injection.

        Returns a formatted string ready to prepend to an agent's system prompt:

        ## Agent Rules (from memory)
        - Rule 1 content here
        - Rule 2 content here
        ...

        If no procedural memories exist, returns empty string.

        Args:
            namespace: Optional namespace filter
        """
        proc_stored = self._encode_tier("procedural")

        where_parts = ["archived = 0", "tier = ?"]
        params: list = [proc_stored]

        if namespace is not None:
            where_parts.append("(namespace = ? OR namespace LIKE ? ESCAPE '\\')")
            params.extend([namespace, _escape_like(namespace) + "/%"])

        where_clause = " AND ".join(where_parts)
        rows = self.db.execute(
            f"SELECT content FROM memories WHERE {where_clause} ORDER BY created_at ASC",
            params,
        ).fetchall()

        if not rows:
            return ""

        lines = ["## Agent Rules (from memory)"]
        for row in rows:
            lines.append(f"- {row[0]}")

        return "\n".join(lines)

    # ================================================================
    # TOOL 15: add_procedure(rule)
    # ================================================================
    def add_procedure(
        self,
        rule: str,
        tags: list[str] | None = None,
        namespace: str = "",
    ) -> RememberResult:
        """
        Add a behavioral rule (procedural memory).

        Convenience wrapper around remember() with tier="procedural".
        Procedural memories are never auto-expired or auto-compacted.

        Args:
            rule: The behavioral rule text (e.g., "Always respond in bullet points")
            tags: Optional tags
            namespace: Optional namespace
        """
        return self.remember(content=rule, tier="procedural", tags=tags, namespace=namespace)

    # ================================================================
    # Conversation Auto-Extraction (regex heuristics only, no LLM)
    # ================================================================

    @staticmethod
    def _extract_facts(text: str) -> list[tuple[str, str, str]]:
        """Extract "X is Y" / "X are Y" style facts. Returns (text, type, tier)."""
        results = []
        # "X is/are/was/were Y" — sentence-level
        for m in re.finditer(
            r'(?:^|\.\s+)([A-Z][^.]*?\s+(?:is|are|was|were)\s+[^.]+\.)',
            text, re.MULTILINE
        ):
            val = m.group(1).strip()
            if len(val) > 10 and len(val) < 500:
                results.append((val, "facts", "learned"))
        # key: value or key = value patterns
        for m in re.finditer(r'(\w[\w\s]{1,30}?)\s*[:=]\s*(\S[^\n]{5,})', text):
            key = m.group(1).strip()
            value = m.group(2).strip()
            full = f"{key}: {value}"
            if len(full) > 10 and len(full) < 500:
                results.append((full, "facts", "learned"))
        return results

    @staticmethod
    def _extract_decisions(text: str) -> list[tuple[str, str, str]]:
        """Extract decisions. Returns (text, type, tier)."""
        results = []
        patterns = [
            r"(?:I|we|I've|we've)\s+decided\s+(?:to\s+)?([^.!]+[.!]?)",
            r"(?:decision|decided|conclusion)\s*:\s*([^\n]+)",
            r"let'?s\s+(?:go\s+with|use|do|try)\s+([^.!]+[.!]?)",
            r"going\s+(?:to|with)\s+([^.!]+[.!]?)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = m.group(1).strip().rstrip(".")
                if len(val) > 5 and len(val) < 500:
                    # Reconstruct with context
                    full_match = m.group(0).strip()
                    results.append((full_match, "decisions", "episodic"))
        return results

    @staticmethod
    def _extract_preferences(text: str) -> list[tuple[str, str, str]]:
        """Extract preferences/rules. Returns (text, type, tier)."""
        results = []
        patterns = [
            r"(?:always|never|must always|must never|should always|should never)\s+([^.!]+[.!]?)",
            r"(?:I|we)\s+(?:prefer|like|want|need)\s+([^.!]+[.!]?)",
            r"(?:don'?t|do not)\s+(?:ever\s+)?(?:use|do|like|want)\s+([^.!]+[.!]?)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                full_match = m.group(0).strip()
                if len(full_match) > 5 and len(full_match) < 500:
                    results.append((full_match, "preferences", "procedural"))
        return results

    @staticmethod
    def _extract_todos(text: str) -> list[tuple[str, str, str]]:
        """Extract action items. Returns (text, type, tier)."""
        results = []
        patterns = [
            r"(?:TODO|FIXME|HACK|XXX)\s*:?\s*([^\n]+)",
            r"(?:need to|needs to|should|must|have to|gotta)\s+([^.!]+[.!]?)",
            r"(?:remember to|don'?t forget)\s+([^.!]+[.!]?)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = m.group(1).strip()
                if len(val) > 3 and len(val) < 500:
                    full_match = m.group(0).strip()
                    results.append((full_match, "todos", "working"))
        return results

    @staticmethod
    def _extract_config(text: str) -> list[tuple[str, str, str]]:
        """Extract configuration values. Returns (text, type, tier)."""
        results = []
        # ENV_VAR=value
        for m in re.finditer(r'([A-Z][A-Z0-9_]{2,})\s*=\s*(\S+)', text):
            full = f"{m.group(1)}={m.group(2)}"
            if len(full) < 500:
                results.append((full, "config", "core"))
        # "set X to Y" / "change X to Y"
        for m in re.finditer(r'(?:set|change|update)\s+(\w+)\s+to\s+([^.!,]+)', text, re.IGNORECASE):
            full = m.group(0).strip()
            if len(full) > 5 and len(full) < 500:
                results.append((full, "config", "core"))
        # "port/host/url/endpoint/key/token/path is/:/= value"
        for m in re.finditer(
            r'(?:port|host|url|endpoint|key|token|path)\s*(?:is|:|=)\s*(\S+)',
            text, re.IGNORECASE
        ):
            full = m.group(0).strip()
            if len(full) > 5 and len(full) < 500:
                results.append((full, "config", "core"))
        return results

    @staticmethod
    def _extract_learnings(text: str) -> list[tuple[str, str, str]]:
        """Extract learnings. Returns (text, type, tier)."""
        results = []
        patterns = [
            r"(?:TIL|today I learned|learned that|turns out|found out|discovered)\s*:?\s*([^\n.!]+[.!]?)",
            r"(?:the\s+(?:trick|solution|fix|answer|key)\s+(?:is|was))\s+([^.!]+[.!]?)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = m.group(1).strip()
                if len(val) > 5 and len(val) < 500:
                    full_match = m.group(0).strip()
                    results.append((full_match, "learnings", "learned"))
        return results

    @staticmethod
    def _extract_important(text: str) -> list[tuple[str, str, str]]:
        """Extract important notes. Returns (text, type, tier)."""
        results = []
        # "important:/note:/NB:/critical:/warning:" patterns
        for m in re.finditer(
            r'(?:important|note|NB|critical|warning|caution|attention)\s*:\s*([^\n]+)',
            text, re.IGNORECASE
        ):
            val = m.group(1).strip()
            if len(val) > 5 and len(val) < 500:
                full_match = m.group(0).strip()
                results.append((full_match, "important", "core"))
        return results

    # Tier mapping for extraction types
    _EXTRACTION_TIER_MAP = {
        "facts": "learned",
        "decisions": "episodic",
        "preferences": "procedural",
        "todos": "working",
        "config": "core",
        "learnings": "learned",
        "important": "core",
    }

    def process_conversation(
        self,
        messages: list[dict[str, str]],
        namespace: str = "",
        source: str = "conversation",
    ) -> ProcessConversationResult:
        """
        Automatically extract and store memories from a conversation history.

        Scans messages for extractable patterns and stores them as appropriate memories.
        Uses ONLY regex heuristics -- no LLM calls.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "text"} dicts.
                      Standard OpenAI/Anthropic message format.
            namespace: Namespace to store extracted memories in
            source: Source label (default "conversation")

        Pattern categories extracted:
        1. FACTS: "X is Y", "X are Y", "X = Y", "X: Y" patterns
        2. DECISIONS: "I decided", "we decided", "decision:", "let's go with"
        3. PREFERENCES: "always", "never", "prefer", "don't like", "I want"
        4. TODOS: "TODO", "FIXME", "need to", "should", "must", "remember to"
        5. CONFIG: "KEY=VALUE", "set X to Y", environment variables
        6. LEARNINGS: "TIL", "learned that", "turns out", "found out", "discovered"
        7. IMPORTANT: "important:", "note:", "NB:", "critical:", "warning:"

        Returns:
            {"extracted": int, "by_type": {"facts": N, "decisions": N, ...}, "memories": [ids]}
        """
        # Order matters: more specific extractors first so their tier/type wins
        # when the same text matches multiple patterns (dedup by seen_texts)
        extractors = [
            self._extract_important,    # "important:", "note:", "warning:" → core
            self._extract_config,       # ENV_VAR=value, "set X to Y" → core
            self._extract_decisions,    # "I decided", "let's go with" → episodic
            self._extract_learnings,    # "learned that", "turns out" → learned
            self._extract_preferences,  # "always", "never", "prefer" → procedural
            self._extract_todos,        # "TODO", "need to", "must" → working
            self._extract_facts,        # "X is Y", "key: value" → learned (most generic)
        ]

        # Collect all extractions: (content, extraction_type, tier)
        all_extractions: list[tuple[str, str, str]] = []
        seen_texts: set[str] = set()  # deduplicate within this batch

        for msg in messages:
            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                continue

            role = msg.get("role", "unknown")

            for extractor in extractors:
                hits = extractor(content)
                for extracted_text, extraction_type, tier in hits:
                    cleaned = extracted_text.strip()
                    if not cleaned:
                        continue
                    # Deduplicate within this extraction run
                    norm = cleaned.lower()
                    if norm in seen_texts:
                        continue
                    seen_texts.add(norm)
                    all_extractions.append((cleaned, extraction_type, tier))

        if not all_extractions:
            return {"extracted": 0, "by_type": {}, "memories": []}

        # Build items for remember_batch
        batch_items = []
        for content_text, extraction_type, tier in all_extractions:
            batch_items.append({
                "content": content_text,
                "tier": tier,
                "tags": [extraction_type, "auto-extracted"],
                "source": source,
                "namespace": namespace,
            })

        # Store via remember_batch for efficiency
        result = self.remember_batch(batch_items, namespace=namespace)

        # Count by extraction type
        by_type: dict[str, int] = {}
        for _, extraction_type, _ in all_extractions:
            by_type[extraction_type] = by_type.get(extraction_type, 0) + 1

        # Retrieve the IDs of stored memories (newly inserted ones)
        # We can approximate by fetching the most recent N memories tagged auto-extracted
        memory_ids: list[int] = []
        if result["imported"] > 0:
            rows = self.db.execute(
                """SELECT id FROM memories
                   WHERE tags LIKE '%auto-extracted%'
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (result["imported"],),
            ).fetchall()
            memory_ids = [r[0] for r in rows]

        return {
            "extracted": len(all_extractions),
            "by_type": by_type,
            "memories": memory_ids,
        }

    def close(self) -> None:
        """Close the store: checkpoint WAL (TRUNCATE) then close connection."""
        if self._closed:
            return
        try:
            self.checkpoint("TRUNCATE")
        except Exception:
            pass  # DB might already be in a bad state; still close
        self.db.close()
        self._closed = True


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
