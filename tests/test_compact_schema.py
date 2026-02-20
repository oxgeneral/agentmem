"""
Test compact schema changes in agentmem/core.py.

Tests:
1. New DB creation uses compact schema (INTEGER tier, BLOB hash, comma tags)
2. Old DB with TEXT tier still works (legacy compatibility)
3. Public API unchanged — tier always returned as string
4. Content hash deduplication works in both modes
5. Storage savings measurement
"""
import os
import sys
import sqlite3
import hashlib
import tempfile
import json
import time

# Ensure we can import core
sys.path.insert(0, os.path.dirname(__file__))

from core import MemoryStore, _TIER_TO_INT, _INT_TO_TIER, TIERS


def test_tier_maps():
    """Tier encoding maps are correct and complete."""
    assert _TIER_TO_INT == {"core": 0, "learned": 1, "episodic": 2, "working": 3}
    assert _INT_TO_TIER == {0: "core", 1: "learned", 2: "episodic", 3: "working"}
    for tier in TIERS:
        assert tier in _TIER_TO_INT, f"Missing tier in map: {tier}"
    print("[PASS] tier maps correct")


def test_new_db_compact_schema():
    """New DB uses INTEGER tier, BLOB hash, comma-separated tags."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store = MemoryStore(db_path=path)
        assert store._schema_mode == "compact", f"Expected compact, got {store._schema_mode}"

        # Verify actual column types in the schema
        col_info = store.db.execute("PRAGMA table_info(memories)").fetchall()
        col_map = {c[1]: c[2] for c in col_info}  # name -> type
        assert "INTEGER" in col_map.get("tier", "").upper(), \
            f"tier column should be INTEGER, got: {col_map.get('tier')}"

        # Insert a memory and check raw DB values
        store.remember("Hello world", tier="episodic", tags=["test", "demo"])
        row = store.db.execute(
            "SELECT tier, tags, content_hash FROM memories WHERE content = 'Hello world'"
        ).fetchone()
        assert row is not None, "Row not found"
        raw_tier, raw_tags, raw_hash = row

        # tier should be stored as integer 2 (episodic)
        assert raw_tier == 2, f"Expected tier=2 (episodic), got {raw_tier!r}"

        # tags should be comma-separated, NOT JSON
        assert raw_tags == "test,demo", f"Expected 'test,demo', got {raw_tags!r}"
        assert not raw_tags.startswith("["), "Tags should NOT be JSON array in compact mode"

        # content_hash should be 8 bytes BLOB, not 16-char hex
        assert isinstance(raw_hash, bytes), f"content_hash should be bytes, got {type(raw_hash)}"
        assert len(raw_hash) == 8, f"content_hash should be 8 bytes, got {len(raw_hash)}"

        # No AUTOINCREMENT — verify schema string
        schema_sql = store.db.execute(
            "SELECT sql FROM sqlite_master WHERE name='memories'"
        ).fetchone()[0]
        assert "AUTOINCREMENT" not in schema_sql, "Compact schema should not use AUTOINCREMENT"

        store.close()
        print("[PASS] new DB compact schema verified")
    finally:
        os.unlink(path)


def test_public_api_returns_string_tier():
    """Public API always returns tier as string, regardless of storage format."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store = MemoryStore(db_path=path)
        assert store._schema_mode == "compact"

        for tier in TIERS:
            result = store.remember(f"Test memory for {tier}", tier=tier)
            assert result["tier"] == tier, f"remember() returned wrong tier: {result['tier']}"

        # recall — no embedding fn, only FTS
        store.remember("searchable memory", tier="core", tags=["findme"])
        results = store.recall("searchable memory")
        if results:
            assert isinstance(results[0]["tier"], str), \
                f"recall() tier should be str, got {type(results[0]['tier'])}"
            assert results[0]["tier"] in TIERS, f"recall() returned unknown tier: {results[0]['tier']}"

        # today()
        today = store.today()
        for r in today:
            assert isinstance(r["tier"], str), f"today() tier should be str, got {type(r['tier'])}"
            assert r["tier"] in TIERS

        store.close()
        print("[PASS] public API returns string tiers")
    finally:
        os.unlink(path)


def test_legacy_db_compatibility():
    """Existing DB with TEXT tier still works without changes."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        # Create legacy DB manually with TEXT tier schema
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE memories (
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
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, source, tags, content='memories', content_rowid='id')
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, source, tags)
                VALUES (new.id, new.content, new.source, new.tags);
            END
        """)

        now = time.time()
        legacy_hash = hashlib.sha256("Legacy memory content".encode()).hexdigest()[:16]
        conn.execute(
            "INSERT INTO memories (content, tier, source, tags, created_at, updated_at, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("Legacy memory content", "episodic", "test", '["old_tag"]', now, now, legacy_hash),
        )
        conn.commit()
        conn.close()

        # Now open with MemoryStore — should detect legacy mode
        store = MemoryStore(db_path=path)
        assert store._schema_mode == "legacy", \
            f"Expected legacy schema, got {store._schema_mode}"

        # Should be able to insert new memories in legacy mode
        result = store.remember("New memory in legacy db", tier="learned")
        assert result["tier"] == "learned"
        assert not result.get("deduplicated")

        # Should be able to recall (FTS)
        results = store.recall("legacy memory content")
        assert any(r["content"] == "Legacy memory content" for r in results), \
            f"Legacy row not found in recall: {results}"
        # Tier should be returned as string "episodic" — not int
        for r in results:
            if r["content"] == "Legacy memory content":
                assert r["tier"] == "episodic", f"Wrong tier for legacy row: {r['tier']}"

        # today() should work
        today = store.today()
        for r in today:
            assert isinstance(r["tier"], str), f"today() tier should be str in legacy mode"

        # stats() should count correctly
        stats = store.stats()
        assert stats["total_memories"] >= 2, f"Expected >= 2 memories, got {stats['total_memories']}"

        store.close()
        print("[PASS] legacy DB (TEXT tier) compatibility verified")
    finally:
        os.unlink(path)


def test_deduplication_compact():
    """Deduplication works in compact mode (BLOB hash uniqueness)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store = MemoryStore(db_path=path)
        r1 = store.remember("Unique content for dedup test", tier="learned")
        r2 = store.remember("Unique content for dedup test", tier="learned")
        assert r1["id"] == r2["id"], "Duplicate content should return same ID"
        assert r2.get("deduplicated") is True, "Second insert should be marked as deduplicated"

        # Different content — should be a new row
        r3 = store.remember("Different content", tier="learned")
        assert r3["id"] != r1["id"], "Different content should get new ID"
        assert not r3.get("deduplicated")

        store.close()
        print("[PASS] deduplication in compact mode works")
    finally:
        os.unlink(path)


def test_tags_roundtrip():
    """Tags survive encode/decode in compact mode."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store = MemoryStore(db_path=path)
        assert store._schema_mode == "compact"

        # Tags with values
        store.remember("memory with tags", tier="learned", tags=["python", "sqlite"])

        # Internal raw storage should be comma-separated
        row = store.db.execute(
            "SELECT tags FROM memories WHERE content = 'memory with tags'"
        ).fetchone()
        assert row[0] == "python,sqlite", f"Unexpected raw tags: {row[0]!r}"

        # Empty tags
        store.remember("memory no tags", tier="learned", tags=[])
        row2 = store.db.execute(
            "SELECT tags FROM memories WHERE content = 'memory no tags'"
        ).fetchone()
        assert row2[0] == "", f"Expected empty string for no tags, got {row2[0]!r}"

        store.close()
        print("[PASS] tags encode/decode roundtrip verified")
    finally:
        os.unlink(path)


def test_storage_savings():
    """Measure and report per-row storage savings."""
    N = 100
    results = {}

    for schema_label, is_compact in [("legacy", False), ("compact", True)]:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            if not is_compact:
                # Create legacy DB manually
                conn = sqlite3.connect(path)
                conn.execute("""
                    CREATE TABLE memories (
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
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(content, source, tags, content='memories', content_rowid='id')
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content, source, tags)
                        VALUES (new.id, new.content, new.source, new.tags);
                    END
                """)
                conn.commit()
                conn.close()

            store = MemoryStore(db_path=path)

            # Insert N memories with typical content
            items = [
                {
                    "content": f"This is a typical agent memory about some important fact number {i}.",
                    "tier": "episodic",
                    "tags": ["fact", "test"],
                    "source": "benchmark",
                }
                for i in range(N)
            ]
            store.remember_batch(items)
            store.close()

            size = os.path.getsize(path)
            results[schema_label] = size
            print(f"  {schema_label:8s}: {size:,} bytes for {N} rows "
                  f"({size // N} bytes/row avg DB file)")
        finally:
            os.unlink(path)

    legacy_size = results["legacy"]
    compact_size = results["compact"]
    savings = legacy_size - compact_size
    pct = savings / legacy_size * 100 if legacy_size else 0
    print(f"\n  Savings: {savings:+,} bytes ({pct:+.1f}%) for {N} rows")
    print(f"  Per row: ~{savings // N:+d} bytes")
    # Compact should be equal or smaller
    assert compact_size <= legacy_size, \
        f"Compact schema ({compact_size}) should be <= legacy ({legacy_size})"
    print("[PASS] storage savings confirmed")


def test_save_state_uses_encoded_tier():
    """save_state() correctly archives working memories in compact mode."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store = MemoryStore(db_path=path)
        assert store._schema_mode == "compact"

        store.remember("old working memory", tier="working")
        # save_state should archive the old working memory
        result = store.save_state("current state snapshot")
        assert result.get("saved") is True

        # The old working memory should be archived
        working_stored = store._encode_tier("working")
        archived_count = store.db.execute(
            "SELECT COUNT(*) FROM memories WHERE tier = ? AND archived = 1",
            (working_stored,),
        ).fetchone()[0]
        assert archived_count >= 1, "Old working memory should be archived"

        # New state should be active
        active_count = store.db.execute(
            "SELECT COUNT(*) FROM memories WHERE tier = ? AND archived = 0",
            (working_stored,),
        ).fetchone()[0]
        assert active_count == 1, f"Expected 1 active working memory, got {active_count}"

        store.close()
        print("[PASS] save_state() with encoded tier works")
    finally:
        os.unlink(path)


def run_all():
    print("=" * 60)
    print("agentmem compact schema tests")
    print("=" * 60)

    test_tier_maps()
    test_new_db_compact_schema()
    test_public_api_returns_string_tier()
    test_legacy_db_compatibility()
    test_deduplication_compact()
    test_tags_roundtrip()
    print("\nStorage benchmark:")
    test_storage_savings()
    test_save_state_uses_encoded_tier()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
