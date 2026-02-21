"""
Comprehensive test suite for agentmem.

Tests cover: CRUD, namespaces, temporal versioning, access tracking,
entity extraction, importance scoring, compact/consolidate, procedural
memory, conversation extraction, edge cases, embeddings, schema
migration, and import/export.

Every test is independent (fresh MemoryStore per test).
Uses HashEmbedding for fast, deterministic embeddings.
"""
import os
import time
import math
import hashlib
import sqlite3
import tempfile
import pytest

# Ensure we can import agentmem from the workspace
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agentmem.core import (
    MemoryStore, TIERS, _chunk_markdown, _TIER_TO_INT, _INT_TO_TIER, SCHEMA_VERSION,
    AgentMemError, MemoryNotFoundError, InvalidTierError, EmbeddingError,
)
from agentmem.embeddings import (
    HashEmbedding, NullEmbedding, LazyEmbedding, get_embedding_model,
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Fresh MemoryStore with HashEmbedding, isolated in a temp dir."""
    db = str(tmp_path / "test.db")
    embed = HashEmbedding(dim=128)
    s = MemoryStore(db_path=db, embedding_dim=embed.dim)
    s.set_embed_fn(embed)
    yield s
    s.close()


@pytest.fixture
def bare_store(tmp_path):
    """MemoryStore without any embedding function (FTS5 only)."""
    db = str(tmp_path / "bare.db")
    s = MemoryStore(db_path=db, embedding_dim=128)
    # Explicitly do NOT call set_embed_fn — this gives FTS5-only search
    yield s
    s.close()


# ================================================================
# A. Basic CRUD
# ================================================================

class TestBasicCRUD:

    def test_remember_stores_and_returns_dict(self, store):
        result = store.remember("The sky is blue", tier="learned")
        assert isinstance(result, dict)
        assert "id" in result
        assert result["tier"] == "learned"
        assert result["embedded"] is True
        assert result["deduplicated"] is False

    def test_remember_deduplicates_by_content_hash(self, store):
        r1 = store.remember("duplicate content", tier="learned")
        r2 = store.remember("duplicate content", tier="learned")
        assert r1["id"] == r2["id"]
        assert r2["deduplicated"] is True

    @pytest.mark.parametrize("tier", TIERS)
    def test_remember_with_all_tiers(self, store, tier):
        result = store.remember(f"memory for {tier}", tier=tier)
        assert result["tier"] == tier

    def test_recall_returns_results_sorted_by_score(self, store):
        store.remember("Python is a programming language")
        store.remember("JavaScript is used for web development")
        store.remember("Python has great data science libraries")
        results = store.recall("Python programming")
        assert len(results) > 0
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recall_with_tier_filter(self, store):
        store.remember("core fact", tier="core")
        store.remember("episodic event", tier="episodic")
        results = store.recall("fact", tier="core")
        for r in results:
            assert r["tier"] == "core"

    def test_recall_empty_query_returns_empty(self, bare_store):
        bare_store.remember("something here")
        results = bare_store.recall("")
        assert results == []

    def test_recall_no_results(self, store):
        results = store.recall("xyzzy_nonexistent_query")
        assert results == []

    def test_forget_archives_memory(self, store):
        r = store.remember("to be forgotten")
        result = store.forget(r["id"])
        assert result["forgotten"] is True
        # Should not appear in recall
        hits = store.recall("forgotten")
        ids = [h["id"] for h in hits]
        assert r["id"] not in ids

    def test_unarchive_restores_memory(self, store):
        r = store.remember("restore me")
        store.forget(r["id"])
        result = store.unarchive(r["id"])
        assert result["unarchived"] is True
        # Should appear again in stats
        stats = store.stats()
        assert stats["total_memories"] >= 1

    def test_remember_batch_bulk_insert(self, store):
        items = [
            {"content": f"batch item {i}", "tier": "learned"} for i in range(10)
        ]
        result = store.remember_batch(items)
        assert result["imported"] == 10
        assert result["deduplicated"] == 0
        assert result["embedded"] == 10

    def test_remember_batch_deduplicates(self, store):
        store.remember("already here")
        items = [
            {"content": "already here"},
            {"content": "new item"},
        ]
        result = store.remember_batch(items)
        assert result["imported"] == 1
        assert result["deduplicated"] == 1

    def test_remember_batch_empty_list(self, store):
        result = store.remember_batch([])
        assert result["imported"] == 0

    def test_remember_invalid_tier_raises_error(self, store):
        with pytest.raises(InvalidTierError, match="nonexistent"):
            store.remember("bad tier", tier="nonexistent")


# ================================================================
# B. Namespace Tests
# ================================================================

class TestNamespaces:

    def test_remember_with_namespace(self, store):
        r = store.remember("namespaced", namespace="agent/alice")
        row = store.db.execute(
            "SELECT namespace FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        assert row[0] == "agent/alice"

    def test_recall_namespace_exact_match(self, store):
        store.remember("alice data", namespace="agent/alice")
        store.remember("bob data", namespace="agent/bob")
        results = store.recall("data", namespace="agent/alice")
        for r in results:
            # verify through DB lookup
            row = store.db.execute(
                "SELECT namespace FROM memories WHERE id = ?", (r["id"],)
            ).fetchone()
            ns = row[0] or ""
            assert ns == "agent/alice" or ns.startswith("agent/alice/")

    def test_recall_namespace_prefix_match(self, store):
        store.remember("alice sub data", namespace="agent/alice/session1")
        store.remember("bob data", namespace="agent/bob")
        results = store.recall("data", namespace="agent")
        ids = [r["id"] for r in results]
        # Both should be reachable under "agent" prefix
        assert len(ids) >= 1

    def test_recall_namespace_none_searches_all(self, store):
        store.remember("global item", namespace="")
        store.remember("ns item", namespace="agent/alice")
        results = store.recall("item", namespace=None)
        assert len(results) >= 2

    def test_save_state_namespace_isolation(self, store):
        store.save_state("state A", namespace="ns1")
        store.save_state("state B", namespace="ns2")
        stats_ns1 = store.stats(namespace="ns1")
        stats_ns2 = store.stats(namespace="ns2")
        assert stats_ns1["total_memories"] >= 1
        assert stats_ns2["total_memories"] >= 1

    def test_stats_per_namespace(self, store):
        store.remember("in ns", namespace="myns")
        store.remember("global")
        stats_ns = store.stats(namespace="myns")
        stats_all = store.stats()
        assert stats_ns["total_memories"] == 1
        assert stats_all["total_memories"] == 2

    def test_forget_namespace_safety_guard(self, store):
        r = store.remember("alice secret", namespace="agent/alice")
        result = store.forget(r["id"], namespace="agent/bob")
        assert result["forgotten"] is False
        assert "mismatch" in result["reason"]

    def test_forget_namespace_guard_allows_correct_ns(self, store):
        r = store.remember("alice secret", namespace="agent/alice")
        result = store.forget(r["id"], namespace="agent/alice")
        assert result["forgotten"] is True


# ================================================================
# C. Temporal Versioning
# ================================================================

class TestTemporalVersioning:

    def test_update_memory_creates_version_chain(self, store):
        r1 = store.remember("version 1")
        r2 = store.update_memory(r1["id"], "version 2")
        assert r2["supersedes"] == r1["id"]
        assert r2["id"] != r1["id"]

    def test_update_memory_archives_old_version(self, store):
        r1 = store.remember("old version")
        store.update_memory(r1["id"], "new version")
        row = store.db.execute(
            "SELECT archived FROM memories WHERE id = ?", (r1["id"],)
        ).fetchone()
        assert row[0] == 1

    def test_history_returns_correct_chain(self, store):
        r1 = store.remember("v1 content")
        r2 = store.update_memory(r1["id"], "v2 content")
        r3 = store.update_memory(r2["id"], "v3 content")
        history = store.history(r3["id"])
        # Newest first
        assert len(history) >= 3
        assert history[0]["id"] == r3["id"]
        contents = [h["content"] for h in history]
        assert "v3 content" in contents
        assert "v2 content" in contents
        assert "v1 content" in contents

    def test_recall_current_only_true_excludes_superseded(self, store):
        r1 = store.remember("old server IP is 1.1.1.1")
        r2 = store.update_memory(r1["id"], "new server IP is 2.2.2.2")
        results = store.recall("server IP", current_only=True)
        ids = [r["id"] for r in results]
        assert r1["id"] not in ids

    def test_recall_current_only_false_includes_superseded(self, store):
        r1 = store.remember("original fact about deployment")
        r2 = store.update_memory(r1["id"], "updated fact about deployment v2")
        results = store.recall("deployment", current_only=False)
        ids = [r["id"] for r in results]
        # The new version should be present, and potentially old versions via chain
        assert r2["id"] in ids

    def test_update_memory_nonexistent_id_raises_error(self, store):
        with pytest.raises(MemoryNotFoundError, match="not found"):
            store.update_memory(99999, "new content")


# ================================================================
# D. Access Tracking & Recency
# ================================================================

class TestAccessTracking:

    def test_recall_increments_access_count(self, store):
        r = store.remember("access tracking test content for lookup")
        store.recall("access tracking test")
        row = store.db.execute(
            "SELECT access_count FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        assert row[0] >= 1

    def test_recall_updates_last_accessed(self, store):
        r = store.remember("last accessed test content for search")
        before = time.time()
        store.recall("last accessed test")
        row = store.db.execute(
            "SELECT last_accessed FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        if row[0] is not None:
            assert row[0] >= before - 1

    def test_multiple_recalls_increment_correctly(self, store):
        r = store.remember("multi access test unique content xyz")
        for _ in range(3):
            store.recall("multi access test unique xyz")
        row = store.db.execute(
            "SELECT access_count FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        assert row[0] >= 3

    def test_recency_weight_zero_disables_recency_boost(self, store):
        store.remember("recency test zero weight item")
        r1 = store.recall("recency test zero", recency_weight=0.0)
        # Should not crash, results should still be returned
        assert isinstance(r1, list)

    def test_recency_weight_extreme_still_works(self, store):
        store.remember("recency extreme test content data")
        results = store.recall("recency extreme test", recency_weight=1.0)
        assert isinstance(results, list)


# ================================================================
# E. Entity Extraction
# ================================================================

class TestEntityExtraction:

    def test_extract_entities_finds_mentions(self):
        entities = MemoryStore._extract_entities("Talk to @admin about this")
        names = [e[0] for e in entities]
        assert "@admin" in names

    def test_extract_entities_finds_urls(self):
        entities = MemoryStore._extract_entities("Visit https://example.com/page")
        types = {e[1] for e in entities}
        assert "url" in types

    def test_extract_entities_finds_ips(self):
        entities = MemoryStore._extract_entities("Server is at 192.168.1.1")
        ips = [e[0] for e in entities if e[1] == "ip"]
        assert "192.168.1.1" in ips

    def test_extract_entities_finds_env_vars(self):
        entities = MemoryStore._extract_entities("Set OPENAI_API_KEY=sk-xxx")
        env_vars = [e[0] for e in entities if e[1] == "env_var"]
        assert "OPENAI_API_KEY" in env_vars

    def test_extract_entities_finds_money(self):
        entities = MemoryStore._extract_entities("Cost is $1,000 per month")
        money = [e[0] for e in entities if e[1] == "money"]
        assert "$1,000" in money

    def test_extract_entities_deduplicates(self):
        entities = MemoryStore._extract_entities("@user @user @user")
        mentions = [e[0] for e in entities if e[1] == "mention"]
        assert len(mentions) == 1

    def test_related_finds_memories_by_entity(self, store):
        store.remember("Contact @admin for help")
        results = store.related("@admin")
        assert len(results) >= 1
        assert results[0]["entity_name"] == "@admin"

    def test_entities_lists_all_with_counts(self, store):
        store.remember("Server at 10.0.0.1 running @mybot")
        store.remember("Also check 10.0.0.1 for logs")
        ents = store.entities()
        assert isinstance(ents, list)
        # ip 10.0.0.1 should appear with count >= 2
        ip_ents = [e for e in ents if e["name"] == "10.0.0.1"]
        assert len(ip_ents) >= 1
        assert ip_ents[0]["memory_count"] >= 2


# ================================================================
# F. Importance Scoring
# ================================================================

class TestImportanceScoring:

    def test_core_tier_scores_higher(self):
        core_score = MemoryStore._compute_importance("some content here for testing", "core")
        learned_score = MemoryStore._compute_importance("some content here for testing", "learned")
        assert core_score > learned_score

    def test_working_tier_scores_lower(self):
        working = MemoryStore._compute_importance("some content here for testing", "working")
        learned = MemoryStore._compute_importance("some content here for testing", "learned")
        assert working < learned

    def test_structured_content_scores_higher(self):
        plain = MemoryStore._compute_importance("just plain text nothing special", "learned")
        structured = MemoryStore._compute_importance("- bullet one\n- bullet two\nkey: value", "learned")
        assert structured > plain

    def test_with_entities_scores_higher(self):
        no_ent = MemoryStore._compute_importance("just some text", "learned", entities_count=0)
        with_ent = MemoryStore._compute_importance("just some text", "learned", entities_count=5)
        assert with_ent > no_ent

    def test_importance_stored_in_db(self, store):
        r = store.remember("important: critical production config for deployment")
        row = store.db.execute(
            "SELECT importance FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        assert row[0] is not None
        assert 0.0 <= row[0] <= 1.0


# ================================================================
# G. Compact & Consolidate
# ================================================================

class TestCompactConsolidate:

    def _add_old_memory(self, store, content, tier="learned", days_old=100):
        """Helper to insert a memory with an artificially old timestamp."""
        r = store.remember(content, tier=tier)
        old_ts = time.time() - days_old * 86400
        store.db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (old_ts, old_ts, r["id"]),
        )
        store.db.commit()
        return r

    def test_compact_archives_old_low_access_memories(self, store):
        self._add_old_memory(store, "old unused memory about weather patterns")
        result = store.compact(max_age_days=90, min_access=0)
        assert result["archived"] >= 1
        assert result["dry_run"] is False

    def test_compact_never_archives_core(self, store):
        self._add_old_memory(store, "core should survive", tier="core", days_old=200)
        result = store.compact(max_age_days=1, min_access=0)
        stats = store.stats()
        assert stats["by_tier"].get("core", 0) >= 1

    def test_compact_never_archives_procedural(self, store):
        self._add_old_memory(store, "rule: always be nice", tier="procedural", days_old=200)
        result = store.compact(max_age_days=1, min_access=0)
        stats = store.stats()
        assert stats["by_tier"].get("procedural", 0) >= 1

    def test_compact_dry_run_returns_count_without_modifying(self, store):
        self._add_old_memory(store, "dry run target memory content")
        result = store.compact(max_age_days=90, min_access=0, dry_run=True)
        assert result["dry_run"] is True
        count = result["archived"]
        # Verify nothing was actually archived
        stats = store.stats()
        assert stats["total_memories"] >= 1

    def test_compact_namespace_filter(self, store):
        self._add_old_memory(store, "ns1 old content")
        # Set namespace manually
        store.db.execute(
            "UPDATE memories SET namespace = 'ns1' WHERE content = 'ns1 old content'"
        )
        store.db.commit()
        self._add_old_memory(store, "ns2 old content")
        store.db.execute(
            "UPDATE memories SET namespace = 'ns2' WHERE content = 'ns2 old content'"
        )
        store.db.commit()
        result = store.compact(max_age_days=90, min_access=0, namespace="ns1")
        # Only ns1 should be affected
        row = store.db.execute(
            "SELECT archived FROM memories WHERE content = 'ns2 old content'"
        ).fetchone()
        assert row[0] == 0  # ns2 should not be archived

    def test_consolidate_groups_similar_memories(self, store):
        # Insert very similar content
        store.remember("Python is a great programming language for data science")
        store.remember("Python is a great programming language for data analysis")
        store.remember("JavaScript is used for web development and frontend apps")
        result = store.consolidate(similarity_threshold=0.7)
        # With hash embeddings, very similar texts should group
        assert isinstance(result, dict)
        assert "groups" in result

    def test_consolidate_dry_run_mode(self, store):
        store.remember("consolidate dry run test item alpha")
        store.remember("consolidate dry run test item alpha beta")
        result = store.consolidate(similarity_threshold=0.5, dry_run=True)
        assert result["dry_run"] is True
        # Verify nothing was archived
        stats = store.stats()
        assert stats["archived"] == 0

    def test_consolidate_keeps_longest_content(self, store):
        store.remember("short")
        store.remember("this is a much longer piece of content that should be kept during consolidation")
        result = store.consolidate(similarity_threshold=0.5)
        if result["groups"] > 0:
            kept_id = result["details"][0]["kept"]
            row = store.db.execute(
                "SELECT content FROM memories WHERE id = ?", (kept_id,)
            ).fetchone()
            assert len(row[0]) > len("short")

    def test_consolidate_without_embed_fn_raises_error(self, bare_store):
        bare_store.remember("test A")
        bare_store.remember("test B")
        with pytest.raises(EmbeddingError, match="set_embed_fn"):
            bare_store.consolidate()


# ================================================================
# H. Procedural Memory
# ================================================================

class TestProceduralMemory:

    def test_add_procedure_stores_with_procedural_tier(self, store):
        r = store.add_procedure("Always respond in bullet points")
        assert r["tier"] == "procedural"

    def test_get_procedures_returns_formatted_string(self, store):
        store.add_procedure("Never expose API keys")
        store.add_procedure("Always validate user input")
        text = store.get_procedures()
        assert "Agent Rules" in text
        assert "Never expose API keys" in text
        assert "Always validate user input" in text

    def test_get_procedures_empty_returns_empty_string(self, store):
        text = store.get_procedures()
        assert text == ""

    def test_procedures_survive_compact(self, store):
        r = store.add_procedure("persistent rule that must survive")
        old_ts = time.time() - 200 * 86400
        store.db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (old_ts, old_ts, r["id"]),
        )
        store.db.commit()
        store.compact(max_age_days=1, min_access=0)
        text = store.get_procedures()
        assert "persistent rule that must survive" in text


# ================================================================
# I. Conversation Extraction
# ================================================================

class TestConversationExtraction:

    def test_process_extracts_config_values(self, store):
        messages = [
            {"role": "user", "content": "Set DATABASE_URL=postgres://localhost:5432/db"},
        ]
        result = store.process_conversation(messages)
        assert result["extracted"] > 0
        assert "config" in result["by_type"]

    def test_process_extracts_decisions(self, store):
        messages = [
            {"role": "assistant", "content": "I decided to use PostgreSQL for the database backend."},
        ]
        result = store.process_conversation(messages)
        assert result["extracted"] > 0
        assert "decisions" in result["by_type"]

    def test_process_extracts_preferences_as_procedural(self, store):
        messages = [
            {"role": "user", "content": "Always use type annotations in Python code"},
        ]
        result = store.process_conversation(messages)
        if result["extracted"] > 0 and "preferences" in result["by_type"]:
            # Check that procedural tier was used
            stats = store.stats()
            assert stats["by_tier"].get("procedural", 0) >= 1

    def test_process_extracts_todos_as_working(self, store):
        messages = [
            {"role": "assistant", "content": "TODO: refactor the authentication module before release"},
        ]
        result = store.process_conversation(messages)
        assert result["extracted"] > 0
        assert "todos" in result["by_type"]

    def test_process_extracts_important_as_core(self, store):
        messages = [
            {"role": "user", "content": "important: the production server must never be accessed directly"},
        ]
        result = store.process_conversation(messages)
        assert result["extracted"] > 0
        assert "important" in result["by_type"]

    def test_process_handles_empty_messages(self, store):
        result = store.process_conversation([])
        assert result["extracted"] == 0

    def test_process_handles_messages_with_empty_content(self, store):
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": None},
        ]
        result = store.process_conversation(messages)
        assert result["extracted"] == 0


# ================================================================
# J. Edge Cases
# ================================================================

class TestEdgeCases:

    def test_empty_string_content_raises_error(self, store):
        # Empty string should raise AgentMemError
        with pytest.raises(AgentMemError, match="Content cannot be empty"):
            store.remember("")

    def test_whitespace_only_content_raises_error(self, store):
        # Whitespace-only content should also raise
        with pytest.raises(AgentMemError, match="Content cannot be empty"):
            store.remember("   \n\t  ")

    def test_very_long_content(self, store):
        long_text = "A" * 10000
        r = store.remember(long_text)
        assert r["id"] > 0
        row = store.db.execute(
            "SELECT content FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        assert len(row[0]) == 10000

    def test_unicode_russian(self, store):
        r = store.remember("Привет мир! Это тест на русском языке.")
        results = store.recall("Привет")
        assert len(results) >= 0  # At least no crash

    def test_unicode_cjk(self, store):
        r = store.remember("这是中文测试内容")
        assert r["id"] > 0

    def test_unicode_emoji(self, store):
        r = store.remember("Testing with emojis: 🚀🎉💡")
        assert r["id"] > 0

    def test_sql_injection_in_content(self, store):
        evil = "Robert'); DROP TABLE memories;--"
        r = store.remember(evil)
        assert r["id"] > 0
        # Table should still exist
        stats = store.stats()
        assert stats["total_memories"] >= 1

    def test_special_fts5_characters_in_query(self, store):
        store.remember("test content for FTS5 special chars")
        # These chars are FTS5 special: ^, *, ", (, ), {, }
        results = store.recall('"test" (content) {special}')
        assert isinstance(results, list)  # No crash

    def test_close_and_reopen_db(self, tmp_path):
        db = str(tmp_path / "reopen.db")
        embed = HashEmbedding(dim=128)

        s1 = MemoryStore(db_path=db, embedding_dim=embed.dim)
        s1.set_embed_fn(embed)
        s1.remember("persistent data across reopens")
        s1.close()

        s2 = MemoryStore(db_path=db, embedding_dim=embed.dim)
        s2.set_embed_fn(embed)
        results = s2.recall("persistent data")
        s2.close()
        assert len(results) >= 1

    def test_save_state_replaces_previous_working(self, store):
        store.save_state("state 1")
        store.save_state("state 2")
        stats = store.stats()
        working_count = stats["by_tier"].get("working", 0)
        assert working_count == 1  # Only the latest state

    def test_today_returns_todays_memories(self, store):
        store.remember("today's memory for test")
        results = store.today()
        assert len(results) >= 1

    def test_stats_returns_expected_keys(self, store):
        store.remember("stats test")
        s = store.stats()
        assert "total_memories" in s
        assert "by_tier" in s
        assert "archived" in s
        assert "db_size_bytes" in s
        assert "db_size_human" in s
        assert "has_vectors" in s
        assert "vec_mode" in s
        assert "avg_importance" in s


# ================================================================
# K. Embeddings
# ================================================================

class TestEmbeddings:

    def test_hash_embedding_consistent_results(self):
        embed = HashEmbedding(dim=128)
        v1 = embed.embed("hello world")
        v2 = embed.embed("hello world")
        assert v1 == v2

    def test_hash_embedding_empty_string(self):
        embed = HashEmbedding(dim=128)
        v = embed.embed("")
        assert len(v) == 128
        assert all(x == 0.0 for x in v)

    def test_hash_embedding_unicode(self):
        embed = HashEmbedding(dim=128)
        v = embed.embed("Привет мир 你好世界 🚀")
        assert len(v) == 128
        # Should not be all zeros (non-empty input)
        assert any(x != 0.0 for x in v)

    def test_hash_embedding_l2_normalized(self):
        embed = HashEmbedding(dim=128)
        v = embed.embed("test normalization")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6

    def test_hash_embedding_different_texts_different_vectors(self):
        embed = HashEmbedding(dim=128)
        v1 = embed.embed("cats are fluffy")
        v2 = embed.embed("database optimization techniques")
        assert v1 != v2

    def test_hash_embedding_batch(self):
        embed = HashEmbedding(dim=128)
        vecs = embed.embed_batch(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 128
        assert len(vecs[1]) == 128

    def test_null_embedding_returns_none(self):
        embed = NullEmbedding()
        assert embed.embed("anything") is None
        assert embed.dim == 0

    def test_null_embedding_batch(self):
        embed = NullEmbedding()
        result = embed.embed_batch(["a", "b", "c"])
        assert result == [None, None, None]

    def test_lazy_embedding_delays_loading(self):
        call_count = [0]

        def factory():
            call_count[0] += 1
            return HashEmbedding(dim=64)

        lazy = LazyEmbedding(factory, known_dim=64)
        assert lazy.dim == 64
        assert call_count[0] == 0  # Not loaded yet
        assert lazy.loaded is False

    def test_lazy_embedding_loads_on_embed(self):
        lazy = LazyEmbedding(lambda: HashEmbedding(dim=64), known_dim=64)
        vec = lazy.embed("trigger load")
        assert lazy.loaded is True
        assert len(vec) == 64

    def test_get_embedding_model_hash(self):
        model = get_embedding_model("hash")
        assert isinstance(model, HashEmbedding)

    def test_get_embedding_model_null(self):
        model = get_embedding_model("null")
        assert isinstance(model, NullEmbedding)


# ================================================================
# L. Schema Migration
# ================================================================

class TestSchemaMigration:

    def test_new_db_creates_compact_schema(self, tmp_path):
        db = str(tmp_path / "new.db")
        s = MemoryStore(db_path=db, embedding_dim=128)
        assert s._schema_mode == "compact"
        # Verify tier column is INTEGER
        cols = s.db.execute("PRAGMA table_info(memories)").fetchall()
        tier_col = [c for c in cols if c[1] == "tier"][0]
        assert "INT" in tier_col[2].upper()
        s.close()

    def test_entities_table_created(self, tmp_path):
        db = str(tmp_path / "entities.db")
        s = MemoryStore(db_path=db, embedding_dim=128)
        # entities table should exist
        row = s.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
        ).fetchone()
        assert row is not None
        s.close()

    def test_migration_adds_columns_to_existing_db(self, tmp_path):
        """Simulate an old DB missing new columns, verify migration adds them."""
        db = str(tmp_path / "migrate.db")
        # Create a minimal old-style table
        conn = sqlite3.connect(db)
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
        conn.commit()
        conn.close()

        # Open with MemoryStore — migration should add missing columns
        s = MemoryStore(db_path=db, embedding_dim=128)
        cols = s.db.execute("PRAGMA table_info(memories)").fetchall()
        col_names = {c[1] for c in cols}
        assert "access_count" in col_names
        assert "last_accessed" in col_names
        assert "namespace" in col_names
        assert "supersedes" in col_names
        assert "importance" in col_names
        s.close()


# ================================================================
# M. Import/Export
# ================================================================

class TestImportExport:

    def test_chunk_markdown_basic(self):
        md = """## Section 1
This is a paragraph with enough content to meet the minimum length requirement.

## Section 2
Another paragraph with enough content to meet the minimum length for chunking.
"""
        chunks = _chunk_markdown(md)
        assert len(chunks) >= 2

    def test_chunk_markdown_handles_bullets(self):
        md = """## Features
- Feature one is very important and serves the primary use case
- Feature two is secondary but still provides significant value
- Feature three rounds out the offerings with extra capability
"""
        chunks = _chunk_markdown(md)
        assert len(chunks) >= 1
        # At least one chunk should contain bullet points
        has_bullet = any("-" in c for c in chunks)
        assert has_bullet

    def test_chunk_markdown_handles_code_blocks(self):
        md = """## Code Example
```python
def hello():
    print("world")
    return True
```
"""
        chunks = _chunk_markdown(md)
        assert len(chunks) >= 1
        has_code = any("```" in c for c in chunks)
        assert has_code

    def test_chunk_markdown_handles_headers(self):
        md = """## Configuration
Port: 8080 on the main server for production usage

## Deployment
Use Docker with compose for the entire deployment stack
"""
        chunks = _chunk_markdown(md)
        assert len(chunks) >= 2

    def test_import_markdown_chunks_correctly(self, store, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("""## Notes
This is an important note about the system configuration and setup.

## Tips
Here is a useful tip for debugging the application in production.
""")
        result = store.import_markdown(str(md_file))
        assert result["chunks"] >= 2
        assert result["imported"] >= 1

    def test_import_markdown_deduplicates_on_reimport(self, store, tmp_path):
        md_file = tmp_path / "dedup.md"
        md_file.write_text("## Section\nThis is important content that should not be duplicated on reimport.\n")
        r1 = store.import_markdown(str(md_file))
        r2 = store.import_markdown(str(md_file))
        assert r2["deduplicated"] >= r1["imported"]
        assert r2["imported"] == 0

    def test_export_markdown_roundtrip(self, store):
        store.remember("exported fact number one", tier="learned")
        store.remember("exported fact number two", tier="core")
        md = store.export_markdown()
        assert "exported fact number one" in md
        assert "exported fact number two" in md
        assert "Agent Memory Export" in md

    def test_export_markdown_tier_filter(self, store):
        store.remember("core export test", tier="core")
        store.remember("learned export test", tier="learned")
        md = store.export_markdown(tier="core")
        assert "core export test" in md
        assert "learned export test" not in md

    def test_import_nonexistent_file(self, store):
        result = store.import_markdown("/nonexistent/path.md")
        assert "error" in result


# ================================================================
# Additional functional tests
# ================================================================

class TestAdditionalFunctionality:

    def test_unarchive_nonexistent_returns_false(self, store):
        result = store.unarchive(99999)
        assert result["unarchived"] is False

    def test_forget_nonexistent_raises_error(self, store):
        with pytest.raises(MemoryNotFoundError, match="not found"):
            store.forget(99999)

    def test_history_nonexistent_returns_empty(self, store):
        result = store.history(99999)
        assert result == []

    def test_related_with_entity_type_filter(self, store):
        store.remember("Contact @admin and check https://example.com")
        results = store.related("@admin", entity_type="mention")
        assert all(r["entity_type"] == "mention" for r in results)

    def test_entities_with_type_filter(self, store):
        store.remember("Server at 10.0.0.1 running @mybot")
        ents = store.entities(entity_type="ip")
        assert all(e["type"] == "ip" for e in ents)

    def test_set_embed_fn_with_callable(self, tmp_path):
        """Test passing a plain callable instead of model object."""
        db = str(tmp_path / "callable.db")
        s = MemoryStore(db_path=db, embedding_dim=128)
        embed = HashEmbedding(dim=128)
        s.set_embed_fn(embed.embed)  # Pass the method, not the object
        r = s.remember("test with callable embed")
        assert r["embedded"] is True
        s.close()

    def test_remember_with_tags(self, store):
        r = store.remember("tagged item", tags=["tag1", "tag2"])
        row = store.db.execute(
            "SELECT tags FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        tags_raw = row[0]
        assert "tag1" in tags_raw
        assert "tag2" in tags_raw

    def test_remember_with_source(self, store):
        r = store.remember("sourced item", source="test_file.py")
        row = store.db.execute(
            "SELECT source FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        assert row[0] == "test_file.py"

    def test_content_hash_compact_mode(self, store):
        """Compact schema uses 8-byte BLOB hash."""
        h = store._content_hash("test")
        assert isinstance(h, bytes)
        assert len(h) == 8

    def test_tier_encoding_decoding(self, store):
        for tier in TIERS:
            encoded = store._encode_tier(tier)
            decoded = store._decode_tier(encoded)
            assert decoded == tier

    def test_tags_encoding_decoding(self, store):
        tags = ["alpha", "beta", "gamma"]
        encoded = store._encode_tags(tags)
        decoded = store._decode_tags(encoded)
        assert decoded == tags

    def test_tags_empty_encoding(self, store):
        encoded = store._encode_tags([])
        decoded = store._decode_tags(encoded)
        assert decoded == []

    def test_cleanup_working_archives_expired(self, store):
        r = store.remember("working memory to expire", tier="working")
        # Set created_at to 2 days ago (beyond WORKING_TTL of 24h)
        old_ts = time.time() - 2 * 86400
        store.db.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (old_ts, r["id"]),
        )
        store.db.commit()
        store.cleanup_working()
        row = store.db.execute(
            "SELECT archived FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        assert row[0] == 1

    def test_vec_index_pure_python_search(self, store):
        """Verify the pure Python vector index works for search."""
        store.remember("alpha dog cat bird unique term xyzzy")
        store.remember("beta fish snake whale unique term xyzzy")
        results = store.recall("alpha dog unique xyzzy")
        assert len(results) >= 1

    def test_remember_batch_within_batch_dedup(self, store):
        """Duplicates within the same batch are handled."""
        items = [
            {"content": "same content batch dedup"},
            {"content": "same content batch dedup"},
            {"content": "different content"},
        ]
        result = store.remember_batch(items)
        # One of the duplicates should be caught
        assert result["imported"] == 2
        assert result["deduplicated"] == 1


# ================================================================
# N. Migration System
# ================================================================

class TestMigrationSystem:

    def test_fresh_db_gets_latest_user_version(self, tmp_path):
        """A brand new DB should have user_version set to SCHEMA_VERSION."""
        db = str(tmp_path / "fresh.db")
        s = MemoryStore(db_path=db, embedding_dim=128)
        version = s.db.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        s.close()

    def test_old_db_without_user_version_gets_migrated(self, tmp_path):
        """An old DB (user_version=0, missing columns) gets migrated to latest."""
        db = str(tmp_path / "old.db")
        # Create a minimal old-style table without the v2 columns
        conn = sqlite3.connect(db)
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
        conn.commit()
        conn.close()

        # Open with MemoryStore — migration should add missing columns
        s = MemoryStore(db_path=db, embedding_dim=128)
        cols = {c[1] for c in s.db.execute("PRAGMA table_info(memories)").fetchall()}
        assert "access_count" in cols
        assert "last_accessed" in cols
        assert "namespace" in cols
        assert "supersedes" in cols
        assert "importance" in cols
        # user_version should be set to latest
        version = s.db.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        s.close()

    def test_already_migrated_db_does_not_re_migrate(self, tmp_path):
        """Opening an already-migrated DB should not alter schema or version."""
        db = str(tmp_path / "migrated.db")
        s1 = MemoryStore(db_path=db, embedding_dim=128)
        s1.remember("test data for migration check")
        v1 = s1.db.execute("PRAGMA user_version").fetchone()[0]
        s1.close()

        # Re-open — should be a no-op migration
        s2 = MemoryStore(db_path=db, embedding_dim=128)
        v2 = s2.db.execute("PRAGMA user_version").fetchone()[0]
        assert v1 == v2 == SCHEMA_VERSION
        # Data should survive
        results = s2.db.execute("SELECT content FROM memories").fetchall()
        assert any("test data" in r[0] for r in results)
        s2.close()

    def test_migration_version_increments_correctly(self, tmp_path):
        """Each migration step should increment user_version by 1."""
        db = str(tmp_path / "incremental.db")
        # Create old-style table and set user_version to 1
        # (simulates a DB that had version 1 but needs v2 migration)
        conn = sqlite3.connect(db)
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
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        # Open with MemoryStore — should apply migration 1→2
        s = MemoryStore(db_path=db, embedding_dim=128)
        version = s.db.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        # v2 columns should be present
        cols = {c[1] for c in s.db.execute("PRAGMA table_info(memories)").fetchall()}
        assert "access_count" in cols
        assert "importance" in cols
        s.close()


# ================================================================
# O. Error Handling
# ================================================================

class TestErrorHandling:
    """Tests for custom exception hierarchy and actionable error messages."""

    def test_invalid_tier_raises_invalid_tier_error(self, store):
        with pytest.raises(InvalidTierError, match="Unknown tier 'bogus'"):
            store.remember("test", tier="bogus")

    def test_invalid_tier_error_lists_valid_tiers(self, store):
        with pytest.raises(InvalidTierError) as exc_info:
            store.remember("test", tier="invalid")
        msg = str(exc_info.value)
        for tier in TIERS:
            assert tier in msg

    def test_invalid_tier_error_is_agentmem_error(self):
        assert issubclass(InvalidTierError, AgentMemError)
        assert issubclass(InvalidTierError, Exception)

    def test_forget_nonexistent_raises_memory_not_found(self, store):
        with pytest.raises(MemoryNotFoundError, match="#99999"):
            store.forget(99999)

    def test_forget_error_suggests_recall(self, store):
        with pytest.raises(MemoryNotFoundError) as exc_info:
            store.forget(99999)
        assert "recall()" in str(exc_info.value)

    def test_update_memory_nonexistent_raises_memory_not_found(self, store):
        with pytest.raises(MemoryNotFoundError, match="#88888"):
            store.update_memory(88888, "new content")

    def test_update_memory_error_suggests_recall(self, store):
        with pytest.raises(MemoryNotFoundError) as exc_info:
            store.update_memory(88888, "new content")
        assert "recall()" in str(exc_info.value)

    def test_memory_not_found_is_agentmem_error(self):
        assert issubclass(MemoryNotFoundError, AgentMemError)
        assert issubclass(MemoryNotFoundError, Exception)

    def test_consolidate_without_embeddings_raises_embedding_error(self, bare_store):
        bare_store.remember("test A")
        bare_store.remember("test B")
        with pytest.raises(EmbeddingError, match="set_embed_fn"):
            bare_store.consolidate()

    def test_embedding_error_is_agentmem_error(self):
        assert issubclass(EmbeddingError, AgentMemError)
        assert issubclass(EmbeddingError, Exception)

    def test_embed_fn_failure_wraps_as_embedding_error(self, tmp_path):
        """When the embed function itself raises, it's wrapped in EmbeddingError."""
        db = str(tmp_path / "embed_fail.db")
        s = MemoryStore(db_path=db, embedding_dim=128)

        def bad_embed(text):
            raise RuntimeError("model crashed")

        s.set_embed_fn(bad_embed)
        with pytest.raises(EmbeddingError, match="model crashed"):
            s.remember("trigger embedding")
        s.close()

    def test_embed_error_includes_text_preview(self, tmp_path):
        """EmbeddingError should include a preview of the text that failed."""
        db = str(tmp_path / "embed_preview.db")
        s = MemoryStore(db_path=db, embedding_dim=128)

        def bad_embed(text):
            raise ValueError("dimension mismatch")

        s.set_embed_fn(bad_embed)
        with pytest.raises(EmbeddingError) as exc_info:
            s.remember("short text here")
        assert "short text here" in str(exc_info.value)
        s.close()

    def test_forget_namespace_mismatch_returns_dict_not_exception(self, store):
        """Namespace mismatch is not a missing memory -- should return dict, not raise."""
        r = store.remember("alice data", namespace="agent/alice")
        result = store.forget(r["id"], namespace="agent/bob")
        assert result["forgotten"] is False
        assert "mismatch" in result["reason"]

    def test_exceptions_importable_from_package(self):
        """Verify exceptions are exported from the agentmem package."""
        from agentmem import AgentMemError, MemoryNotFoundError, InvalidTierError, EmbeddingError
        assert issubclass(MemoryNotFoundError, AgentMemError)
        assert issubclass(InvalidTierError, AgentMemError)
        assert issubclass(EmbeddingError, AgentMemError)


# ================================================================
# WAL Checkpoint & Transaction Safety
# ================================================================

class TestWALAndTransactions:

    def test_context_manager(self, tmp_path):
        """MemoryStore works as a context manager (with ... as s)."""
        db = str(tmp_path / "ctx.db")
        embed = HashEmbedding(dim=128)
        with MemoryStore(db_path=db, embedding_dim=embed.dim) as s:
            s.set_embed_fn(embed)
            r = s.remember("context manager test")
            assert r["id"] >= 1
        # After exiting, the store should be closed (double-close safe)
        assert s._closed is True

    def test_close_idempotent(self, tmp_path):
        """Calling close() twice does not raise."""
        db = str(tmp_path / "close2.db")
        s = MemoryStore(db_path=db, embedding_dim=128)
        s.close()
        s.close()  # second call should not error
        assert s._closed is True

    def test_transaction_rollback_on_error(self, store):
        """Transaction rolls back all changes when an exception occurs."""
        store.remember("before transaction")
        initial_count = store.db.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 0"
        ).fetchone()[0]

        with pytest.raises(RuntimeError, match="intentional"):
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO memories (content, tier, source, tags, namespace, created_at, updated_at, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("should be rolled back", 1, "", "", "", time.time(), time.time(), b"fakehash_rollback"),
                )
                raise RuntimeError("intentional failure")

        after_count = store.db.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 0"
        ).fetchone()[0]
        assert after_count == initial_count, "Transaction should have rolled back"

    def test_remember_batch_atomicity(self, tmp_path):
        """If remember_batch fails midway, none of the batch is stored."""
        db = str(tmp_path / "batch_atom.db")

        call_count = 0

        class FailingEmbedding:
            dim = 128
            def embed(self, text):
                nonlocal call_count
                call_count += 1
                return [0.1] * 128
            def embed_batch(self, texts):
                # Fail after returning some vectors — the DB writes happen after embedding
                raise RuntimeError("embedding explosion")

        s = MemoryStore(db_path=db, embedding_dim=128)
        s.set_embed_fn(FailingEmbedding())

        items = [
            {"content": f"batch item {i}", "tier": "learned"}
            for i in range(5)
        ]

        with pytest.raises(RuntimeError, match="embedding explosion"):
            s.remember_batch(items)

        # Nothing should have been stored because embedding failed before the transaction started
        count = s.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert count == 0, f"Expected 0 memories after failed batch, got {count}"
        s.close()

    def test_checkpoint_does_not_error(self, store):
        """checkpoint() with all three modes completes without error."""
        store.remember("checkpoint test data")
        for mode in ("PASSIVE", "FULL", "TRUNCATE"):
            result = store.checkpoint(mode)
            assert isinstance(result, tuple)
            assert len(result) == 3

    def test_write_counter_increments_and_resets(self, tmp_path):
        """Write counter tracks writes and resets after checkpoint."""
        db = str(tmp_path / "writes.db")
        embed = HashEmbedding(dim=128)
        s = MemoryStore(db_path=db, embedding_dim=embed.dim, checkpoint_interval=0)
        s.set_embed_fn(embed)

        assert s._writes_since_checkpoint == 0

        s.remember("write 1")
        assert s._writes_since_checkpoint == 1

        s.remember("write 2")
        assert s._writes_since_checkpoint == 2

        r = s.remember("write 3")
        s.forget(r["id"])
        assert s._writes_since_checkpoint == 4

        # Manual checkpoint should reset the counter
        s.checkpoint("PASSIVE")
        assert s._writes_since_checkpoint == 0

        s.close()


# ================================================================
# P. Bug Fix Regression Tests
# ================================================================

class TestNamespaceIsolation:
    """BUG 1: Same content in different namespaces must NOT be deduplicated."""

    def test_same_content_different_namespaces_both_stored(self, store):
        """Same content in two namespaces should create two separate memories."""
        r1 = store.remember("shared content here", namespace="ns1")
        r2 = store.remember("shared content here", namespace="ns2")
        assert r1["id"] != r2["id"], "Same content in different namespaces should have different IDs"
        assert r1["deduplicated"] is False
        assert r2["deduplicated"] is False

    def test_same_content_same_namespace_deduplicated(self, store):
        """Same content in the same namespace should be deduplicated."""
        r1 = store.remember("dedup content", namespace="ns1")
        r2 = store.remember("dedup content", namespace="ns1")
        assert r1["id"] == r2["id"]
        assert r2["deduplicated"] is True

    def test_same_content_global_and_namespaced_both_stored(self, store):
        """Same content in global namespace and a named namespace: both stored."""
        r1 = store.remember("global and ns content", namespace="")
        r2 = store.remember("global and ns content", namespace="myns")
        assert r1["id"] != r2["id"]

    def test_namespace_isolation_in_batch(self, store):
        """remember_batch should also respect namespace isolation."""
        items = [
            {"content": "batch shared", "namespace": "ns1"},
            {"content": "batch shared", "namespace": "ns2"},
        ]
        result = store.remember_batch(items)
        assert result["imported"] == 2
        assert result["deduplicated"] == 0


class TestUpdateMemoryDedup:
    """BUG 2: update_memory must archive old even when dedup triggers."""

    def test_update_memory_dedup_still_archives_old(self, store):
        """When update_memory's new_content matches an existing memory,
        the old memory should still be archived."""
        r_existing = store.remember("existing content for dedup")
        r_old = store.remember("old version to replace")
        result = store.update_memory(r_old["id"], "existing content for dedup")
        # Old memory should be archived
        row = store.db.execute(
            "SELECT archived FROM memories WHERE id = ?", (r_old["id"],)
        ).fetchone()
        assert row[0] == 1, "Old memory should be archived after update_memory even on dedup"
        # Result should point to the existing memory
        assert result["id"] == r_existing["id"]
        assert result["supersedes"] == r_old["id"]

    def test_update_memory_dedup_sets_supersedes_on_existing(self, store):
        """The existing deduplicated memory should have supersedes set to old_id."""
        r_existing = store.remember("target content for chain")
        r_old = store.remember("old content to update")
        store.update_memory(r_old["id"], "target content for chain")
        # The existing memory should now have supersedes pointing to old
        row = store.db.execute(
            "SELECT supersedes FROM memories WHERE id = ?", (r_existing["id"],)
        ).fetchone()
        assert row[0] == r_old["id"]


class TestEmptyContentValidation:
    """BUG 4: Empty content must be rejected."""

    def test_empty_string_raises(self, store):
        with pytest.raises(AgentMemError, match="Content cannot be empty"):
            store.remember("")

    def test_whitespace_only_raises(self, store):
        with pytest.raises(AgentMemError, match="Content cannot be empty"):
            store.remember("   ")

    def test_newlines_only_raises(self, store):
        with pytest.raises(AgentMemError, match="Content cannot be empty"):
            store.remember("\n\n\t\n")

    def test_non_empty_content_works(self, store):
        r = store.remember("valid content")
        assert r["id"] > 0


class TestLikeWildcardEscape:
    """BUG 5: Namespace with % or _ in name should be matched literally."""

    def test_namespace_with_percent_literal(self, store):
        """% in namespace name should not act as wildcard in LIKE."""
        store.remember("data in weird ns", namespace="agent%special")
        store.remember("data in normal ns", namespace="agentzspecial")
        stats = store.stats(namespace="agent%special")
        assert stats["total_memories"] == 1

    def test_namespace_with_underscore_literal(self, store):
        """_ in namespace name should not act as wildcard in LIKE."""
        store.remember("data in underscored ns", namespace="agent_special")
        store.remember("data in similar ns", namespace="agentXspecial")
        stats = store.stats(namespace="agent_special")
        assert stats["total_memories"] == 1
