"""
agentmem benchmarks — reproducible performance measurements.

Run: python -m agentmem.benchmarks.bench

Measures cold start, import speed, query latency, throughput,
maintenance ops, and storage efficiency with realistic agent memory data.
All timings use time.perf_counter(); each benchmark runs 3+ times and
reports the median.  Output is both a human-readable table and a JSON file.

Dependencies: stdlib only (time, json, tempfile, statistics, os, sys, random, platform).
"""

import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Ensure the package is importable when run as  python -m agentmem.benchmarks.bench
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agentmem import __version__
from agentmem.core import MemoryStore
from agentmem.embeddings import HashEmbedding, NullEmbedding

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RUNS = 5          # repetitions per benchmark (median of RUNS)
SEED = 42         # for reproducibility
random.seed(SEED)

# ---------------------------------------------------------------------------
# Realistic test-data generator
# ---------------------------------------------------------------------------
_TEMPLATES_CORE = [
    "Server IP is 192.168.1.{i} running Ubuntu 24.04 LTS with 4GB RAM",
    "API endpoint https://api.example.com/v{i} returns JSON with pagination",
    "Deployed version 2.{i}.0 to production at 2026-02-{day:02d} 14:{minute:02d}:00 UTC",
    "OPENAI_API_KEY must be set before running bot.py — stored in .env",
    "Database connection pool max_size={n}, increase to {n2} for production loads",
    "Bug fix: SQL injection in search handler on line {line}, use parameterized queries",
    "Backup cron runs at 03:00 UTC daily: pg_dump postgres://10.0.0.{i}:5432/main > /backups/db.sql",
    "Rate limit: {n} requests per minute per IP, 429 response after that",
    "Redis cache TTL is {n} seconds for session data, {n2} for static assets",
    "SSH key fingerprint SHA256:xYz{i}AbC for deploy@prod-{i}.example.com",
]

_TEMPLATES_LEARNED = [
    "The user prefers dark mode in all applications — set THEME=dark globally",
    "Python virtualenvs go in ~/.venvs/{name}, not project-local .venv directories",
    "Alexander's Telegram ID is 252708838 — use for admin bypass checks",
    "When pip install fails with externally-managed, use --break-system-packages flag",
    "SQLite FTS5 does not support prefix queries shorter than 3 characters by default",
    "model2vec potion-base-8M outputs 256-dimensional vectors, not 384",
    "Telegram Stars payments: min withdraw 1000 Stars, 21-day wait, ~65% net via Fragment",
    "For nohup processes: use cron watchdog, not systemd — survives service restarts",
    "browser-use input command timeouts with long text; use click + type + Ctrl+A instead",
    "DeepInfra free tier: image generation works without API key, chat returns 403",
]

_TEMPLATES_EPISODIC = [
    "Submitted bug report #{i} to Patchstack for XSS in WordPress plugin search handler",
    "User @test_user_{i} signed up via deep link ?start=habr on 2026-02-{day:02d}",
    "Bot crashed at {hour:02d}:{minute:02d} — nohup killed when OpenClaw restarted",
    "Published Telegraph article '{title}' — got {n} views in first hour",
    "Ran migration script v{i}.sql — added namespace column to memories table",
    "Channel @workonhuman reached {n} subscribers after diary post #{i}",
    "AgentNet received first external crawler hit from {ip} at 03:42 UTC",
    "Pixie bot: user {uid} generated AI image with prompt 'sunset over mountains'",
    "vc.ru article got {n} views but 0 clicks — visibility without conversion",
    "Deployed astro-bot v1.{i}.0 with horoscope card image generation feature",
]

_TEMPLATES_PROCEDURAL = [
    "To restart Pixie bot: pkill -f 'python3 bot.py' && sleep 20 && cd toolbox-bot && export $(cat .env | xargs) && nohup python3 bot.py &",
    "Submitting to Patchstack: base64 encode textarea content, use atob() in JS, native setter + dispatchEvent",
    "Creating new MCP tool: add function to server.py, register in TOOLS dict, update manifest.json",
    "Database backup procedure: sqlite3 memory.db '.backup /tmp/memory_backup.db' && gzip /tmp/memory_backup.db",
    "Publishing to Telegraph: POST https://api.telegra.ph/createPage with access_token, title, content as Node array",
    "Bot deployment checklist: 1. Update version in pyproject.toml 2. Run tests 3. pkill old 4. Wait 20s 5. Start new",
    "Setting up cron watchdog: crontab -e, add '* * * * * /path/to/watchdog.sh >> /tmp/watchdog.log 2>&1'",
    "Image generation pipeline: translate RU prompt -> EN via deep-translator, POST to DeepInfra FLUX, decode b64 response",
    "MCP server registration: mcp-publisher CLI -> server.json -> smithery deploy -> verify on smithery.ai",
    "Inline bot setup: implement InlineQueryHandler, return InlineQueryResultArticle with InputTextMessageContent",
]

_TITLES = [
    "AI autonomy diary", "bot deployment guide", "server migration notes",
    "API integration docs", "performance optimization", "security audit report",
    "user growth analysis", "revenue tracking sheet", "bug bounty findings",
    "infrastructure overview",
]

_NAMES = ["pixie", "astro", "toolbox", "agent", "worker", "monitor", "scheduler"]


def _generate_memories(count: int) -> list[str]:
    """Generate `count` realistic agent memory texts."""
    all_templates = (
        _TEMPLATES_CORE + _TEMPLATES_LEARNED
        + _TEMPLATES_EPISODIC + _TEMPLATES_PROCEDURAL
    )
    memories = []
    for idx in range(count):
        template = all_templates[idx % len(all_templates)]
        text = template.format(
            i=idx % 255 + 1,
            n=random.randint(10, 10000),
            n2=random.randint(100, 50000),
            day=random.randint(1, 28),
            hour=random.randint(0, 23),
            minute=random.randint(0, 59),
            line=random.randint(10, 500),
            ip=f"185.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            title=random.choice(_TITLES),
            uid=random.randint(100000, 999999),
            name=random.choice(_NAMES),
        )
        memories.append(text)
    return memories


def _tier_for(idx: int) -> str:
    """Assign a realistic tier based on index."""
    tiers = ["core", "learned", "episodic", "working", "procedural"]
    # Weighted: more learned/episodic, fewer core/procedural
    weights = [0.10, 0.30, 0.35, 0.15, 0.10]
    r = (idx * 7 + 13) % 100 / 100.0  # deterministic pseudo-random
    cumulative = 0.0
    for tier, w in zip(tiers, weights):
        cumulative += w
        if r < cumulative:
            return tier
    return "learned"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _median_time(fn, runs: int = RUNS) -> float:
    """Run fn() `runs` times, return median elapsed seconds."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _ms(seconds: float) -> str:
    """Format seconds as human-readable milliseconds string."""
    ms_val = seconds * 1000
    if ms_val < 0.01:
        return f"{ms_val:.4f} ms"
    if ms_val < 1:
        return f"{ms_val:.3f} ms"
    if ms_val < 100:
        return f"{ms_val:.1f} ms"
    return f"{ms_val:.0f} ms"


def _per_sec(count: int, seconds: float) -> str:
    """Format as operations/second."""
    if seconds <= 0:
        return "inf"
    rate = count / seconds
    if rate >= 10000:
        return f"{rate:.0f}"
    return f"{rate:.0f}"


# ---------------------------------------------------------------------------
# Create a temp store helper
# ---------------------------------------------------------------------------
def _make_store(tmpdir: str, name: str = "bench.db", embed=None, dim: int = 128):
    """Create a MemoryStore in a temp dir with optional embedding."""
    path = os.path.join(tmpdir, name)
    store = MemoryStore(path, embedding_dim=dim)
    if embed is not None:
        store.set_embed_fn(embed)
    return store


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_cold_start(tmpdir: str, memories_data: list[str]) -> dict:
    """
    1. Cold start time: create MemoryStore + HashEmbedding from scratch.
    Measures empty DB, 100 memories, 1000 memories.
    """
    embed = HashEmbedding(dim=128)
    results = {}

    # Empty DB
    counter = [0]
    def make_empty():
        counter[0] += 1
        _make_store(tmpdir, f"cold_empty_{counter[0]}.db", embed)
    results["empty_db_s"] = _median_time(make_empty)

    # Pre-populate 100 memories
    def cold_100():
        counter[0] += 1
        s = _make_store(tmpdir, f"cold_100_{counter[0]}.db", embed)
        items = [{"content": t, "tier": _tier_for(i)} for i, t in enumerate(memories_data[:100])]
        s.remember_batch(items)
        del s
        # Now measure re-opening
        t0 = time.perf_counter()
        s2 = MemoryStore(os.path.join(tmpdir, f"cold_100_{counter[0]}.db"), embedding_dim=128)
        s2.set_embed_fn(embed)
        return time.perf_counter() - t0

    # For 100 memories, we measure the reopen time
    reopen_times_100 = []
    for _ in range(RUNS):
        reopen_times_100.append(cold_100())
    results["with_100_s"] = statistics.median(reopen_times_100)

    # Pre-populate 1000 memories
    def cold_1000():
        counter[0] += 1
        s = _make_store(tmpdir, f"cold_1000_{counter[0]}.db", embed)
        items = [{"content": t, "tier": _tier_for(i)} for i, t in enumerate(memories_data[:1000])]
        s.remember_batch(items)
        del s
        t0 = time.perf_counter()
        s2 = MemoryStore(os.path.join(tmpdir, f"cold_1000_{counter[0]}.db"), embedding_dim=128)
        s2.set_embed_fn(embed)
        return time.perf_counter() - t0

    reopen_times_1000 = []
    for _ in range(RUNS):
        reopen_times_1000.append(cold_1000())
    results["with_1000_s"] = statistics.median(reopen_times_1000)

    return results


def bench_import(tmpdir: str, memories_data: list[str]) -> dict:
    """
    2. Import speed: time per chunk for batch vs single insert.
    """
    embed = HashEmbedding(dim=128)
    results = {}

    # Batch import 100 chunks
    def batch_100():
        s = _make_store(tmpdir, f"import_batch_{time.monotonic_ns()}.db", embed)
        items = [{"content": t, "tier": _tier_for(i)} for i, t in enumerate(memories_data[:100])]
        t0 = time.perf_counter()
        s.remember_batch(items)
        elapsed = time.perf_counter() - t0
        return elapsed

    batch_times = [batch_100() for _ in range(RUNS)]
    batch_median = statistics.median(batch_times)
    results["batch_per_chunk_s"] = batch_median / 100
    results["batch_100_total_s"] = batch_median

    # Single insert 100 chunks
    def single_100():
        s = _make_store(tmpdir, f"import_single_{time.monotonic_ns()}.db", embed)
        t0 = time.perf_counter()
        for i, t in enumerate(memories_data[:100]):
            s.remember(t, tier=_tier_for(i))
        elapsed = time.perf_counter() - t0
        return elapsed

    single_times = [single_100() for _ in range(RUNS)]
    single_median = statistics.median(single_times)
    results["single_per_chunk_s"] = single_median / 100
    results["single_100_total_s"] = single_median

    return results


def bench_query(tmpdir: str, memories_data: list[str]) -> dict:
    """
    3-5. Query benchmarks: hybrid, FTS5-only, vector-only at 100 and 1000 scale.
    """
    embed = HashEmbedding(dim=128)
    null_embed = NullEmbedding()

    # Queries that exercise different search paths
    queries = [
        "What is the database connection pool size?",   # keyword-heavy
        "how to deploy the bot to production",           # semantic
        "192.168.1",                                     # exact keyword (IP)
        "Telegram Stars payment withdraw",               # mixed
        "OPENAI_API_KEY environment variable",            # env var keyword
    ]

    results = {}

    for scale in (100, 1000):
        # -- Hybrid store --
        hybrid_store = _make_store(tmpdir, f"query_hybrid_{scale}.db", embed)
        items = [{"content": t, "tier": _tier_for(i)} for i, t in enumerate(memories_data[:scale])]
        hybrid_store.remember_batch(items)

        def hybrid_recall():
            for q in queries:
                hybrid_store.recall(q, limit=5)

        hybrid_s = _median_time(hybrid_recall) / len(queries)
        results[f"hybrid_{scale}_s"] = hybrid_s

        # -- FTS5-only store (no embed fn set => recall uses FTS5 only) --
        fts_store = _make_store(tmpdir, f"query_fts_{scale}.db", dim=128)
        # Insert without embeddings (no embed fn set)
        for i, t in enumerate(memories_data[:scale]):
            fts_store.db.execute(
                """INSERT OR IGNORE INTO memories (content, tier, source, tags, namespace, created_at, updated_at, content_hash)
                   VALUES (?, ?, '', '', '', ?, ?, ?)""",
                (t, fts_store._encode_tier(_tier_for(i)), time.time(), time.time(),
                 fts_store._content_hash(t)),
            )
        fts_store.db.commit()

        def fts_recall():
            for q in queries:
                fts_store.recall(q, limit=5)

        fts_s = _median_time(fts_recall) / len(queries)
        results[f"fts_{scale}_s"] = fts_s

        # -- Vector-only timing (direct _vec_search) --
        def vec_recall():
            for q in queries:
                hybrid_store._vec_search(q, limit=5)

        vec_s = _median_time(vec_recall) / len(queries)
        results[f"vec_{scale}_s"] = vec_s

    return results


def bench_throughput(tmpdir: str, memories_data: list[str]) -> dict:
    """
    6-7. Remember throughput: single inserts/sec and batch inserts/sec.
    """
    embed = HashEmbedding(dim=128)
    count = 200  # enough to get stable throughput numbers
    results = {}

    # Single remember() throughput
    def single_throughput():
        s = _make_store(tmpdir, f"tp_single_{time.monotonic_ns()}.db", embed)
        t0 = time.perf_counter()
        for i in range(count):
            s.remember(memories_data[i], tier=_tier_for(i))
        return time.perf_counter() - t0

    single_times = [single_throughput() for _ in range(RUNS)]
    single_median = statistics.median(single_times)
    results["remember_per_sec"] = count / single_median

    # Batch remember_batch() throughput
    def batch_throughput():
        s = _make_store(tmpdir, f"tp_batch_{time.monotonic_ns()}.db", embed)
        items = [{"content": memories_data[i], "tier": _tier_for(i)} for i in range(count)]
        t0 = time.perf_counter()
        s.remember_batch(items)
        return time.perf_counter() - t0

    batch_times = [batch_throughput() for _ in range(RUNS)]
    batch_median = statistics.median(batch_times)
    results["batch_per_sec"] = count / batch_median

    return results


def bench_entity_extraction(memories_data: list[str]) -> dict:
    """
    8. Entity extraction time per text.
    """
    sample = memories_data[:200]

    def extract_all():
        for t in sample:
            MemoryStore._extract_entities(t)

    elapsed = _median_time(extract_all)
    return {"per_text_s": elapsed / len(sample)}


def bench_importance_scoring(memories_data: list[str]) -> dict:
    """
    9. Importance scoring time per text.
    """
    sample = memories_data[:200]
    # Pre-extract entity counts
    entity_counts = [len(MemoryStore._extract_entities(t)) for t in sample]

    def score_all():
        for t, ec in zip(sample, entity_counts):
            MemoryStore._compute_importance(t, "learned", ec)

    elapsed = _median_time(score_all)
    return {"per_text_s": elapsed / len(sample)}


def bench_compact(tmpdir: str, memories_data: list[str]) -> dict:
    """
    10. Compact time on various DB sizes.
    """
    embed = HashEmbedding(dim=128)
    results = {}

    for scale in (100, 500, 1000):
        s = _make_store(tmpdir, f"compact_{scale}.db", embed)
        items = [{"content": t, "tier": _tier_for(i)} for i, t in enumerate(memories_data[:scale])]
        s.remember_batch(items)

        # compact with max_age_days=0 to target all memories
        def do_compact():
            # Reset: unarchive everything first
            s.db.execute("UPDATE memories SET archived = 0")
            s.db.commit()
            s.compact(max_age_days=0, min_access=999999)

        elapsed = _median_time(do_compact)
        results[f"compact_{scale}_s"] = elapsed

    return results


def bench_consolidate(tmpdir: str, memories_data: list[str]) -> dict:
    """
    11. Consolidate time on various DB sizes (showing O(n^2) cost).
    """
    embed = HashEmbedding(dim=128)
    results = {}

    for scale in (50, 100, 200):
        s = _make_store(tmpdir, f"consolidate_{scale}.db", embed)
        items = [{"content": t, "tier": _tier_for(i)} for i, t in enumerate(memories_data[:scale])]
        s.remember_batch(items)

        def do_consolidate():
            # Reset: unarchive everything first
            s.db.execute("UPDATE memories SET archived = 0")
            s.db.commit()
            s.consolidate(similarity_threshold=0.95, dry_run=True)

        elapsed = _median_time(do_consolidate, runs=3)
        results[f"consolidate_{scale}_s"] = elapsed

    return results


def bench_storage(tmpdir: str, memories_data: list[str]) -> dict:
    """
    12. DB size in bytes at various scales.
    """
    embed = HashEmbedding(dim=128)
    results = {}

    for scale in (10, 100, 1000):
        db_path = os.path.join(tmpdir, f"storage_{scale}.db")
        s = MemoryStore(db_path, embedding_dim=128)
        s.set_embed_fn(embed)
        items = [{"content": t, "tier": _tier_for(i)} for i, t in enumerate(memories_data[:scale])]
        s.remember_batch(items)
        # Force WAL checkpoint so file size is accurate
        s.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        del s
        size = os.path.getsize(db_path)
        results[f"db_{scale}_bytes"] = size

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"agentmem v{__version__} benchmarks")
    print("\u2550" * 55)

    # System info
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_info = platform.platform()
    print(f"System: {os_info}, Python {py_ver}, HashEmbedding 128d")
    print()

    # Generate test data
    print("Generating 1000 realistic test memories...")
    memories = _generate_memories(1000)
    print()

    all_results = {}

    with tempfile.TemporaryDirectory(prefix="agentmem_bench_") as tmpdir:

        # 1. Cold Start
        print("Cold Start")
        cs = bench_cold_start(tmpdir, memories)
        all_results["cold_start"] = cs
        print(f"  Empty DB init:          {_ms(cs['empty_db_s'])}")
        print(f"  Reopen with 100 mem:    {_ms(cs['with_100_s'])}")
        print(f"  Reopen with 1000 mem:   {_ms(cs['with_1000_s'])}")
        print()

        # 2. Import
        print("Import")
        imp = bench_import(tmpdir, memories)
        all_results["import"] = imp
        print(f"  Per chunk (batch):      {_ms(imp['batch_per_chunk_s'])}")
        print(f"  Per chunk (single):     {_ms(imp['single_per_chunk_s'])}")
        print(f"  100 chunks batch:       {_ms(imp['batch_100_total_s'])}")
        print(f"  100 chunks single:      {_ms(imp['single_100_total_s'])}")
        print()

        # 3-5. Query
        print("Query (100 memories)")
        q = bench_query(tmpdir, memories)
        all_results["query"] = q
        print(f"  Hybrid recall:          {_ms(q['hybrid_100_s'])}")
        print(f"  FTS5-only:              {_ms(q['fts_100_s'])}")
        print(f"  Vector-only:            {_ms(q['vec_100_s'])}")
        print()
        print("Query (1000 memories)")
        print(f"  Hybrid recall:          {_ms(q['hybrid_1000_s'])}")
        print(f"  FTS5-only:              {_ms(q['fts_1000_s'])}")
        print(f"  Vector-only:            {_ms(q['vec_1000_s'])}")
        print()

        # 6-7. Throughput
        print("Throughput")
        tp = bench_throughput(tmpdir, memories)
        all_results["throughput"] = tp
        print(f"  remember() / sec:       {_per_sec(1, 1/tp['remember_per_sec'])}")
        print(f"  remember_batch() / sec: {_per_sec(1, 1/tp['batch_per_sec'])}")
        print()

        # 8. Entity extraction
        print("Entity Extraction")
        ee = bench_entity_extraction(memories)
        all_results["entity_extraction"] = ee
        print(f"  _extract_entities():    {_ms(ee['per_text_s'])} / text")
        print()

        # 9. Importance scoring
        print("Importance Scoring")
        isc = bench_importance_scoring(memories)
        all_results["importance_scoring"] = isc
        print(f"  _compute_importance():  {_ms(isc['per_text_s'])} / text")
        print()

        # 10. Compact
        print("Maintenance — compact")
        cm = bench_compact(tmpdir, memories)
        all_results["compact"] = cm
        print(f"  compact(100 memories):  {_ms(cm['compact_100_s'])}")
        print(f"  compact(500 memories):  {_ms(cm['compact_500_s'])}")
        print(f"  compact(1000 memories): {_ms(cm['compact_1000_s'])}")
        print()

        # 11. Consolidate
        print("Maintenance — consolidate")
        cn = bench_consolidate(tmpdir, memories)
        all_results["consolidate"] = cn
        print(f"  consolidate(50):        {_ms(cn['consolidate_50_s'])}")
        print(f"  consolidate(100):       {_ms(cn['consolidate_100_s'])}")
        print(f"  consolidate(200):       {_ms(cn['consolidate_200_s'])}")
        # Show O(n^2) ratio
        if cn['consolidate_50_s'] > 0:
            ratio_100 = cn['consolidate_100_s'] / cn['consolidate_50_s']
            ratio_200 = cn['consolidate_200_s'] / cn['consolidate_50_s']
            print(f"  Scaling: 50->100 = {ratio_100:.1f}x, 50->200 = {ratio_200:.1f}x  (O(n\u00b2) expected: 4x, 16x)")
        print()

        # 12. Storage
        print("Storage")
        st = bench_storage(tmpdir, memories)
        all_results["storage"] = st
        def _human_bytes(b):
            if b < 1024:
                return f"{b} B"
            if b < 1024 * 1024:
                return f"{b/1024:.1f} KB"
            return f"{b/(1024*1024):.1f} MB"

        print(f"  10 memories:            {_human_bytes(st['db_10_bytes'])}")
        print(f"  100 memories:           {_human_bytes(st['db_100_bytes'])}")
        print(f"  1000 memories:          {_human_bytes(st['db_1000_bytes'])}")
        avg_per_mem = st['db_1000_bytes'] / 1000
        print(f"  Per memory avg:         ~{_human_bytes(int(avg_per_mem))}")

    print("\u2550" * 55)

    # Write JSON results
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "version": __version__,
            "python": py_ver,
            "platform": os_info,
            "embedding": "HashEmbedding 128d",
            "runs_per_benchmark": RUNS,
            "results": all_results,
        }, f, indent=2)
    print(f"\nJSON results written to: {json_path}")


# ---------------------------------------------------------------------------
# Allow  python -m agentmem.benchmarks.bench
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
