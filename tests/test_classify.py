"""
Test adaptive query classification in agentmem.core.

Creates an in-memory MemoryStore (no embed fn needed for classifier),
then calls _classify_query() directly on each test case and shows results.
Also stores a few sample memories and does full recall() to verify the
method label flows through end-to-end.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core import MemoryStore

# ----------------------------------------------------------------
# Part 1: unit-test _classify_query directly
# ----------------------------------------------------------------
TEST_CASES = [
    # (query, expected_strategy_hint)
    ("252708838",                               "keyword"),
    ("how to handle bot crashes gracefully",    "semantic"),
    ("Какой токен?",                            "keyword"),
    ("@toolbox_utils_bot",                      "keyword"),
    ("deployment strategies and growth",        "semantic"),
    ("bot token",                               "balanced/keyword"),
    # Extra cases to verify edge rules
    ('"exact phrase search"',                   "keyword (quoted)"),
    ("https://api.telegram.org/bot/sendMessage","keyword (URL)"),
    ("AABBCCDDEE11223344",                      "keyword (hex)"),
    ("Александр Telegram ID 252708838",         "keyword (Russian+number)"),
    ("why does the scheduler fail at midnight", "semantic"),
    ("agentmem",                                "keyword (1 word)"),
    ("MCP server API endpoint config",          "balanced (tech terms)"),
]

store = MemoryStore(":memory:")

print("=" * 70)
print(f"{'QUERY':<45} {'FTS_W':>6} {'VEC_W':>6}  STRATEGY")
print("=" * 70)

for query, hint in TEST_CASES:
    fts_w, vec_w = store._classify_query(query)
    strategy = (
        "keyword"  if fts_w > 0.6 else
        "semantic" if vec_w > 0.6 else
        "hybrid"
    )
    print(f"{query:<45} {fts_w:>6.2f} {vec_w:>6.2f}  {strategy:<10}  (expected: {hint})")

# ----------------------------------------------------------------
# Part 2: end-to-end recall() — verify method label in results
# ----------------------------------------------------------------
print()
print("=" * 70)
print("End-to-end recall() — method label in results")
print("=" * 70)

# Store a few memories (no embed fn = keyword-only path)
store.remember("Alexander's Telegram ID is 252708838", tier="core")
store.remember("To restart the bot use pkill -f bot.py and wait 20s", tier="learned")
store.remember("Astro bot token is in astro-bot/.env as ASTRO_BOT_TOKEN", tier="learned")
store.remember("Deployment strategy: watchdog.sh cron every minute", tier="learned")
store.remember("bot crashes when OpenClaw restarts — use cron cgroup fix", tier="learned")

recall_tests = [
    "252708838",
    "how to handle bot crashes gracefully",
    "@toolbox_utils_bot",
    "deployment strategies and growth",
    "bot token",
]

for q in recall_tests:
    results = store.recall(q, limit=2)
    fts_w, vec_w = store._classify_query(q)
    strategy = (
        "keyword"  if fts_w > 0.6 else
        "semantic" if vec_w > 0.6 else
        "hybrid"
    )
    print(f"\nQuery: {q!r}  →  strategy={strategy} (fts={fts_w}, vec={vec_w})")
    if results:
        for r in results:
            snippet = r["content"][:70].replace("\n", " ")
            print(f"  [{r['method']}] score={r['score']:.4f} | {snippet}")
    else:
        print("  (no results — FTS only, no matches)")

store.close()
print()
print("Done.")
