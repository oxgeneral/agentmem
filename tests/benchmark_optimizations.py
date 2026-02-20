"""
benchmark_optimizations.py — Measure improvements from Iteration 6 & 7.

A. Lazy Loading (Iteration 6):
   Measure time to reach "usable state" (MemoryStore ready for stats/today/forget)
   WITH and WITHOUT model loading.

B. Batch Operations (Iteration 7):
   Measure time to import MEMORY.md via old per-item approach vs new batch approach.

Run from workspace root:
    python3 agentmem/benchmark_optimizations.py
"""
import sys
import time
import tempfile
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentmem.core import MemoryStore
from agentmem.embeddings import get_embedding_model, LazyEmbedding

MEMORY_MD = "/home/openclaw/.claude/projects/-home-openclaw--openclaw-workspace/memory/MEMORY.md"
SEPARATOR = "=" * 60


def _make_store(tmp_dir: str, suffix: str = "") -> str:
    return os.path.join(tmp_dir, f"bench{suffix}.db")


# ============================================================
# A. Lazy Loading Benchmark
# ============================================================

def bench_lazy():
    print(f"\n{SEPARATOR}")
    print("A. LAZY LOADING (Iteration 6)")
    print(SEPARATOR)

    with tempfile.TemporaryDirectory() as tmp:
        # --- A1: Old approach — eager model load at startup ---
        t0 = time.perf_counter()

        # Old code path: model loads immediately inside get_embedding_model()
        embed_eager = get_embedding_model("auto", lazy=False)
        store_eager = MemoryStore(db_path=_make_store(tmp, "_eager"), embedding_dim=embed_eager.dim or 256)
        if embed_eager.dim > 0:
            store_eager.set_embed_fn(embed_eager)
        # Simulate: caller just wants stats(), model was not needed
        _ = store_eager.stats()
        store_eager.close()

        t_eager = time.perf_counter() - t0

        # --- A2: New approach — lazy model, stats() without loading model ---
        t0 = time.perf_counter()

        embed_lazy = get_embedding_model("auto", lazy=True)
        store_lazy = MemoryStore(db_path=_make_store(tmp, "_lazy"), embedding_dim=embed_lazy.dim or 256)
        # With lazy=True, embed_lazy.dim returns 256 WITHOUT loading the model
        if embed_lazy.dim > 0:
            store_lazy.set_embed_fn(embed_lazy)
        # stats() does NOT call _embed(), so model stays unloaded
        _ = store_lazy.stats()
        model_was_loaded = embed_lazy.loaded  # should be False

        t_lazy_nostruct = time.perf_counter() - t0

        # --- A3: Lazy — first embed() call (model loads on demand) ---
        t0 = time.perf_counter()
        _ = embed_lazy.embed("trigger model load")  # first actual embed
        t_first_embed = time.perf_counter() - t0
        store_lazy.close()

        print(f"\nEager init (model + schema + stats):  {t_eager*1000:.1f}ms")
        print(f"Lazy  init (schema + stats only):      {t_lazy_nostruct*1000:.1f}ms")
        print(f"  Model was already loaded? {model_was_loaded}")
        print(f"First embed() call (model loads here): {t_first_embed*1000:.1f}ms")

        speedup_init = t_eager / t_lazy_nostruct if t_lazy_nostruct > 0 else 0
        saved_ms = (t_eager - t_lazy_nostruct) * 1000
        print(f"\nInit speedup:  {speedup_init:.1f}x  ({saved_ms:.0f}ms saved on cold start)")
        print(f"Model load deferred to first embed() call — only paid when actually needed.")


# ============================================================
# B. Batch Import Benchmark
# ============================================================

def bench_batch():
    print(f"\n{SEPARATOR}")
    print("B. BATCH IMPORT (Iteration 7)")
    print(SEPARATOR)

    if not os.path.exists(MEMORY_MD):
        print(f"MEMORY.md not found at {MEMORY_MD}, skipping batch benchmark")
        return

    with tempfile.TemporaryDirectory() as tmp:
        embed = get_embedding_model("auto", lazy=False)  # eager for fair comparison

        # --- B1: Old approach — remember() called per chunk ---
        db_old = _make_store(tmp, "_old")
        store_old = MemoryStore(db_path=db_old, embedding_dim=embed.dim or 256)
        store_old.set_embed_fn(embed)

        # Count chunks first (without timing)
        from agentmem.core import _chunk_markdown
        from pathlib import Path
        text = Path(MEMORY_MD).read_text(encoding="utf-8")
        chunks = _chunk_markdown(text)
        source = "MEMORY.md"
        n_chunks = len(chunks)

        print(f"\nFile: MEMORY.md")
        print(f"Chunks to import: {n_chunks}")

        t0 = time.perf_counter()
        imported_old = 0
        for chunk in chunks:
            result = store_old.remember(content=chunk, tier="learned", source=source)
            if not result.get("deduplicated"):
                imported_old += 1
        t_old = time.perf_counter() - t0
        store_old.close()

        # --- B2: New approach — remember_batch() ---
        db_new = _make_store(tmp, "_new")
        store_new = MemoryStore(db_path=db_new, embedding_dim=embed.dim or 256)
        store_new.set_embed_fn(embed)

        items = [{"content": c, "tier": "learned", "source": source} for c in chunks]

        t0 = time.perf_counter()
        result_new = store_new.remember_batch(items)
        t_new = time.perf_counter() - t0
        store_new.close()

        # --- B3: Second run (test dedup) ---
        db_dedup = _make_store(tmp, "_dedup")
        store_dedup = MemoryStore(db_path=db_dedup, embedding_dim=embed.dim or 256)
        store_dedup.set_embed_fn(embed)
        # Pre-populate
        store_dedup.remember_batch(items)
        # Now re-import (should be all dedup)
        t0 = time.perf_counter()
        result_dedup = store_dedup.remember_batch(items)
        t_dedup = time.perf_counter() - t0
        store_dedup.close()

        # --- Report ---
        print(f"\nOld (per-item remember()):     {t_old*1000:.1f}ms  ({imported_old} imported)")
        print(f"New (remember_batch()):         {t_new*1000:.1f}ms  ({result_new['imported']} imported, {result_new['embedded']} embedded)")

        speedup = t_old / t_new if t_new > 0 else 0
        saved_ms = (t_old - t_new) * 1000
        per_chunk_old = t_old * 1000 / n_chunks
        per_chunk_new = t_new * 1000 / n_chunks

        print(f"\nSpeedup:          {speedup:.1f}x  ({saved_ms:.0f}ms saved)")
        print(f"Per chunk (old):  {per_chunk_old:.2f}ms")
        print(f"Per chunk (new):  {per_chunk_new:.2f}ms")
        print(f"\nDedup re-import:  {t_dedup*1000:.1f}ms  ({result_dedup['deduplicated']} deduplicated)")


# ============================================================
# C. End-to-end: cold stats() call
# ============================================================

def bench_cold_stats():
    print(f"\n{SEPARATOR}")
    print("C. COLD STATS() — no model needed")
    print(SEPARATOR)

    with tempfile.TemporaryDirectory() as tmp:
        # Simulate what the CLI 'agentmem stats' command does

        # Old: loads model eagerly
        t0 = time.perf_counter()
        embed = get_embedding_model("null")  # CLI stats uses null backend, so fair
        store = MemoryStore(db_path=_make_store(tmp, "_stats"), embedding_dim=256)
        _ = store.stats()
        store.close()
        t_null = time.perf_counter() - t0

        # With lazy auto backend (realistic: user has model2vec installed)
        t0 = time.perf_counter()
        embed_lazy = get_embedding_model("auto", lazy=True)
        store2 = MemoryStore(db_path=_make_store(tmp, "_stats2"), embedding_dim=embed_lazy.dim or 256)
        store2.set_embed_fn(embed_lazy)
        _ = store2.stats()  # no embed called
        store2.close()
        t_lazy_auto = time.perf_counter() - t0

        # Eager auto (old behavior)
        t0 = time.perf_counter()
        embed_eager = get_embedding_model("auto", lazy=False)
        store3 = MemoryStore(db_path=_make_store(tmp, "_stats3"), embedding_dim=embed_eager.dim or 256)
        store3.set_embed_fn(embed_eager)
        _ = store3.stats()
        store3.close()
        t_eager_auto = time.perf_counter() - t0

        print(f"\nnull backend (no model):    {t_null*1000:.1f}ms")
        print(f"auto + lazy (new):          {t_lazy_auto*1000:.1f}ms  (model NOT loaded)")
        print(f"auto + eager (old):         {t_eager_auto*1000:.1f}ms  (model loaded)")

        if t_lazy_auto > 0 and t_eager_auto > 0:
            speedup = t_eager_auto / t_lazy_auto
            print(f"\nLazy vs eager speedup for stats(): {speedup:.1f}x")


if __name__ == "__main__":
    print("agentmem optimization benchmarks")
    print("Iteration 6: Lazy Loading | Iteration 7: Batch Operations")

    bench_lazy()
    bench_batch()
    bench_cold_stats()

    print(f"\n{SEPARATOR}")
    print("Done.")
