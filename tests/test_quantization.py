"""
test_quantization.py — Benchmark and correctness test for int8 quantization in agentmem.

Tests:
1. Quantization round-trip accuracy
2. Storage savings (bytes on disk / in-memory blob size)
3. Search quality: top-k recall comparison (float32 vs int8)
4. Speed comparison: insert and search timing
"""
import sys
import os
import time
import math
import random
import tempfile
import struct

# Make sure agentmem is importable from the workspace
sys.path.insert(0, os.path.dirname(__file__))

from core import (
    _quantize_f32_to_i8,
    _dequantize_i8_to_f32,
    _VecIndex,
    MemoryStore,
)

# ---- reproducible randomness ------------------------------------------------
random.seed(42)

DIM = 256
N_VECTORS = 500
N_QUERIES = 20
K = 5


def rand_vec(dim=DIM):
    """Random unit-norm vector (typical for sentence embeddings)."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


# =============================================================================
# TEST 1: round-trip quantization accuracy
# =============================================================================
def test_roundtrip():
    print("=" * 60)
    print("TEST 1: Quantization round-trip accuracy")
    print("=" * 60)

    errors = []
    for _ in range(200):
        vec = rand_vec()
        blob = _quantize_f32_to_i8(vec)
        restored = _dequantize_i8_to_f32(blob, DIM)

        # Max absolute error per element
        max_err = max(abs(a - b) for a, b in zip(vec, restored))
        errors.append(max_err)

    avg_err = sum(errors) / len(errors)
    max_err_all = max(errors)

    print(f"  Vectors tested:        200")
    print(f"  Avg max element error: {avg_err:.6f}")
    print(f"  Worst max error:       {max_err_all:.6f}")
    print(f"  (Theoretical max for 8-bit: ~{1/255:.6f} per unit range)")
    print()
    assert avg_err < 0.02, f"Average error too high: {avg_err}"
    print("  PASS")
    print()


# =============================================================================
# TEST 2: Storage savings
# =============================================================================
def test_storage():
    print("=" * 60)
    print("TEST 2: Storage savings")
    print("=" * 60)

    vec = rand_vec()

    # float32 blob
    f32_blob = struct.pack(f'{DIM}f', *vec)
    # int8 blob (8-byte header + dim bytes)
    i8_blob = _quantize_f32_to_i8(vec)

    f32_bytes = len(f32_blob)
    i8_bytes = len(i8_blob)
    reduction = f32_bytes / i8_bytes

    print(f"  Dimension:           {DIM}")
    print(f"  float32 blob size:   {f32_bytes} bytes")
    print(f"  int8 blob size:      {i8_bytes} bytes  (8-byte header + {DIM} body)")
    print(f"  Reduction factor:    {reduction:.2f}x")
    print()

    # For large N_VECTORS
    f32_total = f32_bytes * N_VECTORS
    i8_total = i8_bytes * N_VECTORS
    print(f"  For {N_VECTORS} vectors:")
    print(f"    float32 total: {f32_total:,} bytes  ({f32_total/1024:.1f} KB)")
    print(f"    int8 total:    {i8_total:,} bytes  ({i8_total/1024:.1f} KB)")
    print(f"    Saved:         {f32_total - i8_total:,} bytes  ({(f32_total - i8_total)/1024:.1f} KB)")
    print()
    assert reduction > 3.5, f"Expected >3.5x reduction, got {reduction:.2f}x"
    print("  PASS")
    print()


# =============================================================================
# TEST 3: Search quality (top-k recall)
# =============================================================================
def test_search_quality():
    print("=" * 60)
    print("TEST 3: Search quality — top-k recall (float32 vs int8)")
    print("=" * 60)

    import sqlite3

    vectors = [rand_vec() for _ in range(N_VECTORS)]
    queries = [rand_vec() for _ in range(N_QUERIES)]

    # Build float32 index
    db_f32 = sqlite3.connect(":memory:")
    idx_f32 = _VecIndex(db_f32, DIM, quantize=False)
    for i, v in enumerate(vectors):
        idx_f32.insert(i + 1, v)
    db_f32.commit()

    # Build int8 index
    db_i8 = sqlite3.connect(":memory:")
    idx_i8 = _VecIndex(db_i8, DIM, quantize=True)
    for i, v in enumerate(vectors):
        idx_i8.insert(i + 1, v)
    db_i8.commit()

    total_recall = 0.0

    for q in queries:
        top_f32 = [rowid for rowid, _ in idx_f32.search(q, K)]
        top_i8 = [rowid for rowid, _ in idx_i8.search(q, K)]

        # Recall@K: how many of the true top-K are in the int8 top-K?
        true_set = set(top_f32)
        found = sum(1 for r in top_i8 if r in true_set)
        recall = found / K
        total_recall += recall

    avg_recall = total_recall / N_QUERIES

    print(f"  Vectors in index:  {N_VECTORS}")
    print(f"  Queries:           {N_QUERIES}")
    print(f"  K:                 {K}")
    print(f"  Average Recall@{K}: {avg_recall:.1%}")
    print()
    assert avg_recall >= 0.80, f"Recall too low: {avg_recall:.1%}"
    print("  PASS")
    print()
    return avg_recall


# =============================================================================
# TEST 4: Speed comparison
# =============================================================================
def test_speed():
    print("=" * 60)
    print("TEST 4: Speed comparison — insert and search")
    print("=" * 60)

    import sqlite3

    vectors = [rand_vec() for _ in range(N_VECTORS)]
    queries = [rand_vec() for _ in range(20)]

    # --- float32 ---
    db_f32 = sqlite3.connect(":memory:")
    idx_f32 = _VecIndex(db_f32, DIM, quantize=False)

    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        idx_f32.insert(i + 1, v)
    db_f32.commit()
    t_insert_f32 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for q in queries:
        idx_f32.search(q, K)
    t_search_f32 = (time.perf_counter() - t0) * 1000 / len(queries)

    # --- int8 ---
    db_i8 = sqlite3.connect(":memory:")
    idx_i8 = _VecIndex(db_i8, DIM, quantize=True)

    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        idx_i8.insert(i + 1, v)
    db_i8.commit()
    t_insert_i8 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for q in queries:
        idx_i8.search(q, K)
    t_search_i8 = (time.perf_counter() - t0) * 1000 / len(queries)

    print(f"  Vectors: {N_VECTORS}, dim={DIM}")
    print()
    print(f"  {'Operation':<25} {'float32':>12} {'int8':>12} {'ratio':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Insert total (ms)':<25} {t_insert_f32:>11.2f} {t_insert_i8:>11.2f} {'N/A':>10}")
    print(f"  {'Search avg (ms/query)':<25} {t_search_f32:>11.2f} {t_search_i8:>11.2f} {t_search_f32/t_search_i8:>9.2f}x")
    print()

    # int8 insert is slightly slower (quantization overhead) but search may be
    # faster due to smaller blobs fitting better in CPU cache
    print("  Note: int8 insert has quantization overhead; search benefits from")
    print("  smaller blobs (better cache utilization) especially at large N.")
    print()
    print("  PASS")
    print()


# =============================================================================
# TEST 5: MemoryStore integration — file on disk, stats report
# =============================================================================
def test_memorystore_integration():
    print("=" * 60)
    print("TEST 5: MemoryStore integration (disk, no real embeddings)")
    print("=" * 60)

    contents = [
        "The agent recalled past context about the user's project deadline.",
        "Python async/await patterns for concurrent agent tasks.",
        "SQLite WAL mode improves concurrent read performance.",
        "int8 quantization reduces vector storage by 4x with minimal quality loss.",
        "Cosine similarity measures angle between two vectors, range [-1, 1].",
        "FTS5 BM25 ranking is excellent for keyword-dense technical queries.",
        "Hybrid search combines keyword and semantic results for best recall.",
        "Working memories expire after 24 hours automatically.",
        "Core tier memories are permanent — never auto-archived.",
        "Model2Vec potion-8M embeds text in ~1ms at 256 dimensions.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path_f32 = os.path.join(tmpdir, "mem_f32.db")
        path_i8 = os.path.join(tmpdir, "mem_i8.db")

        store_f32 = MemoryStore(path_f32, embedding_dim=DIM, quantize=False)
        store_i8 = MemoryStore(path_i8, embedding_dim=DIM, quantize=True)

        # No embedding function — just text storage + FTS
        for c in contents:
            store_f32.remember(c, tier="learned")
            store_i8.remember(c, tier="learned")

        stats_f32 = store_f32.stats()
        stats_i8 = store_i8.stats()

        store_f32.close()
        store_i8.close()

    print(f"  Memories stored:  {len(contents)}")
    print()
    print(f"  float32 store:")
    print(f"    quantize:         {stats_f32['quantize']}")
    print(f"    bytes_per_vector: {stats_f32['bytes_per_vector']}")
    print(f"    total_memories:   {stats_f32['total_memories']}")
    print()
    print(f"  int8 store:")
    print(f"    quantize:         {stats_i8['quantize']}")
    print(f"    bytes_per_vector: {stats_i8['bytes_per_vector']}")
    print(f"    total_memories:   {stats_i8['total_memories']}")
    print()

    assert stats_i8["quantize"] is True
    assert stats_f32["quantize"] is False
    assert stats_i8["bytes_per_vector"] == 8 + DIM   # header + body
    assert stats_f32["bytes_per_vector"] == DIM * 4
    assert stats_i8["total_memories"] == len(contents)
    assert stats_f32["total_memories"] == len(contents)

    savings_pct = (1 - stats_i8["bytes_per_vector"] / stats_f32["bytes_per_vector"]) * 100
    print(f"  Vector storage reduction: {savings_pct:.1f}%")
    print()
    print("  PASS")
    print()


# =============================================================================
# Run all tests
# =============================================================================
if __name__ == "__main__":
    print()
    print("agentmem int8 quantization test suite")
    print("Python", sys.version.split()[0])
    print()

    test_roundtrip()
    test_storage()
    recall = test_search_quality()
    test_speed()
    test_memorystore_integration()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print(f"Search Recall@{K}: {recall:.1%}")
    print("=" * 60)
