"""
agentmem.embeddings — Lightweight embedding backends.

Default: model2vec (potion-base-8M) — 8MB, numpy-only, 30K+ sent/sec.
Fallback: HashEmbedding — pure Python stdlib, zero deps, ~70-80% quality.
Null: no embeddings (FTS5 keyword search only).

The goal: semantic search WITHOUT PyTorch (1.5GB).
"""
from __future__ import annotations

from typing import Callable, Protocol
import hashlib
import math
import struct
import re
from functools import lru_cache


class EmbeddingModel(Protocol):
    """Interface for embedding models."""
    dim: int

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# HashEmbedding — pure Python, zero external dependencies
# ---------------------------------------------------------------------------

class HashEmbedding:
    """
    Random-Projection Hash Vectorizer — pure Python stdlib, zero dependencies.

    Algorithm:
      1. Tokenize text into word unigrams (weighted 3x) + char n-grams (2-3 chars)
      2. Hash each token with MD5 to get a stable 16-byte digest
      3. Decode the digest as 4 signed 32-bit ints → use as ±1 sparse signature
         then map those 4 indices into the output vector (SimHash-style)
      4. L2-normalize the accumulated result

    The key optimisation over a naive LCG loop: instead of generating `dim`
    random numbers per token, we extract 4 index+sign pairs from the MD5
    digest directly.  This reduces per-token work from O(dim) to O(1) while
    preserving enough variance for decent nearest-neighbour retrieval.

    Quality: ~70-80% of transformer models for keyword / entity retrieval.
    Speed:   <0.1 ms per embedding (typically <0.01 ms for short texts).
    Deps:    hashlib, struct, math, re — all Python stdlib.
    Unicode: handles Russian, CJK, and any UTF-8 text natively.
    """

    # Word tokens count this many times more than n-gram tokens when
    # accumulating the projection vector. Higher = more "topic" signal,
    # less sub-word noise that bleeds across unrelated domains.
    _WORD_WEIGHT = 3.0
    _NGRAM_WEIGHT = 1.0

    # Hard cap on tokens processed per text.
    # Prevents O(n) blowup on pathological long-word inputs (e.g. 500-char words).
    # Normal sentences are <200 tokens even with n-grams; this only clips edge cases.
    _MAX_TOKENS = 200

    def __init__(self, dim: int = 128, ngram_range: tuple[int, int] = (2, 3)):
        self.dim = dim
        self._ngram_min, self._ngram_max = ngram_range
        # Number of hash indices we extract per token (must be <= dim/2)
        # Each MD5 gives us 16 bytes = four 32-bit words; we use all four.
        self._k = min(4, dim // 2)
        # Precompute a mask for mapping hash values into [0, dim)
        # We need dim to be usable as modulus — any value works.
        self._dim = dim
        # Bind the cached helper so each HashEmbedding instance shares
        # the same cache (keyed on token string + dim + k).
        self._cached_hash = self.__class__._hash_to_indices_signs_cached

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[tuple[str, float]]:
        """
        Return (token, weight) pairs.
        Words have weight _WORD_WEIGHT; n-grams have weight _NGRAM_WEIGHT.
        """
        text = text.lower().strip()
        # Split on whitespace and punctuation (unicode-aware)
        words = re.split(r"[\s\W]+", text, flags=re.UNICODE)
        words = [w for w in words if len(w) >= 2]  # skip 1-char noise

        tokens: list[tuple[str, float]] = [
            (w, self._WORD_WEIGHT) for w in words
        ]

        for word in words:
            padded = f"#{word}#"
            for n in range(self._ngram_min, self._ngram_max + 1):
                for i in range(len(padded) - n + 1):
                    tokens.append((padded[i:i + n], self._NGRAM_WEIGHT))

        # Cap to avoid quadratic blowup on pathological inputs
        return tokens[:self._MAX_TOKENS]

    # ------------------------------------------------------------------
    # Hashing & sparse projection
    # ------------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=4096)
    def _hash_to_indices_signs_cached(token: str, dim: int, k: int) -> tuple[tuple[int, float], ...]:
        """
        Map a token to k (index, sign) pairs using its MD5 digest.

        MD5 produces 16 bytes.  We split into four 32-bit little-endian
        unsigned ints.  Each int gives:
          index = value % dim        (which output dimension to update)
          sign  = +1 if bit 16 set, else -1

        This is a SimHash-style projection: O(1) per token regardless of dim.
        Cached via lru_cache: repeated tokens (stop-words, n-grams) cost ~0 ns.
        Returns a tuple (not list) to satisfy lru_cache hashability requirement.
        """
        digest = hashlib.md5(token.encode("utf-8")).digest()
        result = []
        for i in range(k):
            val = struct.unpack_from("<I", digest, i * 4)[0]
            idx = val % dim
            sign = 1.0 if (val & 0x10000) else -1.0
            result.append((idx, sign))
        return tuple(result)

    def _hash_to_indices_signs(self, token: str) -> tuple[tuple[int, float], ...]:
        return self._hash_to_indices_signs_cached(token, self._dim, self._k)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed text into a dense L2-normalised vector of length self.dim."""
        if not text or not text.strip():
            return [0.0] * self.dim

        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        for token, weight in tokens:
            for idx, sign in self._hash_to_indices_signs(token):
                vec[idx] += sign * weight

        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec))
        if norm < 1e-9:
            return vec
        inv = 1.0 / norm
        return [v * inv for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# model2vec backend (optional, best quality)
# ---------------------------------------------------------------------------

class Model2VecEmbedding:
    """
    model2vec: Static embeddings distilled from transformers.
    8MB model, numpy-only, 30,000+ sentences/sec on CPU.
    English-focused but works for mixed content.
    """

    def __init__(self, model_name: str = "minishlab/potion-base-8M") -> None:
        from model2vec import StaticModel
        self._model = StaticModel.from_pretrained(model_name)
        # Get dim from a test embedding
        test = self._model.encode(["test"])
        self.dim: int = test.shape[1]

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode([text])[0]
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts)
        return vecs.tolist()


# ---------------------------------------------------------------------------
# Null backend (FTS5 only)
# ---------------------------------------------------------------------------

class NullEmbedding:
    """No embeddings — FTS5 keyword search only. Zero dependencies."""

    dim: int = 0

    def embed(self, text: str) -> list[float] | None:
        return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# LazyEmbedding — delays model loading until first embed() call
# ---------------------------------------------------------------------------

class LazyEmbedding:
    """
    Wraps any embedding model factory and delays actual loading until first use.

    This gives zero cold-start cost for operations that don't need embeddings
    (stats, today, forget, FTS-only recall). The model is loaded exactly once,
    on the first call to embed() or embed_batch().

    Usage:
        lazy = LazyEmbedding(lambda: Model2VecEmbedding(), known_dim=256)
        # Model NOT loaded yet — MemoryStore init is instant
        vec = lazy.embed("some text")  # Model loads here, first time only

    Args:
        factory_fn: Zero-argument callable that returns an EmbeddingModel.
        known_dim:  If provided, .dim returns this value without loading the
                    model. Use when you know the output dimension ahead of time
                    (e.g. model2vec potion-base-8M is always 256d).
    """

    def __init__(self, factory_fn: Callable[[], EmbeddingModel], known_dim: int | None = None):
        self._factory = factory_fn
        self._model: EmbeddingModel | None = None
        self._known_dim = known_dim

    @property
    def dim(self) -> int:
        """
        Dimension of the embedding vectors.
        If known_dim was provided at construction, returns it immediately
        without loading the model. Otherwise triggers model load.
        """
        if self._known_dim is not None:
            return self._known_dim
        return self._get_model().dim

    def _get_model(self) -> EmbeddingModel:
        """Load model on first access (thread-safe-enough for single-process use)."""
        if self._model is None:
            self._model = self._factory()
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Triggers model load on first call."""
        return self._get_model().embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Triggers model load on first call."""
        return self._get_model().embed_batch(texts)

    @property
    def loaded(self) -> bool:
        """True if the model has been loaded into memory."""
        return self._model is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_model(backend: str = "auto", lazy: bool = True) -> EmbeddingModel:
    """
    Get the best available embedding model.

    Priority (auto mode):
      1. model2vec (if installed) — 8MB, fast, best quality
      2. HashEmbedding             — pure Python stdlib, zero deps, decent quality
      3. null                      — FTS5 keyword search only (explicit opt-in)

    Args:
        backend: "auto" | "model2vec" | "hash" | "null"
        lazy:    If True (default), wrap heavy models in LazyEmbedding so the
                 model file is only loaded on the first actual embed() call.
                 Has no effect for "null" or "hash" backends (they are instant).
    """
    if backend == "null":
        return NullEmbedding()

    if backend == "hash":
        # HashEmbedding is pure Python, ~0ms init — no point wrapping
        return HashEmbedding()

    if backend in ("auto", "model2vec"):
        # Check if model2vec is importable without actually loading the model
        try:
            import importlib
            importlib.util.find_spec("model2vec")
            has_model2vec = True
        except (ImportError, ValueError):
            has_model2vec = False

        if has_model2vec or backend == "model2vec":
            if backend == "model2vec" and not has_model2vec:
                raise ImportError(
                    "model2vec not installed. Run: pip install model2vec"
                )
            if lazy:
                # known_dim=256: potion-base-8M output is always 256d.
                # This lets MemoryStore.__init__ set up the schema without
                # triggering the model load (which costs ~500ms).
                return LazyEmbedding(lambda: Model2VecEmbedding(), known_dim=256)
            else:
                return Model2VecEmbedding()

        # model2vec not available — fall through to hash (instant, no lazy needed)
        return HashEmbedding()

    # Unknown backend — default to hash (safe, zero deps)
    return HashEmbedding()
