"""
agentmem — Lightweight persistent memory for AI agents.

One SQLite file. Hybrid search (keywords + semantics). 12MB total.
No PyTorch. No cloud. No server. Just memory.

Usage:
    from agentmem import MemoryStore, get_embedding_model

    # Initialize
    embed = get_embedding_model()  # auto-selects best available
    store = MemoryStore("memory.db", embedding_dim=embed.dim)
    store.set_embed_fn(embed.embed)

    # Store
    store.remember("Alexander's Telegram ID is 252708838", tier="core")

    # Recall
    results = store.recall("What is Alexander's Telegram ID?")

    # Save state before context compression
    store.save_state("Working on feature X, step 3 of 5, blocked by Y")
"""
from .core import MemoryStore, TIERS
from .embeddings import get_embedding_model

__version__ = "0.1.0"
__all__ = ["MemoryStore", "get_embedding_model", "TIERS"]
