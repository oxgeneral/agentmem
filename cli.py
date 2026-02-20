"""
agentmem.cli — Command-line interface.

Commands:
    agentmem init [--db memory.db]           Create new memory database
    agentmem import <file.md> [--tier learned]  Import markdown file
    agentmem import-dir <dir/> [--tier learned] Import all .md files from directory
    agentmem search <query> [--limit 5]      Search memories
    agentmem export [--tier core]            Export as markdown
    agentmem stats                           Show statistics
    agentmem serve                           Start MCP server (stdio)
"""
import argparse
import sys
import os
from pathlib import Path


def cmd_init(args):
    """Initialize a new memory database."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model(args.backend)
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    store.close()
    print(f"Created: {args.db}")
    print(f"Backend: {'model2vec' if embed.dim > 0 else 'FTS5 only'} ({embed.dim}d)")


def cmd_import(args):
    """Import a markdown file."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model(args.backend)
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    if embed.dim > 0:
        store.set_embed_fn(embed)  # pass model object for batch support

    result = store.import_markdown(args.file, tier=args.tier)
    store.close()

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Imported: {result['file']}")
    print(f"  Chunks: {result['chunks']}")
    print(f"  New: {result['imported']}")
    print(f"  Deduplicated: {result['deduplicated']}")


def cmd_import_dir(args):
    """Import all .md files from a directory."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model(args.backend)
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    if embed.dim > 0:
        store.set_embed_fn(embed)  # pass model object for batch support

    dir_path = Path(args.directory)
    if not dir_path.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    total_imported = 0
    total_dedup = 0
    for md_file in sorted(dir_path.glob("**/*.md")):
        result = store.import_markdown(str(md_file), tier=args.tier)
        if "error" not in result:
            print(f"  {md_file.name}: {result['imported']} new, {result['deduplicated']} dedup")
            total_imported += result["imported"]
            total_dedup += result["deduplicated"]

    store.close()
    print(f"\nTotal: {total_imported} imported, {total_dedup} deduplicated")


def cmd_search(args):
    """Search memories."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model(args.backend)
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    if embed.dim > 0:
        store.set_embed_fn(embed)  # pass model object for batch support

    results = store.recall(query=args.query, limit=args.limit, tier=args.tier)
    store.close()

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.3f} tier={r['tier']} method={r['method']}")
        content = r["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"  {content}")
        if r["source"]:
            print(f"  — source: {r['source']}")


def cmd_export(args):
    """Export memories as markdown."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model("null")  # Don't need embeddings for export
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    md = store.export_markdown(tier=args.tier)
    store.close()
    print(md)


def cmd_stats(args):
    """Show memory statistics."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model("null")
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    stats = store.stats()
    store.close()

    print(f"Database: {args.db}")
    print(f"Total memories: {stats['total_memories']}")
    print(f"Archived: {stats['archived']}")
    print(f"DB size: {stats['db_size_human']}")
    print(f"Vectors: {'yes' if stats['has_vectors'] else 'no'} ({stats['embedding_dim']}d)")
    print(f"\nBy tier:")
    for tier, count in stats["by_tier"].items():
        print(f"  {tier}: {count}")


def cmd_serve(args):
    """Start MCP server."""
    os.environ["AGENTMEM_DB"] = args.db
    os.environ["AGENTMEM_BACKEND"] = args.backend
    from .server import main as server_main
    server_main()


def main():
    parser = argparse.ArgumentParser(
        prog="agentmem",
        description="Lightweight persistent memory for AI agents",
    )
    parser.add_argument("--db", default="memory.db", help="Database file path")
    parser.add_argument("--backend", default="auto", choices=["auto", "model2vec", "null"])

    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Create new memory database")

    # import
    p_import = sub.add_parser("import", help="Import markdown file")
    p_import.add_argument("file", help="Markdown file to import")
    p_import.add_argument("--tier", default="learned", choices=["core", "learned", "episodic"])

    # import-dir
    p_import_dir = sub.add_parser("import-dir", help="Import all .md from directory")
    p_import_dir.add_argument("directory", help="Directory with .md files")
    p_import_dir.add_argument("--tier", default="learned", choices=["core", "learned", "episodic"])

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument("--tier", choices=["core", "learned", "episodic", "working"])

    # export
    p_export = sub.add_parser("export", help="Export as markdown")
    p_export.add_argument("--tier", choices=["core", "learned", "episodic", "working"])

    # stats
    sub.add_parser("stats", help="Show statistics")

    # serve
    sub.add_parser("serve", help="Start MCP server (stdio)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "init": cmd_init,
        "import": cmd_import,
        "import-dir": cmd_import_dir,
        "search": cmd_search,
        "export": cmd_export,
        "stats": cmd_stats,
        "serve": cmd_serve,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
