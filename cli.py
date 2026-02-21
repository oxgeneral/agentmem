"""
agentmem.cli — Command-line interface.

Commands:
    agentmem init [--db memory.db]           Create new memory database
    agentmem import <file.md> [--tier learned]  Import markdown file
    agentmem import-dir <dir/> [--tier learned] Import all .md files from directory
    agentmem search <query> [--limit 5]      Search memories
    agentmem export [--tier core]            Export as markdown
    agentmem stats                           Show statistics
    agentmem compact [--max-age-days 90]     Archive low-value memories
    agentmem history <memory_id>             Show version history of a memory
    agentmem consolidate [--threshold 0.85]  Find and merge near-duplicate memories
    agentmem related <entity> [--type mention]  Find memories related to an entity
    agentmem entities [--type url]           List all known entities
    agentmem procedures [-n namespace]       Show all procedural memories (agent rules)
    agentmem add-procedure "rule" [-n ns]    Add a behavioral rule
    agentmem process <chat.json> [-n ns]     Extract memories from conversation JSON
    agentmem serve                           Start MCP server (stdio)
    agentmem serve-http [--port 8422]        Start HTTP REST API server
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

    result = store.import_markdown(args.file, tier=args.tier,
                                    namespace=getattr(args, "namespace", ""))
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
        result = store.import_markdown(str(md_file), tier=args.tier,
                                       namespace=getattr(args, "namespace", ""))
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

    current_only = not getattr(args, "all_versions", False)
    results = store.recall(
        query=args.query, limit=args.limit, tier=args.tier,
        recency_weight=args.recency_weight,
        namespace=getattr(args, "namespace", None),
        current_only=current_only,
    )
    store.close()

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        imp_str = f" imp={r['importance']:.2f}" if "importance" in r else ""
        print(f"\n[{i}] score={r['score']:.3f} tier={r['tier']} method={r['method']}{imp_str}")
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
    stats = store.stats(namespace=getattr(args, "namespace", None))
    store.close()

    print(f"Database: {args.db}")
    print(f"Total memories: {stats['total_memories']}")
    print(f"Archived: {stats['archived']}")
    print(f"DB size: {stats['db_size_human']}")
    print(f"Vectors: {'yes' if stats['has_vectors'] else 'no'} ({stats['embedding_dim']}d)")
    if "avg_importance" in stats:
        print(f"Avg importance: {stats['avg_importance']:.2f}")
    print(f"\nBy tier:")
    for tier, count in stats["by_tier"].items():
        print(f"  {tier}: {count}")


def cmd_compact(args):
    """Archive low-value memories."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model("null")  # Don't need embeddings for compact
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    result = store.compact(
        max_age_days=args.max_age_days,
        min_access=args.min_access,
        tier=args.tier,
        namespace=args.namespace,
        dry_run=args.dry_run,
    )
    store.close()

    if result["dry_run"]:
        print(f"Dry run: {result['archived']} memories would be archived")
    else:
        print(f"Archived: {result['archived']} memories")


def cmd_history(args):
    """Show version history of a memory."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model
    import datetime

    embed = get_embedding_model("null")  # Don't need embeddings for history
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)

    results = store.history(memory_id=args.memory_id)
    store.close()

    if not results:
        print("No history found.")
        return

    print(f"Version history ({len(results)} versions, newest first):\n")
    for i, r in enumerate(results, 1):
        ts = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
        status = " [archived]" if r["archived"] else " [current]"
        print(f"  [{i}] id={r['id']}  {ts}{status}")
        content = r["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"      {content}")
        print()


def cmd_consolidate(args):
    """Find and merge near-duplicate memories."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model(args.backend)
    if embed.dim == 0:
        print("Error: consolidation requires an embedding backend (not null).", file=sys.stderr)
        sys.exit(1)

    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim)
    store.set_embed_fn(embed)

    result = store.consolidate(
        similarity_threshold=args.threshold,
        namespace=args.namespace,
        tier=args.tier,
        dry_run=args.dry_run,
    )
    store.close()

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if result["groups"] == 0:
        print("No duplicate groups found.")
        return

    prefix = "Dry run: " if result["dry_run"] else ""
    print(f"{prefix}{result['groups']} group(s), {result['archived']} memories {'would be ' if result['dry_run'] else ''}archived.\n")

    for i, detail in enumerate(result["details"], 1):
        print(f"Group {i}: kept id={detail['kept']}, archived {len(detail['archived_ids'])} memories")
        for preview in detail["contents_preview"]:
            print(f"  - {preview}")
        print()


def cmd_related(args):
    """Find memories related to an entity."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model("null")  # Don't need embeddings for entity lookup
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)

    results = store.related(
        entity=args.entity,
        entity_type=getattr(args, "type", None),
        limit=args.limit,
        namespace=getattr(args, "namespace", None),
    )
    store.close()

    if not results:
        print(f"No memories found related to '{args.entity}'.")
        return

    print(f"Memories related to '{args.entity}' ({len(results)} found):\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] id={r['id']} tier={r['tier']} entity={r['entity_name']} ({r['entity_type']})")
        content = r["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"  {content}")
        if r["source"]:
            print(f"  — source: {r['source']}")
        print()


def cmd_entities(args):
    """List all known entities."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model("null")  # Don't need embeddings for entity listing
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)

    results = store.entities(
        entity_type=getattr(args, "type", None),
        limit=args.limit,
    )
    store.close()

    if not results:
        print("No entities found.")
        return

    print(f"Known entities ({len(results)}):\n")
    for r in results:
        print(f"  {r['name']} ({r['type']}) — {r['memory_count']} memories")


def cmd_procedures(args):
    """Show all procedural memories (agent rules)."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model("null")  # Don't need embeddings for listing
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)

    text = store.get_procedures(namespace=getattr(args, "namespace", None))
    store.close()

    if not text:
        print("No procedural memories found.")
    else:
        print(text)


def cmd_add_procedure(args):
    """Add a behavioral rule (procedural memory)."""
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed = get_embedding_model(args.backend)
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    if embed.dim > 0:
        store.set_embed_fn(embed)

    result = store.add_procedure(
        rule=args.rule,
        namespace=getattr(args, "namespace", ""),
    )
    store.close()

    if result.get("deduplicated"):
        print(f"Rule already exists (id={result['id']})")
    else:
        print(f"Added procedural memory (id={result['id']})")


def cmd_process(args):
    """Process a conversation JSON file and extract memories."""
    import json as _json
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    # Read and parse JSON
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: {args.file} not found", file=sys.stderr)
        sys.exit(1)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except _json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Support both {"messages": [...]} and bare [...]
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    else:
        print("Error: JSON must be a list of messages or {\"messages\": [...]}", file=sys.stderr)
        sys.exit(1)

    embed = get_embedding_model(args.backend)
    store = MemoryStore(db_path=args.db, embedding_dim=embed.dim or 256)
    if embed.dim > 0:
        store.set_embed_fn(embed)

    result = store.process_conversation(
        messages=messages,
        namespace=getattr(args, "namespace", ""),
        source=args.source,
    )
    store.close()

    if result["extracted"] == 0:
        print("No extractable patterns found.")
        return

    print(f"Extracted: {result['extracted']} memories")
    print(f"\nBy type:")
    for etype, count in sorted(result["by_type"].items()):
        print(f"  {etype}: {count}")
    if result["memories"]:
        print(f"\nStored IDs: {result['memories']}")


def cmd_serve(args):
    """Start MCP server."""
    os.environ["AGENTMEM_DB"] = args.db
    os.environ["AGENTMEM_BACKEND"] = args.backend
    from .server import main as server_main
    server_main()


def cmd_serve_http(args):
    """Start HTTP REST API server."""
    from .http_server import run_http
    run_http(port=args.port, db_path=args.db, backend=args.backend)


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
    p_import.add_argument("-n", "--namespace", default="",
                          help="Namespace for imported memories")

    # import-dir
    p_import_dir = sub.add_parser("import-dir", help="Import all .md from directory")
    p_import_dir.add_argument("directory", help="Directory with .md files")
    p_import_dir.add_argument("--tier", default="learned", choices=["core", "learned", "episodic"])
    p_import_dir.add_argument("-n", "--namespace", default="",
                              help="Namespace for imported memories")

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument("--tier", choices=["core", "learned", "episodic", "working", "procedural"])
    p_search.add_argument("--recency-weight", type=float, default=None,
                          help="Recency boost weight (0.0-1.0, default 0.1)")
    p_search.add_argument("-n", "--namespace", default=None,
                          help="Filter by namespace (prefix match)")
    p_search.add_argument("--all-versions", action="store_true",
                          help="Include superseded memories (default: only latest versions)")

    # export
    p_export = sub.add_parser("export", help="Export as markdown")
    p_export.add_argument("--tier", choices=["core", "learned", "episodic", "working"])

    # stats
    p_stats = sub.add_parser("stats", help="Show statistics")
    p_stats.add_argument("-n", "--namespace", default=None,
                         help="Filter stats by namespace (prefix match)")

    # compact
    p_compact = sub.add_parser("compact", help="Archive low-value memories")
    p_compact.add_argument("--max-age-days", type=int, default=90,
                           help="Archive memories older than this many days (default 90)")
    p_compact.add_argument("--min-access", type=int, default=0,
                           help="Archive memories accessed this many times or less (default 0)")
    p_compact.add_argument("--tier", choices=["learned", "episodic", "working"],
                           default=None, help="Only compact this tier (core and procedural are never compacted)")
    p_compact.add_argument("-n", "--namespace", default=None,
                           help="Only compact this namespace (prefix match)")
    p_compact.add_argument("--dry-run", action="store_true",
                           help="Show count without archiving")

    # history
    p_history = sub.add_parser("history", help="Show version history of a memory")
    p_history.add_argument("memory_id", type=int, help="Memory ID to get history for")

    # consolidate
    p_consolidate = sub.add_parser("consolidate", help="Find and merge near-duplicate memories")
    p_consolidate.add_argument("--threshold", type=float, default=0.85,
                               help="Cosine similarity threshold (default 0.85)")
    p_consolidate.add_argument("-n", "--namespace", default=None,
                               help="Only consolidate within this namespace")
    p_consolidate.add_argument("--tier", choices=["core", "learned", "episodic", "working", "procedural"],
                               default=None, help="Only consolidate this tier")
    p_consolidate.add_argument("--dry-run", action="store_true",
                               help="Preview groups without merging")

    # related
    p_related = sub.add_parser("related", help="Find memories related to an entity")
    p_related.add_argument("entity", help="Entity name to search for (e.g. @username, 10.0.0.1)")
    p_related.add_argument("--type", default=None,
                           choices=["mention", "url", "email", "hashtag", "ip", "port",
                                    "path", "money", "number_unit", "env_var"],
                           help="Filter by entity type")
    p_related.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    p_related.add_argument("-n", "--namespace", default=None,
                           help="Filter by namespace (prefix match)")

    # entities
    p_entities = sub.add_parser("entities", help="List all known entities")
    p_entities.add_argument("--type", default=None,
                            choices=["mention", "url", "email", "hashtag", "ip", "port",
                                     "path", "money", "number_unit", "env_var"],
                            help="Filter by entity type")
    p_entities.add_argument("--limit", type=int, default=50, help="Max results (default 50)")

    # procedures
    p_procedures = sub.add_parser("procedures", help="Show all procedural memories (agent rules)")
    p_procedures.add_argument("-n", "--namespace", default=None,
                              help="Filter by namespace (prefix match)")

    # add-procedure
    p_add_proc = sub.add_parser("add-procedure", help="Add a behavioral rule (procedural memory)")
    p_add_proc.add_argument("rule", help="The behavioral rule text")
    p_add_proc.add_argument("-n", "--namespace", default="",
                            help="Namespace for rule isolation")

    # process
    p_process = sub.add_parser("process", help="Extract memories from conversation JSON file")
    p_process.add_argument("file", help="JSON file with conversation messages")
    p_process.add_argument("-n", "--namespace", default="",
                           help="Namespace for extracted memories")
    p_process.add_argument("--source", default="conversation",
                           help="Source label (default 'conversation')")

    # serve
    sub.add_parser("serve", help="Start MCP server (stdio)")

    # serve-http
    p_serve_http = sub.add_parser("serve-http", help="Start HTTP REST API server")
    p_serve_http.add_argument("--port", type=int, default=8422,
                              help="Port to listen on (default 8422)")

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
        "compact": cmd_compact,
        "history": cmd_history,
        "consolidate": cmd_consolidate,
        "related": cmd_related,
        "entities": cmd_entities,
        "procedures": cmd_procedures,
        "add-procedure": cmd_add_procedure,
        "process": cmd_process,
        "serve": cmd_serve,
        "serve-http": cmd_serve_http,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
