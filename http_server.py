"""
agentmem.http_server — HTTP REST API for agent memory.

Exposes all agentmem tools as REST endpoints:
    GET  /recall?query=...&limit=5&tier=...&namespace=...
    POST /remember  {content, tier, tags, source, namespace}
    POST /save_state  {state, namespace}
    GET  /today?namespace=...
    POST /forget  {memory_id, namespace}
    GET  /stats?namespace=...
    POST /compact  {max_age_days, min_access, tier, namespace, dry_run}
    POST /consolidate  {similarity_threshold, namespace, tier, dry_run}
    GET  /related?entity=...&entity_type=...&limit=...&namespace=...
    GET  /entities?entity_type=...&limit=...
    POST /update_memory  {old_id, new_content, tier, tags, namespace}
    GET  /history?memory_id=...
    POST /unarchive  {memory_id}
    GET  /health  — simple health check

All responses are JSON. CORS headers included for browser access.

Zero new dependencies — uses only Python stdlib (http.server, json, urllib.parse).

Usage:
    # As a module
    python -m agentmem.http_server --port 8422 --db memory.db

    # As CLI entry point
    agentmem-http --port 8422 --db memory.db

    # Programmatically
    from agentmem.http_server import create_server
    server = create_server(port=8422, db_path="memory.db", backend="auto")
    server.serve_forever()
"""
import json
import sys
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class _LazyStore:
    """
    Lazy-initializing wrapper for MemoryStore.

    Creates the SQLite-backed store on first access, ensuring
    the connection is created in the thread that will use it.
    This avoids SQLite's "created in a different thread" error
    when the server is started from a background thread.
    """

    def __init__(self, db_path, backend):
        self._db_path = db_path
        self._backend = backend
        self._store = None

    def _init(self):
        if self._store is not None:
            return
        from .core import MemoryStore
        from .embeddings import get_embedding_model

        embed_model = get_embedding_model(self._backend)
        self._store = MemoryStore(
            db_path=self._db_path, embedding_dim=embed_model.dim or 256
        )
        if embed_model.dim > 0:
            self._store.set_embed_fn(embed_model)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._init()
        return getattr(self._store, name)


def _make_handler(store):
    """Create a request handler class bound to a (possibly lazy) store."""

    class AgentMemHandler(BaseHTTPRequestHandler):
        """HTTP request handler for agentmem REST API."""

        # Suppress default stderr logging for each request
        def log_message(self, format, *args):
            sys.stderr.write(
                "[agentmem-http] %s - %s\n"
                % (self.client_address[0], format % args)
            )

        def _set_cors_headers(self):
            """Set CORS headers on every response."""
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers", "Content-Type, Authorization"
            )

        def _send_json(self, data, status=200):
            """Send a JSON response with proper headers."""
            body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._set_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status, message):
            """Send a JSON error response."""
            self._send_json({"error": message}, status=status)

        def _read_json_body(self):
            """Read and parse JSON body from POST request."""
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}
            raw = self.rfile.read(content_length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"Invalid JSON body: {e}")

        def _get_params(self):
            """Parse query string parameters from URL."""
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query, keep_blank_values=False)
            # parse_qs returns lists; unwrap single values
            return {k: v[0] if len(v) == 1 else v for k, v in params.items()}

        def _get_path(self):
            """Get clean path without query string."""
            return urlparse(self.path).path.rstrip("/") or "/"

        # ==============================================================
        # HTTP Methods
        # ==============================================================

        def do_OPTIONS(self):
            """Handle CORS preflight requests."""
            self.send_response(204)
            self._set_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            """Route GET requests."""
            path = self._get_path()
            params = self._get_params()

            try:
                if path == "/health":
                    self._handle_health()
                elif path == "/recall":
                    self._handle_recall(params)
                elif path == "/today":
                    self._handle_today(params)
                elif path == "/stats":
                    self._handle_stats(params)
                elif path == "/related":
                    self._handle_related(params)
                elif path == "/entities":
                    self._handle_entities(params)
                elif path == "/history":
                    self._handle_history(params)
                else:
                    self._send_error_json(404, f"Not found: {path}")
            except Exception as e:
                self._send_error_json(500, str(e))

        def do_POST(self):
            """Route POST requests."""
            path = self._get_path()

            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_error_json(400, str(e))
                return

            try:
                if path == "/remember":
                    self._handle_remember(body)
                elif path == "/save_state":
                    self._handle_save_state(body)
                elif path == "/forget":
                    self._handle_forget(body)
                elif path == "/compact":
                    self._handle_compact(body)
                elif path == "/consolidate":
                    self._handle_consolidate(body)
                elif path == "/update_memory":
                    self._handle_update_memory(body)
                elif path == "/unarchive":
                    self._handle_unarchive(body)
                else:
                    self._send_error_json(404, f"Not found: {path}")
            except Exception as e:
                self._send_error_json(500, str(e))

        # ==============================================================
        # GET Handlers
        # ==============================================================

        def _handle_health(self):
            """GET /health — simple health check."""
            stats = store.stats()
            self._send_json(
                {
                    "status": "ok",
                    "memories": stats["total_memories"],
                    "db_size": stats.get("db_size_human", "unknown"),
                }
            )

        def _handle_recall(self, params):
            """GET /recall?query=...&limit=5&tier=...&namespace=...&recency_weight=...&current_only=..."""
            query = params.get("query")
            if not query:
                self._send_error_json(400, "Missing required parameter: query")
                return

            limit = int(params.get("limit", 5))
            tier = params.get("tier")
            namespace = params.get("namespace")
            recency_weight = params.get("recency_weight")
            if recency_weight is not None:
                recency_weight = float(recency_weight)
            current_only_str = params.get("current_only", "true").lower()
            current_only = current_only_str not in ("false", "0", "no")

            results = store.recall(
                query=query,
                limit=limit,
                tier=tier,
                recency_weight=recency_weight,
                namespace=namespace,
                current_only=current_only,
            )
            self._send_json(results)

        def _handle_today(self, params):
            """GET /today?namespace=..."""
            namespace = params.get("namespace")
            results = store.today(namespace=namespace)
            self._send_json(results)

        def _handle_stats(self, params):
            """GET /stats?namespace=..."""
            namespace = params.get("namespace")
            result = store.stats(namespace=namespace)
            self._send_json(result)

        def _handle_related(self, params):
            """GET /related?entity=...&entity_type=...&limit=...&namespace=..."""
            entity = params.get("entity")
            if not entity:
                self._send_error_json(400, "Missing required parameter: entity")
                return

            entity_type = params.get("entity_type")
            limit = int(params.get("limit", 10))
            namespace = params.get("namespace")

            results = store.related(
                entity=entity,
                entity_type=entity_type,
                limit=limit,
                namespace=namespace,
            )
            self._send_json(results)

        def _handle_entities(self, params):
            """GET /entities?entity_type=...&limit=..."""
            entity_type = params.get("entity_type")
            limit = int(params.get("limit", 50))

            results = store.entities(
                entity_type=entity_type,
                limit=limit,
            )
            self._send_json(results)

        def _handle_history(self, params):
            """GET /history?memory_id=..."""
            memory_id = params.get("memory_id")
            if memory_id is None:
                self._send_error_json(400, "Missing required parameter: memory_id")
                return

            results = store.history(memory_id=int(memory_id))
            self._send_json(results)

        # ==============================================================
        # POST Handlers
        # ==============================================================

        def _handle_remember(self, body):
            """POST /remember {content, tier, tags, source, namespace}"""
            content = body.get("content")
            if not content:
                self._send_error_json(400, "Missing required field: content")
                return

            result = store.remember(
                content=content,
                tier=body.get("tier", "learned"),
                tags=body.get("tags", []),
                source=body.get("source", ""),
                namespace=body.get("namespace", ""),
            )
            self._send_json(result, status=201)

        def _handle_save_state(self, body):
            """POST /save_state {state, namespace}"""
            state = body.get("state")
            if not state:
                self._send_error_json(400, "Missing required field: state")
                return

            result = store.save_state(
                state=state,
                namespace=body.get("namespace", ""),
            )
            self._send_json(result)

        def _handle_forget(self, body):
            """POST /forget {memory_id, namespace}"""
            memory_id = body.get("memory_id")
            if memory_id is None:
                self._send_error_json(400, "Missing required field: memory_id")
                return

            result = store.forget(
                memory_id=int(memory_id),
                namespace=body.get("namespace"),
            )
            self._send_json(result)

        def _handle_compact(self, body):
            """POST /compact {max_age_days, min_access, tier, namespace, dry_run}"""
            result = store.compact(
                max_age_days=body.get("max_age_days", 90),
                min_access=body.get("min_access", 0),
                tier=body.get("tier"),
                namespace=body.get("namespace"),
                dry_run=body.get("dry_run", False),
            )
            self._send_json(result)

        def _handle_consolidate(self, body):
            """POST /consolidate {similarity_threshold, namespace, tier, dry_run}"""
            result = store.consolidate(
                similarity_threshold=body.get("similarity_threshold", 0.85),
                namespace=body.get("namespace"),
                tier=body.get("tier"),
                dry_run=body.get("dry_run", False),
            )
            self._send_json(result)

        def _handle_update_memory(self, body):
            """POST /update_memory {old_id, new_content, tier, tags, namespace}"""
            old_id = body.get("old_id")
            new_content = body.get("new_content")
            if old_id is None:
                self._send_error_json(400, "Missing required field: old_id")
                return
            if not new_content:
                self._send_error_json(400, "Missing required field: new_content")
                return

            result = store.update_memory(
                old_id=int(old_id),
                new_content=new_content,
                tier=body.get("tier"),
                tags=body.get("tags"),
                namespace=body.get("namespace"),
            )
            self._send_json(result)

        def _handle_unarchive(self, body):
            """POST /unarchive {memory_id}"""
            memory_id = body.get("memory_id")
            if memory_id is None:
                self._send_error_json(400, "Missing required field: memory_id")
                return

            result = store.unarchive(memory_id=int(memory_id))
            self._send_json(result)

    return AgentMemHandler


def create_server(port=8422, db_path="memory.db", backend="auto"):
    """
    Create and return an HTTPServer instance (without starting it).

    The MemoryStore is lazily initialized on the first request,
    so the SQLite connection is always created in the serving thread.
    This avoids "SQLite objects created in a thread can only be used
    in that same thread" errors when serve_forever() runs in a
    background thread.

    Args:
        port: TCP port to listen on.
        db_path: Path to SQLite database file.
        backend: Embedding backend ("auto", "model2vec", "hash", "null").

    Returns:
        HTTPServer instance. Call .serve_forever() to start.
    """
    lazy_store = _LazyStore(db_path, backend)
    handler_class = _make_handler(lazy_store)
    server = HTTPServer(("0.0.0.0", port), handler_class)
    # Attach lazy store for external access
    server._agentmem_store = lazy_store
    return server


def run_http(port=8422, db_path="memory.db", backend="auto"):
    """
    Start HTTP server and block forever.

    Args:
        port: TCP port to listen on.
        db_path: Path to SQLite database file.
        backend: Embedding backend ("auto", "model2vec", "hash", "null").
    """
    # Eagerly init the store for the startup message (same thread)
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed_model = get_embedding_model(backend)
    store = MemoryStore(db_path=db_path, embedding_dim=embed_model.dim or 256)
    if embed_model.dim > 0:
        store.set_embed_fn(embed_model)

    stats = store.stats()
    print(
        f"agentmem HTTP server listening on http://0.0.0.0:{port}",
        file=sys.stderr,
    )
    print(
        f"  {stats['total_memories']} memories, {stats['db_size_human']}",
        file=sys.stderr,
    )
    print(
        f"  Endpoints: /health /recall /remember /stats /today ...",
        file=sys.stderr,
    )

    handler_class = _make_handler(store)
    server = HTTPServer(("0.0.0.0", port), handler_class)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
    finally:
        server.server_close()


def main():
    """CLI entry point for agentmem-http."""
    parser = argparse.ArgumentParser(
        prog="agentmem-http",
        description="agentmem HTTP REST API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8422,
        help="Port to listen on (default: 8422)",
    )
    parser.add_argument(
        "--db",
        default="memory.db",
        help="Database file path (default: memory.db)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "model2vec", "hash", "null"],
        help="Embedding backend (default: auto)",
    )
    args = parser.parse_args()
    run_http(port=args.port, db_path=args.db, backend=args.backend)


if __name__ == "__main__":
    main()
