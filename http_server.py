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

Concurrent: uses ThreadingHTTPServer so multiple clients can query simultaneously.
Thread-safe: each thread gets its own SQLite connection via threading.local().

Zero new dependencies — uses only Python stdlib (http.server, json, urllib.parse, threading, signal).

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
import signal
import sys
import time
import argparse
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .core import MemoryNotFoundError, InvalidTierError, EmbeddingError, AgentMemError


# Max request body size: 10 MB
MAX_BODY_SIZE = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Per-thread MemoryStore (SQLite connections are NOT thread-safe)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_thread_store(db_path, backend, embed_dim):
    """
    Get or create a per-thread MemoryStore instance.

    SQLite connections cannot be shared across threads. This function
    uses threading.local() to ensure each worker thread gets its own
    MemoryStore (and thus its own SQLite connection).
    """
    if not hasattr(_thread_local, "store"):
        from .core import MemoryStore
        from .embeddings import get_embedding_model

        embed = get_embedding_model(backend)
        store = MemoryStore(db_path=db_path, embedding_dim=embed.dim or embed_dim)
        if embed.dim > 0:
            store.set_embed_fn(embed)
        _thread_local.store = store
    return _thread_local.store


# ---------------------------------------------------------------------------
# Helper: safe int/float parsing
# ---------------------------------------------------------------------------

def _safe_int(value, name="parameter"):
    """Parse string to int, raising ValueError with a clear message on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid integer for '{name}': {value!r}")


def _safe_float(value, name="parameter"):
    """Parse string to float, raising ValueError with a clear message on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid number for '{name}': {value!r}")


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------


class MemoryHandler(BaseHTTPRequestHandler):
    """HTTP request handler for agentmem REST API.

    Thread-safe: each request gets a per-thread MemoryStore via
    _get_thread_store(). Class variables db_path/backend/embed_dim
    are set by create_server() / run_http() before serving starts.
    """

    # Config — set as class variables before serving
    db_path = "memory.db"
    backend = "auto"
    embed_dim = 256
    cors_origin = "*"

    @property
    def store(self):
        """Get the per-thread MemoryStore for this request."""
        return _get_thread_store(self.db_path, self.backend, self.embed_dim)

    # ------------------------------------------------------------------
    # Logging with response time
    # ------------------------------------------------------------------

    def log_message(self, format, *args):
        """Log with client address and thread name."""
        elapsed = ""
        if hasattr(self, "_request_start"):
            dt = (time.monotonic() - self._request_start) * 1000
            elapsed = f" ({dt:.1f}ms)"
        sys.stderr.write(
            "[agentmem-http] %s [%s] %s%s\n"
            % (
                self.client_address[0],
                threading.current_thread().name,
                format % args,
                elapsed,
            )
        )

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _set_cors_headers(self):
        """Set CORS headers on every response."""
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )

    def _send_json(self, data, status=200):
        """Send a JSON response with proper headers and Content-Length."""
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status, message, hint=None):
        """Send a JSON error response with optional hint."""
        body = {"error": message}
        if hint:
            body["hint"] = hint
        self._send_json(body, status=status)

    def _handle_exception(self, exc):
        """Map agentmem exceptions to proper HTTP status codes and hints."""
        if isinstance(exc, ValueError):
            # BUG 6: int/float parse errors → 400
            self._send_error_json(400, str(exc))
        elif isinstance(exc, MemoryNotFoundError):
            self._send_error_json(404, str(exc),
                                  hint="Use GET /recall?query=... to find memory IDs.")
        elif isinstance(exc, InvalidTierError):
            self._send_error_json(400, str(exc),
                                  hint="Valid tiers: core, learned, episodic, working, procedural")
        elif isinstance(exc, EmbeddingError):
            self._send_error_json(500, str(exc),
                                  hint="Check that the embedding backend is configured correctly.")
        elif isinstance(exc, AgentMemError):
            self._send_error_json(400, str(exc))
        else:
            # Don't expose internal tracebacks to HTTP clients
            self._send_error_json(500, f"Internal server error: {type(exc).__name__}: {exc}")

    def _read_json_body(self):
        """Read and parse JSON body from POST request. Enforces max body size."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        if content_length > MAX_BODY_SIZE:
            raise ValueError(
                f"Request body too large: {content_length} bytes "
                f"(max {MAX_BODY_SIZE // (1024 * 1024)} MB)"
            )
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
        self._request_start = time.monotonic()
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        """Route GET requests."""
        self._request_start = time.monotonic()
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
                self._send_error_json(404, f"Not found: {path}",
                                      hint="Available GET endpoints: /health, /recall, /today, /stats, /related, /entities, /history")
        except Exception as e:
            self._handle_exception(e)

    def do_POST(self):
        """Route POST requests."""
        self._request_start = time.monotonic()
        path = self._get_path()

        try:
            body = self._read_json_body()
        except ValueError as e:
            self._send_error_json(400, str(e),
                                  hint="Send a valid JSON body with Content-Type: application/json")
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
                self._send_error_json(404, f"Not found: {path}",
                                      hint="Available POST endpoints: /remember, /save_state, /forget, /compact, /consolidate, /update_memory, /unarchive")
        except Exception as e:
            self._handle_exception(e)

    # ==============================================================
    # GET Handlers
    # ==============================================================

    def _handle_health(self):
        """GET /health — simple health check."""
        stats = self.store.stats()
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

        limit = _safe_int(params.get("limit", 5), "limit")
        tier = params.get("tier")
        namespace = params.get("namespace")
        recency_weight = params.get("recency_weight")
        if recency_weight is not None:
            recency_weight = _safe_float(recency_weight, "recency_weight")
        current_only_str = params.get("current_only", "true").lower()
        current_only = current_only_str not in ("false", "0", "no")

        results = self.store.recall(
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
        results = self.store.today(namespace=namespace)
        self._send_json(results)

    def _handle_stats(self, params):
        """GET /stats?namespace=..."""
        namespace = params.get("namespace")
        result = self.store.stats(namespace=namespace)
        self._send_json(result)

    def _handle_related(self, params):
        """GET /related?entity=...&entity_type=...&limit=...&namespace=..."""
        entity = params.get("entity")
        if not entity:
            self._send_error_json(400, "Missing required parameter: entity")
            return

        entity_type = params.get("entity_type")
        limit = _safe_int(params.get("limit", 10), "limit")
        namespace = params.get("namespace")

        results = self.store.related(
            entity=entity,
            entity_type=entity_type,
            limit=limit,
            namespace=namespace,
        )
        self._send_json(results)

    def _handle_entities(self, params):
        """GET /entities?entity_type=...&limit=..."""
        entity_type = params.get("entity_type")
        limit = _safe_int(params.get("limit", 50), "limit")

        results = self.store.entities(
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

        results = self.store.history(memory_id=_safe_int(memory_id, "memory_id"))
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

        result = self.store.remember(
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

        result = self.store.save_state(
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

        result = self.store.forget(
            memory_id=_safe_int(memory_id, "memory_id"),
            namespace=body.get("namespace"),
        )
        self._send_json(result)

    def _handle_compact(self, body):
        """POST /compact {max_age_days, min_access, tier, namespace, dry_run}"""
        result = self.store.compact(
            max_age_days=body.get("max_age_days", 90),
            min_access=body.get("min_access", 0),
            tier=body.get("tier"),
            namespace=body.get("namespace"),
            dry_run=body.get("dry_run", False),
        )
        self._send_json(result)

    def _handle_consolidate(self, body):
        """POST /consolidate {similarity_threshold, namespace, tier, dry_run}"""
        result = self.store.consolidate(
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

        result = self.store.update_memory(
            old_id=_safe_int(old_id, "old_id"),
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

        result = self.store.unarchive(memory_id=_safe_int(memory_id, "memory_id"))
        self._send_json(result)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server(port=8422, db_path="memory.db", backend="auto", host="127.0.0.1",
                  cors_origin="*"):
    """
    Create and return a ThreadingHTTPServer instance (without starting it).

    Each request handler thread gets its own SQLite connection via
    threading.local(), so the server can handle concurrent requests
    without "SQLite objects created in a different thread" errors.

    Args:
        port: TCP port to listen on.
        db_path: Path to SQLite database file.
        backend: Embedding backend ("auto", "model2vec", "hash", "null").
        host: Bind address (default: "127.0.0.1"). Use "0.0.0.0" for external access.
        cors_origin: Access-Control-Allow-Origin header value (default: "*").

    Returns:
        ThreadingHTTPServer instance. Call .serve_forever() to start.
    """
    # Configure handler class with connection parameters
    handler_class = type(
        "ConfiguredMemoryHandler",
        (MemoryHandler,),
        {
            "db_path": db_path,
            "backend": backend,
            "embed_dim": 256,
            "cors_origin": cors_origin,
        },
    )
    server = ThreadingHTTPServer((host, port), handler_class)
    server.daemon_threads = True  # Don't block shutdown on worker threads
    return server


def run_http(port=8422, db_path="memory.db", backend="auto", host="127.0.0.1",
             cors_origin="*"):
    """
    Start HTTP server and block forever.

    Handles SIGINT and SIGTERM for graceful shutdown.

    Args:
        port: TCP port to listen on.
        db_path: Path to SQLite database file.
        backend: Embedding backend ("auto", "model2vec", "hash", "null").
        host: Bind address (default: "127.0.0.1"). Use "0.0.0.0" for external access.
        cors_origin: Access-Control-Allow-Origin header value (default: "*").
    """
    # Eagerly init one store for the startup message (same thread)
    from .core import MemoryStore
    from .embeddings import get_embedding_model

    embed_model = get_embedding_model(backend)
    store = MemoryStore(db_path=db_path, embedding_dim=embed_model.dim or 256)
    if embed_model.dim > 0:
        store.set_embed_fn(embed_model)

    stats = store.stats()
    store.close()

    print(
        f"agentmem HTTP server listening on http://{host}:{port}",
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
    print(
        f"  Threading: enabled (concurrent requests supported)",
        file=sys.stderr,
    )

    server = create_server(port=port, db_path=db_path, backend=backend, host=host,
                           cors_origin=cors_origin)

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(signum, frame):
        signame = signal.Signals(signum).name
        print(f"\nReceived {signame}, shutting down...", file=sys.stderr)
        # shutdown() must be called from a non-serving thread
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        print("Server stopped.", file=sys.stderr)


def main():
    """CLI entry point for agentmem-http."""
    parser = argparse.ArgumentParser(
        prog="agentmem-http",
        description="agentmem HTTP REST API server (threaded)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8422,
        help="Port to listen on (default: 8422)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 for external access.",
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
    parser.add_argument(
        "--cors-origin",
        default="*",
        help="Access-Control-Allow-Origin header value (default: *)",
    )
    args = parser.parse_args()
    run_http(port=args.port, db_path=args.db, backend=args.backend, host=args.host,
             cors_origin=args.cors_origin)


if __name__ == "__main__":
    main()
