"""
Tests for agentmem.http_server — concurrent ThreadingHTTPServer.

Covers:
    - Health endpoint
    - Remember + recall roundtrip
    - CORS headers
    - Concurrent requests (multiple threads hitting the server simultaneously)
    - 404 for unknown path
    - 400 for missing required params
    - Content-Length header presence
    - OPTIONS preflight
"""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error

import pytest

# Ensure agentmem is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agentmem.http_server import create_server, MemoryHandler


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture()
def server_url(tmp_path):
    """
    Start a threaded agentmem HTTP server on a random port,
    yield the base URL, then shut it down.
    """
    db_path = str(tmp_path / "test_http.db")
    # Use hash backend for fast, deterministic tests (no model download)
    server = create_server(port=0, db_path=db_path, backend="hash")
    # port=0 lets the OS pick a free port
    host, port = server.server_address
    base_url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Wait for the server to be ready
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=1)
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.05)
    else:
        pytest.fail("Server did not start within 2.5 seconds")

    yield base_url

    server.shutdown()
    server.server_close()


def _get(url, timeout=5):
    """Helper: GET request, return (status, headers, parsed JSON body)."""
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = json.loads(resp.read().decode("utf-8"))
        return resp.status, resp.headers, body
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
        return e.code, e.headers, body


def _post(url, data=None, timeout=5):
    """Helper: POST JSON, return (status, headers, parsed JSON body)."""
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = json.loads(resp.read().decode("utf-8"))
        return resp.status, resp.headers, body
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
        return e.code, e.headers, body


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------


class TestHealth:
    """GET /health endpoint."""

    def test_health_returns_ok(self, server_url):
        status, headers, body = _get(f"{server_url}/health")
        assert status == 200
        assert body["status"] == "ok"
        assert "memories" in body
        assert "db_size" in body

    def test_health_has_content_length(self, server_url):
        status, headers, body = _get(f"{server_url}/health")
        cl = headers.get("Content-Length")
        assert cl is not None
        assert int(cl) > 0

    def test_health_json_content_type(self, server_url):
        status, headers, body = _get(f"{server_url}/health")
        ct = headers.get("Content-Type", "")
        assert "application/json" in ct


class TestRememberRecall:
    """POST /remember + GET /recall roundtrip."""

    def test_remember_and_recall(self, server_url):
        # Store a memory
        status, _, body = _post(
            f"{server_url}/remember",
            {"content": "The capital of France is Paris", "tier": "core"},
        )
        assert status == 201
        assert "id" in body

        # Recall it
        status, _, results = _get(
            f"{server_url}/recall?query=capital+of+France&limit=3"
        )
        assert status == 200
        assert isinstance(results, list)
        assert len(results) >= 1
        # The stored memory should appear in results
        contents = [r.get("content", "") for r in results]
        assert any("Paris" in c for c in contents)

    def test_remember_returns_201(self, server_url):
        status, _, body = _post(
            f"{server_url}/remember",
            {"content": "Test memory for status code check"},
        )
        assert status == 201

    def test_recall_with_no_results(self, server_url):
        # Empty DB, query for something specific
        status, _, results = _get(
            f"{server_url}/recall?query=xyzzynonexistent42&limit=1"
        )
        assert status == 200
        assert isinstance(results, list)


class TestCORS:
    """CORS headers on responses."""

    def test_cors_on_get(self, server_url):
        status, headers, _ = _get(f"{server_url}/health")
        assert headers.get("Access-Control-Allow-Origin") == "*"
        assert "GET" in headers.get("Access-Control-Allow-Methods", "")
        assert "POST" in headers.get("Access-Control-Allow-Methods", "")

    def test_cors_on_post(self, server_url):
        status, headers, _ = _post(
            f"{server_url}/remember",
            {"content": "CORS test memory"},
        )
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_options_preflight(self, server_url):
        """OPTIONS request for CORS preflight."""
        req = urllib.request.Request(
            f"{server_url}/remember",
            method="OPTIONS",
        )
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert resp.headers.get("Content-Length") == "0"


class TestConcurrent:
    """Concurrent requests — multiple threads hitting the server."""

    def test_concurrent_health_checks(self, server_url):
        """3 threads requesting /health simultaneously."""
        results = [None, None, None]
        errors = []

        def _worker(idx):
            try:
                status, _, body = _get(f"{server_url}/health")
                results[idx] = (status, body)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent errors: {errors}"
        for i, r in enumerate(results):
            assert r is not None, f"Thread {i} got no result"
            assert r[0] == 200
            assert r[1]["status"] == "ok"

    def test_concurrent_remember_and_recall(self, server_url):
        """3 threads storing memories, then main thread recalls all."""
        memories = [
            "Python was created by Guido van Rossum",
            "Rust was created by Graydon Hoare",
            "Go was created by Robert Griesemer",
        ]
        errors = []

        def _store(content):
            try:
                status, _, body = _post(
                    f"{server_url}/remember",
                    {"content": content, "tier": "learned"},
                )
                assert status == 201
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_store, args=(m,)) for m in memories]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent store errors: {errors}"

        # Now recall — all 3 should be findable
        status, _, results = _get(
            f"{server_url}/recall?query=programming+language+creator&limit=5&current_only=false"
        )
        assert status == 200
        assert len(results) >= 2  # at least 2 of 3 should match

    def test_concurrent_mixed_operations(self, server_url):
        """Concurrent reads and writes don't crash."""
        errors = []

        def _writer():
            try:
                for i in range(5):
                    _post(
                        f"{server_url}/remember",
                        {"content": f"Concurrent write test #{i}"},
                    )
            except Exception as e:
                errors.append(e)

        def _reader():
            try:
                for _ in range(5):
                    _get(f"{server_url}/health")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_writer),
            threading.Thread(target=_reader),
            threading.Thread(target=_reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Concurrent mixed errors: {errors}"


class TestErrorHandling:
    """Error responses for bad requests."""

    def test_404_unknown_path(self, server_url):
        status, _, body = _get(f"{server_url}/nonexistent")
        assert status == 404
        assert "error" in body
        assert "Not found" in body["error"]

    def test_404_unknown_post(self, server_url):
        status, _, body = _post(f"{server_url}/nonexistent", {})
        assert status == 404
        assert "error" in body

    def test_400_recall_missing_query(self, server_url):
        status, _, body = _get(f"{server_url}/recall")
        assert status == 400
        assert "error" in body
        assert "query" in body["error"].lower()

    def test_400_remember_missing_content(self, server_url):
        status, _, body = _post(f"{server_url}/remember", {})
        assert status == 400
        assert "error" in body
        assert "content" in body["error"].lower()

    def test_400_forget_missing_memory_id(self, server_url):
        status, _, body = _post(f"{server_url}/forget", {})
        assert status == 400
        assert "error" in body
        assert "memory_id" in body["error"].lower()

    def test_400_save_state_missing_state(self, server_url):
        status, _, body = _post(f"{server_url}/save_state", {})
        assert status == 400
        assert "error" in body
        assert "state" in body["error"].lower()

    def test_400_history_missing_memory_id(self, server_url):
        status, _, body = _get(f"{server_url}/history")
        assert status == 400
        assert "error" in body
        assert "memory_id" in body["error"].lower()

    def test_400_related_missing_entity(self, server_url):
        status, _, body = _get(f"{server_url}/related")
        assert status == 400
        assert "error" in body
        assert "entity" in body["error"].lower()


class TestContentLength:
    """HTTP/1.1 compliance: Content-Length on all responses."""

    def test_content_length_on_success(self, server_url):
        _, headers, _ = _get(f"{server_url}/health")
        assert headers.get("Content-Length") is not None

    def test_content_length_on_error(self, server_url):
        _, headers, _ = _get(f"{server_url}/nonexistent")
        assert headers.get("Content-Length") is not None

    def test_content_length_on_post(self, server_url):
        _, headers, _ = _post(
            f"{server_url}/remember",
            {"content": "Content-Length test"},
        )
        assert headers.get("Content-Length") is not None

    def test_content_length_matches_body(self, server_url):
        """Content-Length matches actual body size."""
        req = urllib.request.Request(f"{server_url}/health")
        resp = urllib.request.urlopen(req, timeout=5)
        raw_body = resp.read()
        claimed_length = int(resp.headers.get("Content-Length", 0))
        assert claimed_length == len(raw_body)


class TestInvalidParams:
    """BUG 6: Invalid int/float params should return 400, not 500."""

    def test_invalid_limit_in_recall(self, server_url):
        status, _, body = _get(f"{server_url}/recall?query=test&limit=notanumber")
        assert status == 400
        assert "error" in body

    def test_invalid_limit_in_related(self, server_url):
        status, _, body = _get(f"{server_url}/related?entity=test&limit=abc")
        assert status == 400
        assert "error" in body

    def test_invalid_limit_in_entities(self, server_url):
        status, _, body = _get(f"{server_url}/entities?limit=xyz")
        assert status == 400
        assert "error" in body

    def test_invalid_memory_id_in_history(self, server_url):
        status, _, body = _get(f"{server_url}/history?memory_id=notint")
        assert status == 400
        assert "error" in body

    def test_invalid_recency_weight_in_recall(self, server_url):
        status, _, body = _get(f"{server_url}/recall?query=test&recency_weight=abc")
        assert status == 400
        assert "error" in body

    def test_invalid_memory_id_in_forget(self, server_url):
        status, _, body = _post(f"{server_url}/forget", {"memory_id": "notint"})
        assert status == 400
        assert "error" in body

    def test_invalid_old_id_in_update(self, server_url):
        status, _, body = _post(
            f"{server_url}/update_memory",
            {"old_id": "notint", "new_content": "test"},
        )
        assert status == 400
        assert "error" in body

    def test_invalid_memory_id_in_unarchive(self, server_url):
        status, _, body = _post(f"{server_url}/unarchive", {"memory_id": "notint"})
        assert status == 400
        assert "error" in body

    def test_valid_int_params_still_work(self, server_url):
        """Ensure valid numeric params still work correctly."""
        # Store something first
        _post(f"{server_url}/remember", {"content": "valid param test"})
        status, _, body = _get(f"{server_url}/recall?query=valid+param&limit=3")
        assert status == 200
