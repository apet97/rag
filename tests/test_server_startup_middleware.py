import json
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


def test_startup_and_middleware_health_readyz():
    # Prepare minimal faux index to satisfy startup validation
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "clockify"
        root.mkdir(parents=True, exist_ok=True)
        # Minimal metadata
        (root / "meta.json").write_text(json.dumps({"model": "stub", "dim": 384, "rows": []}), encoding="utf-8")
        # Touch index file (existence only)
        (root / "index.bin").write_bytes(b"")

        os.environ['RAG_INDEX_ROOT'] = str(Path(tmp))
        os.environ['NAMESPACES'] = 'clockify'
        os.environ['EMBEDDINGS_BACKEND'] = 'stub'

        from src import server

        # Monkeypatch IndexManager to avoid loading FAISS
        class DummyIM:
            def __init__(self, *_args, **_kwargs):
                self._indexes = {"clockify": {"index": type("I", (), {"ntotal": 0})(), "dim": 384, "metas": []}}

            def ensure_loaded(self):
                return None

            def get_all_indexes(self):
                return self._indexes

        server.IndexManager = DummyIM  # type: ignore

        with TestClient(server.app) as client:
            r1 = client.get("/healthz")
            assert r1.status_code == 200
            body = r1.json()
            assert body.get("ok") is True
            assert body.get("namespace") in ("clockify", None)

            r2 = client.get("/readyz")
            assert r2.status_code == 200
            assert "ok" in r2.json()

