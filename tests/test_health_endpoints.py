import os
from typing import Dict


def test_healthz_structure():
    # Import lazily to avoid triggering FastAPI lifespan startup logic
    from src import server

    data: Dict = server.healthz()  # Call the function directly
    assert isinstance(data, dict)
    # Required keys present
    for key in ("ok", "namespace", "index_present", "index_digest", "lexical_weight", "chunk_strategy"):
        assert key in data, f"healthz missing key: {key}"


def test_readyz_structure():
    from src import server

    data: Dict = server.readyz()  # Call the function directly
    assert isinstance(data, dict)
    for key in ("ok", "namespace", "index_digest"):
        assert key in data, f"readyz missing key: {key}"

