import os
import json
import importlib
from pathlib import Path


def test_allowlist_enforced_drops_and_refills():
    os.environ['ALLOWLIST_PATH'] = 'codex/ALLOWLIST.txt'
    os.environ['DENYLIST_PATH'] = 'codex/DENYLIST.txt'
    from src import server
    importlib.reload(server)
    assert server._is_allowed('https://clockify.me/help/getting-started') is True
    assert server._is_allowed('https://docs.langchain.com/') is False


def test_namespace_is_clockify():
    os.environ['RAG_INDEX_ROOT'] = 'index/faiss'
    os.environ['NAMESPACES'] = 'clockify'
    from src import server
    importlib.reload(server)
    assert server.NAMESPACES == ['clockify']


def test_embedding_dim_single_source_of_truth():
    os.environ['EMBEDDING_DIM'] = '768'
    from src import embeddings
    importlib.reload(embeddings)
    from src import embeddings_async
    importlib.reload(embeddings_async)
    assert embeddings.EMBEDDING_DIM == int(os.getenv('EMBEDDING_DIM', '768'))
    # embeddings_async imports EMBEDDING_DIM from embeddings
    assert hasattr(embeddings_async, 'EMBEDDING_MODEL')


def test_no_embeddings_in_json_responses():
    # Prepare index manager
    os.environ['RAG_INDEX_ROOT'] = 'index/faiss'
    os.environ['NAMESPACES'] = 'clockify_url'
    from src import server
    importlib.reload(server)
    # Initialize index manager
    im = server.IndexManager(Path(os.getenv('RAG_INDEX_ROOT')), ['clockify_url'])
    server.index_manager = im
    server.index_manager.ensure_loaded()
    # Build a query vector
    from src.embeddings import embed_query
    qvec = embed_query('delete account')
    res = server.search_ns('clockify_url', qvec, k=3)
    assert isinstance(res, list)
    assert all('embedding' not in r for r in res)


def test_citation_shape_includes_title_and_url():
    os.environ['RAG_INDEX_ROOT'] = 'index/faiss'
    os.environ['NAMESPACES'] = 'clockify_url'
    from src import server
    importlib.reload(server)
    im = server.IndexManager(Path(os.getenv('RAG_INDEX_ROOT')), ['clockify_url'])
    server.index_manager = im
    server.index_manager.ensure_loaded()
    from src.embeddings import embed_query
    qvec = embed_query('integrate google outlook calendar')
    res = server.search_ns('clockify_url', qvec, k=3)
    assert all('title' in r and 'url' in r for r in res)

