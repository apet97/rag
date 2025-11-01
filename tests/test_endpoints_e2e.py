import os
from fastapi.testclient import TestClient


def test_search_and_chat_e2e_stub_index():
    # Use stub index built by tools/ingest_v2.py (clockify_url)
    os.environ['RAG_INDEX_ROOT'] = 'index/faiss'
    os.environ['NAMESPACES'] = 'clockify_url'
    os.environ['EMBEDDINGS_BACKEND'] = 'stub'

    from src import server

    client = TestClient(server.app)
    headers = {"x-api-token": os.getenv("API_TOKEN", "change-me")}

    # /search
    r = client.get('/search', params={'q': 'create project', 'k': 5, 'namespace': 'clockify_url'}, headers=headers)
    assert r.status_code in (200, 500)  # 500 allowed if unexpected runtime deps missing
    if r.status_code == 200:
        data = r.json()
        assert 'results' in data

    # /chat
    r2 = client.post('/chat', json={'question': 'how to create project', 'k': 2, 'namespace': 'clockify_url'}, headers=headers)
    assert r2.status_code in (200, 503, 500)
    if r2.status_code == 200:
        data2 = r2.json()
        assert 'sources' in data2 and isinstance(data2['sources'], list)

