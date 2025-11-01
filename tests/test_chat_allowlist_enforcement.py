import os
import importlib
from types import SimpleNamespace


def test_chat_drops_denied_and_refills_with_allowed(monkeypatch):
    # Configure offline-safe env
    os.environ['MOCK_LLM'] = 'true'
    os.environ['EMBEDDINGS_BACKEND'] = 'stub'
    os.environ['ALLOWLIST_PATH'] = 'codex/ALLOWLIST.txt'
    os.environ['DENYLIST_PATH'] = 'codex/DENYLIST.txt'

    from src import server
    importlib.reload(server)

    # Dummy index manager to satisfy namespace derivation
    class DummyIM:
        def ensure_loaded(self):
            return None

        def get_all_indexes(self):
            return {'clockify': {}}

    server.index_manager = DummyIM()

    # Candidates include a denied domain and two allowed ones
    candidates = [
        {
            'url': 'https://docs.langchain.com/en/latest/',  # denied
            'title': 'LangChain Docs',
            'text': 'not allowed',
            'score': 0.95,
            'chunk_id': 'x1',
            'namespace': 'clockify',
        },
        {
            'url': 'https://clockify.me/help/projects',
            'title': 'Projects',
            'text': 'allowed',
            'score': 0.90,
            'chunk_id': 'x2',
            'namespace': 'clockify',
        },
        {
            'url': 'https://clockify.me/help/integrations/google-calendar',
            'title': 'Google Calendar',
            'text': 'allowed',
            'score': 0.85,
            'chunk_id': 'x3',
            'namespace': 'clockify',
        },
    ]

    # Patch both hybrid and vector search paths to return our candidates
    monkeypatch.setattr(server, 'search_ns_hybrid', lambda *args, **kwargs: candidates)
    monkeypatch.setattr(server, 'search_ns', lambda *args, **kwargs: candidates)

    # Build request objects
    req = server.ChatRequest(question='how to create project', k=2, namespace='clockify')
    dummy_request = SimpleNamespace(client=SimpleNamespace(host='127.0.0.1'))

    # Call the endpoint function directly (bypass FastAPI routing)
    resp = server.chat(req, dummy_request, decomposition_off=True, x_api_token='change-me')

    # Response should include only allowlisted sources and maintain size k
    assert resp.success is True
    assert len(resp.sources) == 2
    urls = [s.get('url', '') for s in resp.sources]
    assert all(url.startswith('https://clockify.me/') for url in urls)
    assert not any('docs.langchain.com' in url for url in urls)

