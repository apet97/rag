import importlib
import os


def test_allowlist_refill_replaces_denied_and_keeps_size():
    os.environ['ALLOWLIST_PATH'] = 'codex/ALLOWLIST.txt'
    os.environ['DENYLIST_PATH'] = 'codex/DENYLIST.txt'
    from src import server
    importlib.reload(server)

    # hits contains one allowed and one denied
    hits = [
        {"url": "https://clockify.me/help/getting-started"},
        {"url": "https://docs.langchain.com/en/latest/"},  # denied
    ]
    # candidates include an extra allowed source that should be used to refill
    candidates = [
        {"url": "https://clockify.me/help/integrations/google-calendar"},
        {"url": "https://clockify.me/help/projects"},
    ]

    out = server._filter_and_refill_for_test(hits, candidates, max_context=2)
    assert len(out) == 2
    assert all(server._is_allowed(x.get('url', '')) for x in out)


def test_allowlist_refill_insufficient_allowed_sources():
    os.environ['ALLOWLIST_PATH'] = 'codex/ALLOWLIST.txt'
    os.environ['DENYLIST_PATH'] = 'codex/DENYLIST.txt'
    from src import server
    importlib.reload(server)

    hits = [
        {"url": "https://docs.langchain.com/bad"},
    ]
    candidates = [
        {"url": "https://docs.langchain.com/also-bad"},
    ]

    out = server._filter_and_refill_for_test(hits, candidates, max_context=2)
    # Nothing allowed, expect 0 results
    assert len(out) == 0

