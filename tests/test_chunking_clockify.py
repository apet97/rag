from src.chunkers.clockify import parse_clockify_html


def test_chunking_h23_splits_sections_and_preserves_lists():
    html = """
    <html><body>
      <article>
        <h2>Getting Started</h2>
        <p>Welcome to Clockify.</p>
        <ul><li>Create workspace</li><li>Add users</li></ul>
        <h3>Time Tracking</h3>
        <p>Track time on tasks.</p>
        <pre>code block</pre>
      </article>
    </body></html>
    """
    out = parse_clockify_html(html, url="https://clockify.me/help/getting-started", title="Getting Started", breadcrumb=["Help", "Getting Started"])
    # Expect multiple chunks due to h2/h3 sections
    assert isinstance(out, list) and len(out) >= 2
    texts = [doc[0]["text"] for doc in out]
    joined = "\n".join(texts)
    # Bullet list preserved with '-' markers from UL
    assert "- Create workspace" in joined
    assert "- Add users" in joined
    # Pre content preserved
    assert "code block" in joined


def test_chunking_no_headings_returns_single_chunk():
    html = "<html><body><article><p>No headings here</p></article></body></html>"
    out = parse_clockify_html(html, url="https://clockify.me/help/plain", title="Plain", breadcrumb=["Help"])
    assert isinstance(out, list) and len(out) == 1
    assert "No headings here" in out[0][0]["text"]

