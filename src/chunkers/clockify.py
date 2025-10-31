"""HTML-aware chunking for Clockify help pages."""
from __future__ import annotations
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString

try:  # Python <3.10 compatibility
    from itertools import pairwise
except ImportError:
    def pairwise(iterable):  # type: ignore
        """Return successive overlapping pairs taken from the input iterable."""
        iterator = iter(iterable)
        try:
            prev = next(iterator)
        except StopIteration:
            return
        for item in iterator:
            yield prev, item
            prev = item


def _clean_heading_text(text: str) -> str:
    """Normalize heading text by trimming decorative glyphs."""
    text = text.strip()
    return text.rstrip('#').strip()


def _element_to_text(el: Tag) -> str:
    """Convert a BeautifulSoup element to readable text preserving lists."""
    if isinstance(el, NavigableString):
        return str(el).strip()

    name = getattr(el, "name", "")
    if name in {"ul", "ol"}:
        parts = []
        for idx, li in enumerate(el.find_all("li", recursive=False), start=1):
            txt = li.get_text(" ", strip=True)
            if not txt:
                continue
            bullet = "-" if name == "ul" else f"{idx}."
            parts.append(f"{bullet} {txt}")
        return "\n".join(parts)
    if name == "table":
        rows = []
        for tr in el.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    if name == "pre":
        return str(el.get_text("\n", strip=True))
    return str(el.get_text(" ", strip=True))


def _breadcrumb_to_str(breadcrumb: str | list | tuple | None) -> str:
    if isinstance(breadcrumb, (list, tuple)):
        return " > ".join([b for b in breadcrumb if b])
    return breadcrumb or ""


def parse_clockify_html(
    html: str,
    url: str,
    title: str,
    breadcrumb: str | list | tuple | None = "",
    updated_at: str | None = None,
) -> list[tuple[dict, dict]]:
    """
    Parse Clockify HTML into semantic chunks based on h2/h3 sections.

    Args:
        html: HTML string
        url: Page URL
        title: Page title
        breadcrumb: Breadcrumb navigation text

    Returns:
        List of (chunk_doc, metadata) tuples
    """
    soup = BeautifulSoup(html, "html.parser")
    content_root = soup.find("article") or soup.find("main") or soup
    breadcrumb_str = _breadcrumb_to_str(breadcrumb)

    # Extract h2/h3 headers
    heads = content_root.select("h2, h3")

    if not heads:
        # Fallback: treat entire page as one chunk
        text = content_root.get_text(" ", strip=True)
        return [(
            {"text": text},
            {
                "url": url,
                "title": title,
                "breadcrumb": breadcrumb_str,
                "section": title,
                "anchor": None,
                "updated_at": updated_at,
            },
        )]

    chunks = []

    # Capture intro block before first heading
    intro_parts = []
    first_head = heads[0]
    for node in content_root.children:
        if node == first_head:
            break
        if isinstance(node, Tag):
            text = _element_to_text(node)
            if text:
                intro_parts.append(text)
    if intro_parts:
        intro_text = "\n\n".join(intro_parts).strip()
        if intro_text:
            chunks.append(
                (
                    {"text": intro_text},
                    {
                        "url": url,
                        "title": title,
                        "breadcrumb": breadcrumb_str,
                        "section": title,
                        "anchor": None,
                        "updated_at": updated_at,
                        "type": "help",
                    },
                )
            )

    # Process each h2/h3 and subsequent content until next h2/h3
    sentinel = soup.new_tag("div")
    for h, nxt in pairwise(heads + [sentinel]):
        block = []
        for el in h.next_siblings:
            if el == nxt:
                break
            if getattr(el, "name", None) in {"script", "style"}:
                continue
            text = _element_to_text(el) if isinstance(el, Tag) else str(el).strip()
            if text:
                block.append(text)

        section_title = _clean_heading_text(h.get_text(" ", strip=True))
        section_text = "\n\n".join([section_title] + block).strip()
        anchor = h.get("id")

        meta = {
            "url": url,
            "title": title,
            "breadcrumb": breadcrumb_str,
            "section": section_title,
            "anchor": anchor,
            "type": "help",
            "updated_at": updated_at,
        }

        chunks.append(({"text": section_text}, meta))

    return chunks
