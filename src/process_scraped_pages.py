#!/usr/bin/env python3
"""
Process scraped Clockify help pages: convert HTML to clean markdown, deduplicate, and validate.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/clockify")
CLEAN_DIR = Path("data/clean/clockify")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _clean_heading_text(text: str) -> str:
    """Normalize heading text by stripping decorative characters."""
    text = text.strip()
    return re.sub(r"#+$", "", text).strip()


def _infer_breadcrumb_from_url(url: str, title: str) -> List[str]:
    """Infer breadcrumb hierarchy from URL path structure.

    Examples:
    - https://clockify.me/help -> ["Clockify Help Center"]
    - https://clockify.me/help/administration -> ["Clockify Help Center", "Administration"]
    - https://clockify.me/help/track-time-and-expenses/kiosk -> ["Clockify Help Center", "Track Time And Expenses", "Kiosk"]
    - https://clockify.me/help/administration/user-roles-and-permissions/who-can-do-what
      -> ["Clockify Help Center", "Administration", "User Roles And Permissions"]
    """
    if not url.startswith("https://clockify.me/help"):
        return ["Clockify Help Center", title]

    # Extract path components after /help
    path_part = url.replace("https://clockify.me/help", "").strip("/")
    if not path_part:
        return ["Clockify Help Center"]

    # Split into components and convert to title case
    components = path_part.split("/")

    # Build breadcrumb: root + category + maybe subcategory
    breadcrumb = ["Clockify Help Center"]

    # Add main category (first component)
    if components and components[0]:
        category = components[0].replace("-", " ").title()
        breadcrumb.append(category)

    # For deeper paths, add second component if it looks like a meaningful subcategory
    # (not a single-word modifier or ID)
    if len(components) > 1 and components[1]:
        second = components[1].replace("-", " ").title()
        # Only add if it's not a single word and doesn't look like an ID/name
        if len(second) > 3 and not second[0].isdigit():
            breadcrumb.append(second)

    return breadcrumb


def extract_content_from_html(html_content: str, url: str) -> Tuple[str, str, str, List[str], str, List[Dict[str, str]]]:
    """Extract title, description, body, breadcrumbs, timestamps, and section metadata."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract title
    title = "Clockify Help"
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
    else:
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text(strip=True)

    # Extract description/meta
    description = ""
    if soup.find("meta", attrs={"name": "description"}):
        description = soup.find("meta", attrs={"name": "description"}).get("content", "")

    # Extract breadcrumbs if available
    breadcrumb_items: List[str] = []
    breadcrumb_container = soup.select_one("div.breadcrumb")
    if breadcrumb_container:
        for node in breadcrumb_container.find_all("a"):
            text = node.get_text(strip=True)
            if text and text not in breadcrumb_items:
                breadcrumb_items.append(text)
        current = breadcrumb_container.select_one("span.breadcrumb--current-page")
        if current:
            current_text = current.get_text(strip=True)
            if current_text and current_text not in breadcrumb_items:
                breadcrumb_items.append(current_text)

    # If no breadcrumbs found in HTML, infer from URL structure
    if not breadcrumb_items:
        breadcrumb_items = _infer_breadcrumb_from_url(url, title)

    # Extract updated timestamp (prefer modified time)
    updated_at = ""
    meta_updated = soup.find("meta", attrs={"property": "article:modified_time"})
    if meta_updated and meta_updated.get("content"):
        updated_at = meta_updated["content"]
    else:
        meta_published = soup.find("meta", attrs={"property": "article:published_time"})
        if meta_published and meta_published.get("content"):
            updated_at = meta_published["content"]

    # Capture section metadata (heading hierarchy)
    section_meta: List[Dict[str, str]] = []
    for heading in soup.select("h2, h3"):
        title_text = _clean_heading_text(heading.get_text(" ", strip=True))
        if not title_text:
            continue
        section_meta.append(
            {
                "level": heading.name,
                "title": title_text,
                "anchor": heading.get("id", ""),
            }
        )

    # Extract main content
    # Remove script, style, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()

    # Get main content
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main:
        # Remove navigation elements
        for nav_elem in main.find_all(["nav", ".sidebar", ".navigation"]):
            nav_elem.decompose()

        # Get text
        body = main.get_text(separator="\n", strip=True)
    else:
        body = soup.get_text(separator="\n", strip=True)

    # Clean whitespace
    body = re.sub(r"\n\n+", "\n", body).strip()

    return title, description, body, breadcrumb_items, updated_at, section_meta


def process_html_file(html_path: Path) -> dict:
    """Process single HTML file into markdown."""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        html = content.get("html", "")
        meta = content.get("meta", {})
        url = meta.get("url", "")

        # Skip non-English URLs
        if any(lang in url.lower() for lang in ["/help/de", "/help/es", "/help/fr", "/help/pt"]):
            logger.info(f"⊘ Skipping non-English: {url}")
            return None

        # Extract content
        title, description, body, breadcrumb, updated_at, section_meta = extract_content_from_html(html, url)

        # Skip if body is too short or empty
        if len(body) < 100:
            logger.warning(f"⊘ Too short ({len(body)} chars): {url}")
            return None

        raw_abs_path = html_path.resolve()
        try:
            raw_rel_path = raw_abs_path.relative_to(PROJECT_ROOT)
        except ValueError:
            raw_rel_path = raw_abs_path

        frontmatter = {
            "url": url,
            "title": title,
            "description": description,
            "breadcrumb": breadcrumb,
            "updated_at": updated_at,
            "sections": section_meta,
            "raw_html_path": str(raw_rel_path),
        }

        markdown_lines = [
            "---",
            json.dumps(frontmatter, ensure_ascii=False),
            "---",
            "",
            f"# {title}",
            "",
        ]
        if description:
            markdown_lines.append(f"> {description}")
            markdown_lines.append("")
        markdown_lines.append(f"**Source:** {url}")
        markdown_lines.append("")
        markdown_lines.append(body)
        markdown = "\n".join(markdown_lines)

        # Create hash for deduplication
        content_hash = hashlib.sha256(body.encode()).hexdigest()

        return {
            "title": title,
            "url": url,
            "description": description,
            "content_hash": content_hash,
            "body_length": len(body),
            "markdown": markdown,
            "file_path": html_path,
        }

    except Exception as e:
        logger.error(f"✗ Failed to process {html_path}: {e}")
        return None


async def main():
    """Process all scraped HTML files."""
    logger.info(f"Processing HTML files from {RAW_DIR}...")

    html_files = list(RAW_DIR.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files")

    processed = []
    seen_hashes = set()
    duplicates = 0

    for i, html_file in enumerate(sorted(html_files)):
        if i % 20 == 0:
            logger.info(f"Processing {i}/{len(html_files)}...")

        result = process_html_file(html_file)

        if result is None:
            continue

        # Check for duplicates
        if result["content_hash"] in seen_hashes:
            logger.debug(f"⊘ Duplicate content: {result['url']}")
            duplicates += 1
            continue

        seen_hashes.add(result["content_hash"])
        processed.append(result)

    logger.info(f"\n=== PROCESSING RESULTS ===")
    logger.info(f"Total HTML files processed: {len(html_files)}")
    logger.info(f"Valid articles extracted: {len(processed)}")
    logger.info(f"Duplicates removed: {duplicates}")
    logger.info(f"Final unique articles: {len(processed)}")

    # Save as markdown files
    logger.info(f"\nSaving to {CLEAN_DIR}...")
    for result in processed:
        # Create safe filename from title
        safe_title = re.sub(r"[^\w\s-]", "", result["title"]).strip()
        safe_title = re.sub(r"[-\s]+", "-", safe_title).lower()
        filename = f"{safe_title}.md"
        result["_clean_filename"] = filename

        filepath = CLEAN_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result["markdown"])

        logger.debug(f"✓ Saved: {filepath}")

    logger.info(f"\n✓ Saved {len(processed)} clean markdown files to {CLEAN_DIR}")

    # Save metadata
    metadata = {
        "total_original": len(html_files),
        "total_processed": len(processed),
        "duplicates_removed": duplicates,
        "articles": [
            {
                "title": r["title"],
                "url": r["url"],
                "file": r.get("_clean_filename", ""),
                "size": r["body_length"],
                "breadcrumb": r.get("breadcrumb", []),
                "updated_at": r.get("updated_at"),
            }
            for r in processed
        ]
    }

    metadata_file = Path("CLOCKIFY_HELP_INGESTION_METADATA.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved to {metadata_file}")

    return len(processed)


if __name__ == "__main__":
    import asyncio
    count = asyncio.run(main())
    logger.info(f"\nReady to ingest {count} articles!")
