#!/usr/bin/env python3
"""
BFS + sitemap scraper for Clockify help articles.
Crawls every URL under `/help`, excludes non-English variants, respects robots.txt.
Outputs clean Markdown and RAG-ready JSONL.
"""

import argparse
import json
import os
import re
import time
import hashlib
import pathlib
import queue
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import tldextract
import urllib.robotparser as robotparser

BASE = "https://clockify.me"
SEED = "https://clockify.me/help/"
SITEMAP = "https://clockify.me/help/wp-sitemap.xml"
UA = "Aleksandar-RAG-Collector/1.0 (+https://clockify.me/help) requests"


def norm(u: str) -> str:
    """Normalize URL to canonical form."""
    p = urlparse(u)
    # strip fragments and queries for canonical content URLs
    p = p._replace(fragment="", query="")
    # strip trailing slash except for /help/
    path = re.sub(r"//+", "/", p.path)
    if path != "/help/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((p.scheme, p.netloc, path, "", "", ""))


def is_same_site(u: str) -> bool:
    """Check if URL is on same site as BASE."""
    return (
        tldextract.extract(u).registered_domain
        == tldextract.extract(BASE).registered_domain
    )


def is_help_path(u: str) -> bool:
    """Check if URL is under /help path."""
    return urlparse(u).path.startswith("/help/")


def lang_excluded(u: str, banned: set) -> bool:
    """Check if URL contains excluded language code."""
    parts = urlparse(u).path.split("/")
    # ['', 'help', 'maybe-lang', ...]
    if len(parts) >= 3:
        seg = parts[2].lower()
        return seg in banned
    return False


def load_urls_from_wp_sitemaps(session: requests.Session) -> set:
    """Load all URLs from WordPress sitemaps."""
    urls = set()
    try:
        print(f"Fetching WordPress sitemaps from {SITEMAP}...")
        r = session.get(SITEMAP, timeout=30)
        r.raise_for_status()
        sx = BeautifulSoup(r.text, "xml")
        sm_urls = [loc.text for loc in sx.select("sitemap > loc")]
        print(f"Found {len(sm_urls)} sitemap indexes")

        for sm in sm_urls:
            try:
                rx = session.get(sm, timeout=30)
                rx.raise_for_status()
                sx2 = BeautifulSoup(rx.text, "xml")
                for loc in sx2.select("url > loc"):
                    u = norm(loc.text.strip())
                    if is_same_site(u) and is_help_path(u):
                        urls.add(u)
            except Exception as e:
                print(f"Warning loading sitemap {sm}: {e}")

        print(f"Loaded {len(urls)} URLs from sitemaps")
    except Exception as e:
        print(f"Sitemap load warning: {e}")
    return urls


def extract_markdown(html: str) -> tuple[str, str]:
    """Extract title and markdown body from HTML."""
    s = BeautifulSoup(html, "lxml")

    # Remove obvious chrome
    for sel in [
        "header",
        "footer",
        "nav",
        ".site-header",
        ".site-footer",
        ".menu",
        ".breadcrumb",
        ".breadcrumbs",
        ".sidebar",
        ".toc",
        ".table-of-contents",
        ".cookie",
        ".cc-window",
        ".newsletter",
        ".hero",
        ".share",
        ".social",
        ".comments",
        ".wp-block-buttons",
    ]:
        for n in s.select(sel):
            n.decompose()

    title = None
    for h in s.select("h1, .entry-title"):
        if h.get_text(strip=True):
            title = h.get_text(" ", strip=True)
            break
    if not title:
        t = s.title.string if s.title else ""
        title = (t or "").strip()

    # Prefer main/article/content wrappers
    for sel in ["main article", ".entry-content", "article", ".post", ".content", "#content"]:
        n = s.select_one(sel)
        if n:
            body_md = md(str(n), heading_style="ATX")
            return title, body_md

    body_md = md(str(s.body or s), heading_style="ATX")
    return title, body_md


def chunk_for_rag(text: str, max_chars: int = 4000) -> list[tuple[str, str]]:
    """Split text into RAG-ready chunks by heading."""
    # Split by top-level headings first
    blocks = re.split(r"(?m)^(#{1,3}\s.+)$", text)
    # Re-stitch to pairs: (heading, content)
    pairs = []
    i = 0
    while i < len(blocks):
        if blocks[i].startswith("#"):
            head = blocks[i].strip()
            content = (blocks[i + 1] if i + 1 < len(blocks) else "").strip()
            pairs.append((head, content))
            i += 2
        else:
            if blocks[i].strip():
                pairs.append(("", blocks[i].strip()))
            i += 1

    # Further chunk long contents
    out = []
    for head, content in pairs:
        if len(content) <= max_chars:
            out.append((head, content))
        else:
            # split on paragraphs
            paras = [p for p in re.split(r"\n{2,}", content) if p.strip()]
            cur = ""
            for p in paras:
                if len(cur) + len(p) + 2 <= max_chars:
                    cur += (("\n\n" + p) if cur else p)
                else:
                    out.append((head, cur))
                    cur = p
            if cur:
                out.append((head, cur))
    return out


def safe_filename(url: str) -> str:
    """Convert URL to safe filename."""
    p = urlparse(url).path.strip("/").replace("/", "__")
    if not p:
        p = "help__index"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", p)


def main():
    ap = argparse.ArgumentParser(
        description="BFS + sitemap scraper for Clockify help articles"
    )
    ap.add_argument("--out", default="clockify-help-dump", help="Output directory")
    ap.add_argument("--delay", type=float, default=0.6, help="Seconds between requests")
    ap.add_argument("--max-pages", type=int, default=5000, help="Maximum pages to crawl")
    ap.add_argument(
        "--exclude",
        default="es,pt,de",
        help="Comma list of language subpaths to exclude",
    )
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    (outdir / "pages").mkdir(parents=True, exist_ok=True)
    rag_fp = open(outdir / "clockify_help.jsonl", "w", encoding="utf-8")

    banned = {x.strip().lower() for x in args.exclude.split(",") if x.strip()}
    print(f"Excluded languages: {sorted(banned)}")

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    # robots.txt compliance
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(urljoin(BASE, "/robots.txt"))
        rp.read()
        print("Loaded robots.txt")
    except Exception as e:
        print(f"Warning: Could not load robots.txt: {e}")

    # Seed set from sitemap
    seed_urls = load_urls_from_wp_sitemaps(session)
    if not seed_urls:
        print("No sitemap URLs found, using /help/ as seed")
        seed_urls = {norm(SEED)}

    q = queue.Queue()
    for u in sorted(seed_urls):
        q.put(u)

    seen = set()
    discovered = set(seed_urls)
    visited_count = 0

    print(f"\nStarting BFS crawl with {len(seed_urls)} seed URLs...")
    print(f"Max pages: {args.max_pages}, Delay: {args.delay}s\n")

    while not q.empty() and visited_count < args.max_pages:
        u = q.get()
        if u in seen:
            continue
        if not is_same_site(u) or not is_help_path(u):
            continue
        if lang_excluded(u, banned):
            continue
        if rp and hasattr(rp, "can_fetch") and not rp.can_fetch(UA, u):
            print(f"✗ robots.txt disallows: {u}")
            seen.add(u)
            continue

        try:
            r = session.get(u, timeout=30)
            r.raise_for_status()
            html = r.text
        except Exception as e:
            print(f"✗ Skip fetch: {u} ({e})")
            seen.add(u)
            continue

        title, body_md = extract_markdown(html)
        fname = safe_filename(u) + ".md"
        with open(outdir / "pages" / fname, "w", encoding="utf-8") as f:
            f.write(f"# {title or 'Clockify Help'}\n\n")
            f.write(f"> URL: {u}\n\n")
            f.write(body_md)

        # JSONL chunks for RAG
        chunks = chunk_for_rag(body_md)
        for head, chunk in chunks:
            doc_id = hashlib.sha1(
                (u + "\n" + head + "\n" + chunk[:64]).encode("utf-8")
            ).hexdigest()[:16]
            rec = {
                "id": doc_id,
                "url": u,
                "title": title,
                "section": head.lstrip("# ").strip() if head else "",
                "content": chunk,
                "lang": "en",
            }
            rag_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        visited_count += 1
        print(f"[{visited_count:3d}] ✓ {u} ({len(chunks)} chunks) -> {fname}")
        seen.add(u)

        # enqueue new links
        try:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a[href]"):
                href = a.get("href")
                if not href:
                    continue
                nu = norm(urljoin(u, href))
                if not is_same_site(nu) or not is_help_path(nu):
                    continue
                if lang_excluded(nu, banned):
                    continue
                if nu not in discovered:
                    discovered.add(nu)
                    q.put(nu)
        except Exception:
            pass

        time.sleep(args.delay)

    rag_fp.close()

    # Write manifest
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed_count": len(seed_urls),
                "discovered": len(discovered),
                "visited": visited_count,
                "excluded_langs": sorted(banned),
                "generated_at": int(time.time()),
            },
            f,
            indent=2,
        )

    # Save URL inventory
    with open(outdir / "urls.txt", "w", encoding="utf-8") as f:
        for u in sorted(seen | discovered):
            f.write(u + "\n")

    print(f"\n{'='*60}")
    print("CRAWL COMPLETE")
    print(f"{'='*60}")
    print(f"Seed URLs: {len(seed_urls)}")
    print(f"Total discovered: {len(discovered)}")
    print(f"Total visited: {visited_count}")
    print(f"Output directory: {outdir}")
    print(f"Markdown pages: {outdir}/pages/")
    print(f"RAG JSONL: {outdir}/clockify_help.jsonl")
    print(f"URL inventory: {outdir}/urls.txt")
    print(f"Manifest: {outdir}/manifest.json")


if __name__ == "__main__":
    main()
