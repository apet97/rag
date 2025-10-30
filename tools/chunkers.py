#!/usr/bin/env python3
"""
Chunkers for v2 ingestion ablation.

Strategies:
  - url_level: one chunk per URL using weighted surrogate text
  - h2_h3_blocks: fetch HTML, split by H2/H3, then pack to CHUNK_SIZE with overlap
"""
import os
import re
from typing import List, Dict, Tuple

from urllib.parse import urlparse

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _pack(text: str, target: int, overlap: int) -> List[str]:
    if _token_count(text) <= target:
        return [text]
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(len(words), i + target)
        chunk_words = words[i:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        # compute overlap start
        i = max(0, end - overlap)
    # Dedup last if identical
    if len(chunks) > 1 and chunks[-1] == chunks[-2]:
        chunks.pop()
    return chunks


def build_surrogate_text(title: str, h1: str, url: str) -> str:
    try:
        path = urlparse(url).path.replace("/", " ")
    except Exception:
        path = ""
    parts = []
    if title:
        parts.append((title + " ") * 3)
    if h1:
        parts.append((h1 + " ") * 2)
    if path:
        parts.append(path)
    parts.append(url)
    return " ".join(parts).strip()


def chunk_url_level(text: str) -> List[str]:
    return [text]


def _fetch_html(url: str, timeout: float = 6.0) -> str:
    from urllib.request import Request, urlopen
    try:
        req = Request(url, headers={'User-Agent': 'Codex-Ingest/1.0'})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
            return data.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def _extract_sections_h23(html: str) -> List[Tuple[str, str]]:
    """
    Return list of (header_text, section_text) for H2/H3 delimited blocks.
    Fallback to full body text if no headers present.
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return [("", _clean_text(html))]
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.body or soup
    # Gather blocks split by h2/h3
    sections = []
    current_header = None
    current_text_parts: List[str] = []
    for tag in body.descendants:
        if getattr(tag, 'name', None) in ('h2', 'h3'):
            # flush previous
            if current_text_parts:
                sections.append((current_header or "", _clean_text(" ".join(current_text_parts))))
                current_text_parts = []
            current_header = tag.get_text(separator=' ', strip=True)
        elif getattr(tag, 'name', None) in ('script', 'style', 'noscript'):
            continue
        elif isinstance(tag, str):
            text = tag.strip()
            if text:
                current_text_parts.append(text)
    if current_text_parts:
        sections.append((current_header or "", _clean_text(" ".join(current_text_parts))))
    if not sections:
        sections = [("", _clean_text(soup.get_text(separator=' ', strip=True)))]
    # Remove extremely short sections
    sections = [(h, t) for h, t in sections if _token_count(t) >= 30]
    return sections


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_h2_h3_blocks(url: str, title: str) -> List[Dict[str, str]]:
    html = _fetch_html(url)
    if not html:
        return [{"section": title or "", "text": build_surrogate_text(title, "", url)}]
    sections = _extract_sections_h23(html)
    chunks: List[Dict[str, str]] = []
    for section_title, section_text in sections:
        for piece in _pack(section_text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"section": section_title or title or "", "text": piece})
    # Fallback
    if not chunks:
        chunks = [{"section": title or "", "text": build_surrogate_text(title, "", url)}]
    return chunks

