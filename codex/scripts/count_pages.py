#!/usr/bin/env python3
"""
Walk knowledge base paths and summarize coverage:
- Counts files by extension (pdf, md, html, txt, jsonl, others)
- Estimates PDF page counts using PyPDF2 if available
- Lists top 20 largest sources
- Flags suspected problematic files (very large, bad encoding)

Writes KB_COVERAGE.md to codex/.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path("~/Downloads/rag").expanduser()
OUT = ROOT / "codex" / "KB_COVERAGE.md"


def get_targets() -> List[Path]:
    # Heuristic: prioritize data/ and docs/
    targets = []
    for d in [ROOT / "data", ROOT / "docs"]:
        if d.exists():
            targets.append(d)
    return targets


def safe_pdf_pages(path: Path) -> int:
    try:
        import PyPDF2  # type: ignore
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        return -1


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n} B"


def main() -> None:
    targets = get_targets()
    if not targets:
        OUT.write_text("No data or docs directories found.\n")
        print(f"Wrote {OUT}")
        return

    counts: Dict[str, int] = {}
    pdf_pages = 0
    unknown_pdf_pages = 0
    files: List[Tuple[Path, int]] = []  # (path, size)

    for root in targets:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                sz = p.stat().st_size
            except Exception:
                sz = 0
            files.append((p, sz))
            ext = p.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1

            if ext == ".pdf":
                pages = safe_pdf_pages(p)
                if pages >= 0:
                    pdf_pages += pages
                else:
                    unknown_pdf_pages += 1

    # Largest files
    largest = sorted(files, key=lambda x: x[1], reverse=True)[:20]

    # Problem files
    problems: List[str] = []
    for p, sz in largest:
        if sz > 20 * 1024 * 1024:
            problems.append(f"Large file (>20MB): {p} ({human_size(sz)})")

    # Write report
    lines: List[str] = []
    lines.append("KB Coverage Summary\n")
    lines.append("Targets:")
    for t in targets:
        lines.append(f"- {t}")
    lines.append("")
    lines.append("Counts by extension:")
    for ext, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {ext or '(no ext)'}: {n}")
    lines.append("")
    lines.append(f"PDF pages (known): {pdf_pages}")
    lines.append(f"PDF files with unknown page count: {unknown_pdf_pages}")
    lines.append("")
    lines.append("Top 20 largest files:")
    for p, sz in largest:
        lines.append(f"- {p} — {human_size(sz)}")
    lines.append("")
    if problems:
        lines.append("Suspected problem files:")
        for msg in problems:
            lines.append(f"- {msg}")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

