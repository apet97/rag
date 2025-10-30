#!/usr/bin/env python3
"""End-to-end ingestion orchestrator for Clockify help content."""

import asyncio
import os
from pathlib import Path

from loguru import logger

from src import process_scraped_pages, chunk, embed


def ingest_disabled() -> bool:
    """Check if ingestion should be skipped (CI/testing mode).

    When RAG_SKIP_INGEST=1, the ingestion pipeline is skipped entirely.
    This allows CI tests to run with lightweight fixtures instead of rebuilding indexes.
    """
    return os.getenv("RAG_SKIP_INGEST", "0") == "1"


async def run_ingestion() -> None:
    # Skip ingestion if RAG_SKIP_INGEST flag is set (CI testing mode)
    if ingest_disabled():
        logger.info("⏭️  RAG_SKIP_INGEST=1: Skipping ingestion pipeline (using pre-built indexes)")
        return

    logger.info("Step 1/3: Processing scraped HTML into clean markdown...")
    processed_count = await process_scraped_pages.main()
    logger.info(f"Processed {processed_count} articles into data/clean.")

    logger.info("Step 2/3: Building hierarchical chunks...")
    await chunk.main()
    chunk_path = Path("data/chunks/clockify.jsonl")
    if chunk_path.exists():
        logger.info(f"Clockify chunk corpus ready: {chunk_path}")

    logger.info("Priming reranker weights (BAAI/bge-reranker-base)...")
    try:
        from FlagEmbedding import FlagReranker  # type: ignore

        FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
        logger.info("✓ Reranker cache prepared")
    except Exception as exc:  # pragma: no cover - informational
        logger.warning(f"Skipping reranker warmup: {exc}")

    logger.info("Step 3/3: Encoding embeddings and writing FAISS indexes...")
    await embed.main()
    logger.info("Ingestion pipeline finished.")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
