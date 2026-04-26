"""
test_speed_comparison.py

Compares query latency between:
  - Python FAISS+BM25 hybrid (original backend)
  - SQLite C++ extension (HybridSQLiteRetriever)

Run with:
  pytest tests/test_speed_comparison.py -v -s
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import faiss
import pytest

ARTIFACTS  = Path("index/sections")
PREFIX     = "textbook_index"
EMB_MODEL  = "models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf"
DB_PATH    = Path("index/tokensmith.db")
EXT_PATH   = Path("extension/build/hybrid_search.so")

QUERY   = "What is a B+ tree and how does it differ from a B-tree?"
POOL    = 20
N_RUNS  = 10


@pytest.fixture(scope="module")
def artifacts():
    if not DB_PATH.exists():
        pytest.skip("tokensmith.db not found — run 'make migrate-db' first")
    if not EXT_PATH.exists():
        pytest.skip("hybrid_search.so not found — run 'make build-extension' first")

    faiss_idx = faiss.read_index(str(ARTIFACTS / f"{PREFIX}.faiss"))
    bm25_idx  = pickle.load(open(ARTIFACTS / f"{PREFIX}_bm25.pkl", "rb"))
    chunks    = pickle.load(open(ARTIFACTS / f"{PREFIX}_chunks.pkl", "rb"))
    return faiss_idx, bm25_idx, chunks


def _time_retriever(retriever, chunks, n_runs: int, warmup: int = 2) -> float:
    """Returns average ms/query after warmup runs."""
    for _ in range(warmup):
        retriever.get_scores(QUERY, POOL, chunks)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        retriever.get_scores(QUERY, POOL, chunks)
    return (time.perf_counter() - t0) / n_runs * 1000


def test_speed_comparison(artifacts):
    from src.retriever import FAISSRetriever, BM25Retriever, HybridSQLiteRetriever, _get_embedder

    faiss_idx, bm25_idx, chunks = artifacts

    # Warm up embedder (shared cache — counts for both backends)
    _get_embedder(EMB_MODEL).encode([QUERY])

    faiss_r  = FAISSRetriever(faiss_idx, EMB_MODEL)
    bm25_r   = BM25Retriever(bm25_idx)
    sqlite_r = HybridSQLiteRetriever(
        db_path=str(DB_PATH),
        extension_path=str(EXT_PATH),
        embed_model=EMB_MODEL,
    )

    # Time each backend
    faiss_ms  = _time_retriever(faiss_r,  chunks, N_RUNS)
    bm25_ms   = _time_retriever(bm25_r,   chunks, N_RUNS)
    sqlite_ms = _time_retriever(sqlite_r, chunks, N_RUNS)

    py_total = faiss_ms + bm25_ms
    ratio    = sqlite_ms / py_total

    print(f"\n{'='*52}")
    print(f"  Query Latency Comparison  ({N_RUNS} runs each)")
    print(f"{'='*52}")
    print(f"  FAISS retriever:          {faiss_ms:>7.1f} ms")
    print(f"  BM25  retriever:          {bm25_ms:>7.1f} ms")
    print(f"  Python FAISS+BM25 total:  {py_total:>7.1f} ms")
    print(f"  SQLite C++ extension:     {sqlite_ms:>7.1f} ms")
    print(f"  Ratio (SQLite / Python):  {ratio:>7.2f}x  "
          f"({'FASTER' if ratio < 1 else 'SLOWER'})")
    print(f"{'='*52}\n")

    # SQLite should be within 3× of the Python hybrid after the fix.
    assert sqlite_ms < py_total * 3, (
        f"SQLite backend ({sqlite_ms:.1f} ms) is more than 3× slower than "
        f"Python hybrid ({py_total:.1f} ms) — persistent connection may not be working"
    )
