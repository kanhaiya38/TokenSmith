"""
test_update_latency.py

Benchmarks incremental index updates via the hybrid_search virtual table's
xUpdate (INSERT) against a full database rebuild (delete + re-migrate).

For each batch size in [1, 10, 100, 500]:
  - Insert N synthetic chunks via INSERT INTO hybrid_search(...)
  - Measure wall-clock time per batch
  - Compare against full rebuild latency

Run with:
  pytest tests/test_update_latency.py -v -s
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

DB_PATH       = Path("index/tokensmith.db")
EXT_PATH      = Path("extension/build/hybrid_search.so")
ARTIFACTS_DIR = Path("index/sections")
INDEX_PREFIX  = "textbook_index"

# Embedding dimension — must match the stored FAISS index
EMB_DIM = 2560

BATCH_SIZES = [1, 10, 100, 500]


def _open_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    conn.load_extension(str(EXT_PATH.resolve()), entrypoint="sqlite3_hybrid_search_init")
    return conn


def _ntotal(conn: sqlite3.Connection) -> int:
    """Return the number of chunks currently in the database."""
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def _insert_batch(conn: sqlite3.Connection, n: int, dim: int) -> float:
    """
    Insert n synthetic chunks via the virtual table's xUpdate (INSERT).
    Returns elapsed seconds.
    """
    rng = np.random.default_rng(42)
    embeddings = rng.random((n, dim), dtype=np.float32)

    sql = (
        "INSERT INTO hybrid_search(query_text, query_embedding, source, section, page_start, page_end) "
        "VALUES(?, ?, ?, ?, ?, ?)"
    )

    t0 = time.perf_counter()
    for i in range(n):
        emb_blob = embeddings[i].tobytes()
        text = f"Synthetic chunk {i} for latency benchmark testing purposes."
        conn.execute(sql, (text, emb_blob, "benchmark.pdf", "Appendix B", 999, 999))
    elapsed = time.perf_counter() - t0
    return elapsed


def _rebuild_latency() -> float:
    """
    Measure full rebuild time (delete DB + re-run index_migration).
    Returns elapsed seconds.
    """
    t0 = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable, "-m", "src.index_migration",
            "--artifacts-dir", str(ARTIFACTS_DIR),
            "--index-prefix", INDEX_PREFIX,
            "--db-path", str(DB_PATH),
            "--extension-path", str(EXT_PATH),
        ],
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Migration failed:\n{result.stderr}")
    return elapsed


@pytest.fixture(scope="module", autouse=True)
def restore_db_after_module():
    """Restore the original DB after all update latency tests."""
    backup = DB_PATH.with_suffix(".bak")
    shutil.copy2(DB_PATH, backup)
    yield
    shutil.copy2(backup, DB_PATH)
    backup.unlink(missing_ok=True)


def test_insert_and_search():
    """Insert a chunk via xUpdate and verify it is retrievable via xFilter."""
    if not DB_PATH.exists():
        pytest.skip("tokensmith.db not found")
    if not EXT_PATH.exists():
        pytest.skip("hybrid_search.so not found")

    conn = _open_conn()
    n_before = _ntotal(conn)

    rng = np.random.default_rng(7)
    emb = rng.random(EMB_DIM, dtype=np.float32)
    text = "This is a unique test chunk inserted by test_insert_and_search."

    conn.execute(
        "INSERT INTO hybrid_search(query_text, query_embedding, source, section, page_start, page_end) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        (text, emb.tobytes(), "test.pdf", "Test Section", 1, 1)
    )

    n_after = _ntotal(conn)
    assert n_after == n_before + 1, f"Expected {n_before + 1} chunks, got {n_after}"

    inserted_id = conn.execute("SELECT MAX(chunk_id) FROM chunks").fetchone()[0]

    # Query back using the same embedding — the inserted chunk should appear in top results
    rows = conn.execute(
        "SELECT chunk_id, score FROM hybrid_search "
        "WHERE query_embedding = ? AND query_text = ? AND top_k = ?",
        (emb.tobytes(), text, 5)
    ).fetchall()

    print(f"\n  Inserted chunk_id: {inserted_id}")
    print(f"  Top-5 results after insert:")
    for chunk_id, score in rows:
        row_text = conn.execute(
            "SELECT SUBSTR(text, 1, 80) FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        print(f"    [{chunk_id}] score={score:.4f} | {row_text[0]!r}")

    retrieved_ids = [r[0] for r in rows]
    assert inserted_id in retrieved_ids, (
        f"Inserted chunk (id={inserted_id}) not found in top-5 results: {retrieved_ids}"
    )

    conn.close()


def test_incremental_update_latency():
    """
    Measure per-chunk INSERT latency for varying batch sizes and compare
    against full rebuild.
    """
    if not DB_PATH.exists():
        pytest.skip("tokensmith.db not found — run 'make migrate-db' first")
    if not EXT_PATH.exists():
        pytest.skip("hybrid_search.so not found — run 'make build-extension' first")

    print(f"\n{'='*64}")
    print(f"  Incremental Insert vs. Full Rebuild Latency")
    print(f"{'='*64}")
    print(f"  {'Batch':>6}  {'Total (s)':>10}  {'Per-chunk (ms)':>15}  {'vs. rebuild':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*15}  {'-'*12}")

    rebuild_time = _rebuild_latency()

    results: list[dict] = []
    for batch_size in BATCH_SIZES:
        conn = _open_conn()
        n_before = _ntotal(conn)

        elapsed = _insert_batch(conn, batch_size, EMB_DIM)
        n_after = _ntotal(conn)
        conn.close()

        assert n_after == n_before + batch_size, (
            f"Expected {n_before + batch_size} chunks, got {n_after}"
        )

        per_chunk_ms = (elapsed / batch_size) * 1000
        speedup = rebuild_time / elapsed

        print(f"  {batch_size:>6}  {elapsed:>10.3f}  {per_chunk_ms:>15.2f}  {speedup:>11.1f}x")

        results.append({
            "batch_size": batch_size,
            "total_s": elapsed,
            "per_chunk_ms": per_chunk_ms,
            "speedup_vs_rebuild": speedup,
        })

        # Restore DB for next batch iteration
        backup = DB_PATH.with_suffix(".bak")
        shutil.copy2(backup, DB_PATH)

    print(f"  {'-'*6}  {'-'*10}  {'-'*15}  {'-'*12}")
    print(f"  {'rebuild':>6}  {rebuild_time:>10.3f}  {'—':>15}  {'1.0x':>12}")
    print(f"{'='*64}")
    print(f"\n  Note: per-row FAISS re-serialization dominates for large batches.")
    print(f"  Each INSERT re-writes the full FAISS BLOB (~18 MB).")
    print(f"  Small batches (N=1) are faster than rebuild; large batches are not.")
    print(f"  Batch xBegin/xCommit could defer serialization to commit time.\n")

    # Correctness: chunk counts must match; small single-insert must beat rebuild
    for r in results:
        if r["batch_size"] == 1:
            assert r["total_s"] < rebuild_time, (
                f"Single-chunk insert ({r['total_s']:.3f}s) must be faster "
                f"than full rebuild ({rebuild_time:.3f}s)"
            )
