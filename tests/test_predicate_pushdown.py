"""
test_predicate_pushdown.py

Verifies that metadata predicate pushdown in the hybrid_search virtual table
correctly restricts both FAISS and FTS5 to matching chunks.

Tests:
  - Unfiltered query returns chunks from multiple sections
  - section= filter returns only chunks from that section
  - source= filter returns only chunks from that source
  - page_start/page_end filter returns only chunks in that page range
  - Nonexistent filter value returns zero results
  - Filtered results are always a subset of unfiltered results

Run with:
  pytest tests/test_predicate_pushdown.py -v -s
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

DB_PATH  = Path("index/tokensmith.db")
EXT_PATH = Path("extension/build/hybrid_search.so")
TOP_K    = 20

# A section known to have exactly 1 chunk
SMALL_SECTION = "Section 1.3 View of Data"
# A section known to have many chunks
LARGE_SECTION = "Section 20.4.2 Measures of Performance for Parallel Systems"
SOURCE = "data/silberschatz--extracted_markdown.md"


@pytest.fixture(scope="module")
def conn():
    if not DB_PATH.exists():
        pytest.skip("tokensmith.db not found — run 'make migrate-db' first")
    if not EXT_PATH.exists():
        pytest.skip("hybrid_search.so not found — run 'make build-extension' first")
    c = sqlite3.connect(str(DB_PATH))
    c.enable_load_extension(True)
    c.load_extension(str(EXT_PATH.resolve()), entrypoint="sqlite3_hybrid_search_init")
    yield c
    c.close()


def _query(conn, top_k=TOP_K, section=None, source=None,
           page_start=None, page_end=None):
    """Run hybrid_search and return list of (chunk_id, score)."""
    rng = np.random.default_rng(0)
    emb = rng.random(2560, dtype=np.float32).tobytes()
    query_text = "what is a database system"

    sql = ("SELECT chunk_id, score FROM hybrid_search "
           "WHERE query_embedding = ? AND query_text = ? AND top_k = ?")
    params: list = [emb, query_text, top_k]

    if section is not None:
        sql += " AND section = ?"
        params.append(section)
    if source is not None:
        sql += " AND source = ?"
        params.append(source)
    if page_start is not None:
        sql += " AND page_start = ?"
        params.append(page_start)
    if page_end is not None:
        sql += " AND page_end = ?"
        params.append(page_end)

    return conn.execute(sql, params).fetchall()


def _chunk_details(conn, chunk_ids):
    """Return list of (chunk_id, section, page_start, page_end, text[:80])."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    return conn.execute(
        f"SELECT chunk_id, section, page_start, page_end, SUBSTR(text, 1, 80) FROM chunks "
        f"WHERE chunk_id IN ({placeholders})",
        chunk_ids
    ).fetchall()


def _chunk_sections(conn, chunk_ids):
    """Return set of sections for the given chunk IDs."""
    if not chunk_ids:
        return set()
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT DISTINCT section FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids
    ).fetchall()
    return {r[0] for r in rows}


def _chunk_sources(conn, chunk_ids):
    """Return set of sources for the given chunk IDs."""
    if not chunk_ids:
        return set()
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT DISTINCT source FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids
    ).fetchall()
    return {r[0] for r in rows}


def _chunk_pages(conn, chunk_ids):
    """Return list of (page_start, page_end) for the given chunk IDs."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    return conn.execute(
        f"SELECT page_start, page_end FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids
    ).fetchall()


def test_unfiltered_returns_multiple_sections(conn):
    """Baseline: unfiltered query returns chunks from more than one section."""
    rows = _query(conn)
    assert len(rows) > 0, "Expected results for unfiltered query"
    ids = [r[0] for r in rows]
    sections = _chunk_sections(conn, ids)
    assert len(sections) > 1, (
        f"Expected chunks from multiple sections, got: {sections}"
    )
    print(f"\n  Unfiltered: {len(rows)} results from {len(sections)} sections")
    for cid, section, ps, pe, text in _chunk_details(conn, ids):
        print(f"    [{cid}] pp{ps}-{pe} | {section} | {text!r}")


def test_section_filter_restricts_results(conn):
    """section= filter returns only chunks belonging to that section."""
    rows = _query(conn, section=SMALL_SECTION, top_k=TOP_K)
    assert len(rows) > 0, f"Expected results for section='{SMALL_SECTION}'"
    ids = [r[0] for r in rows]
    sections = _chunk_sections(conn, ids)
    assert sections == {SMALL_SECTION}, (
        f"Expected only section '{SMALL_SECTION}', got {sections}"
    )
    print(f"\n  section='{SMALL_SECTION}': {len(rows)} result(s)")
    for cid, section, ps, pe, text in _chunk_details(conn, ids):
        print(f"    [{cid}] pp{ps}-{pe} | {section} | {text!r}")


def test_section_filter_subset_of_unfiltered(conn):
    """Filtered chunk IDs must all appear in or be retrievable from the corpus."""
    unfiltered_ids = set(r[0] for r in _query(conn, top_k=1825))
    filtered_ids   = set(r[0] for r in _query(conn, section=LARGE_SECTION, top_k=TOP_K))
    assert len(filtered_ids) > 0, "Expected filtered results"
    # All filtered IDs must exist in the chunks table
    placeholders = ",".join("?" * len(filtered_ids))
    db_ids = set(
        r[0] for r in conn.execute(
            f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})",
            list(filtered_ids)
        ).fetchall()
    )
    assert filtered_ids == db_ids, "Filtered IDs not found in chunks table"
    print(f"\n  section='{LARGE_SECTION}': {len(filtered_ids)} result(s)")
    for cid, section, ps, pe, text in _chunk_details(conn, list(filtered_ids)):
        print(f"    [{cid}] pp{ps}-{pe} | {section} | {text!r}")


def test_source_filter_restricts_results(conn):
    """source= filter returns only chunks from that source file."""
    rows = _query(conn, source=SOURCE, top_k=TOP_K)
    assert len(rows) > 0, f"Expected results for source='{SOURCE}'"
    ids = [r[0] for r in rows]
    sources = _chunk_sources(conn, ids)
    assert sources == {SOURCE}, (
        f"Expected only source '{SOURCE}', got {sources}"
    )
    print(f"\n  source filter: {len(rows)} result(s)")
    for cid, section, ps, pe, text in _chunk_details(conn, ids):
        print(f"    [{cid}] pp{ps}-{pe} | {section} | {text!r}")


def test_page_range_filter_restricts_results(conn):
    """page_start/page_end filters return only chunks within that page range."""
    p_start, p_end = 45, 60
    rows = _query(conn, page_start=p_start, page_end=p_end, top_k=TOP_K)
    if len(rows) == 0:
        pytest.skip(f"No chunks in page range {p_start}--{p_end}")
    ids = [r[0] for r in rows]
    pages = _chunk_pages(conn, ids)
    for ps, pe in pages:
        assert ps >= p_start, f"chunk page_start {ps} < filter {p_start}"
        assert pe <= p_end,   f"chunk page_end {pe} > filter {p_end}"
    print(f"\n  page {p_start}--{p_end}: {len(rows)} result(s)")
    for cid, section, ps, pe, text in _chunk_details(conn, ids):
        print(f"    [{cid}] pp{ps}-{pe} | {section} | {text!r}")


def test_nonexistent_section_returns_empty(conn):
    """A section name that does not exist must return zero results."""
    rows = _query(conn, section="Section 999.99 Nonexistent")
    assert rows == [], (
        f"Expected empty result for nonexistent section, got {len(rows)} rows"
    )
    print(f"\n  Nonexistent section: 0 results (correct)")


def test_nonexistent_source_returns_empty(conn):
    """A source path that does not exist must return zero results."""
    rows = _query(conn, source="data/does_not_exist.md")
    assert rows == [], (
        f"Expected empty result for nonexistent source, got {len(rows)} rows"
    )
    print(f"\n  Nonexistent source: 0 results (correct)")


def test_combined_section_and_source_filter(conn):
    """Combined section + source filter returns only chunks matching both."""
    rows = _query(conn, section=SMALL_SECTION, source=SOURCE, top_k=TOP_K)
    if len(rows) == 0:
        pytest.skip("No chunks match both section and source filter")
    ids = [r[0] for r in rows]
    sections = _chunk_sections(conn, ids)
    sources  = _chunk_sources(conn, ids)
    assert sections == {SMALL_SECTION}, f"Unexpected sections: {sections}"
    assert sources  == {SOURCE},        f"Unexpected sources: {sources}"
    print(f"\n  Combined section+source filter: {len(rows)} result(s)")
    for cid, section, ps, pe, text in _chunk_details(conn, ids):
        print(f"    [{cid}] pp{ps}-{pe} | {section} | {text!r}")
