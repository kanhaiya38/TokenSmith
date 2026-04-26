"""
index_migration.py

Converts the scattered TokenSmith artifact files into a single SQLite database
(index/tokensmith.db) that the HybridSearch C++ extension can query.

Schema created:
  chunks        -- chunk text + metadata (chunk_id is the row index, 0-based)
  chunks_fts    -- FTS5 virtual table over chunks.text (content table)
  hybrid_search -- hybridsearch virtual table (created last, after data is loaded)

Usage:
  python -m src.index_migration [--artifacts-dir index/sections]
                                [--index-prefix textbook_index]
                                [--db-path index/tokensmith.db]
                                [--extension-path extension/build/hybrid_search.so]
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sqlite3
import sys


# helpers

def _page_range(meta: dict) -> tuple[int | None, int | None]:
    pages = meta.get("page_numbers")
    if not pages:
        return None, None
    return int(min(pages)), int(max(pages))


# main

def migrate(
    artifacts_dir: pathlib.Path,
    index_prefix: str,
    db_path: pathlib.Path,
    extension_path: pathlib.Path | None,
) -> None:
    # load artifacts
    print(f"Loading artifacts from {artifacts_dir} with prefix '{index_prefix}' ...")

    chunks_path  = artifacts_dir / f"{index_prefix}_chunks.pkl"
    meta_path    = artifacts_dir / f"{index_prefix}_meta.pkl"
    sources_path = artifacts_dir / f"{index_prefix}_sources.pkl"

    for p in (chunks_path, meta_path):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    import faiss as _faiss

    faiss_path = artifacts_dir / f"{index_prefix}.faiss"
    if not faiss_path.exists():
        print(f"ERROR: required file not found: {faiss_path}", file=sys.stderr)
        sys.exit(1)

    chunks: list[str]    = pickle.load(open(chunks_path, "rb"))
    metadata: list[dict] = pickle.load(open(meta_path, "rb"))
    sources: list[str]   = (
        pickle.load(open(sources_path, "rb"))
        if sources_path.exists()
        else [""] * len(chunks)
    )
    faiss_index = _faiss.read_index(str(faiss_path))

    n = len(chunks)
    print(f"  {n} chunks loaded.")
    print(f"  FAISS index: {faiss_index.ntotal} vectors, dim={faiss_index.d}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
        print(f"Removed existing database at {db_path}")

    print(f"Creating {db_path} ...")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # chunks table
    conn.execute("""
        CREATE TABLE chunks (
            chunk_id   INTEGER PRIMARY KEY,
            text       TEXT    NOT NULL,
            source     TEXT,
            section    TEXT,
            page_start INTEGER,
            page_end   INTEGER
        )
    """)

    rows = []
    for i, (text, meta, src) in enumerate(zip(chunks, metadata, sources)):
        ps, pe = _page_range(meta)
        rows.append((
            i,
            text,
            src or None,
            meta.get("section") or meta.get("section_path") or None,
            ps,
            pe,
        ))

    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", rows)
    print(f"  Inserted {len(rows)} rows into chunks.")

    # FTS5 virtual table
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='chunk_id'
        )
    """)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    print("  Built FTS5 index (chunks_fts).")

    conn.execute("""
        CREATE TABLE faiss_index (
            id   INTEGER PRIMARY KEY,
            data BLOB    NOT NULL
        )
    """)
    faiss_blob = _faiss.serialize_index(faiss_index).tobytes()
    conn.execute("INSERT INTO faiss_index(id, data) VALUES(1, ?)", (faiss_blob,))
    print(f"  Stored FAISS index BLOB ({len(faiss_blob) / 1024:.1f} KB) in faiss_index.")

    conn.commit()

    # hybrid_search virtual table
    # must be created after chunks_fts is populated
    if extension_path and extension_path.exists():
        print(f"Loading extension from {extension_path} ...")
        conn.enable_load_extension(True)
        conn.load_extension(str(extension_path.resolve()), entrypoint="sqlite3_hybrid_search_init")
        conn.execute("CREATE VIRTUAL TABLE hybrid_search USING hybridsearch()")
        conn.commit()
        print("  Created hybrid_search virtual table.")
    else:
        print(
            "  Skipping hybrid_search virtual table creation "
            "(extension not found -- run 'make build-extension' first)."
        )

    conn.close()
    print(f"\nDone. Database written to {db_path.resolve()}")
    size_mb = db_path.stat().st_size / 1_048_576
    print(f"  tokensmith.db size: {size_mb:.1f} MB")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate TokenSmith FAISS/BM25 artifacts to SQLite."
    )
    parser.add_argument(
        "--artifacts-dir", default="index/sections",
        help="Directory containing .faiss/.pkl artifact files."
    )
    parser.add_argument(
        "--index-prefix", default="textbook_index",
        help="Prefix for artifact filenames."
    )
    parser.add_argument(
        "--db-path", default="index/tokensmith.db",
        help="Output SQLite database path."
    )
    parser.add_argument(
        "--extension-path", default="extension/build/hybrid_search.so",
        help="Path to compiled hybrid_search.so (used to create the vtable)."
    )
    args = parser.parse_args()

    migrate(
        artifacts_dir=pathlib.Path(args.artifacts_dir),
        index_prefix=args.index_prefix,
        db_path=pathlib.Path(args.db_path),
        extension_path=pathlib.Path(args.extension_path),
    )


if __name__ == "__main__":
    main()
