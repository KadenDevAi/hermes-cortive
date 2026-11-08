"""Build a synthetic Cortive library for tests and demos. No real user data.

Usage: python fixtures/make_fixture.py OUT.db

The schema mirrors Cortive's library at schema version 11 (captures, the standalone
captures_fts index, collections, deleted_captures, embeddings). Cortive keeps the FTS
index up to date in application code, with no triggers, so rows go into both tables here.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE captures (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL, nonce TEXT,
  url TEXT NOT NULL, domain TEXT NOT NULL, source TEXT NOT NULL, tags TEXT,
  timestamp TEXT NOT NULL, content_hash TEXT, screenshot_blob BLOB, title_encrypted BLOB,
  content_encrypted BLOB, is_encrypted INTEGER DEFAULT 0, screenshot_blob_encrypted BLOB,
  browser TEXT, source_context TEXT, truncated_from INTEGER, note TEXT,
  title_custom INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE captures_fts USING fts5(
  id, title, url, domain, tags, source, content, note, tokenize='unicode61'
);
CREATE TABLE embeddings (
  id TEXT PRIMARY KEY, capture_id TEXT NOT NULL, modality TEXT NOT NULL DEFAULT 'text',
  chunk_index INTEGER NOT NULL DEFAULT 0, chunk_start_token INTEGER NOT NULL DEFAULT 0,
  chunk_end_token INTEGER NOT NULL DEFAULT 0, total_chunks INTEGER NOT NULL DEFAULT 1,
  page_number INTEGER NOT NULL DEFAULT 1, embedding BLOB NOT NULL, model_version TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE collections (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, name_locked INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0, vector_sum BLOB, member_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE capture_collections (
  capture_id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, score REAL NOT NULL
);
CREATE TABLE deleted_captures (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL, source TEXT NOT NULL,
  deleted_at TEXT NOT NULL, snapshot TEXT NOT NULL
);
"""

# A 1x1 transparent PNG, enough to test screenshot export.
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def _ts(days_ago: float, now: datetime) -> str:
    return (now - timedelta(days=days_ago)).isoformat()


def rows(now: datetime) -> list[dict]:
    """Twelve captures covering every case the tests check."""
    big = ("Quarterly logistics ledger. " * 18000)[:500_000] + " zeppelin manifest appendix."
    return [
        dict(id="hvac-notes", title="Henderson HVAC — intake notes", source="browser_extension",
             url="https://crm.example.com/henderson", domain="crm.example.com", days=14,
             tags=["client", "hvac"], hash="h-hvac",
             content="Henderson HVAC intake call. Missed-call text-back recovered 4,200/mo in booked jobs."),
        dict(id="dup-a", title="Sourdough starter guide", source="browser_extension",
             url="https://bread.example.com/starter", domain="bread.example.com", days=30, hash="h-dup",
             content="Feed the sourdough starter twice daily with equal flour and water."),
        dict(id="dup-b", title="Sourdough starter guide", source="browser_extension",
             url="https://bread.example.com/starter", domain="bread.example.com", days=30, hash="h-dup",
             content="Feed the sourdough starter twice daily with equal flour and water."),
        dict(id="dup-c", title="Sourdough starter guide", source="browser_extension",
             url="https://bread.example.com/starter", domain="bread.example.com", days=30, hash="h-dup",
             content="Feed the sourdough starter twice daily with equal flour and water."),
        dict(id="secret", title="Encrypted volcano diary", source="markdown_import",
             url="file:///diary.md", domain="local", days=3, hash="h-secret", encrypted=True,
             content="volcano expedition secret plans"),
        dict(id="empty", title="Blank glacier capture", source="clipboard_capture",
             url="", domain="", days=2, hash="h-empty", content=""),
        dict(id="tagged-1", title="Tomato growing notes", source="pdf_import",
             url="file:///garden.pdf", domain="local", days=60, tags=["gardening", "tomato"], hash="h-tag1",
             content="Prune tomato suckers weekly; water deeply at the base."),
        dict(id="tagged-2", title="Garden plan", source="markdown_import",
             url="file:///plan.md", domain="local", days=200, tags=["planning"], hash="h-tag2",
             content="Plan the raised beds: tomato, basil, peppers.", note="Order seeds in March"),
        dict(id="big", title="Logistics ledger dump", source="pdf_import",
             url="file:///ledger.pdf", domain="local", days=10, hash="h-big", content=big),
        dict(id="old-kayak", title="Kayak rental comparison", source="browser_extension",
             url="https://paddle.example.com/old", domain="paddle.example.com", days=700, hash="h-k1",
             content="Kayak rental prices along the lakeshore, compared by hour."),
        dict(id="new-kayak", title="Kayak rental comparison", source="browser_extension",
             url="https://paddle.example.com/new", domain="paddle.example.com", days=5, hash="h-k2",
             content="Kayak rental prices along the lakeshore, compared by hour."),
        dict(id="deleted-one", title="Deleted meteor note", source="browser_extension",
             url="https://sky.example.com", domain="sky.example.com", days=1, hash="h-del",
             content="meteor shower viewing times", screenshot=True),
        dict(id="shot", title="Whiteboard screenshot", source="screenshot_capture",
             url="", domain="", days=7, hash="h-shot", screenshot=True,
             content="Whiteboard: quokka launch checklist"),
    ]


def build(path: Path, *, user_version: int = 11, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for r in rows(now):
        tags = json.dumps(r.get("tags", []))
        ts = _ts(r["days"], now)
        shot = TINY_PNG if r.get("screenshot") else None
        conn.execute(
            "INSERT INTO captures (id, title, content, url, domain, source, tags, timestamp, content_hash, "
            "screenshot_blob, is_encrypted, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["id"], r["title"], r["content"], r["url"], r["domain"], r["source"], tags, ts, r["hash"],
             shot, 1 if r.get("encrypted") else 0, r.get("note")),
        )
        conn.execute(
            "INSERT INTO captures_fts (id, title, url, domain, tags, source, content, note) VALUES (?,?,?,?,?,?,?,?)",
            (r["id"], r["title"], r["url"], r["domain"], tags, r["source"], r["content"], r.get("note")),
        )
    conn.execute("INSERT INTO deleted_captures VALUES (?,?,?,?,?,?)",
                 ("deleted-one", "Deleted meteor note", "https://sky.example.com", "browser_extension",
                  now.isoformat(), "{}"))
    stamp = now.isoformat()
    conn.executemany(
        "INSERT INTO collections (id, name, pinned, member_count, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        [("col-garden", "Garden", 1, 2, stamp, stamp), ("col-bread", "Baking", 0, 3, stamp, stamp),
         ("col-work", "Clients", 0, 2, stamp, stamp), ("col-empty", "Empty", 0, 0, stamp, stamp)],
    )
    conn.executemany(
        "INSERT INTO capture_collections (capture_id, collection_id, score) VALUES (?,?,?)",
        [("tagged-1", "col-garden", 0.9), ("tagged-2", "col-garden", 0.8),
         ("dup-a", "col-bread", 0.9), ("dup-b", "col-bread", 0.9), ("dup-c", "col-bread", 0.9),
         ("hvac-notes", "col-work", 0.95), ("secret", "col-work", 0.5), ("deleted-one", "col-work", 0.4),
         # collection_id '' is Cortive's marker for a capture the user kept out of collections.
         ("new-kayak", "", 0.0)],
    )
    conn.execute(f"PRAGMA user_version = {int(user_version)}")
    conn.commit()
    conn.close()
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python fixtures/make_fixture.py OUT.db")
    print(build(Path(sys.argv[1])))
