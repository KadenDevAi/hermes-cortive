"""Stress test and benchmark for the Cortive memory provider. Standard library only.

    python3.12 bench/stress.py                      # real library (read-only) + synthetic scale runs
    python3.12 bench/stress.py --sizes 1000 10000   # choose synthetic library sizes
    python3.12 bench/stress.py --skip-real          # synthetic only

It never writes to your Cortive library. The lock test runs on a temporary copy. Output is
timings and counts only: no titles or text from your library are printed.

What it measures:
  1. Real library: latency of recall (prefetch) and cortive_search over many varied queries.
  2. Hostile input: FTS syntax, SQL fragments, unicode, emoji, 100 KB queries; every call must
     return valid JSON and never raise.
  3. Concurrency: many threads searching at once.
  4. Lock contention: a writer holds the library while recall runs (on a copy).
  5. Scale: synthetic libraries of N captures, with a plain-files baseline (the way a Markdown
     vault without an index is searched: open every note and look for the words).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERMES_DIR = Path(os.environ.get("HERMES_AGENT_DIR") or Path.home() / ".hermes" / "hermes-agent")
sys.path.insert(0, str(HERMES_DIR))
sys.path.insert(0, str(ROOT / "fixtures"))
import make_fixture  # noqa: E402

spec = importlib.util.spec_from_file_location("cortive_bench", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

GENERIC_QUERIES = [
    "what did I save about pricing", "notes from the meeting last week", "that article about sleep",
    "screenshot of the error message", "recipe", "how to set up the router", "invoice",
    "the pdf about taxes", "flight times", "something about machine learning",
    "remember the thing about startups?", "password reset instructions", "budget for the trip",
]

HOSTILE = [
    '* OR title:', '"', "(", ")", "NEAR(a b)", "a AND (", "-", "^x", "col:val*", '"unterminated',
    "'; DROP TABLE captures; --", "1 OR 1=1", "\\", "%", "_", "\x00null\x00byte", chr(0x202E) + "right-to-left",
    "🦀🔥 emoji only", "日本語の検索", "Ünïcödé façade naïve", "a" * 100_000, "word " * 20_000,
    " ", "\n\t", "OR OR OR", "AND", "NOT", "*", "{title} : x", "title:", "x" * 300,
]


def pct(values, p):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[k]


def fmt(ms_values):
    return (f"p50 {pct(ms_values, 50):7.2f} ms   p95 {pct(ms_values, 95):7.2f} ms   "
            f"max {max(ms_values):7.2f} ms   (n={len(ms_values)})")


def make_provider(db: Path, home: Path):
    p = plugin.CortiveMemoryProvider(db_path=str(db))
    if not p.is_available():
        raise SystemExit(p.unavailable_reason())
    p.initialize("bench", hermes_home=str(home), platform="cli")
    return p


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t) * 1000


def queries_from_library(db: Path, n: int, rng: random.Random) -> list[str]:
    """Queries built from words that occur in the library, so most of them have real answers."""
    conn = plugin.CortiveMemoryProvider._connect(db, 5000)
    try:
        rows = conn.execute("SELECT title, substr(content, 1, 2000) FROM captures "
                            "WHERE COALESCE(is_encrypted,0)=0 AND content != ''").fetchall()
    finally:
        conn.close()
    words = []
    for title, body in rows:
        words.extend(t for t in plugin.query_terms(f"{title} {body}") if len(t) > 3 and not t.isdigit())
    if not words:
        return []
    out = []
    for _ in range(n):
        k = rng.choice([1, 2, 2, 3, 4])
        out.append(" ".join(rng.choice(words) for _ in range(k)))
    return out


# -- 1-4: real library ------------------------------------------------------------------


def run_real(db: Path, home: Path, rng: random.Random) -> None:
    print(f"\n== Real library (read-only) ==")
    p = make_provider(db, home)
    print(f"captures readable: {p._capture_count}")
    before = db.stat().st_mtime_ns

    queries = queries_from_library(db, 300, rng) + GENERIC_QUERIES
    pre_ms, search_ms, hits, recalled = [], [], 0, 0
    for i, q in enumerate(queries):
        p._prefetch_cache.clear()
        block, ms = timed(p.prefetch, q, session_id=f"s{i}")
        pre_ms.append(ms)
        recalled += bool(block)
        assert len(block) <= plugin.PREFETCH_BUDGET, "recall over budget"
        raw, ms = timed(p.handle_tool_call, "cortive_search", {"query": q, "limit": 5})
        search_ms.append(ms)
        out = json.loads(raw)
        assert out["success"], out
        hits += out["count"] > 0
    print(f"1. {len(queries)} queries")
    print(f"   recall (prefetch)  {fmt(pre_ms)}   injected on {recalled}/{len(queries)}")
    print(f"   cortive_search     {fmt(search_ms)}   had results on {hits}/{len(queries)}")

    errors = 0
    host_ms = []
    for q in HOSTILE:
        for tool, args in (("cortive_search", {"query": q}), ("cortive_get", {"id": q}),
                           ("cortive_search", {"query": q, "source": q, "since_days": q, "limit": q})):
            try:
                raw, ms = timed(p.handle_tool_call, tool, args)
                json.loads(raw)
                host_ms.append(ms)
            except Exception as exc:  # the test fails if anything escapes
                errors += 1
                print(f"   ESCAPED: {tool} {q[:30]!r}: {exc!r}")
        try:
            p.prefetch(q, session_id="hostile")
        except Exception as exc:
            errors += 1
            print(f"   prefetch ESCAPED on {q[:30]!r}: {exc!r}")
    print(f"2. hostile inputs: {len(HOSTILE) * 4} calls, {errors} exceptions escaped   {fmt(host_ms)}")

    threads, per_thread, lat, failures = 16, 40, [], []
    lock = threading.Lock()

    def worker(seed: int) -> None:
        r = random.Random(seed)
        for _ in range(per_thread):
            q = r.choice(queries)
            try:
                raw, ms = timed(p.handle_tool_call, "cortive_search", {"query": q})
                ok = json.loads(raw)["success"]
                with lock:
                    lat.append(ms)
                    if not ok:
                        failures.append(raw)
            except Exception as exc:
                with lock:
                    failures.append(repr(exc))

    started = time.perf_counter()
    pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    wall = time.perf_counter() - started
    print(f"3. concurrency: {threads} threads x {per_thread} searches, {len(failures)} failures, "
          f"{len(lat) / wall:.0f} searches/s   {fmt(lat)}")

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "copy.db"
        shutil.copy2(db, copy)
        pc = make_provider(copy, home)
        writer = sqlite3.connect(copy, isolation_level=None)
        writer.execute("BEGIN EXCLUSIVE")
        block, ms_pre = timed(pc.prefetch, "budget notes meeting", session_id="lock")
        raw, ms_tool = timed(pc.handle_tool_call, "cortive_search", {"query": "budget"})
        writer.execute("ROLLBACK")
        writer.close()
        raw_after, _ = timed(pc.handle_tool_call, "cortive_search", {"query": queries[0]})
        print(f"4. locked library: recall returned {'nothing' if not block else 'TEXT'} after {ms_pre:.0f} ms; "
              f"search said {json.loads(raw).get('error', 'ok')!r} after {ms_tool:.0f} ms; "
              f"search works again after unlock: {'yes' if json.loads(raw_after)['success'] else 'NO'}")

    assert db.stat().st_mtime_ns == before, "the real library was modified!"
    print("   real library unchanged: yes")


# -- 5: scale ----------------------------------------------------------------------------


def build_synthetic(path: Path, n: int, rng: random.Random, vault: Path | None):
    words = [w.strip().lower() for w in Path("/usr/share/dict/words").read_text().split() if w.isalpha()]
    rng.shuffle(words)
    vocab = words[:30000]
    weights = [1 / (i + 1) ** 0.9 for i in range(len(vocab))]  # Zipf-like: a few common words, many rare
    rank = {w: i for i, w in enumerate(vocab)}
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(path)
    conn.executescript(make_fixture.SCHEMA)
    conn.execute("PRAGMA journal_mode = DELETE")
    sources = ["browser_extension", "pdf_import", "screenshot_capture", "clipboard_capture", "watched_folder"]
    targets = []
    batch_c, batch_f = [], []
    for i in range(n):
        length = int(rng.lognormvariate(6.2, 0.8))  # median ~500 words, long tail of big documents
        length = max(20, min(length, 20000))
        body_words = rng.choices(vocab, weights=weights, k=length)
        title = " ".join(rng.choices(vocab, weights=weights, k=rng.randint(2, 7))).title()
        body = " ".join(body_words)
        cid = f"syn-{i:06d}"
        ts = (now - timedelta(days=rng.uniform(0, 1000))).isoformat()
        src = rng.choice(sources)
        tags = json.dumps(rng.sample(vocab[:200], rng.randint(0, 3)))
        batch_c.append((cid, title, body, "", "", src, tags, ts, f"h{i}"))
        batch_f.append((cid, title, "", "", tags, src, body, None))
        if rng.random() < 0.02 and len(targets) < 60:
            # A query someone might type to find this capture: 2-3 of its less common words.
            rare = sorted(w for w in set(body_words) if rank[w] > 2000)
            if len(rare) >= 3:
                targets.append((cid, " ".join(rng.sample(rare, rng.choice([2, 3])))))
        if vault is not None:
            (vault / f"{cid}.md").write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        if len(batch_c) >= 2000:
            _flush(conn, batch_c, batch_f)
    _flush(conn, batch_c, batch_f)
    conn.execute("PRAGMA user_version = 11")
    conn.commit()
    conn.close()
    return targets


def _flush(conn, batch_c, batch_f):
    conn.executemany("INSERT INTO captures (id,title,content,url,domain,source,tags,timestamp,content_hash) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", batch_c)
    conn.executemany("INSERT INTO captures_fts (id,title,url,domain,tags,source,content,note) "
                     "VALUES (?,?,?,?,?,?,?,?)", batch_f)
    conn.commit()
    batch_c.clear()
    batch_f.clear()


def vault_search(vault: Path, terms: list[str], limit: int = 5):
    """Unindexed note search: read every file, keep notes containing every word, rank by count."""
    hits = []
    for f in vault.iterdir():
        text = f.read_text(encoding="utf-8").lower()
        if all(t in text for t in terms):
            hits.append((sum(text.count(t) for t in terms), f.stem))
    hits.sort(reverse=True)
    return [h[1] for h in hits[:limit]]


def run_scale(sizes: list[int], home: Path, rng: random.Random, vault_max: int) -> None:
    print("\n== Scale (synthetic libraries) ==")
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cortive.db"
            vault = Path(tmp) / "vault" if n <= vault_max else None
            if vault:
                vault.mkdir()
            t = time.perf_counter()
            targets = build_synthetic(db, n, rng, vault)
            build_s = time.perf_counter() - t
            p = make_provider(db, home)
            search_ms, pre_ms, found = [], [], 0
            for cid, q in targets:
                raw, ms = timed(p.handle_tool_call, "cortive_search", {"query": q, "limit": 5})
                search_ms.append(ms)
                found += cid in [r["id"] for r in json.loads(raw)["results"]]
                p._prefetch_cache.clear()
                _, ms = timed(p.prefetch, q, session_id="x")
                pre_ms.append(ms)
            size_mb = db.stat().st_size / 1e6
            print(f"\n{n:,} captures  ({size_mb:,.0f} MB library, built in {build_s:.0f} s)")
            print(f"   cortive_search     {fmt(search_ms)}")
            print(f"   recall (prefetch)  {fmt(pre_ms)}")
            print(f"   target capture in top 5: {found}/{len(targets)}")
            if vault:
                v_ms, v_found = [], 0
                for cid, q in targets[:15]:
                    res, ms = timed(vault_search, vault, q.split())
                    v_ms.append(ms)
                    v_found += cid in res
                print(f"   plain-files scan   {fmt(v_ms)}   target in top 5: {v_found}/{min(15, len(targets))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(plugin.default_db_path()))
    ap.add_argument("--sizes", type=int, nargs="*", default=[1000, 10000, 50000])
    ap.add_argument("--vault-max", type=int, default=10000, help="largest size to also run the plain-files baseline on")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory() as home:
        if not args.skip_real:
            db = Path(args.db)
            if db.is_file():
                run_real(db, Path(home), rng)
            else:
                print(f"No Cortive library at {db}; skipping the real-library tests.")
        if args.sizes:
            run_scale(args.sizes, Path(home), rng, args.vault_max)


if __name__ == "__main__":
    main()
