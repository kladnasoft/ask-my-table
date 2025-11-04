# gen_metasample.py — SuperTable version (no MSSQL)
# - Reads SUPERTABLE_* env (dotenv supported by _connect_supertable)
# - Uses _connect_supertable.SupertableClient to call REST API
# - Iterates all tables, samples rows, computes lightweight metadata
# - Writes metasample/<org>_<super>_<table>.json
# - Parallel workers + adaptive progress HUD preserved

from __future__ import annotations

import os
import sys
import json
import time
import queue
import threading
import shutil
from pathlib import Path
from collections import Counter
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from _connect_supertable import SupertableClient, STContext  # <- REST connector

# ========================= Config =============================
OUTPUT_DIR  = os.getenv("METASAMPLE_DIR", "metasample")
TOPN        = int(os.getenv("TOPN", "5"))
HIST_BINS   = int(os.getenv("HIST_BINS", "5"))
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", "50"))  # small fixed sample
MAX_WORKERS = os.getenv("MAX_WORKERS")             # optional override

# HUD (single-line) paging controls (used only in non-multiline consoles)
HUD_PAGE_SIZE     = int(os.getenv("HUD_PAGE_SIZE", "3"))
HUD_PAGE_INTERVAL = float(os.getenv("HUD_PAGE_INTERVAL", "0.8"))
HUD_LABEL_MAX     = int(os.getenv("HUD_LABEL_MAX", "16"))

# ========================= Console capability =================
def supports_multiline():
    # Manual overrides first
    if os.getenv("FORCE_MULTILINE_PROGRESS") == "1":
        return True
    if os.getenv("FORCE_SIMPLE_PROGRESS") == "1":
        return False
    # PyCharm default console is usually not a TTY; terminals are.
    return sys.stdout.isatty()

MULTILINE = supports_multiline()

def enable_ansi_win():
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass

if MULTILINE:
    enable_ansi_win()

def term_width() -> int:
    try:
        return shutil.get_terminal_size((120, 24)).columns
    except Exception:
        return 120

def fmt_dur(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def bar_str(current: int, total: int, width: int) -> str:
    current = max(0, min(current, max(1, total)))
    frac = (current / total) if total else 0.0
    filled = int(frac * width)
    arrow = "=" * max(0, filled - 1) + (">" if filled > 0 else "")
    pad = " " * (width - len(arrow))
    return f"[{arrow}{pad}] {int(frac*100):3d}%"

# ======= Fixed-line multiline writer =======
_ml_initialized = False
_ml_line_count = 0

def write_lines_multiline(lines: list[str]):
    """Reserve a fixed number of lines once, then rewrite in place using ANSI cursor positioning."""
    global _ml_initialized, _ml_line_count
    if not _ml_initialized:
        _ml_line_count = len(lines)
        sys.stdout.write("\n" * _ml_line_count)  # reserve block
        sys.stdout.flush()
        _ml_initialized = True
    # Move cursor to start of block (cursor up by N lines)
    sys.stdout.write(f"\x1b[{_ml_line_count}F")
    w = term_width()
    for i, line in enumerate(lines):
        sys.stdout.write("\r\x1b[K")   # clear line
        sys.stdout.write(line[:w] + ("\n" if i < len(lines) - 1 else ""))  # avoid extra newline last line
    sys.stdout.flush()

def write_line_singleline(line: str):
    sys.stdout.write("\r" + line[:term_width()])
    sys.stdout.flush()

# ========================= Utils ===============================
def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def try_float(v):
    try:
        return float(v)
    except Exception:
        return None

def patternize(value) -> str:
    if value is None:
        return "NULL"
    s = str(value)
    out = []
    for ch in s:
        if "0" <= ch <= "9":
            out.append("9")
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            out.append("A")
        else:
            out.append(ch)
    return "".join(out)

def make_histogram(values, bins=5):
    vals = [v for v in values if v is not None]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [{"bin": 1, "lower": lo, "upper": hi, "count": len(vals), "pct": 100.0}]
    step = (hi - lo) / bins if bins else (hi - lo)
    if step == 0:
        return [{"bin": 1, "lower": lo, "upper": hi, "count": len(vals), "pct": 100.0}]
    edges = [lo + i * step for i in range(bins)] + [hi]
    counts = [0] * bins
    for v in vals:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    total = len(vals)
    return [
        {"bin": i + 1, "lower": edges[i], "upper": edges[i + 1], "count": counts[i], "pct": round(100.0 * counts[i] / total, 2)}
        for i in range(bins)
    ]

def rows_to_dicts(cols: List[str], rows: List[List[Any]], limit=50):
    return [{cols[j]: row[j] for j in range(len(cols))} for row in rows[:limit]]

# ========================= REST helpers (schema/stats) =========
def get_table_schema(client: SupertableClient, ctx: STContext, table: str) -> List[Dict[str, Any]]:
    """
    Calls /meta/schema to retrieve columns for a simple table name.
    Returns a list of dicts like: {"name": str, "data_type": str, "is_nullable": bool, ...}
    Falls back to empty list if unavailable.
    """
    from urllib.parse import urlencode
    url = (
        f"{client.base_url}/meta/schema?"
        + urlencode({
            "organization": ctx.organization,
            "super_name": ctx.super_name,
            "table": table,
            "user_hash": ctx.user_hash,
        })
    )
    r = client.session.get(url)
    r.raise_for_status()
    data = r.json()
    # Expected shape: { ok: true, columns: [...] } — normalize robustly
    cols = data.get("columns") or data.get("schema") or data.get("cols") or []
    out = []
    for c in cols:
        # accept various keys; normalize to a common set
        name = c.get("name") or c.get("column") or c.get("colname")
        dtype = (c.get("data_type") or c.get("dtype") or c.get("type") or "").lower()
        nullable = bool(c.get("is_nullable")) if "is_nullable" in c else (not c.get("not_null", False))
        out.append({"name": name, "data_type": dtype, "is_nullable": nullable, **c})
    return out

def get_table_stats(client: SupertableClient, ctx: STContext, table: str) -> Dict[str, Any]:
    """Calls /meta/stats to retrieve lightweight stats for a simple table name."""
    from urllib.parse import urlencode
    url = (
        f"{client.base_url}/meta/stats?"
        + urlencode({
            "organization": ctx.organization,
            "super_name": ctx.super_name,
            "table": table,
            "user_hash": ctx.user_hash,
        })
    )
    r = client.session.get(url)
    r.raise_for_status()
    return r.json()

def execute_sample(client: SupertableClient, ctx: STContext, table: str, n: int) -> Tuple[List[str], List[List[Any]], Dict[str, Any]]:
    """
    Executes: SELECT * FROM <table> LIMIT n
    Returns (columns, rows_preview, meta)
    """
    res = client.execute(
        f"SELECT * FROM {table} LIMIT {int(n)}",
        organization=ctx.organization,
        super_name=ctx.super_name,
        user_hash=ctx.user_hash,
        engine="DUCKDB",
        with_scan=False,
        preview_rows=10,
    )
    cols = res.get("columns") or []
    rows = res.get("rows_preview") or res.get("rows") or []
    meta = res.get("meta") or {}
    return cols, rows, meta

# ========================= Rendering state =====================
class State:
    def __init__(self, total_tables: int, workers: int, ctx: STContext):
        self.total = total_tables
        self.done = 0
        self.failed = 0
        self.workers = workers
        self.ctx = ctx
        # per-worker fields
        self.label   = ["idle"] * workers
        self.cur     = [0] * workers
        self.tot     = [1] * workers      # avoids div-by-zero; will set later
        self.status  = ["idle"] * workers # idle / running / done / error
        self.start_ts= [0.0] * workers
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()

    def start_table(self, wid: int, label: str, provisional_total: int = 6):
        with self.lock:
            self.label[wid] = label
            self.cur[wid] = 0
            self.tot[wid] = max(1, provisional_total)
            self.status[wid] = "running"
            self.start_ts[wid] = time.perf_counter()

    def set_total(self, wid: int, total_units: int):
        with self.lock:
            self.tot[wid] = max(1, total_units)

    def inc(self, wid: int, n: int = 1):
        with self.lock:
            self.cur[wid] = min(self.tot[wid], self.cur[wid] + n)

    def done_table(self, wid: int, ok: bool = True):
        with self.lock:
            self.status[wid] = "done" if ok else "error"
            self.done += 1
            if not ok:
                self.failed += 1

    def snapshot(self):
        with self.lock:
            return {
                "total": self.total, "done": self.done, "failed": self.failed,
                "label": list(self.label), "cur": list(self.cur), "tot": list(self.tot),
                "status": list(self.status), "start_ts": list(self.start_ts),
                "elapsed": time.perf_counter() - self.t0
            }

# ========================= Renderer ============================
def render_loop(state: State):
    # Paging state for single-line HUD
    render_loop._page = 0
    render_loop._last_switch = time.time()

    # For multiline, allocate fixed lines once
    if MULTILINE:
        write_lines_multiline([""] * (state.workers + 2))

    while True:
        snap = state.snapshot()
        if MULTILINE:
            header = (
                f"MetaSample — Org:{state.ctx.organization}  Super:{state.ctx.super_name}  "
                f"CPU:{os.cpu_count()}  Elapsed:{fmt_dur(snap['elapsed'])}  Workers:{state.workers}"
            )
            gbar = bar_str(snap["done"], snap["total"], width=min(60, max(20, term_width() - 40)))
            lines = [header, f"TOTAL {gbar}  failed:{snap['failed']}"]
            for wid in range(state.workers):
                lab = snap["label"][wid]
                cur, tot = snap["cur"][wid], snap["tot"][wid]
                status = snap["status"][wid]
                prefix = f"T{wid+1} {lab}"
                bar = bar_str(cur, tot, width=min(50, max(20, term_width() - len(prefix) - 20)))
                lines.append(f"{prefix} {bar}  {status}")
            write_lines_multiline(lines)
        else:
            # Single-line HUD with paging (works in PyCharm default console)
            labels, cur, tot = snap["label"], snap["cur"], snap["tot"]
            n = len(labels)
            start = render_loop._page * HUD_PAGE_SIZE
            end = min(start + HUD_PAGE_SIZE, n)

            header = (
                f"ALL {snap['done']}/{snap['total']} fail:{snap['failed']} "
                f"{fmt_dur(snap['elapsed'])}  Workers:{state.workers}  "
                f"{state.ctx.organization}/{state.ctx.super_name}"
            )
            items = []
            for wid in range(start, end):
                lab_full = labels[wid].split(".")[-1]  # tail of table name
                lab = lab_full[-HUD_LABEL_MAX:] if len(lab_full) > HUD_LABEL_MAX else lab_full
                pct = int((cur[wid] / tot[wid]) * 100) if tot[wid] else 0
                items.append(f"T{wid+1} {pct:3d}% {lab}")

            write_line_singleline(" | ".join([header] + items))

            # rotate page
            if time.time() - render_loop._last_switch >= HUD_PAGE_INTERVAL:
                render_loop._page = 0 if end >= n else render_loop._page + 1
                render_loop._last_switch = time.time()

        time.sleep(0.12)

# ========================= Worker ==============================
def build_profiles(cols_meta: List[Dict[str, Any]], cols: List[str], rows: List[List[Any]]):
    """Compute column profiles from a sampled result set."""
    data = {c["name"]: [] for c in cols_meta}
    for rw in rows:
        rd = dict(zip(cols, rw))
        for c in cols_meta:
            data[c["name"]].append(rd.get(c["name"]))

    profiles = {}
    for c in cols_meta:
        name = c["name"]
        # normalize dtype names
        dtype = (c.get("data_type") or c.get("dtype") or "").lower()
        vals = data[name]
        n = len(vals)
        nulls = sum(v is None for v in vals)
        non_null = n - nulls
        non_null_vals = [v for v in vals if v is not None]

        approx_distinct = len(set(non_null_vals))
        uniq_ratio = round(100.0 * approx_distinct / non_null, 2) if non_null else 0.0
        top_values = [{"value": v, "count": int(cnt)} for v, cnt in Counter(non_null_vals).most_common(TOPN)] if non_null_vals else []

        patterns = None
        if any(x in dtype for x in ("char", "text", "string")):
            patterns = [{"pattern": p, "count": int(cnt)} for p, cnt in Counter(patternize(v) for v in non_null_vals).most_common(TOPN)]

        def cast_hist(v):
            if v is None:
                return None
            # try numeric
            f = try_float(v)
            if f is not None:
                return f
            # datetime-like => try timestamp()
            try:
                return v.timestamp() if hasattr(v, "timestamp") else None
            except Exception:
                return None

        histogram = None
        if any(x in dtype for x in ("int", "num", "dec", "float", "real", "money", "date", "time")):
            casted = [cast_hist(v) for v in non_null_vals]
            casted = [v for v in casted if v is not None]
            if casted:
                histogram = make_histogram(casted, bins=HIST_BINS)

        string_len = None
        if any(x in dtype for x in ("char", "text", "string")):
            lens = [len(str(v)) for v in non_null_vals]
            if lens:
                string_len = {"avg": round(sum(lens) / len(lens), 2), "min": min(lens), "max": max(lens)}

        profiles[name] = {
            "null_pct": round(100.0 * nulls / n, 2) if n else 0.0,
            "sample_count": n,
            "sample_non_null": non_null,
            "sample_approx_distinct": approx_distinct,
            "sample_unique_ratio_pct": uniq_ratio,
            "top_values": top_values or None,
            "patterns": patterns,
            "histogram": histogram,
            "string_len": string_len
        }
    return profiles

def worker_loop(wid: int, q: "queue.Queue[tuple[int,str]]", state: State, out_dir: Path, client: SupertableClient, ctx: STContext):
    while True:
        try:
            idx, table = q.get_nowait()
        except queue.Empty:
            break

        label = f"{table}"
        state.start_table(wid, label, provisional_total=5)  # schema + stats + sample + profile + write
        ok = True

        try:
            # 1) schema
            cols_meta = get_table_schema(client, ctx, table)
            if not cols_meta:
                # Fallback: run a tiny sample to get columns
                cols, rows, _ = execute_sample(client, ctx, table, n=min(5, SAMPLE_ROWS))
                cols_meta = [{"name": c, "data_type": ""} for c in cols]
            state.inc(wid)

            # Adjust total: schema(1) + stats(1) + sample(1) + profile(len(cols_meta)) + write(1)
            total_units = 4 + len(cols_meta)
            state.set_total(wid, total_units)

            # 2) stats (row count estimate if available)
            stats = {}
            try:
                stats = get_table_stats(client, ctx, table) or {}
            except Exception:
                stats = {}
            rowcount = (
                stats.get("row_count")
                or stats.get("approx_row_count")
                or stats.get("rows")
                or None
            )
            state.inc(wid)

            # 3) sample
            cols, rows, meta = execute_sample(client, ctx, table, n=SAMPLE_ROWS)
            state.inc(wid)

            # Ensure cols_meta has names present in 'cols'
            if cols and (not cols_meta or any(c.get("name") not in cols for c in cols_meta)):
                # rebuild meta from result columns if mismatch
                cols_meta = [{"name": c, "data_type": ""} for c in cols]

            # 4) profile (per-column progress)
            profiles = build_profiles(cols_meta, cols, rows)
            # give progress ticks per column
            state.inc(wid, len(cols_meta))

            # 5) write
            doc = {
                "organization": ctx.organization,
                "super_name":  ctx.super_name,
                "table":       table,
                "row_count_estimate": rowcount,
                "columns": [
                    {**c, "is_primary_key": False, "profile": profiles.get(c["name"], {})}
                    for c in cols_meta
                ],
                "sample_rows": rows_to_dicts(cols, rows, limit=min(50, SAMPLE_ROWS)),
                "meta": meta or None,
            }
            path = out_dir / safe_filename(f"{ctx.organization}_{ctx.super_name}_{table}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2, default=str)

            state.inc(wid)  # final bump for "write"
        except Exception:
            ok = False
        finally:
            state.done_table(wid, ok=ok)
            q.task_done()

# ========================= Main ================================
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Connect/login + discover defaults (org/super/user_hash)
    with SupertableClient() as st:
        ctx = st.get_default_context()
        print(f"Using SuperTable: org={ctx.organization} super={ctx.super_name}")

        # Enumerate tables
        tables = st.list_tables(ctx.organization, ctx.super_name)

        total = len(tables)
        if total == 0:
            print("No tables found.")
            return

        cpu = os.cpu_count() or 4
        workers = int(MAX_WORKERS) if MAX_WORKERS else min(cpu, total)
        print(f"Workers: {workers} (CPU: {cpu})")

        # Queue tasks
        q: "queue.Queue[tuple[int,str]]" = queue.Queue()
        for i, table in enumerate(tables, start=1):
            q.put((i, table))

        # Shared state + renderer
        state = State(total_tables=total, workers=workers, ctx=ctx)
        renderer = threading.Thread(target=render_loop, args=(state,), daemon=True)
        renderer.start()

        # Workers
        threads = []
        for wid in range(workers):
            t = threading.Thread(target=worker_loop, args=(wid, q, state, out_dir, st, ctx), daemon=True)
            t.start()
            threads.append(t)

        # Wait
        q.join()
        for t in threads:
            t.join()

        # Final newline if single-line HUD
        if not MULTILINE:
            sys.stdout.write("\n")
            sys.stdout.flush()

        snap = state.snapshot()
        print(f"Done. OK={snap['done'] - snap['failed']}  Failed={snap['failed']}  Elapsed={fmt_dur(snap['elapsed'])}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
