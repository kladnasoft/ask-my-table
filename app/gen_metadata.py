# gen_ai_metadata_assistant.py
"""
Batch-convert metasample/*.json into metaschema/*__ai_meta.json using OpenAI Assistants v2,
but with a *strong* run-level system prompt that yields rich, semantic metadata.

- Prints ALL HTTP requests (sanitized).
- Preflight validates key via /models and /assistants.
- Injects INSTRUCTIONS_META_V2 at run-time (no need to edit assistant in dashboard).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

# ─────────────────────────── env helpers ───────────────────────────
def _read_env_file(path: str = ".env") -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if not os.path.exists(path):
        return vals
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            vals[k.strip()] = v.strip()
    return vals

def _unquote_strip(val: str | None) -> str:
    if not val:
        return ""
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    v = "".join(ch for ch in v if 32 <= ord(ch) <= 126)
    return v

def load_env_override() -> Dict[str, str]:
    before = dict(os.environ)
    file_vals = _read_env_file(".env")
    for k, v in file_vals.items():
        os.environ[k] = v
    cfg = {k: os.getenv(k) for k in [
        "OPENAI_API_KEY", "OPENAI_METABUILDER_ID", "OPENAI_BASE_URL",
        "OPENAI_ORG", "OPENAI_PROJECT", "METASAMPLE_DIR", "OUTPUT_DIR",
        "SAMPLE_ROWS_IN_PROMPT", "ASSISTANT_POLL_INTERVAL_SEC",
        "ASSISTANT_MAX_WAIT_SEC", "ASSISTANT_HTTP_TIMEOUT",
        "HTTPX_LOG_LEVEL", "PRINT_HTTP_BODIES", "MAX_PRINT_CHARS"
    ]}
    cfg["OPENAI_API_KEY"] = _unquote_strip(cfg["OPENAI_API_KEY"])
    cfg["OPENAI_METABUILDER_ID"] = _unquote_strip(cfg["OPENAI_METABUILDER_ID"])
    cfg["OPENAI_BASE_URL"] = _unquote_strip(cfg["OPENAI_BASE_URL"] or "") or "https://api.openai.com/v1"
    cfg["OPENAI_ORG"] = _unquote_strip(cfg["OPENAI_ORG"])
    cfg["OPENAI_PROJECT"] = _unquote_strip(cfg["OPENAI_PROJECT"])
    key_src = "unknown"
    if "OPENAI_API_KEY" in file_vals and "OPENAI_API_KEY" in before:
        key_src = ".env (overrode OS)"
    elif "OPENAI_API_KEY" in file_vals:
        key_src = ".env"
    elif "OPENAI_API_KEY" in before:
        key_src = "OS"
    cfg["_KEY_SOURCE"] = key_src
    return cfg

CFG = load_env_override()

# ─────────────────────────── logging ───────────────────────────
logger = logging.getLogger("ask-my-table")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s ask-my-table: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

_httpx_level = (CFG.get("HTTPX_LOG_LEVEL") or "").upper()
if _httpx_level in {"DEBUG", "INFO", "WARNING"}:
    logging.getLogger("httpx").setLevel(getattr(logging, _httpx_level))
    logging.getLogger("httpcore").setLevel(getattr(logging, _httpx_level))
    if not logging.getLogger("httpx").handlers:
        logging.getLogger("httpx").addHandler(handler)
    if not logging.getLogger("httpcore").handlers:
        logging.getLogger("httpcore").addHandler(handler)

# ─────────────────────────── config ───────────────────────────
OPENAI_API_KEY = CFG["OPENAI_API_KEY"] or ""
OPENAI_METABUILDER_ID = CFG["OPENAI_METABUILDER_ID"] or ""
OPENAI_BASE_URL = CFG["OPENAI_BASE_URL"] or "https://api.openai.com/v1"
OPENAI_ORG = CFG["OPENAI_ORG"] or ""
OPENAI_PROJECT = CFG["OPENAI_PROJECT"] or ""

METASAMPLE_DIR = CFG.get("METASAMPLE_DIR") or "metasample"
OUTPUT_DIR = CFG.get("METASSCHEMA_DIR") or "metaschema"
SAMPLE_ROWS_IN_PROMPT = int(CFG.get("SAMPLE_ROWS_IN_PROMPT") or "15")

ASSISTANT_POLL_INTERVAL_SEC = float(CFG.get("ASSISTANT_POLL_INTERVAL_SEC") or "0.8")
ASSISTANT_MAX_WAIT_SEC = float(CFG.get("ASSISTANT_MAX_WAIT_SEC") or "180.0")
ASSISTANT_HTTP_TIMEOUT = float(CFG.get("ASSISTANT_HTTP_TIMEOUT") or "120.0")

PRINT_HTTP_BODIES = (CFG.get("PRINT_HTTP_BODIES") or "1") == "1"
MAX_PRINT_CHARS = int(CFG.get("MAX_PRINT_CHARS") or "8000")

if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY in .env")
if not OPENAI_METABUILDER_ID:
    raise SystemExit("Missing OPENAI_METABUILDER_ID in .env")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2",
}
if OPENAI_ORG:
    HEADERS["OpenAI-Organization"] = OPENAI_ORG
if OPENAI_PROJECT:
    HEADERS["OpenAI-Project"] = OPENAI_PROJECT

def _sanitize_key(k: str) -> str:
    k = (k or "").strip()
    return f"{k[:6]}…{k[-4:]}" if len(k) > 12 else k or "(none)"

def _print_request(label: str, method: str, url: str, headers: Dict[str, str], body: Dict[str, Any] | None):
    hdrs = dict(headers)
    if "Authorization" in hdrs:
        token = hdrs["Authorization"].replace("Bearer ", "")
        hdrs["Authorization"] = f"Bearer {_sanitize_key(token)}"
    logger.info("%s: %s %s", label, method.upper(), url)
    logger.info("%s headers: %s", label, json.dumps(hdrs, ensure_ascii=False))
    if body is not None:
        text = json.dumps(body, ensure_ascii=False, indent=2)
        if not PRINT_HTTP_BODIES and len(text) > 400:
            logger.info("%s body: %s… [truncated]", label, text[:400])
        else:
            if len(text) > MAX_PRINT_CHARS:
                logger.info("%s body: %s… [truncated %d chars]", label, text[:MAX_PRINT_CHARS], len(text) - MAX_PRINT_CHARS)
            else:
                logger.info("%s body: %s", label, text)

# ─────────────────────── system/run instructions ───────────────────────
INSTRUCTIONS_META_V2 = """
You are SQL Metadata Builder++. Transform a metasample JSON (schema + basic profiling + sample_rows) into a compact, *semantic* metadata object for NL→T-SQL (SQL Server).

CRITICAL:
- Output ONE JSON object only. No markdown/prose.
- Use the schema "ai_sql_meta_v2" below. Prefer cautious inference WITH confidence + signals over omitting fields.
- Never invent columns; keep original names/case.

OUTPUT (ai_sql_meta_v2):
{
  "spec_version": "ai_sql_meta_v2",
  "identity": {
    "database": "<db>", "schema": "<schema>", "table": "<table>",
    "table_title": "<short name>", "table_summary": "<what a row represents>"
  },
  "stats": {
    "row_count_estimate": <num>,
    "freshness": { "created_at_col": "<col>", "updated_at_col": "<col>", "min_date": "<iso8601>", "max_date": "<iso8601>" }
  },
  "columns": [
    {
      "name": "<exact col>",
      "sql_type": "<type>",
      "nullable": true/false,
      "primary_key": true/false,
      "candidate_key": true/false,
      "role": "dimension|measure|time|flag|geo|text|id",
      "semantics": "<what this field means>",
      "pii": "none|name|phone|email|address|postcode|id|geo|other",
      "units": "<if any>",
      "enum_values": ["..."],                      // when categorical/bool-like
      "bool_mapping": { "true": ["Y","Yes","True","1"], "false": ["N","No","False","0"] }, // if applicable
      "value_range": { "min": <num|iso8601>, "max": <num|iso8601> }, // when numeric/date
      "patterns": ["AA9 9AA"],                    // e.g., UK postcode shapes
      "uniqueness_ratio_est": <0-100>,
      "missing_pct_est": <0-100>,
      "inference": { "confidence": "low|medium|high", "signals": ["why you inferred this"] }
    }
  ],
  "keys_indexes": {
    "primary_key": ["colA", "colB"],
    "unique_indexes": [ {"name":"<guess_or_pk>","columns":["..."]} ],
    "nonclustered_indexes": [ {"name":"<suggest>","columns":["..."],"reason":"common filter/sort"} ]
  },
  "joins": [
    { "to": "<schema.table or pattern like dbo.<*Contact*|*Account*> >",
      "on": [{"left":"<this_col>","right":"<that_col>"}],
      "reason": "<name/type/value overlap>", "confidence": "low|medium|high" }
  ],
  "synonyms": { "table": ["..."], "columns": { "<col>": ["syn1","syn2"] } },
  "filters": [ { "column":"<col>", "operators":["=","IN","BETWEEN","LIKE"], "examples":["Y","TN12PB","2025-01-01..2025-01-31"] } ],
  "geo": { "lat_col":"<col>", "lon_col":"<col>", "point_wkt_expr":"POINT([lon],[lat])" },
  "nl2sql_guidance": {
    "group_bys": ["status","postcode","group_name"],
    "time_cols": ["created_at","updated_at","deleted_at","expires_at"],
    "sortable_cols": ["created_at","name","reference"],
    "default_row_limit": 500,
    "order_by_preference": ["created_at DESC","name ASC"]
  },
  "typical_queries": [
    { "name":"Counts by status last 30 days",
      "sql":"SELECT [Status], COUNT(*) AS cnt FROM [<schema>].[<table>] WHERE [CreatedAt]>=DATEADD(DAY,-30,SYSDATETIME()) GROUP BY [Status] ORDER BY cnt DESC",
      "notes":"Prefer sargable date predicate; avoid SELECT *" }
  ],
  "caveats": ["Booleans stored as strings ('Y'/'N')", "Avoid non-sargable predicates on indexed cols"]
}

HEURISTICS:
- IDs: name "ID" or "*_ID" → role=id; high uniqueness → candidate_key=true.
- Time: CREATED_AT/UPDATED_AT/DELETED_AT/EXPIRES_AT → role=time; use sample/histogram to bound min/max dates.
- Flags: columns starting "IS_" or short NVARCHAR holding {Y,N,Yes,No,True,False,0,1} → role=flag + bool_mapping.
- Postcode: patterns like "AA9 9AA", "AA99AA" → pii=postcode; role=dimension.
- Geo: names LATITUDE/LONGITUDE → role=geo; if both exist, provide point_wkt_expr.
- Measures: numeric with wide/continuous hist → role=measure; provide value_range.
- Categorical: small distinct sets → enum_values.
- Joins: columns like PARENT_CONTACT_ID or ACCOUNT_REFERENCE → propose joins (pattern targets ok) with reasons + confidence.

Return ONLY the JSON object (no code fences).
"""

# ─────────────────────── payload building ───────────────────────
def trim_metasample(ms: Dict[str, Any]) -> Dict[str, Any]:
    ms2 = dict(ms)
    rows = ms2.get("sample_rows") or []
    cap = int(SAMPLE_ROWS_IN_PROMPT)
    if rows and len(rows) > cap:
        ms2["sample_rows"] = rows[:cap]
    return ms2

def build_user_payload(metasample: Dict[str, Any]) -> str:
    # Just the task + the trimmed data. All structural rules live in run-level instructions.
    return "Please build ai_sql_meta_v2 for the following metasample:\n\n" + json.dumps(trim_metasample(metasample), ensure_ascii=False)

def ensure_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        raise

# ─────────────────────── HTTP helpers ───────────────────────
def _key_healthcheck():
    key = OPENAI_API_KEY
    if len(key) < 40:
        raise SystemExit("OPENAI_API_KEY appears too short. Ensure .env contains the full Platform key (no quotes).")

async def _post_json(client: httpx.AsyncClient, url: str, body: Dict[str, Any], label: str, req_id: str) -> httpx.Response:
    _print_request(f"[{req_id}] {label}", "POST", url, HEADERS, body)
    resp = await client.post(url, headers=HEADERS, json=body)
    logger.info("[%s] %s: response %s", req_id, label, resp.status_code)
    resp.raise_for_status()
    return resp

async def _get(client: httpx.AsyncClient, url: str, label: str, req_id: str, params: Dict[str, Any] | None = None) -> httpx.Response:
    _print_request(f"[{req_id}] {label}", "GET", url, HEADERS, None)
    resp = await client.get(url, headers=HEADERS, params=params)
    logger.info("[%s] %s: response %s", req_id, label, resp.status_code)
    resp.raise_for_status()
    return resp

async def preflight_validate_key() -> None:
    timeout = httpx.Timeout(ASSISTANT_HTTP_TIMEOUT, connect=20.0, read=ASSISTANT_HTTP_TIMEOUT, write=60.0)
    req_id = "preflight"
    async with httpx.AsyncClient(timeout=timeout) as client:
        # /models (no beta)
        models_headers = dict(HEADERS); models_headers.pop("OpenAI-Beta", None)
        _print_request(f"[{req_id}] Models list (preflight)", "GET", f"{OPENAI_BASE_URL}/models", models_headers, None)
        respA = await client.get(f"{OPENAI_BASE_URL}/models", headers=models_headers)
        logger.info("[%s] Models list (preflight): response %s", req_id, respA.status_code)
        respA.raise_for_status()

        # /assistants (beta)
        urlB = f"{OPENAI_BASE_URL}/assistants"
        respB = await _get(client, urlB, "Assistants list (preflight)", req_id, params={"limit": 1})
        _ = respB.json()

# ─────────────────────── assistants v2 run ───────────────────────
async def run_metabuilder(user_text: str, req_id: str) -> str:
    timeout = httpx.Timeout(ASSISTANT_HTTP_TIMEOUT, connect=20.0, read=ASSISTANT_HTTP_TIMEOUT, write=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # 1) thread
        t_url = f"{OPENAI_BASE_URL}/threads"
        t_body = {"messages": [{"role": "user", "content": user_text}]}
        t_resp = await _post_json(client, t_url, t_body, "Assistants thread", req_id)
        thread_id = t_resp.json()["id"]
        logger.info("[%s] Assistants: thread created id=%s", req_id, thread_id)

        # 2) run with strong instructions
        r_url = f"{OPENAI_BASE_URL}/threads/{thread_id}/runs"
        r_body = {
            "assistant_id": OPENAI_METABUILDER_ID,
            "instructions": INSTRUCTIONS_META_V2.strip()
        }
        r_resp = await _post_json(client, r_url, r_body, "Assistants run", req_id)
        run = r_resp.json()
        run_id = run["id"]
        status = run.get("status", "queued")
        logger.info("[%s] Assistants: run created id=%s (status=%s)", req_id, run_id, status)

        # 3) poll
        poll_url = f"{OPENAI_BASE_URL}/threads/{thread_id}/runs/{run_id}"
        start = time.monotonic()
        while status not in {"completed", "failed", "cancelled", "expired"}:
            if time.monotonic() - start > ASSISTANT_MAX_WAIT_SEC:
                raise RuntimeError("Assistant run timed out.")
            await asyncio.sleep(ASSISTANT_POLL_INTERVAL_SEC)
            logger.info("[%s] Assistants poll: GET %s", req_id, poll_url)
            poll = await client.get(poll_url, headers=HEADERS)
            poll.raise_for_status()
            run = poll.json()
            new_status = run.get("status", "unknown")
            if new_status != status:
                logger.info("[%s] Assistants: run status %s -> %s", req_id, status, new_status)
                status = new_status
            if status == "requires_action":
                raise RuntimeError("Assistant requires tool actions not enabled by this script.")

        if status != "completed":
            err_msg = (run.get("last_error") or {}).get("message") if isinstance(run.get("last_error"), dict) else None
            raise RuntimeError(f"Assistant run status: {status}. {err_msg or ''}".strip())

        # 4) messages
        m_url = f"{OPENAI_BASE_URL}/threads/{thread_id}/messages"
        logger.info("[%s] Assistants messages: GET %s?limit=20", req_id, m_url)
        m_resp = await client.get(m_url, headers=HEADERS, params={"limit": 20})
        m_resp.raise_for_status()
        msgs = m_resp.json()
        logger.info("[%s] Assistants: fetched messages", req_id)

    # 5) extract assistant text
    for m in msgs.get("data", []):
        if m.get("role") != "assistant":
            continue
        for c in m.get("content", []):
            if c.get("type") == "text" and "text" in c:
                content = (c["text"]["value"] or "").strip()
                if not content:
                    continue
                mcode = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", content, flags=re.IGNORECASE)
                if mcode:
                    content = mcode.group(1).strip()
                return content
    raise RuntimeError("Assistant returned no text.")

# ─────────────────────── batch ───────────────────────
def rows_to_process(in_dir: Path) -> List[Path]:
    return sorted(in_dir.glob("*.json"))

async def process_file(path: Path, out_dir: Path) -> Tuple[bool, str]:
    req_id = uuid.uuid4().hex[:8]
    logger.info("[%s] Processing metasample: %s", req_id, path.name)

    try:
        metasample = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("[%s] Failed to read/parse JSON: %s", req_id, e)
        return False, f"{path.name}: read/parse error: {e}"

    user_text = build_user_payload(metasample)
    logger.info("[%s] Request preview: base_url=%s key=%s len=%d", req_id, OPENAI_BASE_URL, _sanitize_key(OPENAI_API_KEY), len(OPENAI_API_KEY))
    logger.info("[%s] Payload sizes: user_text_len=%d", req_id, len(user_text))

    t0 = time.monotonic()
    for attempt in range(1, 5):
        try:
            content = await run_metabuilder(user_text, req_id)
            break
        except httpx.HTTPStatusError as e:
            logger.error("[%s] HTTP %s: %s", req_id, e.response.status_code, e.response.text[:400])
            if e.response.status_code in (401, 403, 404):
                return False, f"{path.name}: HTTP {e.response.status_code}"
            await asyncio.sleep(min(0.7 * attempt, 5.0))
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            logger.warning("[%s] Network error (%s). Retrying...", req_id, type(e).__name__)
            await asyncio.sleep(min(0.7 * attempt, 5.0))
        except Exception as e:
            logger.error("[%s] Assistant run failed: %s", req_id, e)
            return False, f"{path.name}: assistant error: {e}"

    try:
        obj = ensure_json(content)
    except Exception as e:
        (out_dir / (path.stem + "__ai_meta_raw.txt")).write_text(content, encoding="utf-8")
        logger.warning("[%s] Non-JSON output. Raw saved.", req_id)
        return False, f"{path.name}: non-JSON output: {e}"

    out_path = out_dir / (path.stem + "__ai_meta.json")
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[%s] Wrote %s (%.1fs)", req_id, out_path.name, time.monotonic() - t0)
    return True, f"OK → {out_path.name}"

async def main_async() -> None:
    _key_healthcheck()
    # Preflight: fail fast if key is bad
    await preflight_validate_key()

    in_dir = Path(CFG.get("METASAMPLE_DIR") or "metasample")
    out_dir = Path(CFG.get("OUTPUT_DIR") or "metaschema")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = rows_to_process(in_dir)
    if not files:
        logger.info("No input files in %s", in_dir.resolve()); return

    logger.info("Found %d metasample files.", len(files))
    ok = 0
    for p in files:
        success, msg = await process_file(p, out_dir)
        logger.info("%s -> %s", p.name, "✅" if success else f"⚠️ {msg}")
        if success:
            ok += 1
    logger.info("Done. Success: %d/%d", ok, len(files))

def main() -> None:
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
