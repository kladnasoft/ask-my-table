# ask_my_table.py — SuperTable edition (DuckDB, bare tables, LIMIT)
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Optional

from dotenv import load_dotenv  # type: ignore
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from prompts import (
    build_plan_prompt,
    build_sql_prompt,
    build_explain_prompt,
    PLAN_SYSTEM_PROMPT,
    SQL_SYSTEM_PROMPT,
    EXPLAIN_SYSTEM_PROMPT,
)
from ai_chat_azure import chat_complete_async
from hot_cache import load_metaschema, build_catalog_brief, build_allowed_schema_json, get_hot_cache_list, find_meta_by_fqn

# SuperTable REST client
from _connect_supertable import SupertableClient, STContext

load_dotenv()
logger = logging.getLogger("ask-my-table")
logger.setLevel(logging.INFO)
router = APIRouter(tags=["ask-my-table"])

# ----------------------------- Config ------------------------------
class AppConfig(BaseModel):
    ai_provider: str = Field(default="azure_openai")

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_api_key: str | None = None

    # (Optional OpenAI keys preserved but unused unless you switch providers)
    openai_api_key: str | None = None
    openai_assistant_id: str | None = None
    openai_metabuilder_id: str | None = None
    openai_sqlgenerator_id: str | None = None
    openai_generic_id: str | None = None

    metaschema_dir: str = Field(default="metaschema")

    # SuperTable context (resolved at runtime if not given)
    supertable_url: str = Field(default="http://0.0.0.0:8000")
    query_max_rows: int = Field(default=1000, ge=1, le=100000)
    max_tables_plan: int = 120
    max_cols_per_table_plan: int = 100
    filter_probe_topn: int = 20
    sql_max_retries: int = Field(default=5, ge=1, le=10)

    # Compatibility with /settings/config UI (expected by settings.py)
    db_mode: str = Field(default="supertable", description="backend mode identifier for UI compatibility")


def _strip_trailing_slash(v: Optional[str]) -> Optional[str]:
    if v and v.endswith("/"):
        return v[:-1]
    return v


def load_config() -> AppConfig:
    cfg = AppConfig(
        ai_provider="azure_openai",
        azure_openai_endpoint=_strip_trailing_slash(os.getenv("AZURE_OPENAI_ENDPOINT")),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano-2"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_assistant_id=os.getenv("OPENAI_ASSISTANT_ID"),
        openai_metabuilder_id=os.getenv("OPENAI_METABUILDER_ID"),
        openai_sqlgenerator_id=os.getenv("OPENAI_SQLGENERATOR_ID"),
        openai_generic_id=os.getenv("OPENAI_GENERIC_ID"),
        metaschema_dir=os.getenv("METASCHEMA_DIR", os.getenv("OUTPUT_DIR", "metaschema")),
        supertable_url=os.getenv("SUPERTABLE_URL", "http://0.0.0.0:8000"),
        query_max_rows=int(os.getenv("QUERY_MAX_ROWS", "1000")),
        max_tables_plan=int(os.getenv("PLAN_MAX_TABLES", "120")),
        max_cols_per_table_plan=int(os.getenv("PLAN_MAX_COLS_PER_TABLE", "100")),
        filter_probe_topn=int(os.getenv("FILTER_PROBE_TOPN", "20")),
        sql_max_retries=int(os.getenv("SQL_MAX_RETRIES", "5")),
        db_mode=os.getenv("DB_MODE", "supertable"),
    )

    missing = [k for k in ["azure_openai_endpoint", "azure_openai_api_key"] if not getattr(cfg, k)]
    if missing:
        raise RuntimeError(f"Missing Azure OpenAI config: {', '.join(missing)}")

    return cfg


CONFIG = load_config()

# ------------------------- Metaschema loading ----------------------
METASCHEMA_BY_FQN, ALLOWED_TABLES = load_metaschema(CONFIG.metaschema_dir)

# ----------------------------- Trace store -------------------------
RUNS: Dict[str, Dict[str, Any]] = {}
RUN_LOCK = asyncio.Lock()


@dataclass
class TraceItem:
    ts: float
    t_ms: int
    dt_ms: int
    phase: str
    msg: str
    status: str = "info"


@dataclass
class Tracer:
    req_id: str
    t0: float = field(default_factory=lambda: time.time())
    items: List[TraceItem] = field(default_factory=list)

    def add(self, phase: str, msg: str, status: str = "info"):
        now = time.time()
        t_ms = int((now - self.t0) * 1000)
        dt_ms = t_ms - (self.items[-1].t_ms if self.items else 0)
        self.items.append(TraceItem(ts=now, t_ms=t_ms, dt_ms=dt_ms, phase=phase, msg=msg, status=status))
        asyncio.create_task(self._flush())

    def export(self) -> List[Dict[str, Any]]:
        return [{"ts": it.ts, "t_ms": it.t_ms, "dt_ms": it.dt_ms, "phase": it.phase, "msg": it.msg, "status": it.status} for it in self.items]

    async def _flush(self):
        async with RUN_LOCK:
            rec = RUNS.get(self.req_id)
            if not rec:
                RUNS[self.req_id] = {"status": "running", "trace": self.export(), "result": None, "short_status": _short_status_from_trace(self.items)}
            else:
                rec["trace"] = self.export()
                rec["short_status"] = _short_status_from_trace(self.items)


def _short_status_from_trace(items: List[TraceItem]) -> str:
    if not items:
        return "starting…"
    lastp = items[-1].phase.upper()
    if lastp.startswith("PLAN"):
        return "analyzing metadata…"
    if lastp.startswith("FILTER_PROBE"):
        return "checking filter values…"
    if lastp.startswith("SQL_ATTEMPT") or lastp.startswith("SQL_GEN") or lastp == "SQL":
        return "generating SQL…"
    if lastp.startswith("SQL_VALIDATE"):
        return "validating SQL…"
    if lastp.startswith("SQL_EXEC"):
        return "executing SQL…"
    if lastp.startswith("EXPLAIN"):
        return "preparing explanation…"
    return "working…"

# ----------------------------- Models -----------------------------
class DataChatRequest(BaseModel):
    text: str


class DataChatResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    meta: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    explanation_html: Optional[str] = None
    chart: Optional[Dict[str, Any]] = None
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    short_status: str = "completed"
    total_duration_ms: int = 0
    attempts: int = 1


class StartResponse(BaseModel):
    req_id: str
    status: str = "started"


# ----------------------------- ST client helpers -------------------
_ST_CLIENT: Optional[SupertableClient] = None
_ST_CTX: Optional[STContext] = None


def _ensure_st_client(tracer: Optional[Tracer] = None) -> STContext:
    global _ST_CLIENT, _ST_CTX
    if _ST_CLIENT is None:
        _ST_CLIENT = SupertableClient(base_url=CONFIG.supertable_url)
        if tracer:
            tracer.add("SQL_EXEC", f"connecting to SuperTable at {CONFIG.supertable_url}")
    if _ST_CTX is None:
        with _ST_CLIENT as st:
            _ST_CTX = st.get_default_context()
    return _ST_CTX  # type: ignore[return-value]


def sanitize_llm_sql(sql: str) -> str:
    """
    Keep it as a single SELECT; avoid semicolons and dangerous tokens.
    """
    s = (sql or "").strip()
    # Strip code fences if any
    m = re.match(r"^```(?:sql)?\s*([\s\S]*?)\s*```$", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    # One statement heuristic: forbid ';' and DDL/DML keywords
    forbidden = re.compile(r"\b(?:insert|update|delete|merge|drop|create|alter|grant|revoke|truncate|use)\b", re.I)
    if ";" in s or forbidden.search(s):
        raise HTTPException(status_code=400, detail="Unsafe SQL detected.")
    if not re.match(r"(?is)^\s*select\b", s):
        raise HTTPException(status_code=400, detail="Only SELECT statements are allowed.")
    return s


def _table_tokens(sql: str) -> List[str]:
    """
    Extract raw table tokens that immediately follow FROM / JOIN.
    It skips subqueries starting with '('.
    """
    tokens: List[str] = []
    for m in re.finditer(r"(?is)\b(from|join)\s+([^\s,()]+)", sql):
        tok = m.group(2).strip()
        if not tok or tok.startswith("("):
            continue
        tokens.append(tok)
    return tokens


def _is_schema_qualified_table_token(tok: str) -> bool:
    """
    Heuristic: in FROM/JOIN tokens, any dot indicates a qualified table
    (schema.table or db.schema.table). We allow bare table names only.
    Handles bracketed/quoted identifiers as the dot remains present.
    """
    s = tok.strip()
    return "." in s


def validate_sql(sql: str) -> None:
    """
    Validates the SQL for Supertable/DuckDB target.
    - Only SELECT (handled in sanitize_llm_sql)
    - Disallow schema- or database-qualified TABLE names in FROM/JOIN
      (but allow alias-qualified columns like a.column elsewhere).
    - Disallow TOP keyword; require LIMIT for row-limiting.
    """
    _ = sanitize_llm_sql(sql)

    # Reject TOP usage (e.g., SELECT TOP 10 ... or SELECT DISTINCT TOP (10) ...)
    if re.search(r"(?is)\bselect\s+(?:distinct\s+)?top\b", sql):
        raise HTTPException(
            status_code=400,
            detail="Use LIMIT, not TOP, for Supertable/DuckDB."
        )

    # Inspect table references only (FROM / JOIN)
    for tok in _table_tokens(sql):
        if _is_schema_qualified_table_token(tok):
            raise HTTPException(
                status_code=400,
                detail="Do not qualify tables with schema/database for Supertable."
            )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        # Try raw JSON
        return json.loads(text)
    except Exception:
        pass
    # Try fenced block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

# ----------------------------- Execution ---------------------------
def execute_select_supertable(sql: str, req_id: str, tracer: Optional[Tracer] = None) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Run SQL via SuperTable /execute (DuckDB). Returns (columns, rows[dict]).
    We request up to QUERY_MAX_ROWS and expect 'rows_preview'.
    """
    ctx = _ensure_st_client(tracer)
    preview = max(1, min(CONFIG.query_max_rows, 100000))

    if tracer:
        tracer.add("SQL_EXEC", f"posting to /execute preview_rows={preview}")

    res = _ST_CLIENT.execute(
        query=sql,
        organization=ctx.organization,
        super_name=ctx.super_name,
        user_hash=ctx.user_hash,
        engine="DUCKDB",
        with_scan=False,
        preview_rows=preview,
    )

    columns = res.get("columns") or []
    matrix = res.get("rows_preview") or res.get("rows") or []

    rows = [{columns[i]: row[i] for i in range(len(columns))} for row in matrix]

    if tracer:
        shape = res.get("shape") or [len(matrix), len(columns)]
        tracer.add("SQL_EXEC", f"received rows={len(rows)} cols={len(columns)} shape={shape}")

    return columns, rows

# ----------------------------- Pipeline ---------------------------
async def run_pipeline(req_id: str, text: str):
    tracer = Tracer(req_id=req_id)
    async with RUN_LOCK:
        RUNS[req_id] = {"status": "running", "trace": [], "result": None, "short_status": "starting…", "attempts": 0}

    tracer.add("RECV", f"text={text!r}")

    # 1) PLAN — catalog brief -> plan
    try:
        tracer.add("PLAN", "building catalog brief")
        catalog = build_catalog_brief(METASCHEMA_BY_FQN, CONFIG.max_tables_plan, CONFIG.max_cols_per_table_plan)
        plan_prompt = build_plan_prompt(text, catalog)
        tracer.add("PLAN", "using deployment (Azure Chat)")
        plan_text = await chat_complete_async(system_prompt=PLAN_SYSTEM_PROMPT, user_content=plan_prompt, tracer=tracer, label="PLAN")
        plan_json = _extract_json(plan_text) or {}
        tracer.add("PLAN", f"plan keys={list(plan_json.keys())}")
    except Exception as exc:
        tracer.add("PLAN", f"plan failed: {exc}", status="error")
        plan_json = {}

    # 2) Resolve selected tables (BARE table names only)
    selected_tables: List[str] = []
    if "tables" in plan_json and isinstance(plan_json["tables"], list):
        for t in plan_json["tables"]:
            name = (t.get("name") if isinstance(t, dict) else None) or ""
            if name:
                selected_tables.append(name.strip())
    if not selected_tables:
        tracer.add("PLAN", "no tables selected; falling back to all")
        selected_tables = sorted({
            (m.get("identity", {}).get("table") or "").strip()
            for m in METASCHEMA_BY_FQN.values()
            if (m.get("identity", {}).get("table") or "").strip()
        })

    # 3) Allowed schema (bare tables) and SQL generation
    allowed_schema = build_allowed_schema_json(METASCHEMA_BY_FQN, selected_tables)

    attempts = 0
    last_error = ""
    sql = ""
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []

    while attempts < CONFIG.sql_max_retries:
        attempts += 1
        async with RUN_LOCK:
            RUNS[req_id]["attempts"] = attempts
        tracer.add("SQL_ATTEMPT", f"attempt {attempts}")

        feedback = ""
        if last_error or sql:
            feedback = f"Previous SQL:\n{sql}\n\nError:\n{last_error}"

        try:
            # Enforce LIMIT N policy for unbounded queries
            prompt = build_sql_prompt(
                text,
                allowed_schema,
                feedback=feedback,
                enforce_row_limit=CONFIG.query_max_rows
            )
            tracer.add("SQL_GEN", "provider=azure_openai")
            tracer.add("SQL_GEN", "using deployment (Azure Chat)")
            text_out = await chat_complete_async(system_prompt=SQL_SYSTEM_PROMPT, user_content=prompt, tracer=tracer, label="SQL")

            m = re.match(r"^```(?:sql)?\s*([\s\S]*?)\s*```$", text_out, flags=re.IGNORECASE)
            sql = m.group(1).strip() if m else text_out.strip()

            sql = sanitize_llm_sql(sql)
            tracer.add("SQL_VALIDATE", "validating SQL…")
            validate_sql(sql)

            tracer.add("SQL_EXEC", "executing…")
            columns, rows = execute_select_supertable(sql, req_id, tracer=tracer)
            break  # success
        except HTTPException as exc:
            last_error = str(exc.detail)
            tracer.add("SQL_VALIDATE", f"validation failed: {last_error}", status="error")
            if attempts >= CONFIG.sql_max_retries:
                total_ms = tracer.items[-1].t_ms if tracer.items else 0
                resp = DataChatResponse(
                    sql=sql, columns=[], rows=[],
                    meta=None, plan=plan_json or None,
                    explanation_html=f"<p><strong>SQL validation failed after {attempts} attempts:</strong><br/>{last_error}</p>",
                    chart=None,
                    trace=tracer.export(), short_status="sql validation failed", total_duration_ms=total_ms, attempts=attempts
                )
                async with RUN_LOCK:
                    RUNS[req_id]["status"] = "completed"
                    RUNS[req_id]["result"] = resp.model_dump()
                    RUNS[req_id]["short_status"] = "sql validation failed"
                return
            continue
        except Exception as exc:
            last_error = str(exc)
            tracer.add("SQL_VALIDATE", f"validation/exec exception: {last_error}", status="error")
            if attempts >= CONFIG.sql_max_retries:
                total_ms = tracer.items[-1].t_ms if tracer.items else 0
                resp = DataChatResponse(
                    sql=sql, columns=[], rows=[],
                    meta=None, plan=plan_json or None,
                    explanation_html=f"<p><strong>SQL generation failed after {attempts} attempts:</strong><br/>{last_error}</p>",
                    chart=None,
                    trace=tracer.export(),
                    short_status="sql generation failed",
                    total_duration_ms=total_ms,
                    attempts=attempts,
                )
                async with RUN_LOCK:
                    RUNS[req_id]["status"] = "completed"
                    RUNS[req_id]["result"] = resp.model_dump()
                    RUNS[req_id]["short_status"] = "sql generation failed"
                return
            continue

    # 4) EXPLAIN
    explanation_html: Optional[str] = None
    chart: Optional[Dict[str, Any]] = None
    try:
        tracer.add("EXPLAIN", "building explanation…")
        explain_prompt = build_explain_prompt(sql, columns, rows)
        tracer.add("EXPLAIN", "using deployment (Azure Chat)")
        explain_text = await chat_complete_async(system_prompt=EXPLAIN_SYSTEM_PROMPT, user_content=explain_prompt, tracer=tracer, label="EXPLAIN")
        explain_json = _extract_json(explain_text) or {}
        explanation_html = (explain_json.get("explanation_html") or "").strip() or None
        chart = explain_json.get("chart") if isinstance(explain_json.get("chart"), dict) else None
        tracer.add("EXPLAIN", "explanation ready")
    except Exception as exc:
        tracer.add("EXPLAIN", f"explain failed: {exc}", status="error")
        explanation_html = "<p>Query executed. (Explanation step failed.)</p>"
        chart = None

    total_ms = tracer.items[-1].t_ms if tracer.items else 0
    resp = DataChatResponse(
        sql=sql,
        columns=columns,
        rows=rows,
        meta=None,
        plan=plan_json or None,
        explanation_html=explanation_html,
        chart=chart,
        trace=tracer.export(),
        short_status="completed",
        total_duration_ms=total_ms,
        attempts=attempts,
    )
    async with RUN_LOCK:
        RUNS[req_id]["status"] = "completed"
        RUNS[req_id]["result"] = resp.model_dump()
        RUNS[req_id]["short_status"] = "completed"

# ----------------------------- Routes -----------------------------
@router.post("/ask-my-table/start")
async def data_chat_start(req: DataChatRequest):
    req_id = uuid.uuid4().hex[:8]
    async with RUN_LOCK:
        RUNS[req_id] = {"status": "running", "trace": [], "result": None, "short_status": "starting…", "attempts": 0}
    asyncio.create_task(run_pipeline(req_id, req.text))
    return {"req_id": req_id, "status": "started"}


@router.get("/ask-my-table/progress/{req_id}")
async def data_chat_progress(req_id: str):
    async with RUN_LOCK:
        rec = RUNS.get(req_id)
        if not rec:
            raise HTTPException(status_code=404, detail="req_id not found")
        trace = rec.get("trace", [])
        status = rec.get("status", "running")
        short_status = rec.get("short_status", "working…")
        attempts = rec.get("attempts", 0)
    total_ms = trace[-1]["t_ms"] if trace else 0
    return {"req_id": req_id, "status": status, "short_status": short_status, "trace": trace, "attempts": attempts, "total_duration_ms": total_ms}


@router.get("/ask-my-table/result/{req_id}")
async def data_chat_result(req_id: str):
    async with RUN_LOCK:
        rec = RUNS.get(req_id)
        if not rec:
            raise HTTPException(status_code=404, detail="req_id not found")
        if rec.get("status") != "completed" or not rec.get("result"):
            raise HTTPException(status_code=425, detail="result not ready")
        data = rec["result"]
    return data


@router.post("/ask-my-table")
async def data_chat(req: DataChatRequest):
    rid = uuid.uuid4().hex[:8]
    await run_pipeline(rid, req.text)
    async with RUN_LOCK:
        data = RUNS[rid]["result"]
    return data


@router.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def get_public_status() -> Dict[str, Any]:
    # Present bare table names in status for Supertable clarity
    tables: Set[str] = set()
    for meta in METASCHEMA_BY_FQN.values():
        ident = meta.get("identity", {})
        tbl = (ident.get("table") or "").strip()
        if tbl:
            tables.add(tbl)

    # Surface resolved SuperTable context if available
    try:
        ctx = _ensure_st_client()
        org = ctx.organization
        sup = ctx.super_name
        user = ctx.user_hash
    except Exception:
        org = os.getenv("SUPERTABLE_ORGANIZATION") or "-"
        sup = os.getenv("SUPER_NAME") or "-"
        user = os.getenv("SUPER_USER_HASH") or "-"

    return {
        "ai_provider": CONFIG.ai_provider,
        "backend": "supertable",
        "db_mode": CONFIG.db_mode,
        "supertable_url": CONFIG.supertable_url,
        "organization": org,
        "super_name": sup,
        "user_hash": user,
        "allowed_tables": sorted(tables),
        "query_max_rows": CONFIG.query_max_rows,
        "metaschema_dir": CONFIG.metaschema_dir,
        "metas_loaded": len(tables),
        "sql_max_retries": CONFIG.sql_max_retries,
    }


# ----------------------------- UI Routes -----------------------------
POLL_INTERVAL_MS = int(os.getenv("UI_POLL_INTERVAL_MS", "800"))
_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "ask.html")


def _load_template() -> str:
    """
    Load the ask.html template from ./templates and inject minimal dynamic values.
    """
    try:
        with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        return "<html><body><h1>Template not found</h1><p>Expected at: {}</p></body></html>".format(_TEMPLATE_PATH)
    # Simple placeholder replacement
    html = html.replace("{{POLL_INTERVAL_MS}}", str(POLL_INTERVAL_MS))
    return html


@router.get("/ui")
def ui_page():
    return HTMLResponse(_load_template())


# ----------------------------- Hot Cache Routes -----------------------------
@router.get("/ask-my-table/cache")
def get_cache() -> Dict[str, Any]:
    tables = get_hot_cache_list(METASCHEMA_BY_FQN)
    return {
        "metaschema_dir": CONFIG.metaschema_dir,
        "count": len(tables),
        "tables": tables,
    }


@router.post("/ask-my-table/cache/reload")
def reload_cache() -> Dict[str, Any]:
    global METASCHEMA_BY_FQN, ALLOWED_TABLES
    METASCHEMA_BY_FQN, ALLOWED_TABLES = load_metaschema(CONFIG.metaschema_dir)
    return get_cache()


@router.get("/ask-my-table/cache/item")
def get_cache_item(fqn: str) -> Dict[str, Any]:
    meta = find_meta_by_fqn(METASCHEMA_BY_FQN, fqn)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Cache item not found: {fqn}")
    # Provide a little helpful envelope
    ident = meta.get("identity", {})
    return {
        "fqn": fqn,
        "schema": ident.get("schema"),
        "table": ident.get("table"),
        "meta": meta,
    }