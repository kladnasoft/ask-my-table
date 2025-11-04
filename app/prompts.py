# prompts.py
from __future__ import annotations

import json
import datetime as _dt
import decimal as _dec
import os
from typing import Any, Dict, List, Optional

# ==============================
# System / instruction blocks
# ==============================

IDENTIFY_INSTRUCTIONS = """You are a laser-focused schema selector.

Objective
- From a natural-language BI/analytics request, return ONLY the minimal set of tables and columns strictly necessary to answer it.

You will be given:
- CATALOG: an array of objects [{ "table": "table", "table_summary": "...", "columns": [{ "name": "col", "semantics": "..." }] }]
- Optional SAMPLES: { "table": { "examples": { "colName": ["v1","v2", ...] } } }

Your single output must be one JSON object (no markdown, no extra text):
{
  "tables": [ { "name": "table", "reason": "5-15 words max" } ],
  "columns": { "table": ["exactColA", "exactColB"] },
  "notes": "≤ 200 chars; optional clarifications or constraints"
}

Hard rules
- Choose the smallest viable subset. Do NOT include unused tables/columns.
- Only select columns that clearly map to the user request via name, semantics, or examples.
- Preserve exact casing and bare table names (no schema/database qualifiers).
- If the user asks for a field that does not exist, omit it silently; do NOT guess or rename.
- Prefer keys and filters (dates, ids, categories) that enable sargable predicates later.
- Keep reasons concise and actionable; avoid restating column names.

Quality bar
- Zero hallucinations. Zero invented names. Zero placeholder schemas.
"""

SQL_SYSTEM = """You are a senior SQL engineer targeting DuckDB/PostgreSQL syntax. Produce one and only one valid SELECT statement that answers the request using strictly the allowed schema payload.

Output format
- Return ONLY raw SQL text for a single SELECT statement. No comments, no markdown, no explanations.

Whitelist constraints
- Use ONLY tables and columns present in the provided "schema" payload (bare table names). Do NOT invent anything.
- Use clear table aliases and fully qualify base columns with those aliases in SELECT/WHERE/JOIN/ORDER/GROUP.
- Do NOT qualify tables with database or schema (no db.schema.table).

Single-statement + purity
- Exactly one SELECT. No temp tables, variables, multiple CTE statements, DDL/DML, MERGE, EXEC, INSERT/UPDATE/DELETE, hints, or USE.

Row limits
- If the result is potentially large or unbounded and the user did not specify limits, add:
    LIMIT 1000
  Never use TOP; it is not supported.

Joins & filters
- Use explicit INNER/LEFT JOIN with clear ON predicates. No CROSS/NATURAL joins.
- Write sargable predicates (e.g., CreatedAt >= CURRENT_DATE - INTERVAL '365 days').
- Do NOT wrap indexed date columns in functions inside WHERE when avoidable.

Date handling & grouping
- For date bucketing to day precision prefer:
    CAST(t.some_datetime AS DATE)
- If you bucket by day, ensure EVERY reference in SELECT/GROUP/ORDER uses the exact same CAST(... AS DATE).

Missing fields
- If the user requests unavailable fields/filters, produce the closest valid query using ONLY allowed columns (omit the unavailable parts without substitutes).

Ordering & nulls
- Provide deterministic ORDER BY when the request implies top/first/last/recent.
- Use COALESCE only when the user asks to replace NULLs.

Style
- Readable aliases (a, b, o, li, etc.).
- No trailing semicolons.

Your job ends at producing the single best SQL that satisfies the request under these constraints.
"""

EXPLAIN_INSTRUCTIONS = """You are a concise data storyteller.

You will receive:
- The executed SQL statement.
- A preview of the result rows as JSON.

Return ONLY one JSON object (no markdown):
{
  "explanation_html": "<p>≤200 words; plain HTML; crisp summary and key bullets if useful.</p>",
  "chart": {
    "type": "bar|line|pie|table",
    "x": "column_for_x_or_null",
    "y": "column_or_array_for_y_or_null",
    "agg": "sum|avg|count|max|min",
    "limit": 20,
    "rationale": "≤160 chars: why this chart"
  }
}

Rules
- If no obvious chart fits (e.g., many text columns or wide table), set type "table" and omit x/y/agg.
- Keep HTML simple and safe (no scripts/styles).
- Be accurate and avoid speculation.
"""

PLAN_SYSTEM_PROMPT = os.getenv(
    "PLAN_SYSTEM_PROMPT",
    "You are a meticulous data catalog planner. Extract relevant tables/columns from the provided catalog and produce a compact JSON plan."
)
SQL_SYSTEM_PROMPT = os.getenv(
    "SQL_SYSTEM_PROMPT",
    "You are a senior SQL developer. Generate a single safe SELECT statement strictly over the allowed schema (DuckDB compatible, use LIMIT not TOP)."
)
EXPLAIN_SYSTEM_PROMPT = os.getenv(
    "EXPLAIN_SYSTEM_PROMPT",
    "You explain query results clearly for business users and optionally propose a simple chart config in JSON."
)

# ==============================
# Serialization helpers
# ==============================

def _json_default(o: Any) -> Any:
    if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
        return o.isoformat()
    if isinstance(o, _dec.Decimal):
        return float(o)
    return str(o)

# ==============================
# Public prompt builders
# ==============================

def build_plan_prompt(
    user_text: str,
    catalog: List[Dict[str, Any]],
    samples: Optional[Dict[str, Any]] = None,
    *,
    intent_hint: Optional[str] = None,
    must_include_tables: Optional[List[str]] = None,
    must_exclude_tables: Optional[List[str]] = None,
) -> str:
    """
    Build the planning (schema selection) prompt.
    """
    payload = {
        "user_request": user_text,
        "intent_hint": intent_hint or "",
        "catalog": catalog,
        "samples": samples or {},
        "constraints": {
            "must_include_tables": must_include_tables or [],
            "must_exclude_tables": must_exclude_tables or [],
        },
    }
    return IDENTIFY_INSTRUCTIONS + "\n\n" + json.dumps(
        payload, ensure_ascii=False, default=_json_default
    )


def build_sql_prompt(
    user_text: str,
    allowed_schema: Dict[str, Any],
    feedback: str = "",
    values_ctx: Optional[Dict[str, List[str]]] = None,
    aux_context: Optional[Dict[str, Any]] = None,
    *,
    enforce_row_limit: int | None = None,
    prefer_recent_days: int | None = None,
) -> str:
    """
    Build the SQL generation prompt with optional extra guidance:
    - enforce_row_limit: force a LIMIT N if the user didn't specify a limit.
    - prefer_recent_days: nudge the model to filter recent data if appropriate (soft hint).
    - values_ctx: example filter values per column to keep predicates precise.
    - aux_context: tiny reference datasets to steer join/filter choices.
    """
    guidance_parts: List[str] = []

    if values_ctx:
        lines: List[str] = []
        for col, vals in values_ctx.items():
            uniq = ", ".join(sorted({str(v) for v in vals}))
            lines.append(f"{col} -> [{uniq}]")
        if lines:
            guidance_parts.append("Filter value guidance:\n" + "\n".join(lines))

    if aux_context:
        snippet = json.dumps(aux_context, ensure_ascii=False, default=_json_default)
        guidance_parts.append("Auxiliary context (compact reference):\n" + snippet[:2500])

    if enforce_row_limit and enforce_row_limit > 0:
        guidance_parts.append(
            f"Row limit policy: If unbounded, add 'LIMIT {enforce_row_limit}'. Never use TOP."
        )

    if prefer_recent_days and prefer_recent_days > 0:
        guidance_parts.append(
            f"Recency preference: When sensible, filter to the last {prefer_recent_days} days using sargable date predicates."
        )

    # Concise schema summary to discourage drift (bare table names)
    summary_lines: List[str] = []
    for t in allowed_schema.get("tables", []):
        tname = t.get("name")
        cols = t.get("columns", [])
        summary_lines.append(f"- {tname}: {', '.join(cols)}")
    schema_summary = "You may use ONLY these tables/columns (bare table names, no schema/database qualifiers):\n" + "\n".join(summary_lines)

    payload = {
        "user_request": user_text,
        "schema": allowed_schema,
        "notes": "Use ONLY listed base columns; derived expressions are allowed.",
    }

    prefix = f"{SQL_SYSTEM}\n\n{schema_summary}\n\n"
    if guidance_parts:
        prefix += "\n".join(guidance_parts) + "\n\n"

    fb = f"\n\nPrevious attempt feedback:\n{feedback.strip()}" if feedback else ""

    suffix = (
        "\n\nIMPORTANT REMINDERS:\n"
        "- Prefer CAST(col AS DATE) for day bucketing.\n"
        "- ORDER BY should precede LIMIT.\n"
        "- Never use TOP; use LIMIT N instead.\n"
    )

    return prefix + json.dumps(payload, ensure_ascii=False, default=_json_default) + fb + suffix


def build_explain_prompt(
    sql: str,
    columns: List[str],
    rows: List[Dict[str, Any]],
    max_rows: int = 40
) -> str:
    """
    Build the explanation prompt for result narration + simple viz suggestion.
    """
    sample_rows = rows[:max_rows]
    payload = {"sql": sql, "columns": columns, "rows_sample": sample_rows}
    return EXPLAIN_INSTRUCTIONS + "\n\n" + json.dumps(
        payload, ensure_ascii=False, default=_json_default
    )
