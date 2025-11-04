# hot_cache.py
from __future__ import annotations

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set


def _norm(s: str) -> str:
    return re.sub(r'[\[\]\s"]', "", (s or "")).lower()


def _fqns(identity: Dict[str, Any]) -> List[str]:
    """
    Build lookup variants. We keep db.schema.table variants for matching,
    but downstream (catalog + allowed schema) we pass ONLY the bare table
    name to ensure Supertable queries don't include a database/schema.
    """
    db = (identity.get("database") or "").strip()
    sch = (identity.get("schema") or "").strip()
    tbl = (identity.get("table") or "").strip()
    names: List[str] = []
    if sch and tbl:
        names += [f"{sch}.{tbl}", f"[{sch}].[{tbl}]"]
    if db and sch and tbl:
        names += [f"{db}.{sch}.{tbl}", f"[{db}].[{sch}].[{tbl}]"]
    # also include bare table for matching convenience
    if tbl:
        names += [tbl, f"[{tbl}]"]
    return names


def _load_one_meta(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def load_metaschema(dir_path: str) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        raise RuntimeError(f"Metaschema directory not found: {dir_path}")

    metas_by_fqn: Dict[str, Dict[str, Any]] = {}
    allowed_variants: Set[str] = set()

    files = [p for p in os.listdir(dir_path) if p.endswith(".json")]
    for fn in sorted(files):
        meta = _load_one_meta(os.path.join(dir_path, fn))
        if not meta:
            continue
        ident = meta.get("identity", {})
        for v in _fqns(ident):
            metas_by_fqn[_norm(v)] = meta
            allowed_variants.add(_norm(v))

    return metas_by_fqn, allowed_variants


def build_catalog_brief(metas_by_fqn: Dict[str, Dict[str, Any]], max_tables: int, max_cols_per_table: int) -> List[Dict[str, Any]]:
    """
    Build the catalog the planner sees. IMPORTANT: only bare table names.
    """
    items: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for meta in metas_by_fqn.values():
        ident = meta.get("identity", {})
        tbl = (ident.get("table") or "").strip()
        if not tbl or tbl in seen:
            continue
        seen.add(tbl)
        cols_src = meta.get("columns") or []
        cols = [{"name": c.get("name"), "semantics": c.get("semantics")} for c in cols_src[:max_cols_per_table]]
        items.append({
            "table": tbl,
            "table_summary": ident.get("table_summary") or "",
            "columns": cols
        })
        if len(items) >= max_tables:
            break
    return items


def build_allowed_schema_json(metas_by_fqn: Dict[str, Dict[str, Any]], selected_tables: List[str]) -> Dict[str, Any]:
    """
    Allowed schema passed to the SQL generator. Bare table names only.
    """
    selected_norm = {t.strip().lower() for t in selected_tables if t}
    out_tables: List[Dict[str, Any]] = []
    for meta in metas_by_fqn.values():
        ident = meta.get("identity", {}) or {}
        tbl = (ident.get("table") or "").strip()
        if not tbl or tbl.lower() not in selected_norm:
            continue
        cols_meta = meta.get("columns") or []
        cols = [c.get("name") for c in cols_meta if c.get("name")]
        types = {c.get("name"): c.get("sql_type") for c in cols_meta if c.get("name")}
        out_tables.append({"name": tbl, "columns": cols, "types": types})
    return {"tables": out_tables}


def _fqn_from_identity(meta: Dict[str, Any]) -> Optional[str]:
    ident = meta.get("identity") or {}
    sch = (ident.get("schema") or "").strip()
    tbl = (ident.get("table") or "").strip()
    if sch and tbl:
        return f"{sch}.{tbl}"
    return None


def get_hot_cache_list(metas_by_fqn: Dict[str, Dict[str, Any]]) -> List[str]:
    tables: set[str] = set()
    for meta in metas_by_fqn.values():
        fqn = _fqn_from_identity(meta)
        if fqn:
            tables.add(fqn)
    return sorted(tables)


def _split_fqn(fqn: str) -> Tuple[str, str]:
    if "." not in fqn:
        return ("", fqn.strip())
    a, b = fqn.split(".", 1)
    return (a.strip(), b.strip())


def find_meta_by_fqn(metas_by_fqn: Dict[str, Dict[str, Any]], fqn: str) -> Optional[Dict[str, Any]]:
    """
    Try to find meta by either:
    - dict key equal to fqn
    - identity.schema + identity.table equal to fqn
    """
    # direct key match
    dct: Dict[str, Any] = metas_by_fqn
    if fqn in dct:
        return dct[fqn]
    sch, tbl = _split_fqn(fqn)
    sch_l, tbl_l = sch.lower(), tbl.lower()
    for meta in dct.values():
        ident = meta.get("identity") or {}
        s = str(ident.get("schema") or "").strip().lower()
        t = str(ident.get("table") or "").strip().lower()
        if s == sch_l and t == tbl_l and s and t:
            return meta
    return None