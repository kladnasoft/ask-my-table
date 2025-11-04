#!/usr/bin/env python3
"""
Reusable SuperTable REST connector.

Environment (dotenv supported):
- SUPERTABLE_URL                  (default: http://0.0.0.0:8000)
- SUPERTABLE_ADMIN_TOKEN          (required)
- SUPERTABLE_ORGANIZATION         (optional; auto-discovered if missing)
- SUPER_NAME                      (optional; auto-discovered if missing)
- SUPER_USER_HASH                 (optional; auto-discovered if missing)

Typical use:
    from _connect_supertable import SupertableClient

    with SupertableClient() as st:
        ctx = st.get_default_context()  # org, super_name, user_hash
        supers = st.list_supers(ctx.organization)
        tables = st.list_tables(ctx.organization, ctx.super_name)
        result = st.execute(
            "SELECT * FROM facts LIMIT 50",
            organization=ctx.organization,
            super_name=ctx.super_name,
            user_hash=ctx.user_hash,
        )
"""

from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

# ---------- Robust .env loading (optional) ----------
def _load_env():
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
        loaded_from = None

        # Walk up from current working directory first
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
            loaded_from = found

        # Also try script directory and parents
        script_dir = Path(__file__).resolve().parent
        for p in (script_dir / ".env", script_dir.parent / ".env", script_dir.parent.parent / ".env"):
            if p.is_file():
                load_dotenv(p.as_posix(), override=False)
                if not loaded_from:
                    loaded_from = p.as_posix()

        if loaded_from:
            print(f"[env] Loaded variables from: {loaded_from}")
        else:
            print("[env] No .env found; relying on process env.")
    except Exception as e:
        print(f"[env] dotenv not available or failed ({e}); relying on process env only.", file=sys.stderr)

_load_env()


@dataclass
class STContext:
    organization: str
    super_name: str
    user_hash: str


class STError(RuntimeError):
    pass


class SupertableClient:
    """
    Thin wrapper around the SuperTable REST API.
    Ensures admin cookie auth for every call.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        admin_token: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("SUPERTABLE_URL", "http://0.0.0.0:8000")).rstrip("/")
        self.admin_token = (admin_token or os.getenv("SUPERTABLE_ADMIN_TOKEN", "")).strip()
        self.session = session or requests.Session()
        self._logged_in = False

    # ---- context manager ----
    def __enter__(self) -> "SupertableClient":
        self.login()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.session.close()
        except Exception:
            pass

    # ---- auth guard for EVERY call ----
    def _ensure_auth(self) -> None:
        if not self._logged_in:
            self.login()

    def login(self) -> None:
        if self._logged_in:
            return
        if not self.admin_token:
            raise STError("SUPERTABLE_ADMIN_TOKEN is required to login")

        url = f"{self.base_url}/admin/login"
        resp = self.session.post(url, data={"token": self.admin_token}, allow_redirects=False)
        if resp.status_code not in (200, 302):
            raise STError(f"Login failed (HTTP {resp.status_code}): {resp.text}")

        # Ensure cookie is set (follow the redirect once if present)
        if "st_admin_token" not in self.session.cookies:
            if resp.is_redirect and "Location" in resp.headers:
                self.session.get(f"{self.base_url}{resp.headers['Location']}")
        if "st_admin_token" not in self.session.cookies:
            raise STError("Login did not set st_admin_token cookie (check SUPERTABLE_ADMIN_TOKEN)")

        self._logged_in = True

    # ---- discovery ----
    def tenants(self) -> Dict[str, Any]:
        self._ensure_auth()
        url = f"{self.base_url}/api/tenants"
        r = self.session.get(url)
        r.raise_for_status()
        return r.json()

    def pick_first_tenant(self) -> tuple[str, str]:
        data = self.tenants()
        tenants = data.get("tenants", [])
        if not tenants:
            raise STError("No tenants discovered via /api/tenants.")
        first = tenants[0]
        return first.get("org"), first.get("sup")

    # ---- API calls ----
    def list_supers(self, organization: str) -> List[str]:
        self._ensure_auth()
        qs = urlencode({"organization": organization})
        url = f"{self.base_url}/meta/supers?{qs}"
        r = self.session.get(url)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("ok") is False:
            raise STError(f"/meta/supers returned error: {json.dumps(data, ensure_ascii=False)}")
        return data.get("supers", []) if isinstance(data, dict) else data

    def list_tables(self, organization: str, super_name: str) -> List[str]:
        self._ensure_auth()
        qs = urlencode({"organization": organization, "super_name": super_name})
        url = f"{self.base_url}/meta/tables?{qs}"
        r = self.session.get(url)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("ok") is False:
            raise STError(f"/meta/tables returned error: {json.dumps(data, ensure_ascii=False)}")
        return data.get("tables", []) if isinstance(data, dict) else data

    def list_users(self, org: str, sup: str) -> List[Dict[str, Any]]:
        self._ensure_auth()
        qs = urlencode({"org": org, "sup": sup})
        url = f"{self.base_url}/api/users?{qs}"
        r = self.session.get(url)
        r.raise_for_status()
        data = r.json()
        return data.get("users", []) if isinstance(data, dict) else data

    # Optional convenience for schema/stats if you want them here too
    def get_table_schema(self, organization: str, super_name: str, table: str, user_hash: str) -> Dict[str, Any]:
        self._ensure_auth()
        qs = urlencode({"organization": organization, "super_name": super_name, "table": table, "user_hash": user_hash})
        url = f"{self.base_url}/meta/schema?{qs}"
        r = self.session.get(url)
        r.raise_for_status()
        return r.json()

    def get_table_stats(self, organization: str, super_name: str, table: str, user_hash: str) -> Dict[str, Any]:
        self._ensure_auth()
        qs = urlencode({"organization": organization, "super_name": super_name, "table": table, "user_hash": user_hash})
        url = f"{self.base_url}/meta/stats?{qs}"
        r = self.session.get(url)
        r.raise_for_status()
        return r.json()

    def execute(
        self,
        query: str,
        *,
        organization: str,
        super_name: str,
        user_hash: str,
        engine: Optional[str] = "DUCKDB",
        with_scan: bool = False,
        preview_rows: int = 10,
    ) -> Dict[str, Any]:
        self._ensure_auth()
        url = f"{self.base_url}/execute"
        payload = {
            "query": query,
            "organization": organization,
            "super_name": super_name,
            "user_hash": user_hash,
            "engine": engine,
            "with_scan": with_scan,
            "preview_rows": preview_rows,
        }
        r = self.session.post(url, json=payload)
        r.raise_for_status()
        return r.json()

    # ---- high-level convenience ----
    def get_default_context(
        self,
        organization: Optional[str] = None,
        super_name: Optional[str] = None,
        user_hash: Optional[str] = None,
    ) -> STContext:
        """
        Resolve organization, super_name, and user_hash:
        - If any are missing, auto-discover.
        - Env overrides: SUPERTABLE_ORGANIZATION, SUPER_NAME, SUPER_USER_HASH.
        """
        self._ensure_auth()

        env_org  = os.getenv("SUPERTABLE_ORGANIZATION") or organization
        env_sup  = os.getenv("SUPER_NAME") or super_name
        env_user = os.getenv("SUPER_USER_HASH") or user_hash

        if not env_org:
            env_org, sup_from_tenants = self.pick_first_tenant()
        else:
            sup_from_tenants = None

        supers = self.list_supers(env_org)
        sup = env_sup or sup_from_tenants or (supers[0] if supers else None)
        if not sup:
            raise STError("Could not determine SUPER_NAME (no supers returned).")

        if env_user:
            uhash = env_user
        else:
            users = self.list_users(env_org, sup)
            if not users or not users[0].get("hash"):
                raise STError("No users found for this tenant (need SUPER_USER_HASH).")
            uhash = users[0]["hash"]

        return STContext(organization=env_org, super_name=sup, user_hash=uhash)
