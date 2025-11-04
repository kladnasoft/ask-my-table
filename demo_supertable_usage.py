#!/usr/bin/env python3
"""
Demo script showing how to use _connect_supertable.SupertableClient.

It reproduces the same flow you showed:
- list supers
- list tables for the chosen super
- pick/get user_hash
- execute "SELECT * FROM facts LIMIT 50"
"""
import json
from _connect_supertable import SupertableClient

def pretty(o): print(json.dumps(o, ensure_ascii=False, indent=2))

def main():
    with SupertableClient() as st:
        # resolve org/super/user_hash (env or auto-discover)
        ctx = st.get_default_context()
        print("\n== Supers ==")
        pretty(st.list_supers(ctx.organization))

        print(f"\n[info] Using SUPER_NAME={ctx.super_name}")

        print("\n== Tables ==")
        pretty(st.list_tables(ctx.organization, ctx.super_name))

        print(f"\n[info] Using SUPER_USER_HASH={ctx.user_hash}")

        print(f"\n== Executing: SELECT * FROM facts LIMIT 50 ==")
        res = st.execute(
            "SELECT * FROM facts LIMIT 50",
            organization=ctx.organization,
            super_name=ctx.super_name,
            user_hash=ctx.user_hash,
            engine="DUCKDB",
            with_scan=False,
            preview_rows=10,
        )
        pretty(res)

if __name__ == "__main__":
    main()
