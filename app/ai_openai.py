# ai_openai.py
from __future__ import annotations

import os
import re
import time
import asyncio
from typing import Any

import httpx
from fastapi import HTTPException

ASSISTANT_POLL_INTERVAL_SEC = float(os.getenv("ASSISTANT_POLL_INTERVAL_SEC", "0.8"))
ASSISTANT_MAX_WAIT_SEC = float(os.getenv("ASSISTANT_MAX_WAIT_SEC", "300.0"))
ASSISTANT_HTTP_TIMEOUT = float(os.getenv("ASSISTANT_HTTP_TIMEOUT", "300.0"))


def _redact(s: str) -> str:
    if not s:
        return s
    return re.sub(r"sk-(proj|live|test|key)[A-Za-z0-9\-_]*", "sk-***REDACTED***", s)


async def assistant_complete(
    prompt: str,
    req_id: str,
    assistant_id: str,
    label: str,
    *,
    api_key: str,
    tracer=None,
) -> str:
    """
    Extracted Assistant API flow (v2): create thread, run, poll, fetch messages.
    """
    base_url = "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    timeout = httpx.Timeout(
        ASSISTANT_HTTP_TIMEOUT,
        connect=30.0,
        read=ASSISTANT_HTTP_TIMEOUT,
        write=90.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        if tracer:
            tracer.add(label.upper(), f"creating thread… (assistant={assistant_id})")
            tracer.add(label.upper(), f"request preview: {_redact(prompt)[:800]}…")

        t_resp = await client.post(
            f"{base_url}/threads",
            headers=headers,
            json={"messages": [{"role": "user", "content": prompt}]},
        )
        if tracer:
            tracer.add(label.upper(), f"thread response {t_resp.status_code}")
        t_resp.raise_for_status()
        thread_id = t_resp.json()["id"]

        if tracer:
            tracer.add(label.upper(), "starting run…")
        r_resp = await client.post(
            f"{base_url}/threads/{thread_id}/runs",
            headers=headers,
            json={"assistant_id": assistant_id},
        )
        if tracer:
            tracer.add(label.upper(), f"run response {r_resp.status_code}")
        r_resp.raise_for_status()
        run = r_resp.json()
        run_id = run["id"]
        status = run.get("status", "queued")

        start = time.monotonic()
        while status not in {"completed", "failed", "cancelled", "expired"}:
            if time.monotonic() - start > ASSISTANT_MAX_WAIT_SEC:
                raise HTTPException(status_code=504, detail=f"Assistant[{label}] timed out.")
            await asyncio.sleep(ASSISTANT_POLL_INTERVAL_SEC)
            poll = await client.get(f"{base_url}/threads/{thread_id}/runs/{run_id}", headers=headers)
            poll.raise_for_status()
            run = poll.json()
            new_status = run.get("status", "unknown")
            if new_status != status:
                if tracer:
                    tracer.add(label.upper(), f"run {status} -> {new_status}")
                status = new_status

        if status != "completed":
            err = (run.get("last_error") or {}).get("message", "").strip()
            raise HTTPException(
                status_code=502,
                detail=f"Assistant[{label}] status={status}. {('Error: '+err) if err else ''}".strip(),
            )

        m_resp = await client.get(
            f"{base_url}/threads/{thread_id}/messages",
            headers=headers,
            params={"limit": 20},
        )
        m_resp.raise_for_status()
        msgs = m_resp.json()

    out = ""
    for m in msgs.get("data", []):
        if m.get("role") != "assistant":
            continue
        for c in m.get("content", []):
            if c.get("type") == "text" and "text" in c:
                out = (c["text"]["value"] or "").strip()
                break
        if out:
            break

    if tracer:
        tracer.add(label.upper(), f"response preview: {_redact(out)[:600]}…")
    if not out:
        raise HTTPException(status_code=502, detail=f"Assistant[{label}] returned no text.")
    return out
