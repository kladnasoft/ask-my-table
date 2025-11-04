# main.py
from __future__ import annotations

import hmac
import logging
import os
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)

from ui import router as ui_router
# Dynamically load the hyphenated module file "ask_my_table.py"
import importlib.util, types, os as _os, sys as _sys
_amt_path = _os.path.join(_os.path.dirname(__file__), "ask_my_table.py")
_spec = importlib.util.spec_from_file_location("ask_my_table_mod", _amt_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to locate ask_my_table.py for import")
_mod = importlib.util.module_from_spec(_spec)
_sys.modules[_spec.name] = _mod  # ensure module is registered for decorators/dataclasses
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
# Expose router and helper for FastAPI include_router
ask_my_table_router = getattr(_mod, "router")
get_public_status = getattr(_mod, "get_public_status")
from settings import router as settings_router

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="Ask My Table", version="1.0.0")

# CORS (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Token auth utilities
# ------------------------------------------------------------------------------
def _get_valid_tokens() -> List[str]:
    """
    Read UI access tokens from env:
      UI_ACCESS_TOKEN="token1, token2"
    Separate by commas or semicolons.
    """
    raw = os.getenv("UI_ACCESS_TOKEN", "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _extract_token(request: Request) -> Optional[str]:
    """
    Accept token via:
      - Authorization: Bearer <token>
      - cookie: ui_token
      - querystring: ?token=<token>
    """
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip() or None

    cookie = request.cookies.get("ui_token")
    if cookie:
        return cookie

    q = request.query_params.get("token")
    if q:
        return q

    return None


class AuthError(HTTPException):
    pass


@app.exception_handler(AuthError)
async def _auth_exception_handler(request: Request, exc: AuthError):
    # If browser requests HTML, redirect to /login (preserve next)
    accept = request.headers.get("accept", "")
    if "text/html" in accept.lower():
        next_url = request.url.path
        if request.url.query:
            next_url += f"?{request.url.query}"
        return RedirectResponse(url=f"/login?next={next_url}", status_code=302)
    # Otherwise return JSON error
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail or "Unauthorized"}
    )


async def require_token(request: Request):
    """
    Hard-enforced: a valid UI_ACCESS_TOKEN **must** be configured and presented.
    Public exceptions: /, /healthz, /login, /logout are open.
    """
    valid = _get_valid_tokens()
    if not valid:
        # Strict mode: refuse if not configured (prevents accidental public exposure)
        raise AuthError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="UI_ACCESS_TOKEN not configured",
        )

    token = _extract_token(request)
    if not token:
        raise AuthError(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token"
        )

    # Constant-time compare
    ok = any(hmac.compare_digest(token, v) for v in valid)
    if not ok:
        raise AuthError(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


# ------------------------------------------------------------------------------
# Public endpoints (no auth)
# ------------------------------------------------------------------------------
@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/login", status_code=302)


@app.get("/healthz", response_class=PlainTextResponse)
def health():
    return "ok"


@app.get("/login", response_class=HTMLResponse)
async def login_form(next: str = "/ui", error: str | None = None):
    cfg = "configured" if _get_valid_tokens() else "NOT CONFIGURED"
    err_html = (
        '<div style="margin:10px 0; padding:10px; border-radius:8px; '
        'background:#3b0d0d; border:1px solid #7f1d1d; color:#fecaca;">'
        "Invalid token. Please try again."
        "</div>"
        if error
        else ""
    )
    return HTMLResponse(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Login</title>
<style>
  body{{font-family:ui-sans-serif,system-ui; background:#0b0f14; color:#e5e7eb; display:grid; place-items:center; min-height:100vh; margin:0}}
  .card{{background:#111827; border:1px solid #1f2937; border-radius:14px; padding:22px; width:min(520px,92vw);
         box-shadow:0 10px 30px rgba(0,0,0,.35)}}
  label{{display:block; margin-bottom:8px; font-weight:700}}
  input{{width:100%; padding:10px 12px; border-radius:10px; border:1px solid #253349; background:#0e2137; color:#e5e7eb}}
  .row{{display:flex; gap:10px; align-items:center; margin-top:12px;}}
  .btn{{appearance:none; border:1px solid #294058; border-radius:10px; padding:10px 16px; background:#0b2540; color:#d7eaff; font-weight:800; cursor:pointer}}
  .tiny{{font-size:12px; color:#9ca3af}}
  .warn{{margin-top:10px; font-size:12px; color:#f59e0b}}
</style></head>
<body>
  <form class="card" method="post" action="/login">
    <h2 style="margin:0 0 12px 0;">Sign in</h2>
    <div class="tiny" style="margin-bottom:8px;">Security: UI_ACCESS_TOKEN is <b>{cfg}</b></div>
    {'<div class="warn">You must set UI_ACCESS_TOKEN in environment to enable access.</div>' if cfg == 'NOT CONFIGURED' else ''}
    {err_html}
    <input type="hidden" name="next" value="{next}">
    <label for="token">Access token</label>
    <input id="token" name="token" type="password" placeholder="Paste your access token" autofocus required>
    <div class="row" style="justify-content:space-between;">
      <span class="tiny">You can also pass <code>?token=…</code> or use <code>Authorization: Bearer …</code></span>
      <button class="btn" type="submit">Continue</button>
    </div>
  </form>
</body></html>"""
    )


@app.post("/login")
async def login_post(token: str = Form(...), next: str = Form("/ui")):
    valid = _get_valid_tokens()
    if not valid:
        # Refuse logins until configured (strict mode)
        return RedirectResponse(url="/login?error=1", status_code=302)

    if not any(hmac.compare_digest(token, v) for v in valid):
        return RedirectResponse(url="/login?error=1", status_code=302)

    resp = RedirectResponse(url=next or "/ui", status_code=302)
    # Secure cookie if TLS_ENABLED=true or COOKIE_SECURE=true
    secure_cookie = (
        os.getenv("COOKIE_SECURE", "auto").lower() == "true"
        or (
            os.getenv("COOKIE_SECURE", "auto").lower() == "auto"
            and os.getenv("TLS_ENABLED", "").lower() == "true"
        )
    )
    resp.set_cookie(
        "ui_token",
        token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )
    return resp


@app.post("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("ui_token")
    return resp


# ------------------------------------------------------------------------------
# Protected routers (UI, data APIs, settings)
# ------------------------------------------------------------------------------
app.include_router(ui_router, dependencies=[Depends(require_token)])
app.include_router(ask_my_table_router, dependencies=[Depends(require_token)])
app.include_router(settings_router, dependencies=[Depends(require_token)])


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Open http://{HOST}:{PORT}/ui")
    logger.info(f"Open http://{HOST}:{PORT}/healthz")
    uvicorn.run(app, host=HOST, port=PORT)