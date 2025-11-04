# ui.py
from __future__ import annotations

import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

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