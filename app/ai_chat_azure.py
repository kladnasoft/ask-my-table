# ai_chat_azure.py
from __future__ import annotations

import os
import traceback
import json
from typing import Dict, Any, Optional, Tuple, List, Generator

from fastapi import HTTPException
from openai import AzureOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()
# --------------------
# ENV mirrors app.py
# --------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano-2").strip()

MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))
MAX_TOKENS_CAP = int(os.getenv("MAX_TOKENS_CAP", "16384"))
# New setting for batch processing
MAX_BATCH_TOKENS = int(os.getenv("MAX_BATCH_TOKENS", "8000"))  # Conservative batch size
CONTINUATION_PROMPT = os.getenv("CONTINUATION_PROMPT", "Please continue from where you left off.").strip()

# Temperature setting removed as it's not properly supported
FORCE_JSON_OUTPUT = os.getenv("FORCE_JSON_OUTPUT", "false").lower() in ("1", "true", "yes")

_client: Optional[AzureOpenAI] = None


def _json_error(e: Exception) -> str:
    return f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}"


def get_client() -> AzureOpenAI:
    global _client
    if _client is not None:
        return _client
    missing = []
    if not AZURE_OPENAI_ENDPOINT: missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY: missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_API_VERSION: missing.append("AZURE_OPENAI_API_VERSION")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    _client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )
    return _client


def _extract_reply_and_reason(resp_obj) -> Tuple[str, Optional[str]]:
    """
    Parse chat.completions responses robustly (compatible with app.py style).
    """
    try:
        finish_reason = None
        reply = ""
        choices = getattr(resp_obj, "choices", None) or (
            resp_obj.get("choices") if isinstance(resp_obj, dict) else None)
        if not choices:
            return "", None
        ch0 = choices[0]
        finish_reason = getattr(ch0, "finish_reason", None) or (
            ch0.get("finish_reason") if isinstance(ch0, dict) else None)
        message = getattr(ch0, "message", None) or (ch0.get("message") if isinstance(ch0, dict) else None)
        if message:
            content = getattr(message, "content", None) or (
                message.get("content") if isinstance(message, dict) else None)
            if isinstance(content, str) and content.strip():
                return content, finish_reason
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                    elif isinstance(p, str):
                        parts.append(p)
                reply = "\n".join([x for x in parts if x])
                return reply, finish_reason
        content = getattr(ch0, "content", None) or (ch0.get("content") if isinstance(ch0, dict) else None)
        if isinstance(content, str) and content.strip():
            return content, finish_reason
        return "", finish_reason
    except Exception:
        return "", None


def _maybe_force_json(kwargs: Dict[str, Any]) -> None:
    if FORCE_JSON_OUTPUT:
        kwargs["response_format"] = {"type": "json_object"}


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token for English text)"""
    return len(text) // 4


def _split_into_batches(system_prompt: str, user_content: str, max_tokens: int = MAX_BATCH_TOKENS) -> List[
    Tuple[str, str]]:
    """
    Split long content into batches that fit within token limits.
    Returns list of (system_prompt, user_content) batches.
    """
    system_tokens = _estimate_tokens(system_prompt)
    user_tokens = _estimate_tokens(user_content)

    # If total is under limit, return as single batch
    if system_tokens + user_tokens <= max_tokens:
        return [(system_prompt, user_content)]

    # If user content is too long, split it
    batches = []
    current_batch = ""
    current_tokens = 0

    # Simple paragraph-based splitting
    paragraphs = user_content.split('\n\n')

    for paragraph in paragraphs:
        para_tokens = _estimate_tokens(paragraph)

        # If single paragraph is too big, split by sentences
        if para_tokens > max_tokens - system_tokens:
            sentences = paragraph.replace('. ', '.\n').split('\n')
            for sentence in sentences:
                sent_tokens = _estimate_tokens(sentence)
                if current_tokens + sent_tokens > max_tokens - system_tokens:
                    if current_batch:
                        batches.append((system_prompt, current_batch))
                        current_batch = ""
                        current_tokens = 0
                current_batch += sentence + " "
                current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > max_tokens - system_tokens:
                if current_batch:
                    batches.append((system_prompt, current_batch))
                    current_batch = ""
                    current_tokens = 0
            current_batch += paragraph + "\n\n"
            current_tokens += para_tokens

    if current_batch:
        batches.append((system_prompt, current_batch))

    return batches


def _chat_complete_blocking(*, system_prompt: str, user_content: str, label: str = "chat",
                            is_continuation: bool = False, previous_response: str = "") -> Tuple[str, bool]:
    """
    Blocking call that hits Azure Chat Completions.
    Returns tuple of (response_text, needs_continuation)
    """
    client = get_client()

    # Adjust system prompt for continuations
    actual_system_prompt = system_prompt
    if is_continuation:
        actual_system_prompt = f"{system_prompt}\n\nThis is a continuation of previous content. Please continue naturally from: '{previous_response[-500:]}...'"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": actual_system_prompt},
        {"role": "user", "content": user_content},
    ]

    kwargs: Dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }

    _maybe_force_json(kwargs)

    try:
        resp = client.chat.completions.create(**kwargs)
        text, finish_reason = _extract_reply_and_reason(resp)

        # Handle length limitations in response
        needs_continuation = finish_reason == "length"

        if not text:
            raise HTTPException(status_code=502, detail=f"Azure[{label}] returned no text.")

        return text, needs_continuation

    except HTTPException:
        raise
    except Exception as e:
        # Handle token limit errors from API
        error_str = str(e).lower()
        if any(term in error_str for term in ["token", "length", "too long"]):
            raise HTTPException(
                status_code=400,
                detail="Input too long. Consider breaking your request into smaller parts."
            )
        raise e


def _handle_response_continuation(*, system_prompt: str, user_content: str, label: str = "chat") -> str:
    """
    Handle responses that need continuation due to length limitations.
    """
    full_response = ""
    max_continuations = 5  # Prevent infinite loops
    continuation_count = 0

    while continuation_count < max_continuations:
        is_continuation = continuation_count > 0
        current_label = f"{label}_cont_{continuation_count}" if is_continuation else label

        try:
            response, needs_continuation = _chat_complete_blocking(
                system_prompt=system_prompt,
                user_content=user_content,
                label=current_label,
                is_continuation=is_continuation,
                previous_response=full_response
            )

            full_response += response

            if not needs_continuation:
                break

            # Prepare for next continuation
            user_content = CONTINUATION_PROMPT
            continuation_count += 1

        except HTTPException as e:
            # If we get a length error on first attempt and have no content yet, re-raise
            if "Response too long" in str(e) and not full_response:
                raise
            # Otherwise, return what we have so far
            break

    return full_response


def _chat_complete_with_batching(*, system_prompt: str, user_content: str, label: str = "chat") -> str:
    """
    Handle chat completion with automatic batching for long content AND long responses.
    """
    # First, handle input batching if user content is too long
    batches = _split_into_batches(system_prompt, user_content)

    if len(batches) == 1:
        # Single batch - handle potential response length issues
        return _handle_response_continuation(
            system_prompt=system_prompt,
            user_content=user_content,
            label=label
        )

    # Multiple batches - process sequentially with response continuation handling
    full_response = ""
    for i, (batch_system, batch_user) in enumerate(batches):
        is_first = i == 0
        is_last = i == len(batches) - 1

        if not is_first:
            # For subsequent batches, use continuation prompt
            batch_user = f"{CONTINUATION_PROMPT}\n\n{batch_user}"

        batch_response = _handle_response_continuation(
            system_prompt=batch_system,
            user_content=batch_user,
            label=f"{label}_batch_{i + 1}"
        )

        full_response += batch_response

        # Add separator between batches (except after last one)
        if not is_last:
            full_response += "\n\n"

    return full_response


def chat_complete(*, system_prompt: str, user_content: str, tracer=None, label: str = "chat") -> str:
    """
    Backwards-compatible sync function with batching and continuation support.
    """
    if tracer:
        tracer.add(label.upper(), f"calling Azure Chat (deployment={AZURE_OPENAI_DEPLOYMENT})")
    try:
        out = _chat_complete_with_batching(
            system_prompt=system_prompt,
            user_content=user_content,
            label=label
        )
    except Exception as e:
        if tracer:
            tracer.add(label.upper(), f"Azure call failed: {_json_error(e)}", status="error")
        raise
    if tracer:
        preview = out[:600] if out else "(empty)"
        tracer.add(label.upper(), f"response preview: {preview}…")
    return out


async def chat_complete_async(*, system_prompt: str, user_content: str, tracer=None, label: str = "chat") -> str:
    """
    Async wrapper with batching and continuation support.
    """
    if tracer:
        tracer.add(label.upper(), f"calling Azure Chat (deployment={AZURE_OPENAI_DEPLOYMENT})")
    try:
        out = await asyncio.to_thread(
            _chat_complete_with_batching,
            system_prompt=system_prompt,
            user_content=user_content,
            label=label,
        )
    except Exception as e:
        if tracer:
            tracer.add(label.upper(), f"Azure call failed: {_json_error(e)}", status="error")
        raise
    if tracer:
        preview = out[:600] if out else "(empty)"
        tracer.add(label.upper(), f"response preview: {preview}…")
    return out


# New function for explicit batch processing
async def chat_complete_batched(*, system_prompt: str, user_contents: List[str],
                                tracer=None, label: str = "chat") -> List[str]:
    """
    Process multiple user contents in parallel batches.
    """
    if tracer:
        tracer.add(label.upper(), f"processing {len(user_contents)} batches in parallel")

    tasks = []
    for i, user_content in enumerate(user_contents):
        task = chat_complete_async(
            system_prompt=system_prompt,
            user_content=user_content,
            tracer=tracer,
            label=f"{label}_batch_{i + 1}"
        )
        tasks.append(task)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if tracer:
                    tracer.add(f"{label.upper()}_BATCH_{i + 1}", f"Batch failed: {result}", status="error")
                raise result
            final_results.append(result)

        return final_results

    except Exception as e:
        if tracer:
            tracer.add(label.upper(), f"Batch processing failed: {_json_error(e)}", status="error")
        raise