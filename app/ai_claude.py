# ai_claude.py
from __future__ import annotations

import os
import time
import asyncio
from typing import Dict, Any, Optional

import httpx
from fastapi import HTTPException


class ClaudeClient:
    """
    Claude (Anthropic) API client using Messages API.
    """

    def __init__(
            self,
            api_key: str,
            model: str = "claude-3-sonnet-20240229",
            base_url: str = "https://api.anthropic.com/v1",
            max_tokens: int = 4096,
            timeout: float = 120.0
    ):
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat_complete(
            self,
            system_prompt: str,
            user_content: str,
            tracer=None,
            label: str = "chat"
    ) -> str:
        """
        Send chat completion request to Claude API.

        Args:
            system_prompt: System message/instructions
            user_content: User message/content
            tracer: Optional tracer for logging
            label: Label for tracing

        Returns:
            Response text from Claude
        """
        if not self._client:
            raise RuntimeError("ClaudeClient must be used as async context manager")

        if tracer:
            tracer.add(label.upper(), f"calling Claude (model={self.model})")
            preview = user_content[:400] + "..." if len(user_content) > 400 else user_content
            tracer.add(label.upper(), f"request preview: {preview}")

        messages = [
            {
                "role": "user",
                "content": user_content
            }
        ]

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages
        }

        try:
            start_time = time.time()
            response = await self._client.post(
                f"{self.base_url}/messages",
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            if tracer:
                duration_ms = int((time.time() - start_time) * 1000)
                tracer.add(label.upper(), f"Claude response received in {duration_ms}ms")

            # Extract text from response
            content_blocks = data.get("content", [])
            text_parts = []

            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            result = "".join(text_parts).strip()

            if not result:
                raise HTTPException(
                    status_code=502,
                    detail=f"Claude[{label}] returned empty response"
                )

            if tracer:
                preview = result[:600] + "…" if len(result) > 600 else result
                tracer.add(label.upper(), f"response preview: {preview}")

            return result

        except httpx.HTTPStatusError as e:
            error_msg = f"Claude API error: {e.response.status_code} - {e.response.text}"
            if tracer:
                tracer.add(label.upper(), error_msg, status="error")
            raise HTTPException(
                status_code=502,
                detail=f"Claude[{label}] API error: {e.response.status_code}"
            )
        except httpx.TimeoutException:
            error_msg = f"Claude[{label}] request timeout"
            if tracer:
                tracer.add(label.upper(), error_msg, status="error")
            raise HTTPException(status_code=504, detail=error_msg)
        except Exception as e:
            error_msg = f"Claude[{label}] unexpected error: {str(e)}"
            if tracer:
                tracer.add(label.upper(), error_msg, status="error")
            raise HTTPException(status_code=500, detail=error_msg)


# Global client instance
_claude_client: Optional[ClaudeClient] = None


def get_claude_client() -> ClaudeClient:
    """
    Get or create Claude client with environment configuration.
    """
    global _claude_client

    if _claude_client is not None:
        return _claude_client

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")

    model = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
    base_url = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1")
    max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
    timeout = float(os.getenv("CLAUDE_TIMEOUT", "120.0"))

    _claude_client = ClaudeClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        timeout=timeout
    )

    return _claude_client


async def chat_complete_async(
        *,
        system_prompt: str,
        user_content: str,
        tracer=None,
        label: str = "chat"
) -> str:
    """
    Async Claude chat completion with same interface as Azure OpenAI.

    This provides a drop-in replacement for ai_chat_azure.chat_complete_async
    """
    client = get_claude_client()

    async with client:
        return await client.chat_complete(
            system_prompt=system_prompt,
            user_content=user_content,
            tracer=tracer,
            label=label
        )


def chat_complete(
        *,
        system_prompt: str,
        user_content: str,
        tracer=None,
        label: str = "chat"
) -> str:
    """
    Sync wrapper for Claude chat completion.
    """
    return asyncio.run(chat_complete_async(
        system_prompt=system_prompt,
        user_content=user_content,
        tracer=tracer,
        label=label
    ))


# Batch processing support
async def chat_complete_batched(
        *,
        system_prompt: str,
        user_contents: list[str],
        tracer=None,
        label: str = "chat"
) -> list[str]:
    """
    Process multiple user contents in parallel.
    """
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
            tracer.add(label.upper(), f"Batch processing failed: {str(e)}", status="error")
        raise