"""
Native Qwen client for Alibaba Cloud Bailian/DashScope.

This module uses the native DashScope Python SDK and does not rely on any
OpenAI-compatible client or endpoint.
"""

from __future__ import annotations

import os
from http import HTTPStatus
from typing import Any

from dashscope import Generation


class QwenConfigurationError(RuntimeError):
    """Raised when required Qwen/DashScope configuration is missing."""


class QwenAPIError(RuntimeError):
    """Raised when the DashScope API returns an error or empty output."""


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()

    return ""


class QwenClient:
    """Small wrapper around the native DashScope Generation API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or _first_env("DASHSCOPE_API_KEY", "BAILIAN_API_KEY")
        if not self.api_key:
            raise QwenConfigurationError(
                "Missing DashScope/Bailian API key. Set DASHSCOPE_API_KEY "
                "or BAILIAN_API_KEY before running the Qwen AML demo."
            )

        self.model = (model.strip() if model and model.strip() else _first_env("QWEN_MODEL")) or "qwen-plus"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = Generation.call(
            model=self.model,
            api_key=self.api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            result_format="message",
        )

        if getattr(response, "status_code", None) != HTTPStatus.OK:
            raise QwenAPIError(
                "DashScope Qwen request failed "
                f"({getattr(response, 'code', 'unknown_error')}): "
                f"{getattr(response, 'message', 'Unknown DashScope error')}"
            )

        content = self._extract_text(response)
        if not content:
            raise QwenAPIError("DashScope Qwen request returned an empty response.")

        return content

    @staticmethod
    def _extract_text(response: Any) -> str:
        output = getattr(response, "output", None)
        if output is None:
            return ""

        choices = getattr(output, "choices", None)
        if choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
            else:
                message = getattr(first_choice, "message", None)

            if isinstance(message, dict):
                return _normalize_content(message.get("content"))

            return _normalize_content(getattr(message, "content", None))

        if isinstance(output, dict):
            return _normalize_content(output.get("text"))

        return _normalize_content(getattr(output, "text", None))
