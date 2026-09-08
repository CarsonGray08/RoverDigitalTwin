"""Provider-agnostic JSON-mode chat clients for the planning advisor.

Two interchangeable backends:

- ``GeminiClient``: Google Gemini API (recommended). Fast (~1-2 s), cheap
  (~$0.0003 per planning cycle with a camera frame on flash-lite), and the
  API enforces the JSON response schema server-side.
- ``OllamaClient``: local Ollama, kept for offline demos. Slow on this
  machine because the GPU is shared with Isaac Sim.

Both speak the same ``complete_json`` interface so the planner does not care
which one it is talking to. Pure stdlib (urllib), matching the rest of the
package.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class LLMClientError(RuntimeError):
    """Raised when a provider cannot return usable JSON content."""


# Local-project convenience: a plain-text key file at the repo root, used when
# $GEMINI_API_KEY is not set. Resolves to <repo>/gemini_api_key.txt both from
# the source tree and from an editable install.
GEMINI_KEY_FILE = Path(__file__).resolve().parents[2] / "gemini_api_key.txt"


def resolve_gemini_api_key(explicit: str | None = None) -> str:
    """Return the Gemini key from the explicit arg, $GEMINI_API_KEY, or the key file."""
    if explicit:
        return explicit
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return GEMINI_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


@dataclass(frozen=True)
class LLMReply:
    text: str
    latency_s: float
    provider: str
    model: str


class LLMClient(Protocol):
    provider: str
    model: str

    def complete_json(
        self,
        system: str,
        user_text: str,
        json_schema: dict[str, Any],
        images_png_base64: list[str] | None = None,
    ) -> LLMReply:
        """Return model output constrained to ``json_schema`` as raw JSON text."""


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class GeminiClient:
    """Gemini ``generateContent`` client with schema-constrained JSON output."""

    model: str = "gemini-2.5-flash-lite"
    api_key: str | None = None  # falls back to $GEMINI_API_KEY, then GEMINI_KEY_FILE
    url_base: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 20.0
    temperature: float = 0.2
    # Gemini 2.5 thinking tokens count against maxOutputTokens; with the old
    # 512 cap, thinking models (e.g. gemini-2.5-flash) spent the whole budget
    # thinking and the JSON answer came back truncated.
    max_output_tokens: int = 4096
    # Explicit thinking budget so spatial reasoning gets some thought without
    # starving the answer. 0 disables thinking; None omits the field.
    thinking_budget: int | None = 1024
    retries: int = 1
    provider: str = "gemini"

    def complete_json(
        self,
        system: str,
        user_text: str,
        json_schema: dict[str, Any],
        images_png_base64: list[str] | None = None,
    ) -> LLMReply:
        key = resolve_gemini_api_key(self.api_key)
        if not key:
            raise LLMClientError(
                "No Gemini API key: set $GEMINI_API_KEY or paste the key into "
                f"{GEMINI_KEY_FILE} (or pass --llm-provider ollama)"
            )
        payload = self.build_payload(system, user_text, json_schema, images_png_base64)
        url = f"{self.url_base}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": key}

        last_error: Exception | None = None
        retry_delay_s = 1.5
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            try:
                response = _post_json(url, payload, headers, self.timeout_s)
                return LLMReply(
                    text=self.response_text(response),
                    latency_s=time.monotonic() - started,
                    provider=self.provider,
                    model=self.model,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = LLMClientError(f"Gemini HTTP {exc.code}: {detail}")
                if exc.code not in (429, 500, 502, 503, 504):
                    raise last_error from exc  # config/auth/request error: retry won't help
                if exc.code == 429:
                    retry_delay_s = 12.0  # free-tier quota is per-minute; give it room
            except (TimeoutError, socket.timeout) as exc:
                last_error = LLMClientError(f"Gemini request timed out after {self.timeout_s:.0f}s")
            except urllib.error.URLError as exc:
                last_error = LLMClientError(f"Could not reach Gemini API: {exc.reason}")
            except json.JSONDecodeError as exc:
                last_error = LLMClientError("Gemini returned a non-JSON HTTP body")
            if attempt < self.retries:
                time.sleep(retry_delay_s)
        raise last_error if last_error is not None else LLMClientError("Gemini request failed")

    def build_payload(
        self,
        system: str,
        user_text: str,
        json_schema: dict[str, Any],
        images_png_base64: list[str] | None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{"text": user_text}]
        for image in images_png_base64 or []:
            parts.append({"inline_data": {"mime_type": "image/png", "data": image}})
        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": to_gemini_schema(json_schema),
        }
        if self.thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        return {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

    @staticmethod
    def response_text(response: dict[str, Any]) -> str:
        feedback = response.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LLMClientError(f"Gemini blocked the prompt: {feedback['blockReason']}")
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise LLMClientError("Gemini response had no candidates")
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text.strip():
            raise LLMClientError(f"Gemini returned no text (finishReason={finish})")
        return text


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a draft-style JSON schema to Gemini's OpenAPI-subset responseSchema.

    Gemini wants uppercase type names and rejects keys like additionalProperties
    and min/maxItems; everything we rely on (properties, required, items, enum)
    carries over directly.
    """
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("additionalProperties", "minItems", "maxItems"):
            continue
        if key == "type" and isinstance(value, str):
            converted[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            converted[key] = {name: to_gemini_schema(sub) for name, sub in value.items()}
        elif key == "items" and isinstance(value, dict):
            converted[key] = to_gemini_schema(value)
        else:
            converted[key] = value
    return converted


@dataclass
class OllamaClient:
    """Local Ollama ``/api/chat`` client with format-constrained JSON output."""

    model: str = "qwen2.5vl:3b"
    url: str = "http://127.0.0.1:11434/api/chat"
    timeout_s: float = 45.0
    temperature: float = 0.1
    num_predict: int = 256
    provider: str = "ollama"

    def complete_json(
        self,
        system: str,
        user_text: str,
        json_schema: dict[str, Any],
        images_png_base64: list[str] | None = None,
    ) -> LLMReply:
        user_message: dict[str, Any] = {"role": "user", "content": user_text}
        if images_png_base64:
            user_message["images"] = list(images_png_base64)
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": json_schema,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            "messages": [{"role": "system", "content": system}, user_message],
        }
        started = time.monotonic()
        try:
            response = _post_json(self.url, payload, {}, self.timeout_s)
        except (TimeoutError, socket.timeout) as exc:
            raise LLMClientError(f"Ollama request timed out after {self.timeout_s:.0f}s") from exc
        except urllib.error.URLError as exc:
            raise LLMClientError(f"Could not reach Ollama at {self.url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMClientError("Ollama returned a non-JSON HTTP body") from exc
        return LLMReply(
            text=self.response_text(response),
            latency_s=time.monotonic() - started,
            provider=self.provider,
            model=self.model,
        )

    @staticmethod
    def response_text(response: dict[str, Any]) -> str:
        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            content = message["content"]
            if not content.strip() and isinstance(message.get("thinking"), str) and message["thinking"].strip():
                raise LLMClientError("Ollama returned only hidden thinking and no JSON content")
            return content
        if isinstance(response.get("response"), str):
            return response["response"]
        raise LLMClientError("Ollama response did not include message.content")
