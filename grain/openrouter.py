"""Minimal streaming client for OpenRouter's chat completions API (standard library only).

Each call is pinned to one provider with fallbacks off, so a result always describes
that provider. Streaming lets us time the first token.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_stream(api_key: str, model: str, provider: str, prompt: str, max_tokens: int,
                reasoning: dict | None = None, timeout_s: float = 180,
                url: str = API_URL) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "provider": {"order": [provider], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if reasoning:
        body["reasoning"] = reasoning
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "grain-probe",
        },
        method="POST",
    )

    started = time.monotonic()
    first_token = last_token = None
    pieces: list[str] = []
    reasoning_chars = 0
    usage = None
    served = None
    finish = None
    http_status = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            http_status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue  # blank keep-alives and ": OPENROUTER PROCESSING" comments
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    error = {"code": chunk["error"].get("code", "stream_error"),
                             "message": str(chunk["error"].get("message", ""))[:300]}
                    break
                served = chunk.get("provider") or served
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning") or ""
                    if text or thought:
                        now = time.monotonic()
                        first_token = first_token or now
                        last_token = now
                    pieces.append(text)
                    reasoning_chars += len(thought)
                    finish = choice.get("finish_reason") or finish
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 - best effort only
            detail = ""
        error = {"code": exc.code, "message": detail}
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        error = {"code": "network", "message": str(exc)[:300]}

    ended = time.monotonic()
    out_tokens = (usage or {}).get("completion_tokens")
    gen_time = (last_token - first_token) if first_token and last_token else None
    return {
        "http_status": http_status,
        "error": error,
        "ttft_s": round(first_token - started, 3) if first_token else None,
        "total_s": round(ended - started, 3),
        "tokens_per_s": round(out_tokens / gen_time, 1) if out_tokens and gen_time else None,
        "text": "".join(pieces),
        "finish_reason": finish,
        "in_tokens": (usage or {}).get("prompt_tokens"),
        "out_tokens": out_tokens,
        "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost_usd": (usage or {}).get("cost"),
        "served_provider": served,
        "reasoning_chars": reasoning_chars,
    }
