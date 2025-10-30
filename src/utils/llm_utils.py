# utils/llm_utils.py
import os
import json
import base64
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests

# ----------------------------------------------------------------------
# Defaults / Env
# ----------------------------------------------------------------------
# Legacy FastAPI server default:
DEFAULT_LLM_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:9000")

# OpenAI-style vLLM server:
VLM_BASE_URL   = os.getenv("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLM_API_KEY    = os.getenv("VLM_API_KEY", "EMPTY")           # vLLM ignores value but header required
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "medgemma-4b-it")

def _is_openai_server(url: str) -> bool:
    """Detect OpenAI-style server: any '/v1' in the base URL."""
    return "/v1" in (url or "")

def _image_part(image_path: str, prefer_file_url: bool = True) -> Dict[str, Any]:
    """Build an OpenAI 'image_url' content part for a file:// or data: URL."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if prefer_file_url:
        return {"type": "image_url", "image_url": {"url": "file://" + str(p.resolve())}}
    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

# ----------------------------------------------------------------------
# Unified helper (preferred)
# ----------------------------------------------------------------------
def request_llm_local(
    prompt: str,
    server_url: Optional[str] = None,
    timeout_sec: int = 600,
    mode: str = "multiple_choice",                 # used only by legacy server
    start_token: Optional[str] = os.getenv("LLM_START_TOKEN", None),
    end_token: Optional[str] = os.getenv("LLM_END_TOKEN", None),
    max_new_tokens: Optional[int] = None,
    use_guided_decoding: Optional[bool] = False,   # legacy server only
    # New optional params (ignored by legacy server if unsupported)
    system: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = 0.1,
    image: Optional[str] = None,
    image_path: Optional[str] = None,
    images: Optional[List[str]] = None,
    prefer_file_url: bool = True,
) -> str:
    """
    If server_url (or env) points to an OpenAI-style vLLM (/v1), call /v1/chat/completions.
    Otherwise, call the legacy FastAPI /generate endpoint.

    Supports an optional image via image/image_path/images[0] for OpenAI mode.
    """
    # Choose base URL
    base = (server_url or DEFAULT_LLM_URL).rstrip("/")
    is_openai = _is_openai_server(base)

    # Normalize image argument
    img = image_path or image or (images[0] if images else None)

    if is_openai:
        # ---------------- OpenAI /v1/chat/completions ----------------
        url = f"{base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLM_API_KEY}",
        }
        model = (model_name or VLM_MODEL_NAME).strip()

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if img:
            content.append(_image_part(img, prefer_file_url=prefer_file_url))

        messages.append({"role": "user", "content": content})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_new_tokens or 512),
            "temperature": float(temperature if temperature is not None else 0.1),
        }

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return json.dumps(data)

    else:
        # ---------------- Legacy FastAPI /generate ----------------
        url = f"{base}/generate"
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "mode": mode,
            "use_guided_decoding": bool(use_guided_decoding),
        }
        if start_token is not None:
            payload["start_token"] = start_token
        if end_token is not None:
            payload["end_token"] = end_token
        if max_new_tokens is not None:
            payload["max_new_tokens"] = int(max_new_tokens)
        # Some legacy builds accept image fields; harmless if ignored
        if img:
            payload.update({
                "image": img,
                "image_path": img,
                "images": [img],
                "prefer_file_url": prefer_file_url,
            })

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "response" in data:
            return (data["response"] or "").strip()
        return str(data)

# ----------------------------------------------------------------------
# Back-compat wrapper (keeps your existing imports working)
# ----------------------------------------------------------------------
def request_llm(
    prompt: str,
    server_url: Optional[str] = None,
    timeout_sec: int = 600,
    mode: str = "multiple_choice",
    start_token: Optional[str] = os.getenv("LLM_START_TOKEN", None),
    end_token: Optional[str] = os.getenv("LLM_END_TOKEN", None),
    max_new_tokens: Optional[int] = None,
    use_guided_decoding: Optional[bool] = False,
) -> str:
    """Backward-compatible wrapper that calls request_llm_local with legacy args."""
    return request_llm_local(
        prompt=prompt,
        server_url=server_url,
        timeout_sec=timeout_sec,
        mode=mode,
        start_token=start_token,
        end_token=end_token,
        max_new_tokens=max_new_tokens,
        use_guided_decoding=use_guided_decoding,
    )

# ----------------------------------------------------------------------
# Quick manual test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Choose which server to hit by setting either:
    #   export LLM_SERVER_URL="http://127.0.0.1:9000"                 # legacy
    # or
    #   export VLM_BASE_URL="http://127.0.0.1:8000/v1"                # OpenAI-style
    #   export VLM_MODEL_NAME="medgemma-4b-it"
    #   export VLM_API_KEY="EMPTY"

    # Auto-detect based on env:
    base = VLM_BASE_URL if _is_openai_server(VLM_BASE_URL) else DEFAULT_LLM_URL
    print(f"[test] Sending to: {base}")

    out = request_llm_local(
        prompt="Say hello",
        server_url=base,
        max_new_tokens=32,
        system="Be concise."
    )
    print("[test] Response:", out)
