#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Tuple, Optional, List

from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)

print(f"HF_HUB_CACHE is set to: {os.getenv('HF_HUB_CACHE')}")
print(f"HF_HOME is set to: {os.getenv('HF_HOME')}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# -------- OpenAI clients for multimodal proxies ----------
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI

def _mk_client(base_url: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)

# =========================
# Multimodal endpoints (one server per model)
# =========================
# Default (e.g., Med-Gemma, or any vision model)
HUATUO_BASE_URL   = os.getenv("HUATUO_BASE_URL",   "http://127.0.0.1:8000/v1")
LLAVA_MED_BASE_URL= os.getenv("LLAVA_MED_BASE_URL","http://127.0.0.1:8000/v1")
MEDGEMMA_BASE_URL = os.getenv("MEDGEMMA_BASE_URL", "http://127.0.0.1:8000/v1")

# Optional shared default (keeps backward compat with your older env vars)
VLM_BASE_URL      = os.getenv("VLM_BASE_URL", HUATUO_BASE_URL)

# Served-model-name strings EXACTLY as you start the vLLM servers with:
HUATUO_SERVED_NAME   = os.getenv("HUATUO_SERVED_NAME",   "huatuo-vision")
LLAVA_MED_SERVED_NAME= os.getenv("LLAVA_MED_SERVED_NAME","llava-med")
MEDGEMMA_SERVED_NAME = os.getenv("MEDGEMMA_SERVED_NAME", "medgemma")   # --served-model-name when launching vLLM

# Construct OpenAI clients per endpoint
_OAI_CLIENTS: Dict[str, OpenAI] = {
    # explicit aliases you’ll pass via `model_name` in /generate_multimodal
    HUATUO_SERVED_NAME:    _mk_client(HUATUO_BASE_URL),
    LLAVA_MED_SERVED_NAME: _mk_client(LLAVA_MED_BASE_URL),
    MEDGEMMA_SERVED_NAME:  _mk_client(MEDGEMMA_BASE_URL),

    # keep a default to avoid KeyError('default')
    "default": _mk_client(VLM_BASE_URL),
}

print("[VLM clients] available model aliases:",
      ", ".join(sorted([k for k in _OAI_CLIENTS.keys() if k != 'default'])))

def _resolve_mm_target(requested_name: Optional[str]) -> tuple[OpenAI, str]:
    """
    Choose which OpenAI client + served model name to use.
    - If `requested_name` is provided and configured, use that client & name.
    - Else fall back to env VLM_MODEL_NAME or HUATUO_SERVED_NAME, and if still missing, 'default'.
    """
    # Prefer the explicit name in the request
    name = (requested_name or os.getenv("VLM_MODEL_NAME") or HUATUO_SERVED_NAME).strip()
    client = _OAI_CLIENTS.get(name)

    if client is None:
        # fall back to default client, but keep the name the user asked for
        client = _OAI_CLIENTS.get("default")
        if client is None:
            # This should never happen now, but keep a clear error if misconfigured
            avail = ", ".join(sorted(k for k in _OAI_CLIENTS.keys()))
            raise HTTPException(
                status_code=500,
                detail=f"No OpenAI clients configured. Available keys: {avail}"
            )
        # Also warn loudly in logs so you notice the mismatch
        print(f"[WARN] Unknown model alias '{name}'. Using 'default' client at {VLM_BASE_URL}. "
              f"Valid aliases: {', '.join(k for k in _OAI_CLIENTS.keys() if k!='default')}")

    return client, name

def _to_data_url(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _image_part_from_path(image_path: str, prefer_file_url: bool = True) -> dict:
    if prefer_file_url:
        url = "file://" + str(Path(image_path).resolve())
    else:
        url = _to_data_url(image_path)
    return {"type": "image_url", "image_url": {"url": url}}

# -----------------------
# Text model registry (this process loads a single TEXT model with vLLM)
# -----------------------
MODEL_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_MEDALPACA = "medalpaca/medalpaca-13b"
MODEL_PMC_LLAMA = "axiong/PMC_LLaMA_13B"
MODEL_QWQ = "Qwen/QwQ-32B"
MODEL_LLAMA_33 = "meta-llama/Llama-3.3-70B-Instruct"

def _model_registry() -> Dict[str, Tuple[str, str]]:
    """Return mapping of model key -> (HF model id, model kind). Only text here."""
    return {
        "llama": (MODEL_LLAMA, "text"),
        "medalpaca": (MODEL_MEDALPACA, "text"),
        "pmc": (MODEL_PMC_LLAMA, "text"),
        "qwq": (MODEL_QWQ, "text"),
        "llama_33": (MODEL_LLAMA_33, "text"),
        # NOTE: multimodal models are *not* loaded here; they are proxied via OpenAI endpoints.
    }

# -----------------------
# FastAPI app & globals
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_key = os.getenv("LLM_MODEL", "llama")
    _init_vllm(model_key)
    yield

app = FastAPI(lifespan=lifespan)

llm: Optional[LLM] = None
tokenizer: Optional[AutoTokenizer] = None
model_kind: str = "text"
model_id: str = ""
default_start_token: Optional[str] = None
default_end_token: Optional[str] = None

# -----------------------
# Helpers
# -----------------------
def _select_model(model_key: str) -> Tuple[str, str]:
    registry = _model_registry()
    key = (model_key or "").lower().strip() or "llama"
    if key not in registry:
        key = "llama"
    return registry[key]


def _init_vllm(model_key: str) -> None:
    global llm, tokenizer, model_kind, model_id, default_start_token, default_end_token

    model_id, model_kind = _select_model(model_key)

    if model_kind != "text":
        print(f"[WARN] Requested model kind '{model_kind}', but this process handles text only.")

    tensor_parallel_size   = int(os.getenv("VLLM_TP", "1"))
    gpu_memory_utilization = float(os.getenv("VLLM_GPU_UTIL", "0.45"))
    dtype                  = os.getenv("VLLM_DTYPE", "float16")
    max_model_len          = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    quantization           = os.getenv("VLLM_QUANTIZATION", "") or None

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=max_model_len,
        quantization=quantization,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    env_start = (os.getenv("LLM_START_TOKEN") or "").strip()
    env_end   = (os.getenv("LLM_END_TOKEN") or "").strip()
    default_start_token = env_start if env_start != "" else getattr(tokenizer, "bos_token", None)
    default_end_token   = env_end   if env_end   != "" else getattr(tokenizer, "eos_token", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[vLLM] Initialized: {model_id} (kind={model_kind}) on {device}, TP={tensor_parallel_size}, dtype={dtype}")

# -----------------------
# Request schemas
# -----------------------
class Query(BaseModel):
    prompt: str
    mode: Optional[str] = "plain_text"  # "plain_text" or "multiple_choice"
    start_token: Optional[str] = None
    end_token: Optional[str] = None
    max_new_tokens: Optional[int] = None
    use_guided_decoding: Optional[bool] = False

class MultimodalQuery(BaseModel):
    prompt: str
    image_path: str
    system: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    prefer_file_url: Optional[bool] = True
    model_name: Optional[str] = None  # 'huatuo', 'llava_med', 'default', or a raw served-model-name

# -----------------------
# Generation (text-only)
# -----------------------
@app.post("/generate")
def generate(query: Query):
    if llm is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")

    mode = (query.mode or "plain_text").lower()
    max_new_tokens = query.max_new_tokens if query.max_new_tokens is not None else (4 if mode == "multiple_choice" else 300)

    start_tok = (query.start_token or "").strip() or default_start_token
    end_tok   = (query.end_token or "").strip() or default_end_token

    prompt_text = f"{start_tok or ''}{query.prompt}"

    stop: List[str] = []
    stop_token_ids: Optional[List[int]] = None

    if end_tok:
        stop.append(end_tok)
        end_ids = tokenizer(end_tok, add_special_tokens=False).input_ids
        if len(end_ids) == 1:
            stop_token_ids = [end_ids[0]]

    guided_decoding_params = None
    if query.use_guided_decoding:
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "maxLength": 500},
                "answer": {"type": "string", "enum": ["A","B","C","D","E","N"]}
            },
            "required": ["reasoning","answer"],
            "additionalProperties": False
        }
        guided_decoding_params = GuidedDecodingParams(json=schema)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=1,
        temperature=0.0 if mode == "multiple_choice" else 0.2,
        top_p=1.0,
        stop=stop if stop else None,
        stop_token_ids=stop_token_ids,
        guided_decoding=guided_decoding_params,
    )

    outputs = llm.generate([prompt_text], sampling_params)
    if not outputs or not outputs[0].outputs:
        return {"response": ""}

    text = outputs[0].outputs[0].text
    return {"response": text}

# -----------------------
# Multimodal (proxied to OpenAI-style vLLM per model)
# -----------------------
@app.post("/generate_multimodal")
def generate_multimodal(q: MultimodalQuery):
    # Choose endpoint + served model
    client, served_model = _resolve_mm_target(q.model_name)

    # Build messages
    system_msg = q.system or "You are a cardiologist. Be concise, medically accurate, and specific."
    messages: List[dict] = [{"role": "system", "content": system_msg}]

    user_content: List[dict] = [{"type": "text", "text": q.prompt}]
    try:
        user_content.append(_image_part_from_path(q.image_path, prefer_file_url=bool(q.prefer_file_url)))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    messages.append({"role": "user", "content": user_content})

    try:
        resp = client.chat.completions.create(
            model=served_model,  # may be 'huatuo-vision', 'llava-med', or your custom served name
            messages=messages,
            temperature=q.temperature if q.temperature is not None else 0.1,
            max_tokens=q.max_new_tokens if q.max_new_tokens is not None else 512,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vision backend error: {e}")

    try:
        text = resp.choices[0].message.content.strip()
    except Exception:
        text = ""

    return {"response": text, "model": served_model}

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start vLLM-backed server (text; multimodal proxied to per-model endpoints).")
    parser.add_argument("-m", "--model", default="llama", choices=list(_model_registry().keys()),
                        help="Text model key to load in this process (default: llama)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on (default: 9000)")
    args = parser.parse_args()

    os.environ["LLM_MODEL"] = args.model
    uvicorn.run(app, host=args.host, port=args.port)
