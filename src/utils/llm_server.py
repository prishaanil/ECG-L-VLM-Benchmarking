from dotenv import load_dotenv
import os
load_dotenv(verbose=True, override=True)
print(f"HF_HUB_CACHE is set to: {os.getenv('HF_HUB_CACHE')}")
print(f"HF_HOME is set to: {os.getenv('HF_HOME')}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from typing import Dict, Tuple, Optional
from PIL import Image
from pathlib import Path
from contextlib import asynccontextmanager


# Usage:
# - As a script (supports -m option):
#   python utils/llm_server.py -m llama --host 127.0.0.1 --port 8000
# - With uvicorn (set env var):
#   LLM_MODEL=lingshu uvicorn utils.llm_server:app --host 127.0.0.1 --port 8000
#   LLM_MODEL=llama uvicorn utils.llm_server:app --host 127.0.0.1 --port 8000

MODEL_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_MEDALPACA = "medalpaca/medalpaca-13b"
MODEL_PMC_LLAMA = "axiong/PMC_LLaMA_13B"
MODEL_MEDGEMMA = "google/medgemma-27b-text-it"
MODEL_LINGSHU = "lingshu-medical-mllm/Lingshu-32B"
MODEL_QWQ = "Qwen/QwQ-32B"
MODEL_LLAMA_33 = "meta-llama/Llama-3.3-70B-Instruct"


def _model_registry() -> Dict[str, Tuple[str, str]]:
    """Return mapping of model key -> (HF model id, model kind).

    Model kinds: "text" (text-only), "multimodal" (vision-language).
    """
    return {
        "llama": (MODEL_LLAMA, "text"),
        "medalpaca": (MODEL_MEDALPACA, "text"),
        "pmc": (MODEL_PMC_LLAMA, "text"),
        "medgemma": (MODEL_MEDGEMMA, "text"),
        "lingshu": (MODEL_LINGSHU, "multimodal"),
        "qwq": (MODEL_QWQ, "text"),
        "llama_33": (MODEL_LLAMA_33, "text"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at app startup using env `LLM_MODEL`."""
    model_key = os.getenv("LLM_MODEL", "llama")
    _load_model(model_key)
    yield

app = FastAPI(lifespan=lifespan)

# Globals populated at startup
model = None
tokenizer = None
processor = None
model_kind = "text"
default_start_token: Optional[str] = "<start>"
default_end_token: Optional[str] = "<end>"


def _select_model(model_key: str) -> Tuple[str, str]:
    """Resolve model key to an HF id and kind.

    Args:
        model_key: Short key like "llama", "medalpaca", "pmc", "medgemma", or "lingshu".

    Returns:
        Tuple of (model_id, model_kind).
    """
    registry = _model_registry()
    key = (model_key or "").lower().strip()
    if key == "":
        key = "llama"
    if key not in registry:
        key = "llama"
    return registry[key]


def _load_model(model_key: str) -> None:
    """Load the requested model and supporting components into globals.

    Supports both causal LM and vision-language models.
    """
    global model, tokenizer, processor, model_kind, default_start_token, default_end_token
    model_id, model_kind = _select_model(model_key)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_kind == "text":
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        processor = None

        # Configure default start/end tokens from env or tokenizer specials
        env_start = os.getenv("LLM_START_TOKEN", "").strip()
        env_end = os.getenv("LLM_END_TOKEN", "").strip()
        default_start_token = env_start if env_start != "" else getattr(tokenizer, "bos_token", None)
        default_end_token = env_end if env_end != "" else getattr(tokenizer, "eos_token", None)
    else:
        # Lazy import to avoid unnecessary dependency costs when unused
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
        model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")
        tokenizer = None

        # For multimodal models, still allow specifying an end token for stopping
        env_end = os.getenv("LLM_END_TOKEN", "").strip()
        default_start_token = None
        default_end_token = env_end if env_end != "" else None


# Startup handled via FastAPI lifespan above


class Query(BaseModel):
    prompt: str
    mode: Optional[str] = "plain_text"  # "plain_text" or "multiple_choice"
    start_token: Optional[str] = None
    end_token: Optional[str] = None
    max_new_tokens: Optional[int] = None


class _SequenceStopper(StoppingCriteria):
    """Stop generation when the end token sequence appears at the end of input_ids."""

    def __init__(self, end_sequence_ids):
        super().__init__()
        self.end_sequence_ids = end_sequence_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        if input_ids.shape[1] < len(self.end_sequence_ids):
            return False
        tail = input_ids[0, -len(self.end_sequence_ids):].tolist()
        return tail == self.end_sequence_ids


@app.post("/generate")
def generate(query: Query):
    """Generate a response for the given prompt using the loaded model."""
    mode = (query.mode or "plain_text").lower()
    max_new_tokens = getattr(query, "max_new_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = 4 if mode == "multiple_choice" else 300

    if model_kind == "text":
        # Resolve start/end tokens (prefer request, then env-configured defaults)
        start_tok = query.start_token if (query.start_token and query.start_token.strip() != "") else default_start_token
        end_tok = query.end_token if (query.end_token and query.end_token.strip() != "") else default_end_token

        # Prepend start token if provided
        prompt_text = f"{start_tok or ''}{query.prompt}"

        # Prepare stopping configuration
        eos_token_id = None
        stopping_criteria = None
        if end_tok:
            end_ids = tokenizer(end_tok, add_special_tokens=False).input_ids
            if len(end_ids) == 1:
                eos_token_id = end_ids[0]
            elif len(end_ids) > 1:
                stopping_criteria = StoppingCriteriaList([_SequenceStopper(end_ids)])

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        # Decode only the generated continuation
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {"response": text}
    else:
        # Vision-language path, text-only prompt supported
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query.prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # Configure stopping using end token if provided or defaulted
        end_tok = query.end_token if (query.end_token and query.end_token.strip() != "") else default_end_token
        eos_token_id = None
        stopping_criteria = None
        if end_tok and hasattr(processor, "tokenizer"):
            end_ids = processor.tokenizer(end_tok, add_special_tokens=False).input_ids  # type: ignore[attr-defined]
            if len(end_ids) == 1:
                eos_token_id = end_ids[0]
            elif len(end_ids) > 1:
                stopping_criteria = StoppingCriteriaList([_SequenceStopper(end_ids)])

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return {"response": text}


class MultimodalQuery(BaseModel):
    """Request body for multimodal generation with a single image.

    Attributes:
        prompt: User question or instruction.
        image_path: Absolute or relative path to the image on disk.
        end_token: Optional explicit end token for stopping.
        max_new_tokens: Optional max length of generated continuation.
    """
    prompt: str
    image_path: str
    end_token: Optional[str] = None
    max_new_tokens: Optional[int] = None


@app.post("/generate_multimodal")
def generate_multimodal(query: MultimodalQuery):
    """Generate a response for a multimodal prompt (text + image).

    Requires a model with kind == "multimodal" to be loaded.
    """
    if model_kind != "multimodal":
        raise HTTPException(status_code=400, detail="Loaded model is not multimodal; use /generate for text-only.")

    max_new_tokens = getattr(query, "max_new_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = 300

    # Resolve image path. If relative, allow prefixing with DATA_STORAGE_HDD_PUBMEDVISION
    image_path = query.image_path
    # Normalize and expand user/env vars; strip file:// prefix if present
    image_path = os.path.expandvars(os.path.expanduser(image_path))
    if image_path.startswith("file://"):
        image_path = image_path[len("file://"):]
    if not os.path.isabs(image_path):
        base_dir = os.getenv("DATA_STORAGE_HDD_PUBMEDVISION", "").strip()
        base_dir = os.path.expandvars(os.path.expanduser(base_dir)) if base_dir else base_dir
        if base_dir:
            image_path = os.path.join(base_dir, image_path)
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {exc}")

    # Build a chat-like message with text+image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query.prompt},
                {"type": "image"},
            ],
        }
    ]

    # Create text prompt via chat template, then pack with image via processor
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # Configure stopping using end token if provided or defaulted
    end_tok = query.end_token if (query.end_token and query.end_token.strip() != "") else default_end_token
    eos_token_id = None
    stopping_criteria = None
    if end_tok and hasattr(processor, "tokenizer"):
        end_ids = processor.tokenizer(end_tok, add_special_tokens=False).input_ids  # type: ignore[attr-defined]
        if len(end_ids) == 1:
            eos_token_id = end_ids[0]
        elif len(end_ids) > 1:
            stopping_criteria = StoppingCriteriaList([_SequenceStopper(end_ids)])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        stopping_criteria=stopping_criteria,
    )

    # Prefer tokenizer.decode when available for consistency
    if hasattr(processor, "tokenizer"):
        text = processor.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)  # type: ignore[attr-defined]
    else:
        text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    return {"response": text}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Start LLM server with selectable model")
    parser.add_argument("-m", "--model", default="llama", choices=list(_model_registry().keys()),
                        help="Model key to load (default: llama)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    # Make choice visible to startup hook
    os.environ["LLM_MODEL"] = args.model

    uvicorn.run(app, host=args.host, port=args.port)