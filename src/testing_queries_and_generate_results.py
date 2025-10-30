import argparse
import ast
import base64
import glob
import inspect
import json
import logging
import mimetypes
import numpy as np
import os
import pandas as pd
import re
import requests
import sqlite3
import string
import time
import torch
import torch.nn.functional as F
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from bert_score import score as bertscore_score
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel

# ---- Prompt templates now MUST contain {patient_block} ----
from prompt_templates import (
    PROMPT_TEMPLATE_VARIANT_1,
    PROMPT_TEMPLATE_VARIANT_2,
    PROMPT_TEMPLATE_VARIANT_3,
    PROMPT_TEMPLATE_VARIANT_4,
    PROMPT_TEMPLATE_VARIANT_5,
)
# Build the patient block from a row, and render {patient_block} into the template
from utils.prompt_utils import build_patient_block, render_prompt
from utils.rag_retriever import RAGRetriever, format_context

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLINICALBERT_ID = "emilyalsentzer/Bio_ClinicalBERT"  # or "medicalai/ClinicalBERT"

# Load once (will be reused)
cbert_tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_ID)
cbert_model = AutoModel.from_pretrained(CLINICALBERT_ID).to(DEVICE)
cbert_model.eval()

log = logging.getLogger("mm")

RETRYABLE = {500, 502, 503, 504}
CONNECT_READ_TIMEOUT = (10, 120)    # connect, read
MAX_RETRIES = 4
BACKOFF = 1.8

ALIAS_MAP = {
    r"\brbbb\b": "right bundle branch block",
    r"\blbbb\b": "left bundle branch block",
    r"\brvh\b": "right ventricular hypertrophy",
    r"\blvh\b": "left ventricular hypertrophy",
    r"\bsinus rhythm\b": "sinus rhythm",
    r"\bnsr\b": "sinus rhythm",
    r"\baf\b|\ba[-\s]?fib\b": "atrial fibrillation",
    r"\bst[-\s]?elev(ation)?\b": "st elevation",
    r"\bst[-\s]?dep(ression)?\b": "st depression",
}

@torch.no_grad()
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.no_grad()
def encode_texts_to_numpy(texts: List[str], max_length: int = 512) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    enc = cbert_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = cbert_model(**enc)
    pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.detach().cpu().numpy()

def canonicalize_ecg_text(s: str) -> str:
    if not s:
        return ""
    s = normalize_medical_text(s)  # your existing cleanup
    for pat, rep in ALIAS_MAP.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_json_or_fallback(resp: str) -> Dict[str, Any]:
    """
    Try to parse strict JSON:
      {"description": "...", "diagnosis":[{"label":"...","evidence":[...]}], "management":[...], "confidence":0-1}
    Fallback to your existing paragraph/regex extractors.
    """
    if not resp:
        return {"description":"", "diagnosis":[], "management":[], "confidence":None}
    try:
        data = json.loads(resp.strip())
        # Minimal schema guard
        if not isinstance(data.get("diagnosis", []), list):
            raise ValueError("diagnosis not list")
        if "description" not in data:
            data["description"] = ""
        if "management" not in data:
            data["management"] = []
        return data
    except Exception:
        secs = extract_sections_from_paragraph(resp)
        return {
            "description": secs.get("description",""),
            "diagnosis": [{"label": extract_diagnosis_sentence(resp) or secs.get("diagnosis",""), "evidence": []}],
            "management": [secs.get("next","")] if secs.get("next") else [],
            "confidence": None,
        }

def parse_scp_label_set(s: Any, threshold: float = 50.0) -> set:
    """
    From 'scp_codes' cell like "{'NORM':100.0,'LVOLT':0.0,...}" -> set of GT labels with score >= threshold.
    """
    d = parse_scp_codes(s)
    if not d:
        return set()
    return {k for k,v in d.items() if float(v) >= threshold}

def _nanmean_or(vals, default=0.0) -> float:
    arr = np.asarray(vals, dtype=float)
    m = np.isfinite(arr)
    return float(arr[m].mean()) if m.any() else float(default)

def set_f1(pred: set, true: set) -> float:
    if not pred or not true:
        return np.nan
    inter = len(pred & true)
    if inter == 0:
        return 0.0
    prec = inter / len(pred)
    rec  = inter / len(true)
    return 2*prec*rec/(prec+rec)

def map_text_to_scp(
    text: str,
    scp_codes: List[str],
    scp_texts: List[str],
    scp2id: Dict[str,int],
    label_embs: np.ndarray,
    top_k: int = 3
) -> List[str]:
    """
    Rank SCP codes by ClinicalBERT cosine similarity against normalized text.
    """
    t = canonicalize_ecg_text(text or "")
    if not t:
        return []
    sim = diagnosis_scores_from_similarity(t, scp_texts, label_embs)
    idx = np.argsort(sim)[::-1][:top_k]
    return [scp_codes[i] for i in idx]

# "Next step" action ontology → coverage score (0..1)
ACTION_ONTOLOGY = {
    "monitoring":  [r"\b(telemetry|holter|ambulatory ecg|24[- ]?hour monitor|repeat ecg)\b"],
    "referral":    [r"\b(cardiology|electrophysiology|ep (?:clinic|consult)|follow[- ]?up)\b"],
    "imaging":     [r"\b(echo|echocardiogram|cardiac mri|ct|ctpa|cxr|chest x[- ]?ray)\b"],
    "labs":        [r"\b(troponin|bmp|cmp|cbc|bnp|d[- ]?dimer)\b"],
    "therapy":     [r"\b(asa|aspirin|anticoagulation|beta[- ]?blocker|ace[- ]?i|statin)\b"],
    "disposition": [r"\b(admit|discharge|observe|observation|icu)\b"],
}

def action_coverage(pred_text: str) -> Tuple[float, Dict[str,int]]:
    t = (pred_text or "").lower()
    cls = {}
    for k, pats in ACTION_ONTOLOGY.items():
        cls[k] = int(any(re.search(p, t) for p in pats))
    cov = float(np.mean(list(cls.values()))) if cls else 0.0
    return cov, cls

# Simple hallucination & formatting checks
NUMERIC_PAT = re.compile(r"\b(\d+(\.\d+)?)\s*(ms|msec|bpm|mm\/?s|mv)\b", re.I)
SYMPTOMS_PAT = re.compile(r"\b(chest pain|dyspnea|syncope|palpitations)\b", re.I)

def hallucination_flags(text: str) -> Dict[str,int]:
    s = text or ""
    return {
        "hallucinated_numbers": int(bool(NUMERIC_PAT.search(s))),
        "hallucinated_symptoms": int(bool(SYMPTOMS_PAT.search(s))),
    }

def format_compliance_score(raw_resp: str) -> float:
    try:
        json.loads((raw_resp or "").strip())
        return 1.0
    except Exception:
        secs = extract_sections_from_paragraph(raw_resp or "")
        present = sum(1 for k in ["description","diagnosis","next"] if secs.get(k))
        return present / 3.0

def _file_to_data_url(p: str) -> str:
    p = str(p)
    mime = mimetypes.guess_type(p)[0] or ("image/png" if p.lower().endswith(".png") else "image/jpeg")
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _is_substantial_text(s: str, min_chars: int = 20) -> bool:
    if not s:
        return False
    s = str(s).strip()
    if len(s) < min_chars:
        return False
    # common failure boilerplate you logged earlier
    bad_markers = [
        "Vision backend error", "Connection error", "Internal Server Error",
        "HTTPException", "Error:", "Traceback (most recent call last)"
    ]
    return not any(m.lower() in s.lower() for m in bad_markers)

def _safe_bertscore(
    preds,
    refs,
    model_type: str,
    rescale_with_baseline: bool,
    lang: Optional[str]
):
    """
    Quiet + robust BERTScore:
      • Auto-disables rescaling for SciBERT (no published baseline)
      • Passes lang only when rescaling
      • Falls back to no-rescale on assertion errors
    Returns: (P_mean, R_mean, F1_mean) as floats
    """
    if not preds or not refs:
        return 0.0, 0.0, 0.0

    # Heuristic: SciBERT has no baseline; avoid rescaling.
    if model_type and "scibert" in model_type.lower():
        rescale_with_baseline = False
        lang = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            P, R, F1 = bertscore_score(
                preds, refs,
                model_type=model_type if model_type else None,
                rescale_with_baseline=rescale_with_baseline,
                **({"lang": lang} if rescale_with_baseline and lang else {})
            )
        except AssertionError:
            # e.g., baseline missing or lang not provided → fall back
            P, R, F1 = bertscore_score(
                preds, refs,
                model_type=model_type if model_type else None,
                rescale_with_baseline=False
            )

    return float(P.mean().item()), float(R.mean().item()), float(F1.mean().item())

def safe_bertscore_means(preds: List[str], refs: List[str],
                         model_type: str, rescale_with_baseline: bool,
                         lang: Optional[str] = "en") -> Dict[str, float]:
    """
    Avoid crashes / warnings: if rescale baseline missing, fall back to no-rescale.
    If lists are empty, return zeros.
    """
    if not preds or not refs:
        return {"P": 0.0, "R": 0.0, "F1": 0.0}
    try:
        # If rescaling, lang must be provided
        P, R, F1 = bertscore_score(preds, refs, model_type=model_type,
                                   rescale_with_baseline=rescale_with_baseline,
                                   lang=(lang if rescale_with_baseline else None))
    except AssertionError:
        # missing baseline; retry without rescaling
        P, R, F1 = bertscore_score(preds, refs, model_type=model_type,
                                   rescale_with_baseline=False)
    return {"P": float(P.mean().item()), "R": float(R.mean().item()), "F1": float(F1.mean().item())}

def call_mm_safe(*, prompt: str, image_path: str, model_name: str,
                 system: Optional[str] = None,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 prefer_file_url: bool = True) -> Tuple[str, int, Optional[str]]:
    """
    Returns: (response_text, mm_ok, error_message)
    """
    try:
        resp = call_mm(prompt=prompt, image_path=image_path, model_name=model_name,
                       system=system, max_new_tokens=max_new_tokens,
                       temperature=temperature, prefer_file_url=prefer_file_url)
        if _is_substantial_text(resp):
            return resp, 1, None
        return resp or "", 0, "empty_or_boilerplate"
    except Exception as e:
        return "", 0, f"{type(e).__name__}: {e}"

def call_mm(prompt: str, image_path: str, **kwargs) -> str:
    """
    FastAPI multimodal proxy (/generate_multimodal) with retries.
    More tolerant of return schemas and with clearer logging.
    """
    mm_endpoint = os.getenv("MM_ENDPOINT", "http://127.0.0.1:9000/generate_multimodal")

    # kwarg wins; otherwise env; otherwise default False (data URLs more robust)
    prefer_file_url = kwargs.pop("prefer_file_url", None)
    if prefer_file_url is None:
        prefer_file_url = os.getenv("MM_PREFER_FILE_URL", "false").lower() == "true"

    session = requests.Session()
    last_err = None

    try_file_path_first = True
    for attempt in range(MAX_RETRIES + 1):
        as_data = (not try_file_path_first)  # first try: file path; if we toggle, switch to data
        payload = {
            "prompt": prompt,
            "image_path": _file_to_data_url(image_path) if as_data else image_path,
            "prefer_file_url": (not as_data),
        }
        for k in ["model_name","system","max_new_tokens","temperature"]:
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        try:
            r = session.post(mm_endpoint, json=payload, timeout=CONNECT_READ_TIMEOUT, headers={"Connection":"keep-alive"})
            if r.ok:
                try:
                    data = r.json()
                except ValueError:
                    raise ValueError(f"Non-JSON 200 from server: {r.text[:200]!r}")
                for key in ("response","text","output"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
                if isinstance(data, dict) and "detail" in data:
                    raise RuntimeError(f"Server returned detail: {data['detail']}")
                raise ValueError(f"Malformed JSON 200: {str(data)[:200]}")
            else:
                try:
                    detail = r.json().get("detail")
                except Exception:
                    detail = None
                print(f"[MM] HTTP {r.status_code} attempt={attempt} as_data={as_data} detail={detail} body[:200]={r.text[:200]!r}")

                # if 500-series: backoff and toggle strategy
                if r.status_code in RETRYABLE:
                    # flip method next time (file path <-> data url)
                    try_file_path_first = not try_file_path_first
                    wait = (BACKOFF ** attempt) + (0.05 * attempt)
                    log.warning("[MM] %s; attempt=%s/%s, toggling method (as_data->%s); retrying in %.2fs",
                                r.status_code, attempt, MAX_RETRIES, (not as_data), wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()

        except requests.exceptions.RequestException as e:
            wait = (BACKOFF ** attempt) + (0.05 * attempt)
            log.warning("[MM] network/proxy error: %s | attempt=%s/%s; retry in %.2fs",
                        repr(e), attempt, MAX_RETRIES, wait)
            time.sleep(wait)
            continue

    raise RuntimeError(f"[MM] exhausted retries; last_err={repr(last_err)}")

# ---------- Paths tied to your repo layout ----------
PROJ_ROOT   = Path(__file__).resolve().parents[1]          # ECG-L-VLM-Benchmarking/
DATA_ROOT   = PROJ_ROOT / "data"
IMAGES_ROOT = DATA_ROOT / "ptb-xl-images"
DB_DEFAULT  = DATA_ROOT / "physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
OUT_DEFAULT = PROJ_ROOT / "results"

# ---------- Prompt variants ----------
PROMPT_VARIANTS = {
    "1": PROMPT_TEMPLATE_VARIANT_1,
    "2": PROMPT_TEMPLATE_VARIANT_2,
    "3": PROMPT_TEMPLATE_VARIANT_3,
    "4": PROMPT_TEMPLATE_VARIANT_4,
    "5": PROMPT_TEMPLATE_VARIANT_5,
}

# ---------- Regex helpers ----------
SENTENCE_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
SECTION_PATTERNS = {
    "description": [
        r"^\s*(?:describe(?: the)? (?:ecg|ecg signal|waveform)|waveform description)\s*[:\-]?\s*(.+)",
        r"^\s*(?:ecg description|signal description)\s*[:\-]?\s*(.+)",
    ],
    "diagnosis": [
        r"^\s*(?:most likely diagnosis|likely diagnosis|diagnosis|dx|impression)\s*[:\-]?\s*(.+)",
        r"^\s*(?:assessment)\s*[:\-]?\s*(.+)",
    ],
    "next": [
        r"^\s*(?:what (?:should be )?done next|management|plan|next steps?|treatment)\s*[:\-]?\s*(.+)",
        r"^\s*(?:recommendation|action)\s*[:\-]?\s*(.+)",
    ],
}
DIAG_HINTS = re.compile(r"\b(most likely (?:dx|diagnosis)|likely diagnosis|diagnosis|dx|impression|conclusion)\b", re.IGNORECASE)
NEXT_HINTS = re.compile(r"\b(next step|management|plan|treatment|recommendation|action)\b", re.IGNORECASE)

# ---------- Utils ----------
def _mm_preflight(mm_endpoint: str, image_path: str, model_name: str):
    import os, requests, base64, mimetypes
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    for as_data in (False, True):  # try file path, then data URL
        payload = {
            "prompt": "ping",
            "image_path": (data_url if as_data else image_path),
            "prefer_file_url": (not as_data),
            "model_name": model_name,
            "max_new_tokens": 8,
            "temperature": 0.0
        }
        try:
            r = requests.post(mm_endpoint, json=payload, timeout=(10, 60))
            print(f"[PREFLIGHT] as_data={as_data} -> {r.status_code} body[:200]={r.text[:200]!r}")
        except Exception as e:
            print(f"[PREFLIGHT] as_data={as_data} -> EXC {e!r}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def thousand_bucket(ecg_id: int) -> str:
    return f"{(ecg_id // 1000) * 1000:05d}"

def build_image_path_for_ecg(ecg_id: int, images_root: Path = IMAGES_ROOT,
                             prefer_quality: str = "lr",
                             try_indices: tuple = (0, 1, 2),
                             also_try_other_quality: bool = True) -> str:
    """
    data/ptb-xl-images/<THOUSAND>/<ID>_<quality>-<idx>.png
    """
    ecg_str = f"{int(ecg_id):05d}"
    folder = images_root / thousand_bucket(int(ecg_id))
    for i in try_indices:
        p = folder / f"{ecg_str}_{prefer_quality}-{i}.png"
        if p.exists():
            return str(p.resolve())
    if also_try_other_quality:
        other = "hr" if prefer_quality == "lr" else "lr"
        for i in try_indices:
            p = folder / f"{ecg_str}_{other}-{i}.png"
            if p.exists():
                return str(p.resolve())
    hits = list((folder).glob(f"{ecg_str}_*.png"))
    return str(hits[0].resolve()) if hits else ""

def resolve_image(row: Dict[str, Any], images_root: Path = IMAGES_ROOT) -> str:
    """
    Prefer ecg_id via bucket rule. Fallback to filename_lr if present.
    """
    ecg_id = row.get("ecg_id")
    if pd.notna(ecg_id):
        try:
            p = build_image_path_for_ecg(int(float(ecg_id)), images_root=images_root, prefer_quality="lr")
            if p:
                return p
        except Exception:
            pass

    fname_lr = row.get("filename_lr")
    if isinstance(fname_lr, str) and fname_lr.strip():
        base = images_root / Path(fname_lr)
        for cand in [base.with_suffix(".png"),
                     Path(str(base) + "-0.png"),
                     Path(str(base) + "-1.png"),
                     Path(str(base) + "-2.png")]:
            if cand.exists():
                return str(cand.resolve())
    return ""

def _first_sentence(text: str) -> str:
    parts = SENTENCE_SPLIT.split(text.strip())
    return parts[0].strip() if parts else text.strip()

def _normalize_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\-\*\d\)\(\.]+\s*", "", s)
    return s.strip()

def extract_sections_from_paragraph(text: str) -> Dict[str, str]:
    out = {"description": "", "diagnosis": "", "next": ""}
    lines = [l for l in text.splitlines() if l.strip()]
    taken = set()
    for key, pats in SECTION_PATTERNS.items():
        for pat in pats:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                out[key] = _normalize_line(m.group(1)); break
        if out[key]:
            for i, line in enumerate(lines):
                if out[key] in line:
                    taken.add(i); break
    if not all(out.values()):
        rem = " ".join(_normalize_line(l) for i, l in enumerate(lines) if i not in taken).strip()
        if rem:
            sents = [s.strip() for s in SENTENCE_SPLIT.split(rem) if s.strip()]
            if not out["description"] and sents: out["description"] = sents[0]
            if not out["diagnosis"] and len(sents) >= 2: out["diagnosis"] = sents[1]
            if not out["next"] and len(sents) >= 3: out["next"] = sents[2]
    for k in out:
        if not out[k]:
            out[k] = _first_sentence(text)
    return out

def extract_diagnosis_sentence(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for raw in lines:
        m = re.search(r"^\s*(?:most likely diagnosis|likely diagnosis|diagnosis|dx|impression)\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
        if m: return _first_sentence(_normalize_line(m.group(1)))
        if DIAG_HINTS.search(raw):
            m2 = re.search(r":\s*(.+)", raw)
            return _first_sentence(_normalize_line(m2.group(1) if m2 else raw))
    sents = [s.strip() for s in SENTENCE_SPLIT.split(" ".join(lines)) if s.strip()]
    if sents:
        if len(sents) >= 2:
            desc_cues = re.compile(r"\b(rate|rhythm|axis|interval|qrs|qt|st|t[- ]?wave|p[- ]?wave|morphology|amplitude)\b", re.IGNORECASE)
            if desc_cues.search(sents[0]) and not (DIAG_HINTS.search(sents[0]) or NEXT_HINTS.search(sents[0])):
                return _first_sentence(_normalize_line(sents[1]))
        return _first_sentence(_normalize_line(sents[0]))
    return text.strip()

def normalize_medical_text(s: str) -> str:
    s = s.strip()
    s = s.rstrip(string.punctuation + " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bST[-\s]?elev(?:ation)?\b", "st elevation", s, flags=re.IGNORECASE)
    s = re.sub(r"\bST[-\s]?dep(?:ression)?\b", "st depression", s, flags=re.IGNORECASE)
    s = re.sub(r"\bA[-\s]?fib\b", "atrial fibrillation", s, flags=re.IGNORECASE)
    s = re.sub(r"\b1st[-\s]?degree\b", "first degree", s, flags=re.IGNORECASE)
    return s

def parse_scp_codes(raw: Any) -> dict:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)): return {}
    s = str(raw).strip()
    if not s: return {}
    try: return ast.literal_eval(s)
    except Exception: return {}

def gt_scp_from_scores(scp_scores: dict, *, threshold: float = None) -> Tuple[Optional[str], List[str]]:
    if not scp_scores: return None, []
    items = sorted([(k, float(v)) for k, v in scp_scores.items()], key=lambda x: x[1], reverse=True)
    top1 = items[0][0]
    multi = [k for k, v in items if threshold is not None and v >= threshold]
    return top1, multi

def diagnosis_scores_from_similarity(
    diagnosis_text: str,
    label_texts: List[str],
    label_embs: np.ndarray = None
) -> np.ndarray:
    diag_vec = encode_texts_to_numpy([diagnosis_text])[0]
    if label_embs is None:
        label_embs = encode_texts_to_numpy(label_texts)
    return label_embs @ diag_vec

def bertscore_mean(cands: List[str], refs: List[str],
                   model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
                   rescale_with_baseline: bool = True) -> Dict[str, float]:
    if not cands: return {"P": 0.0, "R": 0.0, "F1": 0.0}
    P, R, F1 = bertscore_score(cands, refs, model_type=model_type, rescale_with_baseline=rescale_with_baseline)
    return {"P": float(P.mean().item()), "R": float(R.mean().item()), "F1": float(F1.mean().item())}

def compute_f1(pred: str, true: str) -> float:
    pt = set(pred.lower().split()); tt = set(true.lower().split())
    if not pt or not tt: return 0.0
    inter = pt & tt
    if not inter: return 0.0
    prec = len(inter)/len(pt); rec = len(inter)/len(tt)
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def embedding_score(pred: str, true: str) -> float:
    if not pred or not true:
        return 0.0
    embs = encode_texts_to_numpy([pred, true])
    v1, v2 = embs[0], embs[1]
    return float(np.dot(v1, v2))

# ---------- IO ----------
def read_table(db_path: str, sql_table: Optional[str] = None) -> pd.DataFrame:
    if db_path.endswith(".csv"): return pd.read_csv(db_path)
    if db_path.endswith(".parquet"): return pd.read_parquet(db_path)
    if db_path.endswith(".db") or db_path.endswith(".sqlite"):
        if not sql_table:
            raise ValueError("SQLite input requires --sql_table.")
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {sql_table}", conn)
    raise ValueError(f"Unsupported input: {db_path}")

# ---------- Core (run + evaluate) ----------
def pick_variant(idx: int, mode: str) -> str:
    if mode != "roundrobin":
        return PROMPT_VARIANTS[mode]
    keys = ["1","2","3","4","5"]
    return PROMPT_VARIANTS[keys[idx % len(keys)]]


def _filtered_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else 0.0

# ==== evaluate_one_model signature & usage ====
def evaluate_one_model(
    model_name: str,
    db_path: str,
    outdir: Path,
    limit: Optional[int],
    scp_csv_path: Optional[str],
    k_list: Tuple[int, ...],
    bert_model: str,
    bert_rescale: bool,
    sql_table: Optional[str],
    variant: str,
    save_json: bool,
    exclude_demographics: Optional[str] = "",
    keymap: Optional[Dict[str, str]] = None,
    bert_lang: str = "en",
    retriever: Optional[RAGRetriever] = None,
    rag_top_k: int = 4,
    rag_off: bool = False,
) -> None:
    df = read_table(db_path, sql_table=sql_table)
    mdir = outdir / model_name
    ensure_dir(mdir)

    # JSONL stream (single file)
    raw_f = None
    raw_path = None
    if save_json:
        raw_path = mdir / "raw.jsonl"
        raw_f = open(raw_path, "w", encoding="utf-8")

    exclude_set = {x.strip().lower() for x in (exclude_demographics or "").split(",") if x.strip()}
    keymap = keymap or {"age": "age", "sex": "sex", "height": "height", "weight": "weight"}

    # Load SCP labels
    use_topk = False
    scp_codes: List[str] = []
    scp_texts: List[str] = []
    scp2id: Dict[str, int] = {}
    label_embs = None
    if scp_csv_path and os.path.exists(scp_csv_path):
        df_s = pd.read_csv(scp_csv_path, index_col=0)
        meaning_col = next((c for c in [
            "SCP-ECG Statement Description", "SCP-ECG statement description", "description", "Description"
        ] if c in df_s.columns), None)
        if meaning_col:
            df_s = df_s.reset_index().rename(columns={"index": "scp_code"})
            df_s["scp_code"] = df_s["scp_code"].astype(str).str.strip()
            df_s[meaning_col] = df_s[meaning_col].astype(str).str.strip()
            df_s = df_s[(df_s["scp_code"] != "") & (df_s[meaning_col] != "")]
            scp_codes = df_s["scp_code"].tolist()
            scp_texts = df_s[meaning_col].tolist()
            scp2id = {c: i for i, c in enumerate(scp_codes)}
            label_embs = encode_texts_to_numpy(scp_texts)
            use_topk = True

    per_sample_rows = []
    y_true_ids, y_score_rows, y_pred_ids_top1 = [], [], []

    bert_desc_preds, bert_desc_refs = [], []
    bert_diag_preds, bert_diag_refs = [], []
    bert_steps_preds, bert_steps_refs = [], []

    pv_lists = {
        "diag_token_f1": defaultdict(list),
        "desc_emb": defaultdict(list),
        "steps_emb": defaultdict(list),
        "bert_desc_preds": defaultdict(list),
        "bert_desc_refs": defaultdict(list),
        "bert_diag_preds": defaultdict(list),
        "bert_diag_refs": defaultdict(list),
        "bert_steps_preds": defaultdict(list),
        "bert_steps_refs": defaultdict(list),
    }

    try:
        # BEFORE the for-loop:
        first_img = ""
        for _, r0 in df.iterrows():
            first_img = resolve_image(r0.to_dict(), IMAGES_ROOT)
            if first_img:
                break
        if first_img:
            _mm_preflight(os.getenv("MM_ENDPOINT", "http://127.0.0.1:9000/generate_multimodal"), first_img, model_name)

        df = df.reset_index(drop=True)

        for i, r in enumerate(df.itertuples(index=False)):
            if limit is not None and i >= limit:
                break

            # --- convert row to a plain dict (works for both itertuples and iterrows)
            row_dict = r._asdict() if hasattr(r, "_asdict") else (r.to_dict() if hasattr(r, "to_dict") else dict(r))

            tpl   = pick_variant(i, variant)
            vname = variant if variant != "roundrobin" else f"rr_{(i % 5) + 1}"

            patient_block = build_patient_block(row_dict, exclude=exclude_set, keymap=keymap)
            prompt        = render_prompt(tpl, patient_block, row=row_dict, keymap=keymap)

            img = resolve_image(row_dict, IMAGES_ROOT)
            if img:
                img = str(Path(img).resolve())
            else:
                print(f"[WARN] no image for ecg_id={row_dict.get('ecg_id')}; skipping row {i}")
                continue

            rag_context, rag_docs = "", []
            if (retriever is not None) and retriever.ready() and (not rag_off):
                query = (
                    f"{patient_block}\n"
                    "Task: ECG interpretation; focus on rhythm, intervals (PR/QRS/QT), ST/T changes, "
                    "bundle branch blocks, ischemia criteria (by lead groups), and SCP code mappings."
                )
                hits = retriever.retrieve(query, top_k=rag_top_k)
                rag_context = format_context(hits)
                rag_docs = [h[1].title for h in hits]

            system_msg = (
                "You are a cardiologist. Use the CONTEXT to guide terminology and criteria. "
                "Do NOT invent history/symptoms or numeric measurements you cannot infer from the ECG. "
                "If unknown, say 'not stated'. Summarize criteria; do not quote verbatim.\n\n"
                f"CONTEXT:\n{rag_context if rag_context else '(none)'}"
            )

            response, mm_ok, mm_err = call_mm_safe(
                prompt=prompt,
                image_path=img,
                model_name=model_name,
                system=system_msg,
                max_new_tokens=512,
                temperature=0.1,
                prefer_file_url=True,   # file path first (works per your preflight)
            )

            rid = row_dict.get("ecg_id") or row_dict.get("id") or i

            gt_diag_text = row_dict.get("report", "") or ""
            gt_desc = gt_diag_text
            gt_steps = ""
            scp_scores = parse_scp_codes(row_dict.get("scp_codes", ""))
            gt_diag_scp, _ = gt_scp_from_scores(scp_scores)

            if raw_f is not None:
                raw_record = {
                    "id": int(rid) if str(rid).isdigit() else rid,
                    "input": {
                        "prompt": prompt,
                        "image": img,
                        "rag_context": rag_context,
                        "rag_docs": rag_docs,
                        "system": system_msg,
                        "patient_block": patient_block,
                        "prompt_variant": vname,
                    },
                    "output": {"response": response, "mm_ok": mm_ok, "mm_error": mm_err},
                }
                raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

            if mm_ok != 1:
                per_sample_rows.append({
                    "id": rid,
                    "model": model_name,
                    "image_path": img,
                    "prompt_variant": vname,
                    "gt_diag_text": gt_diag_text,
                    "gt_diag_scp": gt_diag_scp,
                    "pred_diag_text": "",
                    "pred_top1_scp": "",
                    "diag_token_f1": np.nan,
                    "desc_emb": np.nan,
                    "steps_emb": np.nan,
                    "rag_used": int(bool(rag_context)),
                    "rag_docs": ";".join(rag_docs),
                    "mm_ok": 0,
                    "mm_error": mm_err or "",          # <— add this
                })
                continue

            # ---- parse and score ----
            sections = extract_sections_from_paragraph(response)
            response_diag  = normalize_medical_text(extract_diagnosis_sentence(response) or sections["diagnosis"])
            response_desc  = normalize_medical_text(sections["description"])
            response_steps = normalize_medical_text(sections["next"])

            diag_f1   = compute_f1(response_diag, gt_diag_text) if gt_diag_text else np.nan
            desc_emb  = embedding_score(response_desc, gt_desc)  if gt_desc      else np.nan
            steps_emb = embedding_score(response_steps, gt_steps) if gt_steps    else np.nan

            topk_dict, pred_top1_code = {}, ""
            if use_topk and response_diag and gt_diag_scp and (gt_diag_scp in scp2id):
                scores_vec = diagnosis_scores_from_similarity(response_diag, scp_texts, label_embs)
                y_score_rows.append(scores_vec)
                y_true_ids.append(scp2id[gt_diag_scp])
                p_idx = int(np.argmax(scores_vec))
                y_pred_ids_top1.append(p_idx)
                pred_top1_code = scp_codes[p_idx]
                for k in k_list:
                    topk = np.argpartition(scores_vec, -k)[-k:]
                    topk_dict[f"top{k}_hit"] = bool(scp2id[gt_diag_scp] in topk)

            per_sample_rows.append({
                "id": rid,
                "model": model_name,
                "image_path": img,
                "prompt_variant": vname,
                "gt_diag_text": gt_diag_text,
                "gt_diag_scp": gt_diag_scp,
                "pred_diag_text": response_diag,
                "pred_top1_scp": pred_top1_code,
                "diag_token_f1": diag_f1,
                "desc_emb": desc_emb,
                "steps_emb": steps_emb,
                "rag_used": int(bool(rag_context)),
                "rag_docs": ";".join(rag_docs),
                "mm_ok": 1,
                **topk_dict
            })

            # ---- global BERT queues ----
            if gt_desc and response_desc:
                bert_desc_preds.append(response_desc); bert_desc_refs.append(gt_desc)
            if gt_diag_text and response_diag:
                bert_diag_preds.append(response_diag); bert_diag_refs.append(gt_diag_text)
            if gt_steps and response_steps:
                bert_steps_preds.append(response_steps); bert_steps_refs.append(gt_steps)

            # ---- per-variant queues ----
            pv_lists["diag_token_f1"][vname].append(diag_f1)
            pv_lists["desc_emb"][vname].append(desc_emb)
            pv_lists["steps_emb"][vname].append(steps_emb)
            if gt_desc and response_desc:
                pv_lists["bert_desc_preds"][vname].append(response_desc)
                pv_lists["bert_desc_refs"][vname].append(gt_desc)
            if gt_diag_text and response_diag:
                pv_lists["bert_diag_preds"][vname].append(response_diag)
                pv_lists["bert_diag_refs"][vname].append(gt_diag_text)
            if gt_steps and response_steps:
                pv_lists["bert_steps_preds"][vname].append(response_steps)
                pv_lists["bert_steps_refs"][vname].append(gt_steps)

    finally:
        if raw_f is not None:
            raw_f.close()
            print(f"[{model_name}] wrote raw JSONL: {raw_path}")

    # ---- Save per-sample with typed columns (keep NaN for numeric) ----
    per_df = pd.DataFrame(per_sample_rows)

    # Ensure numeric cols are numeric (NaN allowed)
    num_cols = ["diag_token_f1","desc_emb","steps_emb"] + [c for c in per_df.columns if re.match(r"top\d+_hit", c)]
    for c in num_cols:
        if c in per_df.columns:
            per_df[c] = pd.to_numeric(per_df[c], errors="coerce")

    # Fill ONLY string-ish columns for readability
    str_cols = ["model","image_path","prompt_variant","gt_diag_text","gt_diag_scp","pred_diag_text","pred_top1_scp","rag_docs"]
    for c in str_cols:
        if c in per_df.columns:
            per_df[c] = per_df[c].fillna("")

    per_df.to_csv(mdir / "per_sample.csv", index=False)

    # ---- Aggregates ----
    ok_df = per_df[per_df.get("mm_ok", 1) == 1].copy()

    agg = {
        "model": model_name,
        "num_samples": int(len(per_df)),
        "valid_samples": int(len(ok_df)),
    }

    # Overall BERTScores (robust)
    P, R, F = _safe_bertscore(bert_desc_preds, bert_desc_refs, bert_model, bert_rescale, bert_lang)
    agg["bertscore_desc_P"], agg["bertscore_desc_R"], agg["bertscore_desc_F1"] = P, R, F

    P, R, F = _safe_bertscore(bert_diag_preds, bert_diag_refs, bert_model, bert_rescale, bert_lang)
    agg["bertscore_diag_P"], agg["bertscore_diag_R"], agg["bertscore_diag_F1"] = P, R, F

    # Only compute steps when GT pairs exist; otherwise leave 0.0 (or you can skip keys)
    if bert_steps_refs:
        P, R, F = _safe_bertscore(bert_steps_preds, bert_steps_refs, bert_model, bert_rescale, bert_lang)
        agg["bertscore_steps_P"], agg["bertscore_steps_R"], agg["bertscore_steps_F1"] = P, R, F
    else:
        agg["bertscore_steps_P"] = agg["bertscore_steps_R"] = agg["bertscore_steps_F1"] = 0.0

    # Means over successful rows only, ignoring NaN
    for col in ["diag_token_f1", "desc_emb", "steps_emb"]:
        if col in ok_df:
            agg[f"{col}_mean"] = _filtered_mean(ok_df[col])

    # Top-k + confusion, reports
    if use_topk and len(y_true_ids) and len(y_score_rows):
        y_scores = np.vstack(y_score_rows)
        y_true   = np.asarray(y_true_ids, dtype=int)
        y_pred1  = np.asarray(y_pred_ids_top1, dtype=int)

        for k in k_list:
            topk_idx = np.argpartition(y_scores, -k, axis=1)[:, -k:]
            hits = [(t in row) for t, row in zip(y_true, topk_idx)]
            agg[f"top{k}_accuracy"] = float(np.mean(hits)) if hits else 0.0

        cm = confusion_matrix(y_true, y_pred1, labels=list(range(len(scp_codes))))
        pd.DataFrame(cm, index=scp_codes, columns=scp_codes).to_csv(mdir / "confusion_matrix.csv")

        report = classification_report(
            y_true, y_pred1, labels=list(range(len(scp_codes))),
            target_names=scp_codes, output_dict=True, zero_division=0
        )

        rows = []
        for cls in scp_codes:
            if cls in report:
                R = report[cls]
                rows.append({
                    "scp": cls,
                    "precision": R.get("precision", 0.0),
                    "recall": R.get("recall", 0.0),
                    "f1": R.get("f1-score", 0.0),
                    "support": R.get("support", 0),
                })
        if rows:
            pd.DataFrame(rows).to_csv(mdir / "per_class.csv", index=False)

        for key in ["macro avg","weighted avg"]:
            if key in report:
                agg[f"{key.replace(' ','_')}_precision"] = report[key]["precision"]
                agg[f"{key.replace(' ','_')}_recall"]    = report[key]["recall"]
                agg[f"{key.replace(' ','_')}_f1"]        = report[key]["f1-score"]

        # Hard cases (optional)
        if not per_df.empty and "gt_diag_scp" in per_df.columns:
            ps = per_df[per_df["gt_diag_scp"].isin(scp_codes)].copy()
            ps["correct_top1"] = ps["gt_diag_scp"] == ps["pred_top1_scp"]
            miss = (ps.groupby("gt_diag_scp")["correct_top1"]
                      .apply(lambda s: 1.0 - s.mean())
                      .reset_index(name="miss_rate"))
            def most_common_pred(g):
                wrong = g[g["correct_top1"]==False]["pred_top1_scp"]
                return wrong.mode().iloc[0] if len(wrong) else ""
            mc = ps.groupby("gt_diag_scp").apply(most_common_pred).reset_index(name="most_common_wrong_pred")
            miss.merge(mc, on="gt_diag_scp").sort_values("miss_rate", ascending=False).to_csv(mdir / "hard_cases.csv", index=False)

    # Save aggregates
    pd.DataFrame([agg]).fillna(0.0).to_csv(mdir / "aggregates.csv", index=False)

    # ---- Per-variant aggregates (single writer) ----
    pv_rows = []
    variants_present = sorted(set(per_df["prompt_variant"].tolist()))
    for vname in variants_present:
        row = {"model": model_name, "prompt_variant": vname}
        # means
        vals = pv_lists["diag_token_f1"][vname]; row["diag_token_f1_mean"] = _nanmean_or(vals, 0.0)
        vals = pv_lists["desc_emb"][vname];      row["desc_emb_mean"]      = _nanmean_or(vals, 0.0)
        vals = pv_lists["steps_emb"][vname];     row["steps_emb_mean"]     = _nanmean_or(vals, 0.0)

        # BERT per variant
        P,R,F = _safe_bertscore(pv_lists["bert_desc_preds"][vname], pv_lists["bert_desc_refs"][vname], bert_model, bert_rescale, bert_lang)
        row["bertscore_desc_P"], row["bertscore_desc_R"], row["bertscore_desc_F1"] = P, R, F

        P,R,F = _safe_bertscore(pv_lists["bert_diag_preds"][vname], pv_lists["bert_diag_refs"][vname], bert_model, bert_rescale, bert_lang)
        row["bertscore_diag_P"], row["bertscore_diag_R"], row["bertscore_diag_F1"] = P, R, F

        P,R,F = _safe_bertscore(pv_lists["bert_steps_preds"][vname], pv_lists["bert_steps_refs"][vname], bert_model, bert_rescale, bert_lang)
        row["bertscore_steps_P"], row["bertscore_steps_R"], row["bertscore_steps_F1"] = P, R, F

        # contributor count (successful rows) for sanity checks
        row["n"] = int(len([x for x in pv_lists["diag_token_f1"][vname] if not (pd.isna(x) or x == "")]))

        pv_rows.append(row)

    if pv_rows:
        pd.DataFrame(pv_rows).fillna(0.0).to_csv(mdir / "per_variant.csv", index=False)

    # Quick per-variant validation
    try:
        pv_df = pd.read_csv(mdir / "per_variant.csv")
        if "n" in pv_df.columns:
            n_sum = int(pv_df["n"].sum())
            valid = int(ok_df.shape[0])
            if n_sum != valid:
                print(f"[WARN] per_variant n-sum {n_sum} != valid_samples {valid}")
    except Exception as e:
        print(f"[WARN] per_variant validation failed: {e}")

    print(f"[{model_name}] wrote results under: {mdir}")


    # overall BERTScore
    def _bs(preds, refs, label):
        if preds and refs:
            P, R, F1 = bertscore_score(preds, refs, model_type=bert_model, rescale_with_baseline=bert_rescale)
            agg[f"bertscore_{label}_P"] = float(P.mean().item())
            agg[f"bertscore_{label}_R"] = float(R.mean().item())
            agg[f"bertscore_{label}_F1"] = float(F1.mean().item())
    _bs(bert_desc_preds, bert_desc_refs, "desc")
    _bs(bert_diag_preds, bert_diag_refs, "diag")
    _bs(bert_steps_preds, bert_steps_refs, "steps")

    for col in ["diag_token_f1", "desc_emb", "steps_emb"]:
        if col in per_df:
            agg[f"{col}_mean"] = float(per_df[col].mean())

    # ---- Top-k + confusion (overall) ----
    if use_topk and len(y_true_ids) and len(y_score_rows):
        y_scores = np.vstack(y_score_rows)
        y_true   = np.asarray(y_true_ids, dtype=int)
        y_pred1  = np.asarray(y_pred_ids_top1, dtype=int)

        for k in k_list:
            topk_idx = np.argpartition(y_scores, -k, axis=1)[:, -k:]
            hits = [(t in row) for t, row in zip(y_true, topk_idx)]
            agg[f"top{k}_accuracy"] = float(np.mean(hits)) if hits else 0.0

        cm = confusion_matrix(y_true, y_pred1, labels=list(range(len(scp_codes))))
        pd.DataFrame(cm, index=scp_codes, columns=scp_codes).to_csv(mdir / "confusion_matrix.csv")

        report = classification_report(
            y_true, y_pred1, labels=list(range(len(scp_codes))),
            target_names=scp_codes, output_dict=True, zero_division=0
        )

        rows = []
        for cls in scp_codes:
            if cls in report:
                R = report[cls]
                rows.append({
                    "scp": cls,
                    "precision": R.get("precision", 0.0),
                    "recall": R.get("recall", 0.0),
                    "f1": R.get("f1-score", 0.0),
                    "support": R.get("support", 0),
                })
        if rows:
            pd.DataFrame(rows).to_csv(mdir / "per_class.csv", index=False)

        for key in ["macro avg","weighted avg"]:
            if key in report:
                agg[f"{key.replace(' ','_')}_precision"] = report[key]["precision"]
                agg[f"{key.replace(' ','_')}_recall"]    = report[key]["recall"]
                agg[f"{key.replace(' ','_')}_f1"]        = report[key]["f1-score"]

    pd.DataFrame([agg]).to_csv(mdir / "aggregates.csv", index=False)

    # ---- NEW: per-variant aggregates ----
    pv_rows = []
    for vname in sorted(set(per_df["prompt_variant"].tolist())):
        row = {"model": model_name, "prompt_variant": vname, "n": int(len(pv_lists["diag_token_f1"][vname]))}
        # simple means
        for col in ["diag_token_f1", "desc_emb", "steps_emb"]:
            vals = pv_lists[col][vname]
            row[f"{col}_mean"] = float(np.mean(vals)) if vals else 0.0
        # BERTScore means per variant
        def _pv_bs(pred_key, ref_key, label_suffix):
            preds = pv_lists[pred_key][vname]; refs = pv_lists[ref_key][vname]
            if preds and refs:
                P, R, F1 = bertscore_score(preds, refs, model_type=bert_model, rescale_with_baseline=bert_rescale)
                row[f"bertscore_{label_suffix}_P"]  = float(P.mean().item())
                row[f"bertscore_{label_suffix}_R"]  = float(R.mean().item())
                row[f"bertscore_{label_suffix}_F1"] = float(F1.mean().item())
            else:
                row[f"bertscore_{label_suffix}_P"]  = 0.0
                row[f"bertscore_{label_suffix}_R"]  = 0.0
                row[f"bertscore_{label_suffix}_F1"] = 0.0
        _pv_bs("bert_desc_preds", "bert_desc_refs", "desc")
        _pv_bs("bert_diag_preds", "bert_diag_refs", "diag")
        _pv_bs("bert_steps_preds","bert_steps_refs","steps")
        pv_rows.append(row)

    if pv_rows:
        pd.DataFrame(pv_rows).to_csv(mdir / "per_variant.csv", index=False)

    def _bs(preds, refs, label):
        if not preds:
            return
        # Try requested model, then robust fallbacks
        candidates = [
            bert_model,                              # user choice
            "allenai/scibert_scivocab_uncased",     # strong for scientific/biomed
            "bert-base-uncased",                    # very available
            None,                                   # let BERTScore default (roberta-large)
        ]
        last_err = None
        for cand in candidates:
            try:
                kwargs = {
                    "cands": preds,
                    "refs": refs,
                    "rescale_with_baseline": bool(bert_rescale),
                    "device": DEVICE,
                }
                if cand is not None:
                    kwargs["model_type"] = cand
                # When rescaling, lang is REQUIRED
                if bert_rescale:
                    kwargs["lang"] = bert_lang
                P, R, F1 = bertscore_score(**kwargs)
                agg[f"bertscore_{label}_model_used"] = cand or "default(roberta-large)"
                agg[f"bertscore_{label}_P"]  = float(P.mean().item())
                agg[f"bertscore_{label}_R"]  = float(R.mean().item())
                agg[f"bertscore_{label}_F1"] = float(F1.mean().item())
                return
            except AssertionError as e:
                # typically missing lang with rescale; try again w/o rescale
                last_err = e
                try:
                    kwargs["rescale_with_baseline"] = False
                    kwargs.pop("lang", None)
                    P, R, F1 = bertscore_score(**kwargs)
                    agg[f"bertscore_{label}_model_used"] = (cand or "default(roberta-large)") + " (no-rescale)"
                    agg[f"bertscore_{label}_P"]  = float(P.mean().item())
                    agg[f"bertscore_{label}_R"]  = float(R.mean().item())
                    agg[f"bertscore_{label}_F1"] = float(F1.mean().item())
                    agg["bertscore_rescale_fallback"] = True
                    return
                except Exception as e2:
                    last_err = e2
                    continue
            except KeyError as e:
                # unsupported model name -> try next candidate
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        # If we exhausted all options, record the error but don't crash the run
        agg[f"bertscore_{label}_error"] = str(last_err) if last_err else "unknown error"

    _bs(bert_desc_preds, bert_desc_refs, "desc")
    _bs(bert_diag_preds, bert_diag_refs, "diag")
    _bs(bert_steps_preds, bert_steps_refs, "steps")


    for col in ["diag_token_f1","desc_emb","steps_emb"]:
        if col in per_df:
            agg[f"{col}_mean"] = float(per_df[col].mean())

    # Top-k + confusion
    if use_topk and len(y_true_ids) and len(y_score_rows):
        y_scores = np.vstack(y_score_rows)
        y_true   = np.asarray(y_true_ids, dtype=int)
        y_pred1  = np.asarray(y_pred_ids_top1, dtype=int)

        for k in k_list:
            topk_idx = np.argpartition(y_scores, -k, axis=1)[:, -k:]
            hits = [(t in row) for t, row in zip(y_true, topk_idx)]
            agg[f"top{k}_accuracy"] = float(np.mean(hits)) if hits else 0.0

        cm = confusion_matrix(y_true, y_pred1, labels=list(range(len(scp_codes))))
        pd.DataFrame(cm, index=scp_codes, columns=scp_codes).to_csv(mdir / "confusion_matrix.csv")

        report = classification_report(
            y_true, y_pred1, labels=list(range(len(scp_codes))),
            target_names=scp_codes, output_dict=True, zero_division=0
        )

        rows = []
        for cls in scp_codes:
            if cls in report:
                R = report[cls]
                rows.append({
                    "scp": cls,
                    "precision": R.get("precision", 0.0),
                    "recall": R.get("recall", 0.0),
                    "f1": R.get("f1-score", 0.0),
                    "support": R.get("support", 0),
                })
        if rows:
            pd.DataFrame(rows).to_csv(mdir / "per_class.csv", index=False)

        for key in ["macro avg","weighted avg"]:
            if key in report:
                agg[f"{key.replace(' ','_')}_precision"] = report[key]["precision"]
                agg[f"{key.replace(' ','_')}_recall"]    = report[key]["recall"]
                agg[f"{key.replace(' ','_')}_f1"]        = report[key]["f1-score"]

        # Hard cases
        if not per_df.empty and "gt_diag_scp" in per_df.columns:
            ps = per_df[per_df["gt_diag_scp"].isin(scp_codes)].copy()
            ps["correct_top1"] = ps["gt_diag_scp"] == ps["pred_top1_scp"]
            miss = (ps.groupby("gt_diag_scp")["correct_top1"]
                      .apply(lambda s: 1.0 - s.mean())
                      .reset_index(name="miss_rate"))
            def most_common_pred(g):
                wrong = g[g["correct_top1"]==False]["pred_top1_scp"]
                return wrong.mode().iloc[0] if len(wrong) else ""
            mc = ps.groupby("gt_diag_scp").apply(most_common_pred).reset_index(name="most_common_wrong_pred")
            miss.merge(mc, on="gt_diag_scp").sort_values("miss_rate", ascending=False).to_csv(mdir / "hard_cases.csv", index=False)

    pd.DataFrame([agg]).to_csv(mdir / "aggregates.csv", index=False)
    print(f"[{model_name}] wrote results under: {mdir}")

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ECG prompts (once) and evaluate.")
    p.add_argument("--db", type=str, default=str(DB_DEFAULT), help="Path to ptbxl_database.csv")
    p.add_argument("--sql_table", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=str(OUT_DEFAULT))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--scp_csv", type=str, default=None, help="CSV mapping of SCP code -> description")
    p.add_argument("--models", type=str, required=True, help="Comma-separated model names, e.g. 'medgemma-4b-it'")
    p.add_argument("--k_list", type=str, default="1,3,5")
    p.add_argument("--bert_model", type=str, default="allenai/scibert_scivocab_uncased",
                   help="Model for BERTScore (e.g., 'allenai/scibert_scivocab_uncased', 'bert-base-uncased').")
    p.add_argument("--bert_no_rescale", action="store_true")
    p.add_argument("--variant", default="roundrobin", choices=["1","2","3","4","5","roundrobin"])
    p.add_argument("--save_json", action="store_true")
    # ---- NEW: demographics controls
    p.add_argument("--exclude_demographics", type=str, default="",
                   help="Comma-separated fields to exclude (e.g., 'bmi,sex')")
    p.add_argument("--age_col", type=str, default="age")
    p.add_argument("--sex_col", type=str, default="sex")
    p.add_argument("--height_col", type=str, default="height")
    p.add_argument("--weight_col", type=str, default="weight")
    p.add_argument("--bert_lang", type=str, default="en",
                   help="Language code for BERTScore baseline when --bert_no_rescale is NOT set.")
    p.add_argument("--rag_corpus", type=str, default="", help="Folder with .txt/.md domain snippets")
    p.add_argument("--rag_top_k", type=int, default=4)
    p.add_argument("--rag_off", action="store_true", help="Disable retrieval even if index exists")
    p.add_argument("--rag_scp_csv", type=str, default="", help="Use scp_statements.csv as RAG corpus")


    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(args.db)

    outdir = Path(args.output_dir); ensure_dir(outdir)
    k_list = tuple(int(x) for x in args.k_list.split(",") if x.strip())

    # Compute once; auto-disable rescale for SciBERT
    bert_rescale = not args.bert_no_rescale
    if "scibert" in args.bert_model.lower():
        bert_rescale = False

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    keymap = {
        "age": args.age_col,
        "sex": args.sex_col,
        "height": args.height_col,
        "weight": args.weight_col,
    }

    # ---- RAG retriever (build once) ----
    idx_path = str(Path(args.output_dir) / "rag_index.faiss")
    retriever = RAGRetriever(index_path=idx_path)
    if args.rag_corpus:
        idx_path = str(Path(args.output_dir) / "rag_index.faiss")
        retriever = RAGRetriever(index_path=idx_path)
        if not (os.path.exists(idx_path) and os.path.exists(idx_path + ".docs.json")):
            print(f"[RAG] building index from {args.rag_corpus}")
            retriever.build_from_folder(args.rag_corpus)
        else:
            retriever.load()
        if not retriever.ready():
            print("[RAG] No documents indexed; RAG will be disabled.")
            retriever = None

    for m in model_list:
        evaluate_one_model(
            model_name=m,
            db_path=args.db,
            outdir=outdir,
            limit=args.limit,
            scp_csv_path=args.scp_csv,
            k_list=k_list,
            bert_model=args.bert_model,
            bert_rescale=bert_rescale,   # <-- use computed value (not "not args.bert_no_rescale")
            sql_table=args.sql_table,
            variant=args.variant,
            save_json=args.save_json,
            exclude_demographics=args.exclude_demographics if hasattr(args, "exclude_demographics") else "",
            keymap=None,
            bert_lang=args.bert_lang,
            retriever=retriever,
            rag_top_k=args.rag_top_k if hasattr(args, "rag_top_k") else 4,
            rag_off=args.rag_off if hasattr(args, "rag_off") else False,
        )
if __name__ == "__main__":
    main()