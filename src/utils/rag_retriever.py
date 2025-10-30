# src/utils/rag_retriever.py
from __future__ import annotations

import os
import re
import json
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import faiss
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ----------------------------
# Config / defaults
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# You can override via ENV: export RAG_EMB_MODEL="allenai/scibert_scivocab_uncased"
EMB_MODEL = os.getenv("RAG_EMB_MODEL", "allenai/scibert_scivocab_uncased")

# ----------------------------
# Utilities
# ----------------------------
@torch.no_grad()
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

@torch.no_grad()
def _encode_texts(model, tok, texts: List[str], max_length: int = 512) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(**enc)
    pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])  # (N, H)
    pooled = F.normalize(pooled, p=2, dim=1)                            # cosine-ready
    return pooled.detach().cpu().numpy()

def _pick_meaning_col(df: pd.DataFrame) -> str:
    for c in ["SCP-ECG Statement Description",
              "SCP-ECG statement description",
              "description",
              "Description"]:
        if c in df.columns:
            return c
    raise ValueError(
        f"No suitable description column found. Columns present: {list(df.columns)}"
    )

# ----------------------------
# Public API
# ----------------------------
@dataclass
class Doc:
    id: int
    title: str   # e.g., SCP code like 'NDT'
    text: str
    meta: dict

class RAGRetriever:
    """
    Minimal FAISS-based retriever for ECG RAG.

    Build directly from scp_statements.csv:
        retriever = RAGRetriever(".../rag_index.faiss")
        retriever.build_from_scp_csv("data/scp_statements.csv")

    Or (optional) build from a folder of .txt/.md files:
        retriever.build_from_folder("data/rag_corpus")
    """
    def __init__(self, index_path: str, emb_model: str = EMB_MODEL):
        self.index_path = index_path
        self.emb_model_id = emb_model
        self.docs: List[Doc] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.model = None
        self.tok = None

    # ----- model init -----
    def _load_model(self):
        if self.model is None:
            self.tok = AutoTokenizer.from_pretrained(self.emb_model_id)
            self.model = AutoModel.from_pretrained(self.emb_model_id).to(DEVICE).eval()

    # ----- build from SCP CSV (Option B) -----
    def build_from_scp_csv(self, csv_path: str) -> None:
        """
        Build FAISS index directly from scp_statements.csv.
        The first column is assumed to be the SCP code index.
        Stores:
          - FAISS index at index_path
          - docs metadata at index_path + ".docs.json"
        """
        self._load_model()

        df = pd.read_csv(csv_path, index_col=0).reset_index().rename(columns={"index": "scp"})
        df["scp"] = df["scp"].astype(str).str.strip()
        meaning_col = _pick_meaning_col(df)

        self.docs = []
        texts: List[str] = []
        for _, r in df.iterrows():
            code = r.get("scp", "").strip()
            if not code:
                continue
            desc = str(r.get(meaning_col, "") or "").strip()
            if not desc:
                continue

            diag_class  = str(r.get("diagnostic_class", "") or "").strip()
            diag_sub    = str(r.get("diagnostic_subclass", "") or "").strip()
            rhythm      = str(r.get("rhythm", "") or "").strip()
            form        = str(r.get("form", "") or "").strip()

            # Compose a compact doc: keep code & description first
            lines = [f"{code} — {desc}"]
            meta_bits = []
            if diag_class: meta_bits.append(f"class={diag_class}")
            if diag_sub:   meta_bits.append(f"subclass={diag_sub}")
            if rhythm:     meta_bits.append(f"rhythm={rhythm}")
            if form:       meta_bits.append(f"form={form}")
            if meta_bits:
                lines.append(" / ".join(meta_bits))
            # Gentle reporting cue to improve model style
            lines.append("Reporting cues: Rate → Rhythm → Axis → Intervals (PR/QRS/QT) → QRS morphology → ST/T → Impression.")
            text = "\n".join(lines)

            doc_id = len(self.docs)
            self.docs.append(Doc(id=doc_id, title=code, text=text, meta={"code": code}))
            texts.append(text)

        if not texts:
            self.index = None
            return

        X = _encode_texts(self.model, self.tok, texts)  # (N, H)
        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(X.astype(np.float32))

        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".docs.json", "w", encoding="utf-8") as f:
            json.dump([d.__dict__ for d in self.docs], f, ensure_ascii=False, indent=2)

    # ----- OPTIONAL: build from folder of .txt/.md -----
    def build_from_folder(self, folder: str) -> None:
        """Build FAISS index from all .txt/.md files in a folder."""
        self._load_model()
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")) +
                       glob.glob(os.path.join(folder, "*.md")))
        self.docs = []
        texts = []
        for i, p in enumerate(paths):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception:
                continue
            if not text:
                continue
            title = os.path.basename(p)
            self.docs.append(Doc(id=len(self.docs), title=title, text=text, meta={"path": p}))
            texts.append(text)

        if not texts:
            self.index = None
            return

        X = _encode_texts(self.model, self.tok, texts)
        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(X.astype(np.float32))

        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".docs.json", "w", encoding="utf-8") as f:
            json.dump([d.__dict__ for d in self.docs], f, ensure_ascii=False, indent=2)

    # ----- load / ready -----
    def load(self) -> None:
        """Load an existing FAISS index + docs metadata."""
        if not (os.path.exists(self.index_path) and os.path.exists(self.index_path + ".docs.json")):
            self.index = None
            self.docs = []
            return
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".docs.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.docs = [Doc(**d) for d in raw]
        self._load_model()

    def ready(self) -> bool:
        return self.index is not None and len(self.docs) > 0

    # ----- search -----
    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[float, Doc]]:
        """
        Returns a list of (similarity, Doc) for top_k hits.
        Similarity is cosine (dot product on L2-normalized embeddings).
        """
        if not self.ready():
            return []
        qv = _encode_texts(self.model, self.tok, [query])  # (1, H)
        D, I = self.index.search(qv.astype(np.float32), top_k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            out.append((float(score), self.docs[int(idx)]))
        return out

# ----------------------------
# Context formatter for prompts
# ----------------------------
def format_context(hits: List[Tuple[float, Doc]], max_chars: int = 1200) -> str:
    """
    Make a compact, readable context block from search hits.
    Keep it short to avoid blowing up the prompt.
    """
    blocks = []
    for s, d in hits:
        snippet = d.text if len(d.text) <= 800 else (d.text[:800] + " …")
        blocks.append(f"- [{d.title}] (sim {s:.2f})\n{snippet}")
    ctx = "\n\n".join(blocks)
    return ctx[:max_chars]

__all__ = [
    "RAGRetriever",
    "Doc",
    "format_context",
]