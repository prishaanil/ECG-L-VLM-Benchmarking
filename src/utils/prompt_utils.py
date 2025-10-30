# utils/prompt_utils.py

from __future__ import annotations
import math
import pandas as pd
import re
from typing import Dict, Iterable, Optional

_UNRESOLVED = re.compile(r"{[^}]+}")

def _clean_num(x):
    # treat "", None, NaN as missing
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _map_sex(v):
    s = "" if v is None else str(v).strip().lower()
    # PTB-XL: 1=male, 0=female
    if s in {"1", "m", "male"}:
        return "Male"
    if s in {"0", "f", "female"}:
        return "Female"
    return None

def build_patient_block(row, *, exclude=None, keymap=None) -> str:
    """
    Builds a compact 'Patient Information' block, skipping any field that is missing.
    exclude: set like {"height","weight"} to remove fields entirely.
    keymap:  maps canonical keys -> row columns {age,sex,height,weight}.
    """
    exclude = exclude or set()
    keymap = keymap or {"age": "age", "sex": "sex", "height": "height", "weight": "weight"}

    age    = _clean_num(row.get(keymap.get("age", "age")))
    sex    = _map_sex(row.get(keymap.get("sex", "sex")))
    height = _clean_num(row.get(keymap.get("height", "height")))
    weight = _clean_num(row.get(keymap.get("weight", "weight")))

    lines = ["Patient Information:"]
    if ("age" not in exclude) and (age is not None):
        lines.append(f"  Age: {int(age) if age.is_integer() else age}")
    if ("sex" not in exclude) and (sex is not None):
        lines.append(f"  Sex: {sex}")
    if ("height" not in exclude) and (height is not None):
        # PTB-XL height often missing; only include if present
        h = int(height) if float(height).is_integer() else height
        lines.append(f"  Height: {h} cm")
    if ("weight" not in exclude) and (weight is not None):
        w = int(weight) if float(weight).is_integer() else weight
        lines.append(f"  Weight: {w} kg")

    # In case all were missing, include one neutral line so the prompt doesn’t look broken.
    if len(lines) == 1:
        lines.append("  (no additional demographics provided)")

    return "\n".join(lines)

def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"

def _fmt_intlike(x) -> Optional[str]:
    if _is_missing(x):
        return None
    try:
        return str(int(round(float(x))))
    except Exception:
        s = str(x).strip()
        return s if s else None

def _fmt_sex(x) -> Optional[str]:
    if _is_missing(x):
        return None
    s = str(x).strip().lower()
    # handle numeric encodings and floats like "1.0"/"0.0"
    if s in {"1", "1.0", "m", "male"}:   return "Male"
    if s in {"0", "0.0", "f", "female"}: return "Female"
    return str(x).strip().title() if str(x).strip() else None

def _maybe_bmi(height_cm: Optional[str], weight_kg: Optional[str]) -> Optional[str]:
    if height_cm is None or weight_kg is None:
        return None
    try:
        h = float(height_cm); w = float(weight_kg)
        if h > 0 and w > 0:
            return f"{w / ((h/100.0)**2):.1f}"
    except Exception:
        return None
    return None

def render_prompt(
    template: str,
    patient_block: str,
    row: dict | None = None,
    keymap: dict | None = None,
    strip_leftovers: bool = True,
) -> str:
    """
    Insert {patient_block}; if legacy placeholders remain, fill them with
    formatted values (age/height/weight int-like; sex normalized).
    """
    keymap = keymap or {"age": "age", "sex": "sex", "height": "height", "weight": "weight"}

    out = template
    if "{patient_block}" in out:
        out = out.format(patient_block=patient_block)

    legacy_keys = ["age", "sex", "height", "weight"]
    if any("{" + k + "}" in out for k in legacy_keys):
        rd = row or {}
        ctx = {
            "age":    _fmt_intlike(rd.get(keymap.get("age", "age"))) or "unknown",
            "sex":    _fmt_sex(rd.get(keymap.get("sex", "sex"))) or "unknown",
            "height": _fmt_intlike(rd.get(keymap.get("height", "height"))) or "unknown",
            "weight": _fmt_intlike(rd.get(keymap.get("weight", "weight"))) or "unknown",
        }
        try:
            out = out.format(**ctx)
        except KeyError:
            if strip_leftovers:
                out = _UNRESOLVED.sub("", out)
            else:
                raise

    if strip_leftovers and _UNRESOLVED.search(out):
        out = _UNRESOLVED.sub("", out)
    return out
