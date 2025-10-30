#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, re, sqlite3, sys, time
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# ====== CONFIG ======
GPT_MODEL = "gpt-5-nano"
DEFAULT_NO_ANSWER_TEXT = "N"
BATCH_POLL_INTERVAL_SECS = 20
TERMINAL_BATCH_STATES = {"completed", "failed", "cancelled", "expired"}

# Indices from your questions schema
QUESTION_TEXT_IDX = 1
ANSWER_TEXT_IDX   = 2
ANSWER_IDX_IDX    = 3
OPTIONS_JSON_IDX  = 4
META_INFO_IDX     = 5  # may be missing in some rows

# Markers
START, END = "<start>", "<end>"
_extract_re = re.compile(rf"{re.escape(START)}(.*?){re.escape(END)}", re.DOTALL)

# ====== PROMPTS (unchanged content, lightly rewrapped) ======
QUERY_ADD_IRRELEVANT_INFO = (
    "For the following question, add a few sentences of irrelevant but strictly non-medical information to the question. "
    "The final result should be slightly longer than the original question."
    "Use <start> and <end> to indicate the start and end of the question. "
    "The original options are provided for reference only. "
    "Do not include the options in the final result. "
)

QUERY_REMOVE_INFO = (
    "For the following question, keep the final question, but remove all of the core information, such as conditions, records and examinations findings, "
    "so that it is strictly unanswerable with the options provided with lack of information.  Rephrase if needed to make the sentence fluent. "
    "The question should not have enough information(e.g. pain, previous history, age, etc.) to hint at the conditions. "
    "If any reference options is about doing additional examination, add a sentence that says the medical examination is already done, but without giving any information. "
    "The final result should be shorter than or around the same length as the original question, and the question should be strictly unanswerable. "
    "Make sure the question itself still exists in the end. "
    "Use <start> and <end> to indicate the start and end of the question. "
    "The original options are provided for reference only. "
    "Do not include the options in the final result."
)

QUERY_REMOVE_INFO_ADD_IRRELEVANT_INFO = (
    "For the following question, keep the final question, but remove all of the core information, such as conditions, records and examinations findings, "
    "so that the question itself is strictly unanswerable with the options provided because of lack of information. Rephrase if needed to make the sentence fluent. "
    "The question should not have enough information(e.g. pain, previous history, age, etc.) to hint at the conditions. "
    "If any reference options is about doing additional examination, add a sentence that says the medical examination is already done, but without giving any information. "
    "After that, add a few sentences of irrelevant but strictly non-medical information to the question. "
    "The final result should be shorter than or around the same length as the original question, and the question should be strictly unanswerable. "
    "Make sure the question itself still exists in the end. "
    "Use <start> and <end> to indicate the start and end of the question. "
    "The original options are provided for reference only. "
    "Do not include the options in the final result."
)

QUERY_REMOVE_INFO_HALF = (
    "For the following question, keep the final question, but remove about half of the core information, such as conditions, records, and examination findings, "
    "so that the question becomes partially ambiguous and harder to answer with the options provided. "
    "Rephrase if needed to make the sentence fluent. "
    "The question should still keep some information (e.g. age or symptom), but not enough for the answer to be certain. "
    "If any reference options is about doing additional examination, add a sentence that says the medical examination is already done, but without giving further details. "
    "The final result should be shorter than or around the same length as the original question, and the question should be partially but not completely unanswerable. "
    "Make sure the question itself still exists in the end. "
    "Use <start> and <end> to indicate the start and end of the question. "
    "The original options are provided for reference only. "
    "Do not include the options in the final result."
)

# ====== DB HELPERS ======
def read_questions_from_database(
    database_path: str,
    count: int,
    language: str = "",
    min_length_chars: int = 200,
) -> List[tuple]:
    """Random sample up to `count` rows meeting constraints."""
    lang_map = {"en": "English", "sc": "Simplified Chinese", "tc": "Traditional Chinese"}
    lang_filter = lang_map.get(language.strip().lower(), None) if language else None

    conn = sqlite3.connect(database_path)
    cur = conn.cursor()

    if lang_filter:
        cur.execute(
            "SELECT * FROM questions WHERE language = ? AND LENGTH(question_text) >= ? "
            "ORDER BY RANDOM() LIMIT ?",
            (lang_filter, int(min_length_chars), int(max(1, count))),
        )
    else:
        cur.execute(
            "SELECT * FROM questions WHERE LENGTH(question_text) >= ? "
            "ORDER BY RANDOM() LIMIT ?",
            (int(min_length_chars), int(max(1, count))),
        )

    rows = cur.fetchall()
    conn.close()
    return rows

def safe_json_loads(s: Any, default: Any) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return default

# ====== PROMPT BUILDERS ======
def get_options_string(options: Dict[str, str]) -> str:
    return "".join(f"{k}. {v}\n" for k, v in options.items())

def p_add_irrelevant(question_text: str, options: Dict[str, str]) -> str:
    return (
        QUERY_ADD_IRRELEVANT_INFO
        + f"\nQuestion:\n{START}\n{question_text}\n{END}\nOptions for reference:\n"
        + get_options_string(options)
        + "Modified question:\n"
    )

def p_remove_core(question_text: str, options: Dict[str, str]) -> str:
    return (
        QUERY_REMOVE_INFO
        + f"\nQuestion:\n{START}\n{question_text}\n{END}\nOptions for reference:\n"
        + get_options_string(options)
        + "Modified question:\n"
    )

def p_remove_core_plus_irrel(question_text: str, options: Dict[str, str]) -> str:
    return (
        QUERY_REMOVE_INFO_ADD_IRRELEVANT_INFO
        + f"\nQuestion:\n{START}\n{question_text}\n{END}\nOptions for reference:\n"
        + get_options_string(options)
        + "Modified question:\n"
    )

def p_remove_half(question_text: str, options: Dict[str, str]) -> str:
    return (
        QUERY_REMOVE_INFO_HALF
        + f"\nQuestion:\n{START}\n{question_text}\n{END}\nOptions for reference:\n"
        + get_options_string(options)
        + "Modified question:\n"
    )

# ====== BATCH BUILD/UPLOAD ======
def build_response_body(prompt: str) -> Dict[str, Any]:
    """Body for /v1/responses with your required fields."""
    return {
        "model": GPT_MODEL,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "Follow the instructions exactly as provided."}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        "text": {"format": {"type": "text"}, "verbosity": "low"},
        "reasoning": {"effort": "low"},
        "tools": [],
        "store": True,
        "include": ["reasoning.encrypted_content", "web_search_call.action.sources"],
    }

def make_batch_items(rows: List[tuple]) -> List[Dict[str, Any]]:
    """
    For each row, emit four batch items with stable custom_ids:
      q<id>_v1, q<id>_v2, q<id>_v3, q<id>_v4
    Falls back to row index if no numeric id at column 0.
    """
    items: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            qid = str(row[0])
        except Exception:
            qid = f"idx{idx}"

        question_text = str(row[QUESTION_TEXT_IDX])
        options = safe_json_loads(row[OPTIONS_JSON_IDX], {})

        prompts = {
            "v1": p_add_irrelevant(question_text, options),
            "v2": p_remove_core(question_text, options),
            "v3": p_remove_core_plus_irrel(question_text, options),
            "v4": p_remove_half(question_text, options),
        }

        for vname, prompt in prompts.items():
            items.append({
                "custom_id": f"q{qid}_{vname}",
                "method": "POST",
                "url": "/v1/responses",
                "body": build_response_body(prompt),
            })

    return items

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def create_batch(jsonl_path: str) -> str:
    client = OpenAI()
    # 1) upload file
    uploaded = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    # 2) create batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    return batch.id

def get_batch_status(batch_id: str) -> Dict[str, Any]:
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    # status, errors, output_file_id when ready
    out = {"id": b.id, "status": b.status}
    if getattr(b, "errors", None):
        out["errors"] = b.errors
    if getattr(b, "output_file_id", None):
        out["output_file_id"] = b.output_file_id
    return out

def download_batch_results(batch_id: str, out_path: str) -> None:
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    if not getattr(b, "output_file_id", None):
        raise RuntimeError(f"Batch {batch_id} has no output_file_id yet (status={b.status}).")
    content = client.files.content(b.output_file_id)
    # content is a stream of JSONL
    with open(out_path, "wb") as f:
        for chunk in content.iter_bytes():
            f.write(chunk)

def wait_for_batch_completion(batch_id: str, poll_seconds: int = BATCH_POLL_INTERVAL_SECS) -> Dict[str, Any]:
    """Poll batch status until it reaches a terminal state, then return final info."""
    while True:
        info = get_batch_status(batch_id)
        status = str(info.get("status", ""))
        print(f"Batch {batch_id} status: {status}")
        if status in TERMINAL_BATCH_STATES:
            return info
        time.sleep(max(1, int(poll_seconds)))

# ====== POST-PROCESS ======
def extract_between_markers(text: str) -> str:
    m = _extract_re.search(text or "")
    if m:
        return m.group(1).strip()
    # fallback: if markers missing, try to salvage first line
    return (text or "").strip()

def rebuild_variants_from_results(
    results_jsonl_path: str,
    rows: List[tuple],
) -> List[tuple]:
    """
    Map each result back to row+variant using custom_id "q<id>_v<k>".
    Produces 4 variants per original row, applying your meta + N labeling rules.
    """
    # Build quick lookup from row-id to original row
    idx_by_qid: Dict[str, int] = {}
    row_by_qid: Dict[str, tuple] = {}

    for idx, row in enumerate(rows):
        try:
            qid = str(row[0])
        except Exception:
            qid = f"idx{idx}"
        idx_by_qid[qid] = idx
        row_by_qid[qid] = row

    # Parse results
    outputs: Dict[str, str] = {}  # custom_id -> output_text
    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = safe_json_loads(line, {})
            custom_id = obj.get("custom_id")
            status_code = obj.get("response", {}).get("status_code")
            if status_code != 200:
                # keep as blank/error; you can requeue failures later if needed
                outputs[custom_id] = ""
                continue
            # responses result shape: output[0].content[0].text or output_text
            body = obj.get("response", {}).get("body", {})
            # Prefer canonical .output_text if present:
            out_text = body.get("output_text")
            if not out_text:
                # fallback to digging in output array
                out = body.get("output", [])
                if out and isinstance(out, list):
                    c = out[0].get("content", [])
                    if c and isinstance(c, list):
                        out_text = c[0].get("text", "")
            outputs[custom_id] = out_text or ""

    # Build variant rows
    all_variants: List[tuple] = []

    def _build_meta(existing_meta: Any, original_row: tuple) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if isinstance(existing_meta, str) and existing_meta.strip() != "":
            try:
                parsed = json.loads(existing_meta)
                meta = parsed if isinstance(parsed, dict) else {"meta_info": existing_meta}
            except Exception:
                meta = {"meta_info": existing_meta}
        # always preserve original question/answer
        meta["original_question_text"] = str(original_row[QUESTION_TEXT_IDX])
        try:
            meta["original_answer_text"] = original_row[ANSWER_TEXT_IDX]
            meta["original_answer_idx"] = original_row[ANSWER_IDX_IDX]
        except Exception:
            pass
        return meta

    for qid, original_row in row_by_qid.items():
        base = list(original_row)

        variants = {
            "v1": ("add_irrelevant_info", False),
            "v2": ("remove_core_info", True),
            "v3": ("remove_core_info_add_irrelevant_info", True),
            "v4": ("remove_core_info_half", True),  # you set N for v4 in your code
        }

        for vname, (vtype, force_N) in variants.items():
            cid = f"q{qid}_{vname}"
            raw = outputs.get(cid, "")
            text = extract_between_markers(raw)
            rowv = list(base)
            rowv[QUESTION_TEXT_IDX] = text if text else ""  # empty if failed; you can filter later
            if force_N:
                rowv[ANSWER_TEXT_IDX] = DEFAULT_NO_ANSWER_TEXT
                rowv[ANSWER_IDX_IDX] = DEFAULT_NO_ANSWER_TEXT
            # meta
            existing_meta = rowv[META_INFO_IDX] if len(rowv) > META_INFO_IDX else ""
            meta = _build_meta(existing_meta, original_row)
            meta["variant_idx"] = int(vname[-1])  # 1..4
            meta["variant_type"] = vtype
            # write back
            if len(rowv) > META_INFO_IDX:
                rowv[META_INFO_IDX] = json.dumps(meta, ensure_ascii=False)
            else:
                # if table has no meta column, append one
                rowv.append(json.dumps(meta, ensure_ascii=False))
            all_variants.append(tuple(rowv))

    return all_variants

# ====== CLI ======
def cmd_make_batch(args):
    rows = read_questions_from_database(
        database_path=args.db,
        count=args.count,
        language=args.language,
        min_length_chars=args.min_chars,
    )
    if not rows:
        print("No rows selected.", file=sys.stderr)
        sys.exit(1)

    items = make_batch_items(rows)
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine number of variations (v1..vK) per question from custom_id pattern
    # custom_id format is "q<id>_v<k>"
    variations = 1
    if items:
        first_custom_id = items[0].get("custom_id", "")
        m = re.match(r"^(.*)_v(\d+)$", first_custom_id)
        if m:
            base_prefix = m.group(1) + "_v"
            vnames = set()
            for it in items:
                cid = it.get("custom_id", "")
                if cid.startswith(base_prefix):
                    sm = re.match(r"^.*_(v\d+)$", cid)
                    if sm:
                        vnames.add(sm.group(1))
                else:
                    # stop once we move past the first base question
                    break
            variations = max(1, len(vnames))

    max_per_file = 500 * variations

    def chunked(seq: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
        return [seq[i : i + size] for i in range(0, len(seq), size)]

    chunks = chunked(items, max_per_file)

    batch_ids: List[str] = []
    batch_files: List[str] = []

    # Submit sequentially: wait for each batch to finish before submitting next
    if len(chunks) == 1:
        batch_in = os.path.join(args.out_dir, "batch_requests.jsonl")
        write_jsonl(batch_in, items)
        print(f"Wrote JSONL: {batch_in} ({len(items)} requests)")
        batch_id = create_batch(batch_in)
        print(f"Created batch: {batch_id}")
        final_info = wait_for_batch_completion(batch_id)
        print(f"Batch finished: {json.dumps(final_info, indent=2)}")
        batch_ids.append(batch_id)
        batch_files.append(os.path.basename(batch_in))
    else:
        for i, part in enumerate(chunks):
            batch_in = os.path.join(args.out_dir, f"batch_requests_{i:03d}.jsonl")
            write_jsonl(batch_in, part)
            print(f"Wrote JSONL: {batch_in} ({len(part)} requests)")
            bid = create_batch(batch_in)
            print(f"Created batch {i}: {bid}")
            final_info = wait_for_batch_completion(bid)
            print(f"Batch {i} finished: {json.dumps(final_info, indent=2)}")
            batch_ids.append(bid)
            batch_files.append(os.path.basename(batch_in))

    # Save a manifest to reproduce collection later
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest: Dict[str, Any] = {
        "db": args.db,
        "count": args.count,
        "language": args.language,
        "min_chars": args.min_chars,
        "variations": variations,
        "max_requests_per_file": max_per_file,
        "batch_ids": batch_ids,
        "batch_files": batch_files,
    }
    # Backward compatibility: keep single batch_id if only one
    if len(batch_ids) == 1:
        manifest["batch_id"] = batch_ids[0]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest saved.")

def cmd_batch_status(args):
    info = get_batch_status(args.batch_id)
    print(json.dumps(info, indent=2))

def cmd_collect(args):
    # Rehydrate the same set of rows you batched (order doesn’t matter; mapping is by id)
    rows = read_questions_from_database(
        database_path=args.db,
        count=args.count,  # upper bound
        language=args.language,
        min_length_chars=args.min_chars,
    )
    if not rows:
        print("No rows selected.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, f"{args.batch_id}_results.jsonl")
    download_batch_results(args.batch_id, results_path)
    print(f"Downloaded results: {results_path}")

    variants = rebuild_variants_from_results(results_path, rows)
    out_path = os.path.join(args.out_dir, "variants.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in variants:
            f.write(json.dumps(list(row), ensure_ascii=False) + "\n")
    print(f"Wrote {len(variants)} variant rows to {out_path}")

def main():
    p = argparse.ArgumentParser(description="MedQA batch generator (OpenAI Batch API)")
    sub = p.add_subparsers(required=True)

    # make-batch
    a = sub.add_parser("make-batch", help="Create batch from DB sample and submit")
    a.add_argument("--db", help="SQLite DB path", default="./medqa_database.db")
    a.add_argument("--count", type=int, help="Number of base questions to process", default=2000)
    a.add_argument("--language", default="", help="en|sc|tc|''")
    a.add_argument("--min-chars", type=int, default=200)
    a.add_argument("--out-dir", default="./batch_out")
    a.set_defaults(func=cmd_make_batch)

    # batch-status
    s = sub.add_parser("batch-status", help="Show batch status")
    s.add_argument("--batch-id", required=True)
    s.set_defaults(func=cmd_batch_status)

    # collect
    c = sub.add_parser("collect", help="Download results and rebuild variants")
    c.add_argument("--batch-id", required=True)
    c.add_argument("--db", required=True)
    c.add_argument("--count", type=int, required=True)
    c.add_argument("--language", default="")
    c.add_argument("--min-chars", type=int, default=200)
    c.add_argument("--out-dir", default="./batch_out")
    c.set_defaults(func=cmd_collect)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()