from typing import List, Dict, Any, Optional
import sqlite3
from utils.llm_utils import request_llm
import json
from dotenv import load_dotenv
import asyncio, random, re
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIStatusError

load_dotenv(override=True)

# Constants
# DEFAULT_LLM_URL = "http://127.0.0.1:8000/generate"
DEFAULT_LANGUAGE = "English"
DEFAULT_NO_ANSWER_TEXT = "N"
DEFAULT_BATCH_SIZE = 3
RESPONSE_SKIP_LINES = 2

GPT_MODEL = "gpt-5-nano"
QUERY_ADD_IRRELEVANT_INFO = "For the following question, add a few sentences of irrelevant but strictly non-medical information to the question. " + \
    "The final result should be slightly longer than the original question." + \
    "Use <start> and <end> to indicate the start and end of the question. " + \
    "The original options are provided for reference only. " + \
    "Do not include the options in the final result. "
QUERY_REMOVE_INFO = "For the following question, keep the final question, but remove all of the core information, such as conditions, records and examinations findings, " + \
    "so that it is strictly unanswerable with the options provided with lack of information.  Rephrase if needed to make the sentence fluent. " + \
    "The question should not have enough information(e.g. pain, previous history, age, etc.) to hint at the conditions. " + \
    "If any reference options is about doing additional examination, add a sentence that says the medical examination is already done, but without giving any information. " + \
    "The final result should be shorter than or around the same length as the original question, and the question should be strictly unanswerable. " + \
    "Make sure the question itself still exists in the end. " + \
    "Use <start> and <end> to indicate the start and end of the question. " + \
    "The original options are provided for reference only. " + \
    "Do not include the options in the final result."
QUERY_REMOVE_INFO_ADD_IRRELEVANT_INFO = "For the following question, keep the final question, but remove all of the core information, such as conditions, records and examinations findings, " + \
    "so that the question itself is strictly unanswerable with the options provided because of lack of information. Rephrase if needed to make the sentence fluent. " + \
    "The question should not have enough information(e.g. pain, previous history, age, etc.) to hint at the conditions. " + \
    "If any reference options is about doing additional examination, add a sentence that says the medical examination is already done, but without giving any information. " + \
    "After that, add a few sentences of irrelevant but strictly non-medical information to the question. " + \
    "The final result should be shorter than or around the same length as the original question, and the question should be strictly unanswerable. " + \
    "Make sure the question itself still exists in the end. " + \
    "Use <start> and <end> to indicate the start and end of the question. " + \
    "The original options are provided for reference only. " + \
    "Do not include the options in the final result."
QUERY_REMOVE_INFO_HALF = "For the following question, keep the final question, but remove about half of the core information, such as conditions, records, and examination findings, " + \
    "so that the question becomes partially ambiguous and harder to answer with the options provided. " + \
    "Rephrase if needed to make the sentence fluent. " + \
    "The question should still keep some information (e.g. age or symptom), but not enough for the answer to be certain. " + \
    "If any reference options is about doing additional examination, add a sentence that says the medical examination is already done, but without giving further details. " + \
    "The final result should be shorter than or around the same length as the original question, and the question should be partially but not completely unanswerable. " + \
    "Make sure the question itself still exists in the end. " + \
    "Use <start> and <end> to indicate the start and end of the question. " + \
    "The original options are provided for reference only. " + \
    "Do not include the options in the final result."

START, END = "<start>", "<end>"
_EXTRACT_RE = re.compile(rf"{re.escape(START)}(.*?){re.escape(END)}", re.DOTALL)

def _extract_between_markers(text: str) -> str:
    if not text:
        return ""
    m = _EXTRACT_RE.search(text)
    return m.group(1).strip() if m else text.strip()

# --- ASYNC OPENAI CALL WITH RETRIES ---
async def _request_gpt_async(client: AsyncOpenAI, prompt: str) -> str:
    resp = await client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "developer",
             "content": [{"type": "input_text", "text": "Follow the instructions exactly as provided."}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": prompt}]},
        ],
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "low"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content", "web_search_call.action.sources"],
    )
    return resp.output_text or ""

async def _request_with_retries(client: AsyncOpenAI, prompt: str, max_retries: int = 6) -> str:
    delay = 0.5
    for attempt in range(max_retries):
        try:
            return await _request_gpt_async(client, prompt)
        except (RateLimitError, APIStatusError) as e:
            # 429 or 5xx -> backoff + retry
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay + random.random() * 0.3)
            delay = min(delay * 2, 8.0)
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay + random.random() * 0.3)
            delay = min(delay * 2, 8.0)

# --- JOB MODEL ---
# For order stability, we prebuild jobs in strict sequential order:
# [ (job_idx, row_idx, "v1", prompt), (job_idx, row_idx, "v2", prompt), ... ]
from typing import NamedTuple
class Job(NamedTuple):
    job_idx: int
    row_idx: int
    variant: str  # "v1" | "v2" | "v3" | "v4"
    prompt: str

def _build_jobs(rows: list) -> list[Job]:
    jobs: list[Job] = []
    job_idx = 0
    for i, original_row in enumerate(rows):
        question_text = original_row[1]
        try:
            options = json.loads(original_row[4])
        except Exception:
            options = {}

        # Your existing prompt builders
        prompt_1 = query_rephrase_question_add_irrelevant_info(question_text, options)       # v1
        prompt_2 = query_rephrase_unanswerable_question_remove_core_info(question_text, options)  # v2
        prompt_3 = query_rephrase_unanswerable_question_remove_core_info_add_irrelevant_info(question_text, options)  # v3
        prompt_4 = query_rephrase_unanswerable_question_remove_core_info_half(question_text, options)  # v4

        jobs.append(Job(job_idx, i, "v1", prompt_1)); job_idx += 1
        jobs.append(Job(job_idx, i, "v2", prompt_2)); job_idx += 1
        jobs.append(Job(job_idx, i, "v3", prompt_3)); job_idx += 1
        jobs.append(Job(job_idx, i, "v4", prompt_4)); job_idx += 1
    return jobs

# --- RUN CONCURRENTLY WITH CAP + PROGRESS ---
async def _run_jobs(jobs: list[Job], max_concurrency: int = 16) -> list[str]:
    """
    Returns a list of raw model outputs aligned by job_idx.
    """
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(max_concurrency)
    total = len(jobs)
    done = 0
    done_lock = asyncio.Lock()
    results: list[Optional[str]] = [None] * total

    async def one(job: Job):
        nonlocal done
        async with sem:
            out = await _request_with_retries(client, job.prompt)
        results[job.job_idx] = out
        async with done_lock:
            done += 1
            # live progress
            if done % 5 == 0:
                print(f"[progress] {done}/{total}", end="\r", flush=True)

    tasks = [asyncio.create_task(one(j)) for j in jobs]
    await asyncio.gather(*tasks)
    print()  # newline after progress line
    return [r or "" for r in results]

def request_gpt(prompt: str) -> str:
    client = OpenAI()
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {
            "role": "developer",
            "content": [
                {
                "type": "input_text",
                "text": "Follow the instructions exactly as provided."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "input_text",
                "text": prompt
                }
            ]
            }
        ],
        text={
            "format": {
            "type": "text"
            },
            "verbosity": "low"
        },
        reasoning={
            "effort": "low"
        },
        tools=[],
        store=True,
        include=[
            "reasoning.encrypted_content",
            "web_search_call.action.sources"
        ]
    )
    return response.output_text

def _add_options_to_query(question_text: str, options: Dict[str, str], use_numbers: bool = False, no_answer_text: str = DEFAULT_NO_ANSWER_TEXT) -> str:
    """Format a question with multiple choice options."""
    options_string = ""
    for key, value in options.items():
        options_string += f"{key}. {value}\n"
    prompt = f"""System: You are an answer selector. Output exactly one uppercase letter from {{A,B,C,D,E,N}}. Use N only if none of the options are relevant or the question cannot be answered from the information given.

User:
    Question:
    {question_text}

    Options:
{options_string}N. Can't be answered from the information given.

    Answer:"""

    return prompt

def get_options_string(options: Dict[str, str]) -> str:
    options_string = ""
    for key, value in options.items():
        options_string += f"{key}. {value}\n"
    return options_string

def query_rephrase_unanswerable_question_remove_core_info(question_text: str, options: Dict[str, str]) -> str:
    """Re-phrase a question to make it unanswerable."""
    options_string = get_options_string(options)
    prompt = QUERY_REMOVE_INFO + \
    f"""
Question:
<start>
{question_text}
<end>
Options for reference:
{options_string}
Modified question:
"""
    return prompt

def query_rephrase_unanswerable_question_remove_core_info_half(question_text: str, options: Dict[str, str]) -> str:
    """Re-phrase a question to make it unanswerable."""
    options_string = get_options_string(options)
    prompt = QUERY_REMOVE_INFO_HALF + \
    f"""
Question:
<start>
{question_text}
<end>
Options for reference:
{options_string}
Modified question:
"""
    return prompt

def query_rephrase_unanswerable_question_remove_core_info_add_irrelevant_info(question_text: str, options: Dict[str, str]) -> str:
    """Re-phrase a question to make it unanswerable."""
    options_string = get_options_string(options)
    prompt = QUERY_REMOVE_INFO_ADD_IRRELEVANT_INFO + \
    f"""
Question:
<start>
{question_text}
<end>
Options for reference:
{options_string}
Modified question:
"""
    return prompt

def query_rephrase_question_add_irrelevant_info(question_text: str, options: Dict[str, str]) -> str:
    """Re-phrase a question to make it unanswerable."""
    options_string = get_options_string(options)
    prompt = QUERY_ADD_IRRELEVANT_INFO + \
    f"""
Question:
<start>
{question_text}
<end>
Options for reference:
{options_string}
Modified question:
"""
    return prompt

def read_questions_from_database(database_path: str, count: int = 1, language: str = "", min_length_chars: int = 200) -> List[tuple]:
    """Retrieve random unique questions from the database.

    Args:
        database_path: Path to the SQLite database file.
        count: Number of random unique questions to return (default: 1). Values < 1 are treated as 1.
        language: Optional language filter. If set to 'en', 'sc', or 'tc', only select questions in that language.
            Mapping:
            - en -> 'English'
            - sc -> 'Simplified Chinese'
            - tc -> 'Traditional Chinese'
        min_length_chars: Minimum number of characters required in question_text (default: 250).

    Returns:
        A list of question rows (each row is a tuple), length up to ``count`` depending on availability.
    """
    if count < 1:
        count = 1

    lang_map = {
        "en": "English",
        "sc": "Simplified Chinese",
        "tc": "Traditional Chinese",
    }
    lang_filter = lang_map.get(language.strip().lower(), None) if language else None

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    if lang_filter:
        cursor.execute(
            "SELECT * FROM questions WHERE language = ? AND LENGTH(question_text) >= ? ORDER BY RANDOM() LIMIT ?",
            (lang_filter, int(min_length_chars), count),
        )
    else:
        cursor.execute(
            "SELECT * FROM questions WHERE LENGTH(question_text) >= ? ORDER BY RANDOM() LIMIT ?",
            (int(min_length_chars), count),
        )

    rows = cursor.fetchall()
    conn.close()
    return rows

def generate_question_variants(database_path: str, num_questions: int = 1, language: str = "", max_concurrency: int = 20) -> List[tuple]:
    """
    Drop-in replacement. Still returns rows in the SAME ORDER as your old sequential code:
    for each selected question: v1, v2, v3, v4.
    """
    rows = read_questions_from_database(database_path, count=num_questions, language=language, min_length_chars=200)
    if not rows:
        return []

    # Build meta-preserver
    def _build_meta(existing_meta: Any, original_row: tuple, variant_idx: int, variant_type: str) -> str:
        meta: Dict[str, Any] = {}
        if isinstance(existing_meta, str) and existing_meta.strip() != "":
            try:
                parsed = json.loads(existing_meta)
                meta = parsed if isinstance(parsed, dict) else {"meta_info": existing_meta}
            except Exception:
                meta = {"meta_info": existing_meta}
        meta["original_question_text"] = original_row[1]
        try:
            meta["original_answer_text"] = original_row[2]
            meta["original_answer_idx"] = original_row[3]
        except Exception:
            pass
        meta["variant_idx"] = variant_idx
        meta["variant_type"] = variant_type
        return json.dumps(meta, ensure_ascii=False)

    # 1) Build all jobs in strict sequential order
    jobs = _build_jobs(rows)  # 4 per row -> len = num_questions*4

    # 2) Fire concurrently (order preserved via job_idx)
    raw_outputs = asyncio.run(_run_jobs(jobs, max_concurrency=max_concurrency))

    # 3) Rebuild rows in the same order as before
    all_variants: List[tuple] = []
    # quick map from (row_idx, variant) -> output_text
    # since jobs are sequential by row and v1..v4, we can just iterate in that order using raw_outputs
    out_iter = iter(raw_outputs)

    for i, original_row in enumerate(rows):
        try:
            options = json.loads(original_row[4])
        except Exception:
            options = {}
        base = list(original_row)

        # v1: add irrelevant info (answer preserved)
        gen1 = _extract_between_markers(next(out_iter, ""))
        row1 = list(base)
        row1[1] = gen1
        meta1 = _build_meta(row1[5] if len(row1) > 5 else "", original_row, 1, "add_irrelevant_info")
        row1[5] = meta1
        all_variants.append(tuple(row1))

        # v2: remove core info (N/N)
        gen2 = _extract_between_markers(next(out_iter, ""))
        row2 = list(base)
        row2[1] = gen2
        row2[2] = DEFAULT_NO_ANSWER_TEXT
        row2[3] = DEFAULT_NO_ANSWER_TEXT
        meta2 = _build_meta(row2[5] if len(row2) > 5 else "", original_row, 2, "remove_core_info")
        row2[5] = meta2
        all_variants.append(tuple(row2))

        # v3: remove core info + add irrelevant info (N/N)
        gen3 = _extract_between_markers(next(out_iter, ""))
        row3 = list(base)
        row3[1] = gen3
        row3[2] = DEFAULT_NO_ANSWER_TEXT
        row3[3] = DEFAULT_NO_ANSWER_TEXT
        meta3 = _build_meta(row3[5] if len(row3) > 5 else "", original_row, 3, "remove_core_info_add_irrelevant_info")
        row3[5] = meta3
        all_variants.append(tuple(row3))

        # v4: remove half core info (you set N/N previously — preserved)
        gen4 = _extract_between_markers(next(out_iter, ""))
        row4 = list(base)
        row4[1] = gen4
        row4[2] = DEFAULT_NO_ANSWER_TEXT
        row4[3] = DEFAULT_NO_ANSWER_TEXT
        meta4 = _build_meta(row4[5] if len(row4) > 5 else "", original_row, 4, "remove_core_info_half")
        row4[5] = meta4
        all_variants.append(tuple(row4))

    return all_variants

def generate_rephrased_questions(question_text: str, options: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[str]:
    """Generate multiple rephrased versions of a question with the same answer options."""
    rephrase_prompt = (
        f"Please rephrase the following question in exactly {str(batch_size)} "
        f"different ways while keeping the same meaning, separate each rephrased "
        f"question with a new line. Do not include any other text in your "
        f"response. \n{question_text}\n"
    )
    response = request_llm(rephrase_prompt, mode="plain_text") # May need to run a normal LLM vs a medical tuned
    # Skip first 2 lines which typically contain formatting/header text
    rephrased_questions = [q for q in response.strip().split("\n")[RESPONSE_SKIP_LINES:] if q.strip()]
    
    queries = []
    for rephrased_q in rephrased_questions[:batch_size]:
        queries.append(_add_options_to_query(rephrased_q, options))
    
    return queries

if __name__ == "__main__":
    pass