import argparse
import os
import sqlite3
from typing import List, Tuple
import json

from utils.query_generation import generate_question_variants, _add_options_to_query


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure minimal schema for mappings only and drop redundant tables if present."""
    schema_sql = """  
    CREATE TABLE IF NOT EXISTS rephrased_unanswerable_mappings (
        id INTEGER PRIMARY KEY,
        original_question_id INTEGER NOT NULL,
        original_prompt_text TEXT NOT NULL,
        new_irrelevant_text TEXT,
        new_unanswerable_text TEXT,
        new_unanswerable_irrelevant_text TEXT,
        new_unanswerable_half_text TEXT,
        options_json TEXT NOT NULL,
        original_answer_text TEXT,
        original_answer_idx TEXT
    );
    """
    conn.executescript(schema_sql)
    conn.commit()

def save_questions(dst_db: str, questions: List[Tuple]) -> int:
    """Write aggregated mapping rows per original question with three variant prompts.

    The mapping table's `id` serves as the new id.
    """
    conn = sqlite3.connect(dst_db)
    try:
        _ensure_schema(conn)
        insert_map_sql = (
            "INSERT INTO rephrased_unanswerable_mappings (original_question_id, original_prompt_text, new_irrelevant_text, new_unanswerable_text, new_unanswerable_irrelevant_text, new_unanswerable_half_text, options_json, original_answer_text, original_answer_idx) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        # Aggregate three variants per original_id
        aggregates = {}
        order_counters = {}
        for row in questions:
            original_id = row[0]
            meta_info_str = row[5] if len(row) > 5 else ""
            try:
                meta_dict = json.loads(meta_info_str) if isinstance(meta_info_str, str) and meta_info_str.strip() != "" else {}
            except Exception:
                meta_dict = {}

            original_question_text = meta_dict.get("original_question_text")
            if not original_question_text:
                original_question_text = row[1] if isinstance(row[1], str) else str(row[1])

            try:
                options_obj = json.loads(row[4]) if isinstance(row[4], str) else row[4]
            except Exception:
                options_obj = {}

            try:
                options_dict = json.loads(row[4]) if isinstance(row[4], str) else (row[4] or {})
            except Exception:
                options_dict = {}

            original_prompt_text = _add_options_to_query(original_question_text, options_dict)
            new_question_text = row[1] if isinstance(row[1], str) else str(row[1])
            new_prompt_text = _add_options_to_query(new_question_text, options_dict)

            original_answer_text = meta_dict.get("original_answer_text") if isinstance(meta_dict, dict) else None
            original_answer_idx = meta_dict.get("original_answer_idx") if isinstance(meta_dict, dict) else None

            entry = aggregates.setdefault(
                original_id,
                {
                    "original_prompt_text": original_prompt_text,
                    "new_irrelevant_text": None,
                    "new_unanswerable_text": None,
                    "new_unanswerable_irrelevant_text": None,
                    "new_unanswerable_half_text": None,
                    "options_json": json.dumps(options_obj, ensure_ascii=False),
                    "original_answer_text": original_answer_text,
                    "original_answer_idx": original_answer_idx,
                },
            )

            # Prefer explicit variant_type; fallback to encounter order 1,2,3
            variant_type = None
            if isinstance(meta_dict, dict):
                variant_type = meta_dict.get("variant_type")
            if variant_type == "add_irrelevant_info":
                entry["new_irrelevant_text"] = new_prompt_text
            elif variant_type == "remove_core_info":
                entry["new_unanswerable_text"] = new_prompt_text
            elif variant_type == "remove_core_info_add_irrelevant_info":
                entry["new_unanswerable_irrelevant_text"] = new_prompt_text
            elif variant_type == "remove_core_info_half":
                entry["new_unanswerable_half_text"] = new_prompt_text
            else:
                # Fallback by order
                idx = order_counters.get(original_id, 0) + 1
                order_counters[original_id] = idx
                if idx == 1:
                    entry["new_irrelevant_text"] = new_prompt_text
                elif idx == 2:
                    entry["new_unanswerable_text"] = new_prompt_text
                elif idx == 3:
                    entry["new_unanswerable_irrelevant_text"] = new_prompt_text
                else:
                    entry["new_unanswerable_half_text"] = new_prompt_text

        # Insert one row per original
        for original_id, e in aggregates.items():
            conn.execute(
                insert_map_sql,
                (
                    original_id,
                    e["original_prompt_text"],
                    e["new_irrelevant_text"],
                    e["new_unanswerable_text"],
                    e["new_unanswerable_irrelevant_text"],
                    e["new_unanswerable_half_text"],
                    e["options_json"],
                    e["original_answer_text"],
                    e["original_answer_idx"],
                ),
            )

        conn.commit()
        return len(aggregates)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rephrased unanswerable MedQA test set")
    parser.add_argument("-n", "--num", type=int, default=2000, help="Number of random questions to sample")
    parser.add_argument("--language", type=str, default="en", help="Language filter: en|sc|tc or blank (default en)")
    parser.add_argument("--src", type=str, default=os.getenv("DATABASE_PATH", "./medqa_database.db"), help="Source DB path")
    parser.add_argument("--dst", type=str, default="./medqa_test_rephrased_unanswerable.db", help="Destination DB path")
    args = parser.parse_args()

    # Enforce English selection regardless of input by forcing language="en"
    # Generate three variants per question in order: 1) add irrelevant info, 2) remove core info, 3) remove core + add irrelevant
    questions = generate_question_variants(args.src, num_questions=args.num, language="en")
    if not questions:
        print("No questions generated.")
        return

    # Create destination DB (minimal schema)
    dst_conn = sqlite3.connect(args.dst)
    try:
        _ensure_schema(dst_conn)
    finally:
        dst_conn.close()

    inserted = save_questions(args.dst, questions)
    print(f"Saved {inserted} generated question variants to {args.dst}")


if __name__ == "__main__":
    main()

