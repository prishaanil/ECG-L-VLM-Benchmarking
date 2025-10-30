from datasets import load_dataset
import os
import random
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import requests
load_dotenv(verbose=True, override=True)

DEFAULT_LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8000")

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("FreedomIntelligence/PubMedVision", "PubMedVision_InstructionTuning_VQA")

def _normalize_and_resolve_image_path(image_path: str) -> str:
    """Normalize and resolve image path using environment base if needed."""
    resolved = os.path.expandvars(os.path.expanduser(image_path or ""))
    if resolved.startswith("file://"):
        resolved = resolved[len("file://"):]
    if not os.path.isabs(resolved):
        base_dir = os.getenv("DATA_STORAGE_HDD_PUBMEDVISION", "").strip()
        base_dir = os.path.expandvars(os.path.expanduser(base_dir)) if base_dir else base_dir
        if base_dir:
            resolved = os.path.join(base_dir, resolved)
    return os.path.abspath(resolved)

def _get_base_url(server_url: Optional[str]) -> str:
    return (server_url or DEFAULT_LLM_SERVER_URL).rstrip("/")

def _call_llm_multimodal(base_url: str, prompt_text: str, image_path: str, max_new_tokens: int = 300) -> Optional[str]:
    url = f"{base_url}/generate_multimodal"
    payload = {
        "prompt": prompt_text,
        "image_path": image_path,
        "max_new_tokens": max_new_tokens,
    }
    try:
        resp = requests.post(url, json=payload, timeout=600)
    except Exception as exc:
        print(f"Request failed: {exc}")
        return None
    if resp.status_code != 200:
        print(f"Server responded with {resp.status_code}: {resp.text}")
        return None
    data = resp.json()
    text = data.get("response") if isinstance(data, dict) else None
    if isinstance(text, str):
        return text
    print("No 'response' field in server reply.")
    return None

def generate_multimodal_query(original_question: str, add_na_response: bool = False):
    # The original questions are already formatted as a question for the LLM. No need to reformat.
    a=" Be short and concise.If the information is not present, respond with a simple N/A."
    return f"{original_question}{a if add_na_response else ''}."

def read_questions_from_dataset_pubmedvision_instructions(count: int = 1) -> List[Dict[str, Any]]:
    """Read random unique samples from PubMedVision InstructionTuning VQA.

    Structure per item:
    - image: list[str] of image location strings
    - conversations: dict with keys 'question' and 'response'
    - id: string
    - modality: string
    - body_part: string
    """
    if count < 1:
        count = 1

    dataset = ds["train"] if isinstance(ds, dict) and "train" in ds else load_dataset(
        "FreedomIntelligence/PubMedVision", "PubMedVision_InstructionTuning_VQA", split="train"
    )

    def normalize_conversations(conversations: Any) -> Dict[str, str]:
        """Normalize into a single question/response pair.

        Returns:
            Dict with keys 'question' and 'response'. Missing parts are empty strings.
        """
        question_text: str = ""
        response_text: str = ""

        def is_human(role_value: Any) -> bool:
            if not isinstance(role_value, str):
                return False
            role = role_value.strip().lower()
            return role in {"human", "user"}

        def is_assistant(role_value: Any) -> bool:
            if not isinstance(role_value, str):
                return False
            role = role_value.strip().lower()
            return role in {"gpt", "assistant"}

        if not conversations:
            return {"question": question_text, "response": response_text}

        if isinstance(conversations, list):
            last_was_question = False
            for turn in conversations:
                if isinstance(turn, dict):
                    text = turn.get("value") or turn.get("content")
                    role = turn.get("from") or turn.get("role")
                    if isinstance(text, str) and text.strip():
                        if not question_text and is_human(role):
                            question_text = text
                            last_was_question = True
                        elif not response_text and (is_assistant(role) or last_was_question):
                            response_text = text
                            break
                elif isinstance(turn, str) and turn.strip():
                    if not question_text:
                        question_text = turn
                        last_was_question = True
                    elif not response_text:
                        response_text = turn
                        break
            return {"question": question_text, "response": response_text}

        if isinstance(conversations, str) and conversations.strip():
            return {"question": conversations, "response": response_text}

        if isinstance(conversations, dict):
            q = conversations.get("question")
            r = conversations.get("response")
            if isinstance(q, str):
                question_text = q
            if isinstance(r, str):
                response_text = r
            return {"question": question_text, "response": response_text}

        return {"question": question_text, "response": response_text}

    def extract_image_ref(image_obj: Any) -> Optional[str]:
        try:
            # datasets.Image can expose .filename; PIL Images may also have .filename
            filename = getattr(image_obj, "filename", None)
            if filename:
                return str(filename)
        except Exception:
            pass
        if isinstance(image_obj, dict):
            for key in ("path", "filename", "url"):
                if key in image_obj and image_obj[key]:
                    return str(image_obj[key])
        if isinstance(image_obj, str):
            return image_obj
        return None

    num_items = len(dataset)
    if num_items == 0:
        return []

    results: List[Dict[str, Any]] = []

    for idx in random.sample(range(num_items), k=min(count, num_items)):
        row = dataset[idx]
        images_raw = row.get("image")
        images: List[str] = []
        if isinstance(images_raw, list):
            for item in images_raw:
                ref = extract_image_ref(item)
                if isinstance(ref, str):
                    images.append(ref)
        else:
            ref = extract_image_ref(images_raw)
            if isinstance(ref, str):
                images.append(ref)

        conversation_pair = normalize_conversations(row.get("conversations"))

        entry: Dict[str, Any] = {
            "id": str(row.get("id")) if row.get("id") is not None else "",
            "modality": str(row.get("modality")) if row.get("modality") is not None else "",
            "body_part": str(row.get("body_part")) if row.get("body_part") is not None else "",
            "image": images,
            "conversations": conversation_pair,
        }
        results.append(entry)

    return results

def generate_unanswerable_query_swapped_body_part(pairs: int = 1):
    """Generate unanswerable queries by swapping the body part with a random body part.
    
    Args:
        pairs: Number of pairs of unanswerable queries to generate.
    """
    if pairs < 1:
        pairs = 1

    # Use the training split by default
    dataset = ds["train"] if isinstance(ds, dict) and "train" in ds else ds

    def extract_image_ref_local(image_obj: Any) -> Optional[str]:
        try:
            filename = getattr(image_obj, "filename", None)
            if filename:
                return str(filename)
        except Exception:
            pass
        if isinstance(image_obj, dict):
            for key in ("path", "filename", "url"):
                if key in image_obj and image_obj[key]:
                    return str(image_obj[key])
        if isinstance(image_obj, str):
            return image_obj
        return None

    def resolve_image_path(ref: str) -> str:
        if not isinstance(ref, str):
            return ""
        image_path = os.path.expandvars(os.path.expanduser(ref))
        if image_path.startswith("file://"):
            image_path = image_path[len("file://"):]
        if not os.path.isabs(image_path):
            base_dir = os.getenv("DATA_STORAGE_HDD_PUBMEDVISION", "").strip()
            base_dir = os.path.expandvars(os.path.expanduser(base_dir)) if base_dir else base_dir
            if base_dir:
                image_path = os.path.join(base_dir, image_path)
        return os.path.abspath(image_path)

    def extract_question_text(conversations: Any) -> str:
        # Supports both raw list-of-dicts and normalized dict {question,response}
        if not conversations:
            return ""
        if isinstance(conversations, dict):
            q = conversations.get("question")
            return q if isinstance(q, str) else ""
        if isinstance(conversations, list):
            for turn in conversations:
                if isinstance(turn, dict):
                    who = str(turn.get("from", "")).lower()
                    if who in {"human", "user"}:
                        text = turn.get("value") or turn.get("content")
                        if isinstance(text, str):
                            return text
                elif isinstance(turn, str) and turn.strip():
                    return turn
            # Fallback to first
            first = conversations[0]
            if isinstance(first, dict):
                return str(first.get("value", ""))
            return str(first)
        if isinstance(conversations, str):
            return conversations
        return ""

    # Bucket items by modality and body_part (brain vs other)
    buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in dataset:
        modality = str(row.get("modality") or "").strip()
        body_part = str(row.get("body_part") or "").strip()
        images_raw = row.get("image")
        image_ref: Optional[str] = None
        if isinstance(images_raw, list):
            for item in images_raw:
                image_ref = extract_image_ref_local(item)
                if isinstance(image_ref, str):
                    break
        else:
            image_ref = extract_image_ref_local(images_raw)

        if not image_ref:
            continue

        question_text = extract_question_text(row.get("conversations"))
        if not isinstance(question_text, str) or not question_text.strip():
            continue

        mod_bucket = buckets.setdefault(modality, {"brain": [], "other": []})
        entry = {
            "body_part": body_part,
            "image_ref": image_ref,
            "question_text": question_text,
        }
        if body_part.lower() == "brain":
            mod_bucket["brain"].append(entry)
        else:
            mod_bucket["other"].append(entry)

    # Build swapped queries
    results: List[Dict[str, Any]] = []
    remaining = pairs
    modalities = list(buckets.keys())
    random.shuffle(modalities)
    for modality in modalities:
        if remaining <= 0:
            break
        brain_list = buckets[modality]["brain"]
        other_list = buckets[modality]["other"]
        if not brain_list or not other_list:
            continue
        random.shuffle(brain_list)
        random.shuffle(other_list)
        num_here = min(remaining, len(brain_list), len(other_list))
        for i in range(num_here):
            b = brain_list[i]
            o = other_list[i]

            # Swap images and build prompts
            prompt_b = generate_multimodal_query(b["question_text"])
            prompt_o = generate_multimodal_query(o["question_text"])
            img_for_b = resolve_image_path(o["image_ref"])  # non-brain image
            img_for_o = resolve_image_path(b["image_ref"])  # brain image

            results.append({"prompt": prompt_b, "image_path": img_for_b})
            results.append({"prompt": prompt_o, "image_path": img_for_o})
        remaining -= num_here

    return results

def generate_unanswerable_query_swapped_modality(pairs: int = 1):
    """Generate unanswerable queries by swapping the modality with a random modality. Use brain for default."""
    pass # TODO: Implement this

def test_read_questions_from_dataset_pubmedvision_instructions(num_items: int = 4) -> None:
    """Fetch and print a sample of items from PubMedVision InstructionTuning VQA.

    Args:
        num_items: Number of items to fetch and print. Defaults to 4.
    """
    items = read_questions_from_dataset_pubmedvision_instructions(count=num_items)
    if not items:
        print("No items returned from dataset.")
        return

    for idx, item in enumerate(items, start=1):
        print("-" * 60)
        print(f"Sample {idx}")
        print(f"id        : {item.get('id', '')}")
        print(f"modality  : {item.get('modality', '')}")
        print(f"body_part : {item.get('body_part', '')}")

        images = item.get("image", []) or []
        print(f"images ({len(images)}):")
        for img in images:
            print(f"  - {img}")

        conversations = item.get("conversations", {}) or {}
        print("conversation pair:")
        print(f"  question: {conversations.get('question', '')}")
        print(f"  response: {conversations.get('response', '')}")
    print("-" * 60)

def test_send_one_multimodal_query_to_llm(server_url: Optional[str] = None) -> Optional[str]:
    """Read one question and send a single multimodal query (text + image).

    The image path is resolved relative to the environment variable
    DATA_STORAGE_HDD_PUBMEDVISION if the dataset provides a relative path.

    Args:
        server_url: Base URL of the running LLM server. Defaults to env LLM_SERVER_URL
            or http://127.0.0.1:8000.

    Returns:
        The response text from the LLM if successful; otherwise None.
    """
    base_url = _get_base_url(server_url)

    items = read_questions_from_dataset_pubmedvision_instructions(count=1)
    if not items:
        print("No items available from dataset.")
        return None

    item = items[0]
    images = item.get("image", []) or []
    conversations = item.get("conversations", {}) or {}
    print(f"Conversations: {conversations}")
    if not images:
        print("Selected item has no images; cannot run multimodal test.")
        return None
    if not conversations or not isinstance(conversations, dict):
        print("Selected item has no conversation text; cannot run multimodal test.")
        return None

    image_ref = images[0]
    image_path = _normalize_and_resolve_image_path(image_ref)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    # Use 'question' as the prompt text.
    prompt_text = str(conversations.get("question", ""))
    print(f"Prompt text: {prompt_text}")
    print(f"Image path: {image_path}")
    response_text = _call_llm_multimodal(base_url, prompt_text, image_path, max_new_tokens=300)
    if isinstance(response_text, str):
        print("LLM response:\n" + response_text)
    return response_text

def test_unanswerable_queries_swapped_body_part(num_pairs: int = 1):
    base_url = _get_base_url(None)
    queries = generate_unanswerable_query_swapped_body_part(pairs=num_pairs)
    if not queries:
        print("No swapped queries generated.")
        return
    print(f"Generated {len(queries)} swapped queries (pairs requested: {num_pairs}).")
    for idx, q in enumerate(queries, start=1):
        prompt_text = str(q.get("prompt", ""))
        image_path = _normalize_and_resolve_image_path(str(q.get("image_path", "")))
        if not prompt_text or not image_path:
            print(f"[{idx}] Skipping due to empty prompt or image path.")
            continue
        if not os.path.exists(image_path):
            print(f"[{idx}] Image not found: {image_path}")
            continue
        print("-" * 60)
        print(f"[{idx}] Prompt: {prompt_text}")
        print(f"[{idx}] Image:  {image_path}")
        response_text = _call_llm_multimodal(base_url, prompt_text, image_path, max_new_tokens=150)
        if isinstance(response_text, str):
            print(f"[{idx}] LLM response:\n{response_text}")
        else:
            print(f"[{idx}] LLM request failed.")

if __name__ == "__main__":
    test_unanswerable_queries_swapped_body_part()