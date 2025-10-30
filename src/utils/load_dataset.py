import os
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)


import json
from typing import List, Dict, Any
from pathlib import Path

class Params:
    def __init__(self, file_path: str, load_size: int, dataset_split: str):
        self.file_path = file_path
        self.load_size = load_size
        self.dataset_split = dataset_split

    
def load_dataset(input_data: str, params: Params):
    """
    Dataset options: medqa, medmcqa, mimic3
    Util method to load different dataset based on input_data.
    """
    match input_data:
        case "medqa":
            return load_medqa(params)
        case "medmcqa":
            return load_medmcqa(params)
        case "mimic3":
            return load_mimic3(params)
        case _:
            raise ValueError(f"Invalid dataset: {input_data}")

def get_medqa_regions(base_dataset_path: str = "./datasets/medqa/questions/") -> List[str]:
    """
    Get available regions for MedQA dataset.
    
    Args:
        base_dataset_path: Base path to the medqa questions directory
        
    Returns:
        List of available regions (e.g., ['US', 'Taiwan', 'Mainland'])
    """
    base_path = Path(base_dataset_path)
    if not base_path.exists():
        raise FileNotFoundError(f"MedQA dataset path not found: {base_path}")
    
    regions = [d.name for d in base_path.iterdir() if d.is_dir() and d.name != "__pycache__"]
    return sorted(regions)

def extract_region_from_path(file_path: str) -> str:
    """
    Extract region name from file path for database queries.
    
    Args:
        file_path: Path containing region information
        
    Returns:
        Region name (US, Taiwan, or Mainland)
    """
    path_str = str(file_path).lower()
    
    if 'us' in path_str:
        return 'US'
    elif 'taiwan' in path_str:
        return 'Taiwan'
    elif 'mainland' in path_str:
        return 'Mainland'
    else:
        # Default fallback - try to extract from path components
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part.upper() in ['US', 'TAIWAN', 'MAINLAND']:
                return part.upper()
        
        raise ValueError(f"Could not determine region from path: {file_path}")

def load_medqa(params: Params, db_path: str = "medqa_database.db") -> List[Dict[str, Any]]:
    """
    Load MedQA dataset from database.
    
    Dataset Structure:
    - Questions in multiple regions: US (English), Taiwan (Chinese), Mainland (Chinese)
    - Standard splits: train, dev, test
    - Each sample contains: question, answer, options (A-E), meta_info, answer_idx
    
    Args:
        params: Params object containing:
            - file_path: Path to the dataset directory (e.g., "./datasets/medqa/questions/US/")
            - load_size: Number of samples to load (if -1, load all)
            - dataset_split: Split to load ("train", "dev", or "test")
        db_path: Path to the MedQA database file
        
    Returns:
        List of dictionaries containing question data
        
    Example:
        >>> params = Params("./datasets/medqa/questions/US/", 10, "dev")
        >>> data = load_medqa(params)
        >>> print(data[0]['question'])
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}. Please create the database first using create_medqa_database().")
    
    try:
        from .database_generator import MedQAQueryHelper
    except ImportError:
        # Handle relative import when run from different contexts
        from database_generator import MedQAQueryHelper
    
    # Extract region from file path
    region = extract_region_from_path(params.file_path)
    
    # Query database
    with MedQAQueryHelper(db_path) as helper:
        questions = helper.get_questions(
            region=region,
            split=params.dataset_split,
            limit=params.load_size if params.load_size != -1 else None
        )
    
    # Transform database format to original JSONL format for backward compatibility
    data = []
    for q in questions:
        # Parse options JSON back to dict
        options = json.loads(q['options_json']) if q['options_json'] else {}
        
        # Create item in original format
        item = {
            'question': q['question_text'],
            'answer': q['answer_text'], 
            'answer_idx': q['answer_idx'],
            'options': options,
            'meta_info': q['meta_info'] or ''
        }
        
        # Add MetaMap phrases if available
        if q['metamap_phrases_json']:
            try:
                item['metamap_phrases'] = json.loads(q['metamap_phrases_json'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        data.append(item)
    
    print(f"✅ Loaded {len(data)} samples from database ({region} {params.dataset_split})")
    return data

def demo_medqa_dataset():
    """
    Demonstrate how to use the MedQA dataset loader.
    """
    print("=== MedQA Dataset Demo ===")
    
    # Show available regions
    try:
        regions = get_medqa_regions()
        print(f"Available regions: {regions}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load sample from each region
    for region in regions[:2]:  # Demo first 2 regions
        print(f"\n--- {region} Region ---")
        try:
            params = Params(f"./datasets/medqa/questions/{region}/", 1, "dev")
            data = load_medqa(params)
            
            if data:
                sample = data[0]
                print(f"Question: {sample['question'][:100]}...")
                print(f"Answer: {sample['answer']}")
                print(f"Options: {list(sample['options'].keys())}")
        except Exception as e:
            print(f"Error loading {region}: {e}")

def load_medmcqa(params: Params):
    pass

def load_mimic3(params: Params):
    pass

if __name__ == "__main__":
    demo_medqa_dataset()