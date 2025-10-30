import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

class MedQADatabase:
    """
    Database manager for MedQA dataset.
    Stores questions and textbook metadata in SQLite database.
    """
    
    def __init__(self, db_path: str = "medqa_database.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to the database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_schema(self):
        """Create database schema for MedQA dataset"""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        schema_sql = """
        -- Regions lookup table
        CREATE TABLE IF NOT EXISTS regions (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            language TEXT NOT NULL,
            description TEXT
        );
        
        -- Question formats lookup table  
        CREATE TABLE IF NOT EXISTS question_formats (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            num_options INTEGER
        );
        
        -- Main questions table
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            answer_idx TEXT NOT NULL,
            options_json TEXT NOT NULL,  -- Store options as JSON
            meta_info TEXT,
            region_id INTEGER NOT NULL,
            split TEXT NOT NULL,  -- train/dev/test
            language TEXT NOT NULL,
            format_id INTEGER NOT NULL,
            has_metamap BOOLEAN DEFAULT FALSE,
            metamap_phrases_json TEXT,  -- Store metamap phrases as JSON
            source_file TEXT,
            file_line_number INTEGER,
            FOREIGN KEY (region_id) REFERENCES regions (id),
            FOREIGN KEY (format_id) REFERENCES question_formats (id)
        );
        
        -- Textbook metadata table
        CREATE TABLE IF NOT EXISTS textbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            language TEXT NOT NULL,
            subject TEXT,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            file_hash TEXT,  -- MD5 hash for integrity
            format TEXT  -- txt, pdf, etc.
        );
        
        -- Indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_questions_region ON questions(region_id);
        CREATE INDEX IF NOT EXISTS idx_questions_split ON questions(split);
        CREATE INDEX IF NOT EXISTS idx_questions_language ON questions(language);
        CREATE INDEX IF NOT EXISTS idx_questions_format ON questions(format_id);
        CREATE INDEX IF NOT EXISTS idx_textbooks_language ON textbooks(language);
        CREATE INDEX IF NOT EXISTS idx_textbooks_subject ON textbooks(subject);
        """
        
        self.conn.executescript(schema_sql)
        self.conn.commit()
        print("✅ Database schema created successfully")
    
    def populate_lookup_tables(self):
        """Populate lookup tables with initial data"""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        # Insert regions
        regions = [
            (1, 'US', 'English', 'United States medical questions'),
            (2, 'Taiwan', 'Traditional Chinese', 'Taiwan medical questions'),
            (3, 'Mainland', 'Simplified Chinese', 'Mainland China medical questions')
        ]
        
        self.conn.executemany(
            "INSERT OR IGNORE INTO regions (id, name, language, description) VALUES (?, ?, ?, ?)",
            regions
        )
        
        # Insert question formats
        formats = [
            (1, 'basic_5_options', 'Standard format with 5 options (A-E)', 5),
            (2, 'basic_4_options', 'Standard format with 4 options (A-D)', 4),
            (3, 'metamap_5_options', 'Enhanced with MetaMap phrases, 5 options', 5),
            (4, 'metamap_4_options', 'Enhanced with MetaMap phrases, 4 options', 4),
            (5, 'translated', 'Translated version (Taiwan to English)', 4)
        ]
        
        self.conn.executemany(
            "INSERT OR IGNORE INTO question_formats (id, name, description, num_options) VALUES (?, ?, ?, ?)",
            formats
        )
        
        self.conn.commit()
        print("✅ Lookup tables populated successfully")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def extract_subject_from_filename(self, filename: str) -> str:
        """Extract subject from textbook filename"""
        # Remove extension and clean up filename
        name = filename.replace('.txt', '').replace('_', ' ')
        
        # Common subject mappings
        subject_mappings = {
            'Surgery': 'Surgery',
            'Psychiatry': 'Psychiatry', 
            'Physiology': 'Physiology',
            'Pharmacology': 'Pharmacology',
            'Pediatrics': 'Pediatrics',
            'Pathology': 'Pathology',
            'Pathoma': 'Pathology',
            'Obstetrics': 'Obstetrics & Gynecology',
            'Gynecology': 'Obstetrics & Gynecology',
            'Neurology': 'Neurology',
            'Internal': 'Internal Medicine',
            'Harrison': 'Internal Medicine',
            'Immunology': 'Immunology',
            'Histology': 'Histology',
            'First Aid': 'USMLE Prep',
            'Cell Biology': 'Cell Biology',
            'Biochemistry': 'Biochemistry',
            'Anatomy': 'Anatomy'
        }
        
        for key, subject in subject_mappings.items():
            if key.lower() in name.lower():
                return subject
                
        # For Chinese textbooks, try to extract from characters
        chinese_subjects = {
            '内科': 'Internal Medicine',
            '外科': 'Surgery', 
            '妇产科': 'Obstetrics & Gynecology',
            '神经病': 'Neurology',
            '精神病': 'Psychiatry',
            '儿科': 'Pediatrics',
            '药理': 'Pharmacology',
            '生理': 'Physiology',
            '解剖': 'Anatomy',
            '组织学': 'Histology',
            '病理': 'Pathology',
            '免疫': 'Immunology',
            '生化': 'Biochemistry',
            '微生物': 'Microbiology'
        }
        
        for key, subject in chinese_subjects.items():
            if key in name:
                return subject
                
        return 'General Medicine'
    
    def determine_format_id(self, options: dict, has_metamap: bool = False, is_translated: bool = False) -> int:
        """Determine format_id based on question characteristics"""
        num_options = len(options)
        
        if is_translated:
            return 5  # translated format
        elif has_metamap:
            return 3 if num_options == 5 else 4  # metamap formats
        else:
            return 1 if num_options == 5 else 2  # basic formats
    
    def load_questions_from_jsonl(self, file_path: Path, region_name: str, split: str, 
                                 has_metamap: bool = False, is_translated: bool = False) -> int:
        """Load questions from a JSONL file into the database"""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        # Get region_id
        region_id = self.conn.execute(
            "SELECT id FROM regions WHERE name = ?", (region_name,)
        ).fetchone()
        
        if not region_id:
            raise ValueError(f"Region '{region_name}' not found in database")
        
        region_id = region_id[0]
        
        # Determine language based on region
        language_map = {'US': 'English', 'Taiwan': 'Traditional Chinese', 'Mainland': 'Simplified Chinese'}
        if is_translated:
            language = 'English'  # Translated to English
        else:
            language = language_map.get(region_name, 'Unknown')
        
        questions_loaded = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                        
                    try:
                        question_data = json.loads(line.strip())
                        
                        # Validate required fields
                        required_fields = ['question', 'answer', 'options', 'answer_idx']
                        if not all(field in question_data for field in required_fields):
                            print(f"⚠️  Skipping malformed question at line {line_num}")
                            continue
                        
                        # Extract metamap phrases if available
                        metamap_phrases = question_data.get('metamap_phrases', [])
                        metamap_json = json.dumps(metamap_phrases, ensure_ascii=False) if metamap_phrases else None
                        
                        # Determine format
                        format_id = self.determine_format_id(
                            question_data['options'], 
                            has_metamap or bool(metamap_phrases),
                            is_translated
                        )
                        
                        # Insert question
                        self.conn.execute("""
                            INSERT INTO questions (
                                question_text, answer_text, answer_idx, options_json,
                                meta_info, region_id, split, language, format_id,
                                has_metamap, metamap_phrases_json, source_file, file_line_number
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            question_data['question'],
                            question_data['answer'],
                            question_data['answer_idx'],
                            json.dumps(question_data['options'], ensure_ascii=False),
                            question_data.get('meta_info', ''),
                            region_id,
                            split,
                            language,
                            format_id,
                            has_metamap or bool(metamap_phrases),
                            metamap_json,
                            str(file_path),
                            line_num
                        ))
                        
                        questions_loaded += 1
                        
                    except json.JSONDecodeError:
                        print(f"⚠️  JSON decode error at line {line_num}")
                        continue
                    except Exception as e:
                        print(f"⚠️  Error processing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            raise RuntimeError(f"Failed to load questions from {file_path}: {e}")
        
        self.conn.commit()
        return questions_loaded
    
    def populate_all_questions(self, dataset_base_path: str):
        """Load questions from all regions and formats"""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        base_path = Path(dataset_base_path)
        questions_path = base_path / "questions"
        
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions directory not found: {questions_path}")
        
        total_loaded = 0
        
        # Define regions and their primary splits
        regions = {
            'US': ['train', 'dev', 'test'],
            'Taiwan': ['train', 'dev', 'test'], 
            'Mainland': ['train', 'dev', 'test']
        }
        
        for region, splits in regions.items():
            region_path = questions_path / region
            if not region_path.exists():
                print(f"⚠️  Region directory not found: {region_path}")
                continue
                
            print(f"\n📁 Loading {region} questions...")
            
            # Load basic format questions
            for split in splits:
                file_path = region_path / f"{split}.jsonl"
                if file_path.exists():
                    loaded = self.load_questions_from_jsonl(file_path, region, split)
                    total_loaded += loaded
                    print(f"   ✅ {split}: {loaded} questions")
                else:
                    print(f"   ⚠️  {split}.jsonl not found")
            
            # Load 4-options format if available
            four_options_path = region_path / "4_options"
            if four_options_path.exists():
                print(f"   📁 Loading 4-options format...")
                for jsonl_file in four_options_path.glob("*.jsonl"):
                    # Determine split from filename
                    if 'train' in jsonl_file.name:
                        split = 'train'
                    elif 'dev' in jsonl_file.name:
                        split = 'dev'
                    elif 'test' in jsonl_file.name:
                        split = 'test'
                    else:
                        split = 'unknown'
                    
                    loaded = self.load_questions_from_jsonl(jsonl_file, region, split, has_metamap=True)
                    total_loaded += loaded
                    print(f"   ✅ {jsonl_file.name}: {loaded} questions")
            
            # Load metamap processed questions if available
            metamap_path = region_path / "metamap"
            if metamap_path.exists():
                print(f"   📁 Loading metamap format...")
                for split_dir in metamap_path.iterdir():
                    if split_dir.is_dir():
                        split = split_dir.name
                        for jsonl_file in split_dir.glob("*.jsonl"):
                            loaded = self.load_questions_from_jsonl(jsonl_file, region, split, has_metamap=True)
                            total_loaded += loaded
                            print(f"   ✅ {split}/{jsonl_file.name}: {loaded} questions")
            
            # Load translated versions if available (Taiwan)
            if region == 'Taiwan':
                translated_path = region_path / "tw_translated_jsonl" / "en"
                if translated_path.exists():
                    print(f"   📁 Loading translated format...")
                    for jsonl_file in translated_path.glob("*.jsonl"):
                        if 'train' in jsonl_file.name:
                            split = 'train'
                        elif 'dev' in jsonl_file.name:
                            split = 'dev'
                        elif 'test' in jsonl_file.name:
                            split = 'test'
                        else:
                            split = 'unknown'
                        
                        loaded = self.load_questions_from_jsonl(jsonl_file, region, split, is_translated=True)
                        total_loaded += loaded
                        print(f"   ✅ translated/{jsonl_file.name}: {loaded} questions")
        
        print(f"\n🎉 Total questions loaded: {total_loaded}")
        return total_loaded
    
    def populate_textbook_metadata(self, dataset_base_path: str):
        """Load textbook metadata into the database"""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        base_path = Path(dataset_base_path)
        textbooks_path = base_path / "textbooks"
        
        if not textbooks_path.exists():
            raise FileNotFoundError(f"Textbooks directory not found: {textbooks_path}")
        
        total_textbooks = 0
        
        # Process English textbooks
        en_path = textbooks_path / "en"
        if en_path.exists():
            print(f"\n📚 Loading English textbook metadata...")
            for txt_file in en_path.glob("*.txt"):
                title = txt_file.stem.replace('_', ' ')
                subject = self.extract_subject_from_filename(txt_file.name)
                file_size = txt_file.stat().st_size
                file_hash = self.get_file_hash(txt_file)
                
                self.conn.execute("""
                    INSERT OR IGNORE INTO textbooks (
                        filename, title, language, subject, file_path, 
                        file_size, file_hash, format
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    txt_file.name,
                    title,
                    'English',
                    subject,
                    str(txt_file.absolute()),
                    file_size,
                    file_hash,
                    'txt'
                ))
                
                total_textbooks += 1
                print(f"   ✅ {title} ({subject})")
        
        # Process Chinese textbooks  
        for lang_dir in ['zh_paragraph', 'zh_sentence']:
            lang_path = textbooks_path / lang_dir
            if lang_path.exists():
                lang_type = 'Chinese (Paragraph)' if 'paragraph' in lang_dir else 'Chinese (Sentence)'
                print(f"\n📚 Loading {lang_type} textbook metadata...")
                
                for txt_file in lang_path.glob("*.txt"):
                    if txt_file.name.startswith('.'):  # Skip hidden files
                        continue
                        
                    title = txt_file.stem
                    subject = self.extract_subject_from_filename(txt_file.name)
                    file_size = txt_file.stat().st_size
                    file_hash = self.get_file_hash(txt_file)
                    
                    self.conn.execute("""
                        INSERT OR IGNORE INTO textbooks (
                            filename, title, language, subject, file_path, 
                            file_size, file_hash, format
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        txt_file.name,
                        title,
                        lang_type,
                        subject,
                        str(txt_file.absolute()),
                        file_size,
                        file_hash,
                        'txt'
                    ))
                    
                    total_textbooks += 1
                    print(f"   ✅ {title} ({subject})")
        
        self.conn.commit()
        print(f"\n📖 Total textbooks loaded: {total_textbooks}")
        return total_textbooks

class MedQAQueryHelper:
    """Helper class for common database queries"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def get_questions(self, region: str = None, split: str = None, 
                     language: str = None, limit: int = None) -> List[Dict]:
        """Get questions with optional filters"""
        query = """
            SELECT q.*, r.name as region_name, qf.name as format_name
            FROM questions q
            JOIN regions r ON q.region_id = r.id
            JOIN question_formats qf ON q.format_id = qf.id
            WHERE 1=1
        """
        params = []
        
        if region:
            query += " AND r.name = ?"
            params.append(region)
        if split:
            query += " AND q.split = ?"
            params.append(split)
        if language:
            query += " AND q.language = ?"
            params.append(language)
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        return [dict(row) for row in self.conn.execute(query, params).fetchall()]
    
    def get_textbooks(self, language: str = None, subject: str = None) -> List[Dict]:
        """Get textbook metadata with optional filters"""
        query = "SELECT * FROM textbooks WHERE 1=1"
        params = []
        
        if language:
            query += " AND language = ?"
            params.append(language)
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        
        return [dict(row) for row in self.conn.execute(query, params).fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        # Total counts
        stats['total_questions'] = self.conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        stats['total_textbooks'] = self.conn.execute("SELECT COUNT(*) FROM textbooks").fetchone()[0]
        
        # Questions by region
        stats['questions_by_region'] = [
            dict(row) for row in self.conn.execute("""
                SELECT r.name, r.language, COUNT(*) as count 
                FROM questions q 
                JOIN regions r ON q.region_id = r.id 
                GROUP BY r.name, r.language
            """).fetchall()
        ]
        
        # Questions by split  
        stats['questions_by_split'] = [
            dict(row) for row in self.conn.execute("""
                SELECT split, COUNT(*) as count 
                FROM questions 
                GROUP BY split
            """).fetchall()
        ]
        
        return stats

def create_medqa_database(dataset_base_path: str = "./datasets/medqa/", 
                         db_path: str = "medqa_database.db",
                         overwrite: bool = False) -> MedQADatabase:
    """
    Create and populate the MedQA database.
    
    Args:
        dataset_base_path: Path to the medqa dataset
        db_path: Path for the database file
        overwrite: Whether to overwrite existing database
        
    Returns:
        MedQADatabase instance
    """
    # Remove existing database if overwrite is True
    if overwrite and os.path.exists(db_path):
        os.remove(db_path)
        print(f"🗑️  Removed existing database: {db_path}")
    
    # Create database instance
    db = MedQADatabase(db_path)
    db.connect()
    
    try:
        # Create schema and populate lookup tables
        db.create_schema()
        db.populate_lookup_tables()
        
        # Populate questions and textbooks
        db.populate_all_questions(dataset_base_path)
        db.populate_textbook_metadata(dataset_base_path)
        
        print(f"\n🎉 Database created successfully: {db_path}")
        return db
        
    except Exception as e:
        db.disconnect()
        raise RuntimeError(f"Failed to create database: {e}")

if __name__ == "__main__":
    # Demo usage - adjust path based on where script is run from
    import sys
    base_path = "../datasets/medqa/" if "utils" in os.getcwd() else "./datasets/medqa/"
    
    db = create_medqa_database(base_path, overwrite=True)
    db.disconnect()
    print("Database generation completed!") 