import os
from typing import List, Dict

def load_document(file_path: str) -> str:
    """Load document from file with error handling."""
    try:
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if not content:
            raise ValueError("Empty file")
            
        return content
        
    except Exception as e:
        print(f"Error loading document: {e}")
        return ""

def format_messages(questions: List[str]) -> List[Dict[str, str]]:
    """Convert questions to message format with validation."""
    try:
        if not questions or not isinstance(questions, list):
            return []
            
        formatted = []
        for q in questions:
            if isinstance(q, str) and q.strip():
                formatted.append({
                    "role": "user",
                    "content": q.strip()
                })
                
        return formatted
        
    except Exception as e:
        print(f"Error formatting messages: {e}")
        return [] 