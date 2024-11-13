import os
from typing import List, Dict
from .custom_logger import CustomLogger

class Utils:
    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger.get_logger("Utils", verbose)

    def load_document(self, file_path: str) -> str:
        """Load document from file with error handling."""
        try:
            if not file_path or not isinstance(file_path, str):
                self.logger.error("Invalid file path", "FILE")
                raise ValueError("Invalid file path")
                
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}", "FILE")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            self.logger.info(f"Loading document: {file_path}", "FILE")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not content:
                self.logger.warning("Empty file", "FILE")
                raise ValueError("Empty file")
                
            self.logger.success(f"Document loaded successfully: {len(content)} chars", "FILE")
            return content
            
        except Exception as e:
            self.logger.error(f"Error loading document: {e}", "FILE")
            return ""

    def format_messages(self, questions: List[str]) -> List[Dict[str, str]]:
        """Convert questions to message format with validation."""
        try:
            if not questions or not isinstance(questions, list):
                self.logger.warning("Invalid questions format", "PROCESS")
                return []
                
            formatted = []
            for q in questions:
                if isinstance(q, str) and q.strip():
                    formatted.append({
                        "role": "user",
                        "content": q.strip()
                    })
                    
            self.logger.info(f"Formatted {len(formatted)} messages", "PROCESS")
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting messages: {e}", "PROCESS")
            return []