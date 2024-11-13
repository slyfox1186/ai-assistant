# src/file_handler.py

import os
import json
from typing import Optional
from .custom_logger import CustomLogger

class FileHandler:
    """Provide basic structure and let model handle content naturally."""
    
    def __init__(self, config_path: str = "data/json/trusted_file_types.json", verbose: bool = False):
        """Initialize with configuration."""
        self.logger = CustomLogger.get_logger("FileHandler", verbose)
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.logger.info("Configuration loaded successfully", "FILE")
        except Exception as e:
            self.logger.error(f"Error loading config: {e}", "FILE")
            self.config = None

    def handle_file_operation(self, message: str, generated_text: str) -> Optional[str]:
        """Let model's natural language guide file operations."""
        try:
            # Extract filename from message
            parts = message.split("'")
            if len(parts) >= 2:
                filename = parts[1]
                self.logger.info(f"Processing file: {filename}", "FILE")
                
                # Validate file type
                extension = os.path.splitext(filename)[1]
                if not self._is_allowed_extension(extension):
                    self.logger.warning(f"File type {extension} is not allowed", "FILE")
                    return f"File type {extension} is not allowed."
                
                # Let model handle the content
                if "create" in message.lower() or "save" in message.lower() or "write" in message.lower():
                    self.logger.info("Creating/writing file", "FILE")
                    content = self._extract_content(generated_text)
                    
                    if content is None:
                        self.logger.error("Could not determine content to write", "FILE")
                        return "Could not determine content to write"
                    
                    try:
                        with open(filename, 'w') as f:
                            f.write(content)
                        self.logger.success(f"File '{filename}' created successfully", "FILE")
                        return f"File '{filename}' has been created successfully!"
                    except Exception as e:
                        self.logger.error(f"Error creating file: {e}", "FILE")
                        return f"Error creating file: {e}"
                
                elif "read" in message.lower() or "open" in message.lower() or "show" in message.lower():
                    self.logger.info("Reading file", "FILE")
                    if os.path.exists(filename):
                        try:
                            with open(filename, 'r') as f:
                                content = f.read()
                            self.logger.success(f"File '{filename}' read successfully", "FILE")
                            return f"\nContents of {filename}:\n\n{content}"
                        except Exception as e:
                            self.logger.error(f"Error reading file: {e}", "FILE")
                            return f"Error reading file: {e}"
                    else:
                        self.logger.warning(f"File '{filename}' does not exist", "FILE")
                        return f"File '{filename}' does not exist."
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error handling file operation: {e}", "FILE")
            return None

    def _extract_content(self, generated_text: str) -> Optional[str]:
        """Extract actual content from model's natural language response."""
        try:
            # Remove common prefixes
            text = generated_text.strip()
            
            # Remove meta text before actual content
            prefixes = [
                "Here's", "Sure!", "I'll", "Let me", "Here is", 
                "Here you go", "Creating", "Writing",
                "Would that be", "you're looking for"
            ]
            
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text.split("\n", 1)[1] if "\n" in text else ""
                    
            # Clean up markdown code blocks
            if "```" in text:
                parts = text.split("```")
                if len(parts) >= 3:  # Has opening and closing ticks
                    text = parts[1]
                    # Remove language identifier if present
                    if "\n" in text:
                        text = text.split("\n", 1)[1]
                        
            # Remove any remaining question marks or meta text
            text = text.strip("?").strip()
            
            # Clean up extra whitespace
            text = "\n".join(line.strip() for line in text.splitlines()).strip()
            
            return text if text else None
            
        except Exception as e:
            print(f"Error extracting content: {e}")
            return None

    def _is_allowed_extension(self, extension: str) -> bool:
        """Basic security check for allowed file types."""
        try:
            if not self.config:
                return False
                
            for category in self.config["trusted_extensions"].values():
                if extension in category["extensions"]:
                    return True
            return False
            
        except Exception as e:
            print(f"Error checking extension: {e}")
            return False