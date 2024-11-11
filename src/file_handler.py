import os
import json
from typing import Optional

class FileHandler:
    """Provide basic structure and let model handle content naturally."""
    
    def __init__(self, config_path: str = "data/json/trusted_file_types.json"):
        """Initialize with configuration."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = None

    def handle_file_operation(self, message: str, generated_text: str) -> Optional[str]:
        """Let model's natural language guide file operations."""
        try:
            # Extract filename from message
            parts = message.split("'")
            if len(parts) >= 2:
                filename = parts[1]
                
                # Validate file type
                extension = os.path.splitext(filename)[1]
                if not self._is_allowed_extension(extension):
                    return f"File type {extension} is not allowed."
                
                # Let model handle the content
                if "create" in message.lower() or "save" in message.lower() or "write" in message.lower():
                    # Extract the actual content from model's response
                    content = self._extract_content(generated_text)
                    
                    if content is None:
                        return "Could not determine content to write"
                    
                    # Create the file
                    try:
                        with open(filename, 'w') as f:
                            f.write(content)
                        return f"File '{filename}' has been created successfully!"
                    except Exception as e:
                        return f"Error creating file: {e}"
                
                # Handle reading
                elif "read" in message.lower() or "open" in message.lower() or "show" in message.lower():
                    if os.path.exists(filename):
                        try:
                            with open(filename, 'r') as f:
                                content = f.read()
                            return f"\nContents of {filename}:\n\n{content}"
                        except Exception as e:
                            return f"Error reading file: {e}"
                    else:
                        return f"File '{filename}' does not exist."
            
            return None
            
        except Exception as e:
            print(f"Error handling file operation: {e}")
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