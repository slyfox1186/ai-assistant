from typing import List, Dict, Optional
import json
import os
import time
from .identity import IdentityManager

class Memory:
    """Simple memory structure for conversation history."""
    def __init__(self, role: str, content: str, memory_type: str = "conversation", context_type: str = None):
        self.role = role
        self.content = content
        self.type = memory_type
        self.context_type = context_type
        self.timestamp = time.time()
        self.access_count = 0
        self.last_access = self.timestamp
        self.importance = 1.0
        
        self.attributes = {
            "is_identity": memory_type == "identity",
            "is_question": "?" in content,
            "needs_context": False,
            "last_accessed": time.time()
        }
        
        self.identity_info = {
            "user_name": None,
            "assistant_name": "Charlotte",  # Always set assistant name
            "mentioned_names": []
        }
        
        self.context = {
            "topic": None,
            "intent": None,
            "key_entities": []
        }
        
        self.associations = []

class BrainMemory:
    """Memory system that lets the model use its natural language abilities."""
    
    def __init__(self, memory_file: str = "data/memory/brain_memory.json"):
        self.memory_file = memory_file
        self.conversations = []
        self.semantic = []
        self.identity_manager = IdentityManager()  # Initialize IdentityManager
        self._load_memories()

    def add(self, role: str, content: str):
        """Add a new memory and let model update identities naturally."""
        memory = Memory(role, content)
        self.conversations.append(memory)
        
        # Check content for identity updates
        content_lower = content.lower()
        
        try:
            if "my name is" in content_lower and role == "user":
                # Extract name properly - handle the full name after "my name is"
                name_start = content_lower.index("my name is") + len("my name is")
                name = content_lower[name_start:].strip()  # Get full name after "my name is"
                
                if name:
                    # Clean up the name - remove punctuation and extra spaces
                    name = ' '.join(word.capitalize() for word in name.split())
                    
                    # Update user identity through IdentityManager
                    if self.identity_manager.update_identity("user", name):
                        # Create identity memory
                        identity_memory = Memory(
                            role="system",
                            content=f"The user's name is {name}.",
                            memory_type="identity"
                        )
                        identity_memory.attributes["is_identity"] = True
                        self.semantic.append(identity_memory)
                        
                        # Update the current memory's identity info
                        memory.identity_info["user_name"] = name
            
            # Always ensure assistant identity is preserved
            assistant_name = self.identity_manager.get_identity("assistant")
            if assistant_name:
                memory.identity_info["assistant_name"] = assistant_name
                
            # Update user name in memory
            user_name = self.identity_manager.get_identity("user")
            if user_name:
                memory.identity_info["user_name"] = user_name
                
        except Exception as e:
            print(f"Error handling identity: {e}")
        
        self._save_memories()

    def get_context(self, max_turns: int = 5) -> str:
        """Get recent conversation context with identities."""
        context_parts = []
        
        # Add identity information
        assistant_name = self.identity_manager.get_identity("assistant")
        user_name = self.identity_manager.get_identity("user")
        
        if assistant_name:
            context_parts.append(f"The assistant's name is {assistant_name}.")
        if user_name:
            context_parts.append(f"The user's name is {user_name}.")
        
        # Add recent conversation turns
        recent = self.conversations[-max_turns:]
        for memory in recent:
            prefix = "User: " if memory.role == "user" else "Assistant: "
            context_parts.append(f"{prefix}{memory.content}")
            
        return "\n\n".join(context_parts)

    def update_identity(self, role: str, name: str):
        """Update identity through IdentityManager."""
        if self.identity_manager.update_identity(role, name):
            # Create identity memory
            identity_memory = Memory(
                role="system",
                content=f"The {role}'s name is {name}.",
                memory_type="identity"
            )
            identity_memory.attributes["is_identity"] = True
            self.semantic.append(identity_memory)
            self._save_memories()

    def get_identity(self, role: str) -> Optional[str]:
        """Get identity."""
        return self.identity_index.get(role)

    def clear(self):
        """Clear conversation history but keep identities."""
        self.conversations = []
        self._save_memories()

    def _save_memories(self):
        """Save memories with proper identity information."""
        try:
            data = {
                "conversations": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "type": m.type,
                        "context_type": m.context_type,
                        "attributes": m.attributes,
                        "identity_info": {
                            "user_name": self.identity_manager.get_identity("user"),
                            "assistant_name": self.identity_manager.get_identity("assistant"),
                            "mentioned_names": m.identity_info.get("mentioned_names", [])
                        },
                        "context": m.context,
                        "associations": m.associations
                    }
                    for m in self.conversations
                ],
                "semantic": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "type": m.type,
                        "attributes": m.attributes
                    }
                    for m in self.semantic
                ]
            }
            
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            print(f"Error saving memories: {e}")

    def _load_memories(self):
        """Load memories from file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    
                    self.conversations = [Memory(
                        m["role"],
                        m["content"],
                        m.get("type", "conversation"),
                        m.get("context_type")
                    ) for m in data.get("conversations", [])]
                    
                    self.semantic = [Memory(
                        m["role"],
                        m["content"],
                        m.get("type", "semantic")
                    ) for m in data.get("semantic", [])]
                    
                    self.identity_index = data.get("identity_index", {})
        except Exception as e:
            print(f"Error loading memories: {e}")

    def get_semantic_context(self, query: str, max_results: int = 3) -> List[str]:
        """Get relevant semantic memories for context."""
        try:
            # Return identity memories first
            context = []
            identity_memories = [m for m in self.semantic if m.attributes.get("is_identity")]
            if identity_memories:
                context.extend([m.content for m in identity_memories])
                
            # Get recent conversations
            recent_convos = self.conversations[-5:] if self.conversations else []
            if recent_convos:
                context.extend([
                    f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" 
                    for m in recent_convos
                ])
                
            return context[:max_results]  # Limit number of context items
            
        except Exception as e:
            print(f"Error getting semantic context: {e}")
            return []

