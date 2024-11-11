from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import os
import time

@dataclass
class Identity:
    """Store identity information."""
    name: str
    role: str
    locked: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

class IdentityManager:
    """Handle identity tracking and persistence."""
    def __init__(self, identity_file: str = "data/json/identities.json"):
        self.identity_file = identity_file
        # Initialize with default values
        self.identities = {
            "assistant": {"name": "Charlotte", "locked": True},
            "user": {"name": None, "locked": False}
        }
        self._load_identities()
        
    def update_identity(self, role: str, name: str) -> bool:
        """Update identity with persistence."""
        try:
            if not name or not isinstance(name, str) or not name.strip():
                return False
                
            if role not in self.identities:
                return False
                
            if self.identities[role].get("locked"):
                return False
                
            # Clean and validate name
            name = name.strip().capitalize()
            
            # Update in memory
            self.identities[role]["name"] = name
            
            # Update file directly
            with open(self.identity_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            data["identities"][role]["name"] = name
            data["metadata"]["last_updated"] = time.time()
            
            with open(self.identity_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error updating identity: {e}")
            return False
            
    def get_identity(self, role: str) -> Optional[str]:
        """Get identity with fallback."""
        try:
            # Always read from file to get latest
            with open(self.identity_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                identity = data["identities"].get(role, {})
                return identity.get("name")
        except Exception as e:
            print(f"Error getting identity: {e}")
            return None
            
    def _load_identities(self):
        """Load identities from file."""
        try:
            if os.path.exists(self.identity_file):
                with open(self.identity_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.identities = data.get("identities", self.identities)
        except Exception as e:
            print(f"Error loading identities: {e}")
            
    def _save_identities(self):
        """Save identities to file."""
        try:
            data = {
                "identities": self.identities,
                "metadata": {
                    "last_updated": time.time(),
                    "version": "1.0"
                }
            }
            os.makedirs(os.path.dirname(self.identity_file), exist_ok=True)
            with open(self.identity_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving identities: {e}")