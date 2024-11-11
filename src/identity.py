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

    def update_name(self, new_name: str) -> bool:
        """Update name if not locked."""
        if self.locked:
            return False
        self.name = new_name.strip()
        self.timestamp = time.time()
        return True

class IdentityManager:
    """Handle identity tracking and persistence."""
    def __init__(self, identity_file: str = "data/json/identities.json"):
        self.identity_file = identity_file
        # Initialize with default values but don't overwrite existing
        self.default_identities = {
            "assistant": {"name": "Charlotte", "locked": True},
            "user": {"name": None, "locked": False}
        }
        self._load_identities()
        
    def update_identity(self, role: str, name: str) -> bool:
        """Update identity with persistence."""
        try:
            if not name or not isinstance(name, str):
                return False
                
            if role not in self.identities:
                return False
                
            # Clean name
            name = name.strip()
            if not name:
                return False
                
            # Don't update if locked
            if self.identities[role].get("locked", False):
                return False
                
            # Update in memory
            self.identities[role]["name"] = name
            
            # Save to file
            self._save_identities()
            
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
        """Load identities from file or initialize with defaults."""
        try:
            if os.path.exists(self.identity_file):
                with open(self.identity_file, 'r') as f:
                    data = json.load(f)
                    self.identities = data.get("identities", {})
            else:
                # Initialize with defaults
                self.identities = self.default_identities.copy()
                self._save_identities()
                
        except Exception as e:
            print(f"Error loading identities: {e}")
            self.identities = self.default_identities.copy()
            
    def _save_identities(self):
        """Save identities with metadata."""
        try:
            data = {
                "identities": self.identities,
                "metadata": {
                    "last_updated": time.time(),
                    "version": "1.0"
                }
            }
            os.makedirs(os.path.dirname(self.identity_file), exist_ok=True)
            with open(self.identity_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving identities: {e}")