from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import os
import time
from .custom_logger import CustomLogger

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
    def __init__(self, identity_file: str = "data/json/identities.json", verbose: bool = False):
        self.identity_file = identity_file
        self.logger = CustomLogger.get_logger("Identity", verbose)
        
        self.identities = {
            "assistant": {"name": "Charlotte", "locked": True},
            "user": {"name": None, "locked": False}
        }
        self._load_identities()
        
    def update_identity(self, role: str, name: str) -> bool:
        """Update identity with persistence."""
        try:
            if not name or not isinstance(name, str):
                self.logger.warning(f"Invalid name format for {role}", "IDENTITY")
                return False
                
            if role not in self.identities:
                self.logger.warning(f"Unknown role: {role}", "IDENTITY")
                return False
                
            # Clean name and validate
            name = name.strip()
            if not name:
                self.logger.warning("Empty name after cleaning", "IDENTITY")
                return False
                
            # Don't update if locked
            if self.identities[role].get("locked", False):
                self.logger.warning(f"Cannot update locked identity for {role}", "IDENTITY")
                return False
                
            # Update in memory
            self.logger.info(f"Updating {role} identity to: {name}", "IDENTITY")
            self.identities[role]["name"] = name
            self.identities[role]["last_updated"] = time.time()
            
            # Save immediately
            self._save_identities()
            self.logger.success(f"Identity updated for {role}", "IDENTITY")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating identity: {e}", "IDENTITY")
            return False

    def get_identity(self, role: str) -> Optional[str]:
        """Get identity with immediate file check."""
        try:
            # Always read fresh from file
            self._load_identities()
            name = self.identities.get(role, {}).get("name")
            self.logger.debug(f"Retrieved identity for {role}: {name}", "IDENTITY")
            return name
        except Exception as e:
            self.logger.error(f"Error getting identity: {e}", "IDENTITY")
            return None
            
    def _load_identities(self):
        """Load identities from file."""
        try:
            if os.path.exists(self.identity_file):
                with open(self.identity_file, 'r') as f:
                    data = json.load(f)
                    self.identities = data.get("identities", self.identities)
                self.logger.info("Identities loaded from file", "IDENTITY")
            else:
                self.logger.warning("Identity file not found, using defaults", "IDENTITY")
        except Exception as e:
            self.logger.error(f"Error loading identities: {e}", "IDENTITY")
            
    def _save_identities(self):
        """Save identities with metadata."""
        try:
            data = {
                "identities": self.identities,
                "metadata": {
                    "last_updated": time.time(),
                    "version": "1.0",
                    "updates_count": 0
                }
            }
            os.makedirs(os.path.dirname(self.identity_file), exist_ok=True)
            with open(self.identity_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.success("Identities saved to file", "IDENTITY")
        except Exception as e:
            self.logger.error(f"Error saving identities: {e}", "IDENTITY")