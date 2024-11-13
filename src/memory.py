from typing import List, Dict, Optional
import json
import os
from datetime import datetime, timezone
from .identity import IdentityManager
from .custom_logger import CustomLogger

class Memory:
    """Enhanced memory structure with multiple memory types and sophisticated storage."""
    def __init__(self, role: str, content: str, memory_type: str = "short", logger: Optional[CustomLogger] = None):
        self.logger = logger
        
        # Load identities
        identity_manager = IdentityManager()
        assistant_name = identity_manager.get_identity("assistant")
        user_name = identity_manager.get_identity("user")
        
        if self.logger:
            self.logger.debug(f"Creating {memory_type} memory with identities - Assistant: {assistant_name}, User: {user_name}", "MEMORY")
        
        # Enhanced identity context for model
        identity_context = {
            "Identity Understanding": {
                "description": "Understand and track user and assistant identities",
                "patterns": [
                    "My name is", "I am", "Call me", "I'm called",
                    "You are", "Your name is", "They call you"
                ]
            }
        }
        
        # Initialize memory with rich metadata and identity context
        current_time = datetime.now(timezone.utc)
        self.data = {
            "role": role,
            "content": content,
            "memory_type": memory_type,
            "timestamp": current_time.timestamp(),
            "formatted_time": current_time.strftime('%B %d, %Y at %I:%M %p %Z'),
            "identity_info": {
                "user_name": user_name,
                "assistant_name": assistant_name,
                "identity_context": identity_context
            },
            "importance_score": self._calculate_importance(content, identity_context),
            "access_count": 0,
            "last_accessed": current_time.timestamp(),
            "emotional_weight": self._analyze_emotional_weight(content),
            "topic_tags": self._extract_topics(content),
            "references": [],
            "current_importance": 1.0,
            "memory_analysis": self._analyze_memory_content(content),
            "decay_rate": self._determine_decay_rate(memory_type)
        }

    def _calculate_importance(self, content: str, identity_context: Dict) -> float:
        """Calculate memory importance with identity awareness."""
        importance = 1.0
        
        # Check for identity-related content
        identity_patterns = identity_context.get("Identity Understanding", {}).get("patterns", [])
        if any(pattern.lower() in content.lower() for pattern in identity_patterns):
            importance += 0.5  # Boost importance for identity-related content
            
        # Additional importance calculations...
        importance += min(len(content) / 1000, 0.5)
        importance += content.count('?') * 0.1
        
        return min(importance, 2.0)

    def _analyze_emotional_weight(self, content: str) -> float:
        """Analyze emotional significance of memory."""
        emotional_indicators = {
            'positive': ['thank', 'great', 'good', 'excellent', 'appreciate'],
            'negative': ['error', 'wrong', 'bad', 'issue', 'problem'],
            'important': ['urgent', 'critical', 'important', 'necessary']
        }
        
        weight = 0.0
        content_lower = content.lower()
        
        for category, words in emotional_indicators.items():
            for word in words:
                if word in content_lower:
                    weight += 0.1
        
        return min(weight, 1.0)

    def _extract_topics(self, content: str) -> List[str]:
        """Extract relevant topics from memory content."""
        topics = []
        
        # Core capability topics
        capability_keywords = {
            "language": ["understand", "language", "context", "nuance"],
            "task": ["assist", "help", "task", "generate", "create"],
            "efficiency": ["quick", "fast", "speed", "efficient"],
            "data": ["analyze", "data", "information", "insight"],
            "learning": ["learn", "update", "enhance", "improve"],
            "access": ["access", "platform", "available"]
        }
        
        content_lower = content.lower()
        for topic, keywords in capability_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics

    def _analyze_memory_content(self, content: str) -> Dict[str, any]:
        """Analyze memory content with identity awareness."""
        analysis = {
            "is_factual": any(indicator in content.lower() for indicator in ["is", "are", "was", "were", "fact", "data"]),
            "is_personal": any(indicator in content.lower() for indicator in ["i", "you", "we", "my", "your", "our"]),
            "is_procedural": any(indicator in content.lower() for indicator in ["how to", "steps", "process", "method"]),
            "contains_question": "?" in content,
            "contains_command": any(cmd in content.lower() for cmd in ["please", "could you", "would you"]),
            "word_count": len(content.split()),
            "is_identity_related": any(phrase in content.lower() for phrase in [
                "my name is", "i am called", "call me", "i'm",
                "your name is", "you are called", "they call you"
            ])
        }
        
        # Determine retention based on content analysis
        analysis["suggested_retention"] = "long" if any([
            analysis["is_identity_related"],
            len(content.split()) > 100,
            "important" in content.lower(),
            "remember" in content.lower(),
            "save" in content.lower()
        ]) else "short"
        
        return analysis

    def _determine_decay_rate(self, memory_type: str) -> float:
        """Determine decay rate based on memory type."""
        decay_rates = {
            "working": 0.5,    # Decays quickly (minutes to hours)
            "short": 0.1,      # Decays moderately (hours to days)
            "long": 0.01       # Decays slowly (days to weeks)
        }
        return decay_rates.get(memory_type, 0.1)

class BrainMemory:
    """Enhanced memory system with multiple memory types and sophisticated storage."""
    
    def __init__(self, memory_file: str = "data/memory/brain_memory.json", verbose: bool = False):
        self.memory_file = memory_file
        self.logger = CustomLogger.get_logger("Memory", verbose)
        
        # Separate memory streams
        self.working_memory = []  # Very recent interactions (last few minutes)
        self.short_term_memory = []  # Recent interactions (last few hours/days)
        self.long_term_memory = []  # Important or frequently accessed memories
        
        # Memory parameters
        self.max_working_memories = 10
        self.max_short_term_memories = 100
        self.max_long_term_memories = 1000
        
        self.working_memory_threshold = 300  # 5 minutes in seconds
        self.short_term_threshold = 86400    # 24 hours in seconds
        
        self._load_state()
        self._prune_memories()

    def add(self, role: str, content: str):
        """Add memory with automatic type determination."""
        try:
            # Create memory and analyze it
            memory = Memory(role, content, "working", self.logger)
            analysis = memory.data["memory_analysis"]
            
            # Check for identity-related content
            if role == "user" and any(phrase in content.lower() for phrase in [
                "my name is", "i am called", "i'm called", "call me"
            ]):
                # Extract name and update identity
                name_parts = content.lower().split()
                for phrase in ["my name is", "i am called", "i'm called", "call me"]:
                    if phrase in content.lower():
                        idx = content.lower().find(phrase) + len(phrase)
                        name = content[idx:].strip().split()[0]
                        identity_manager = IdentityManager()
                        if identity_manager.update_identity("user", name):
                            memory.data["importance_score"] = 2.0  # Boost importance for identity info
                            memory.data["memory_type"] = "long"  # Store in long-term memory
            
            # Determine memory type based on analysis
            if analysis["suggested_retention"] == "long" or analysis["is_factual"]:
                memory.data["memory_type"] = "long"
                self.long_term_memory.append(memory.data)
            else:
                memory.data["memory_type"] = "short"
                self.short_term_memory.append(memory.data)
            
            # Also add to working memory
            self.working_memory.append(memory.data)
            
            self._prune_memories()
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {e}", "MEMORY")

    def get_context(self, max_turns: Optional[int] = None) -> str:
        """Get context from all memory types."""
        try:
            context_parts = []
            
            # Get recent working memory
            working_context = self._format_memories(
                self._select_important_memories(self.working_memory, max_turns),
                prefix="Recent: "
            )
            if working_context:
                context_parts.append(working_context)
            
            # Get relevant short-term memory
            short_term_context = self._format_memories(
                self._select_important_memories(self.short_term_memory, max_turns),
                prefix="Earlier: "
            )
            if short_term_context:
                context_parts.append(short_term_context)
            
            # Get relevant long-term memory
            long_term_context = self._format_memories(
                self._select_important_memories(self.long_term_memory, max_turns),
                prefix="Previously: "
            )
            if long_term_context:
                context_parts.append(long_term_context)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error getting context: {e}", "MEMORY")
            return ""

    def _format_memories(self, memories: List[Dict], prefix: str = "") -> str:
        """Format memories with optional prefix."""
        if not memories:
            return ""
            
        formatted = []
        for memory in memories:
            name = "User" if memory["role"] == "user" else memory.get("identity_info", {}).get("assistant_name", "Assistant")
            formatted.append(f"{name}: {memory['content']}")
        
        return prefix + "\n".join(formatted)

    def _prune_memories(self):
        """Prune memories and handle memory transitions."""
        try:
            current_time = datetime.now(timezone.utc).timestamp()
            
            # Prune working memory
            self.working_memory = [
                m for m in self.working_memory 
                if (current_time - m["timestamp"]) < self.working_memory_threshold
            ][:self.max_working_memories]
            
            # Move aging memories from short-term to long-term
            for memory in self.short_term_memory[:]:
                age = current_time - memory["timestamp"]
                importance = memory.get("current_importance", 0)
                
                if age > self.short_term_threshold and importance > 0.7:
                    self.short_term_memory.remove(memory)
                    self.long_term_memory.append(memory)
            
            # Prune based on size limits
            self.short_term_memory = self._select_important_memories(
                self.short_term_memory, 
                self.max_short_term_memories
            )
            
            self.long_term_memory = self._select_important_memories(
                self.long_term_memory, 
                self.max_long_term_memories
            )
            
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"Error pruning memories: {e}", "MEMORY")

    def _select_important_memories(self, memories: List[Dict], max_count: Optional[int] = None) -> List[Dict]:
        """Select important memories with decay."""
        try:
            current_time = datetime.now(timezone.utc).timestamp()
            
            # Calculate current importance with decay
            for memory in memories:
                hours_old = (current_time - memory["timestamp"]) / 3600
                decay = memory.get("decay_rate", 0.1) * hours_old
                
                base_importance = memory.get("importance_score", 1.0)
                emotional_weight = memory.get("emotional_weight", 0.0)
                access_bonus = memory.get("access_count", 0) * 0.1
                
                # Calculate decayed importance
                importance = base_importance * (1.0 - decay)
                importance *= (1.0 + emotional_weight)
                importance *= (1.0 + access_bonus)
                
                # Boost importance based on analysis
                if memory.get("memory_analysis", {}).get("is_factual", False):
                    importance *= 1.2
                if memory.get("memory_analysis", {}).get("is_personal", False):
                    importance *= 1.1
                
                memory["current_importance"] = max(0, importance)
            
            # Sort by importance and limit count
            sorted_memories = sorted(
                memories,
                key=lambda x: x.get("current_importance", 0),
                reverse=True
            )
            
            return sorted_memories[:max_count] if max_count else sorted_memories
            
        except Exception as e:
            self.logger.error(f"Error selecting memories: {e}", "MEMORY")
            return []

    def _save_state(self):
        """Save all memory types."""
        try:
            data = {
                "working_memory": self.working_memory,
                "short_term_memory": self.short_term_memory,
                "long_term_memory": self.long_term_memory,
                "metadata": {
                    "last_updated": datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p %Z'),
                    "total_memories": {
                        "working": len(self.working_memory),
                        "short_term": len(self.short_term_memory),
                        "long_term": len(self.long_term_memory)
                    }
                }
            }
            
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving state: {e}", "MEMORY")

    def _load_state(self):
        """Load all memory types."""
        try:
            if os.path.exists(self.memory_file):
                try:
                    with open(self.memory_file, 'r') as f:
                        data = json.load(f)
                        self.working_memory = data.get("working_memory", [])
                        self.short_term_memory = data.get("short_term_memory", [])
                        self.long_term_memory = data.get("long_term_memory", [])
                        
                    self.logger.info(
                        f"Loaded memories - Working: {len(self.working_memory)}, "
                        f"Short-term: {len(self.short_term_memory)}, "
                        f"Long-term: {len(self.long_term_memory)}", 
                        "MEMORY"
                    )
                except json.JSONDecodeError:
                    self.logger.warning("Invalid memory file format, initializing empty memories", "MEMORY")
                    self._initialize_empty()
            else:
                self.logger.info("No existing memory file, starting fresh", "MEMORY")
                self._initialize_empty()
                
        except Exception as e:
            self.logger.error(f"Error loading state: {e}", "MEMORY")
            self._initialize_empty()

    def _initialize_empty(self):
        """Initialize empty memory streams."""
        self.working_memory = []
        self.short_term_memory = []
        self.long_term_memory = []
        self._save_state()

