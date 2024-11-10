from typing import List, Dict, Optional
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os

class MemoryType(Enum):
    WORKING = "working"     # Active conversation context
    SHORT_TERM = "short"    # Recent conversations
    LONG_TERM = "long"      # Important/repeated information

@dataclass
class Memory:
    """Neural memory structure that enhances model's natural abilities."""
    content: Dict
    timestamp: float
    memory_type: MemoryType
    importance: float = 1.0
    last_access: float = field(default_factory=time.time)

class BrainMemory:
    """Brain-inspired memory system that works with model's natural language abilities."""
    
    def __init__(self, memory_file: str = "brain_memory.json"):
        """Initialize brain-like memory system with persistence."""
        self.memory_file = memory_file
        self.memories = {
            MemoryType.WORKING: [],   # Active context (last few exchanges)
            MemoryType.SHORT_TERM: [], # Recent history (last hour)
            MemoryType.LONG_TERM: []   # Important memories (repeated/emotional content)
        }
        
        # Memory consolidation parameters
        self.working_capacity = 5     # Number of recent exchanges to keep active
        self.short_term_hours = 1     # Hours to keep in short-term memory
        self.importance_threshold = 0.7  # Threshold for long-term storage
        
        # Load existing memories
        self._load_memories()
        
    def _load_memories(self):
        """Load memories from persistent storage."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    
                for memory_type in MemoryType:
                    type_str = str(memory_type.value)
                    if type_str in data:
                        self.memories[memory_type] = [
                            Memory(
                                content=m["content"],
                                timestamp=m["timestamp"],
                                memory_type=memory_type,
                                importance=m.get("importance", 0.5),
                                last_access=m.get("last_access", time.time())
                            )
                            for m in data[type_str]
                        ]
        except Exception as e:
            print(f"Error loading memories: {e}")

    def _save_memories(self):
        """Save memories to persistent storage."""
        try:
            data = {}
            for memory_type, memories in self.memories.items():
                data[str(memory_type.value)] = [
                    {
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "importance": m.importance,
                        "last_access": m.last_access
                    }
                    for m in memories
                ]
                
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving memories: {e}")

    def add(self, role: str, content: str):
        """Add new memory, letting model's natural language handle the content."""
        try:
            # Create new memory
            memory = Memory(
                content={"role": role, "content": content},
                timestamp=time.time(),
                memory_type=MemoryType.WORKING,
                importance=self._estimate_importance(content)
            )
            
            # Add to working memory
            self.memories[MemoryType.WORKING].append(memory)
            
            # Consolidate memories naturally
            self._consolidate_memories()
            
            # Save updated memories
            self._save_memories()
            
        except Exception as e:
            print(f"Error adding memory: {e}")

    def get_context(self) -> List[Dict]:
        """Get conversation context in a way that enhances model's abilities."""
        try:
            context = []
            current_time = time.time()
            
            # Get active working memory (most recent exchanges)
            context.extend(m.content for m in self.memories[MemoryType.WORKING][-self.working_capacity:])
            
            # Add relevant short-term memories
            recent_cutoff = current_time - (self.short_term_hours * 3600)
            short_term = [
                m.content for m in self.memories[MemoryType.SHORT_TERM]
                if m.timestamp > recent_cutoff
            ]
            context.extend(short_term)
            
            # Add important long-term memories
            long_term = [
                m.content for m in self.memories[MemoryType.LONG_TERM]
                if m.importance > self.importance_threshold
            ]
            context.extend(long_term)
            
            return context
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return []

    def _estimate_importance(self, content: str) -> float:
        """Estimate memory importance based on content characteristics."""
        try:
            importance = 0.5  # Base importance
            
            # Increase importance for potential key information
            if any(word in content.lower() for word in [
                "name", "remember", "important", "must", "need",
                "forget", "never", "always", "key", "critical"
            ]):
                importance += 0.2
                
            # Increase for emotional content
            if any(word in content.lower() for word in [
                "love", "hate", "happy", "sad", "angry",
                "excited", "worried", "sorry", "thank", "please"
            ]):
                importance += 0.2
                
            # Increase for questions (likely important for context)
            if "?" in content:
                importance += 0.1
                
            return min(1.0, importance)
            
        except Exception as e:
            print(f"Error estimating importance: {e}")
            return 0.5

    def _consolidate_memories(self):
        """Consolidate memories between stores, letting model handle content naturally."""
        try:
            current_time = time.time()
            
            # Move older working memories to short-term
            working_cutoff = current_time - (5 * 60)  # 5 minutes
            for memory in self.memories[MemoryType.WORKING]:
                if memory.timestamp < working_cutoff:
                    memory.memory_type = MemoryType.SHORT_TERM
                    self.memories[MemoryType.SHORT_TERM].append(memory)
            
            # Move important short-term memories to long-term
            for memory in self.memories[MemoryType.SHORT_TERM]:
                if memory.importance > self.importance_threshold:
                    memory.memory_type = MemoryType.LONG_TERM
                    self.memories[MemoryType.LONG_TERM].append(memory)
            
            # Clean up moved memories
            self.memories[MemoryType.WORKING] = [
                m for m in self.memories[MemoryType.WORKING]
                if m.memory_type == MemoryType.WORKING
            ]
            self.memories[MemoryType.SHORT_TERM] = [
                m for m in self.memories[MemoryType.SHORT_TERM]
                if m.memory_type == MemoryType.SHORT_TERM
            ]
            
            # Maintain reasonable memory sizes
            self._prune_memories()
            
        except Exception as e:
            print(f"Error consolidating memories: {e}")

    def _prune_memories(self):
        """Prune memories while preserving important ones."""
        try:
            # Keep only recent working memories
            self.memories[MemoryType.WORKING] = self.memories[MemoryType.WORKING][-self.working_capacity:]
            
            # Remove old short-term memories
            current_time = time.time()
            short_term_cutoff = current_time - (self.short_term_hours * 3600)
            self.memories[MemoryType.SHORT_TERM] = [
                m for m in self.memories[MemoryType.SHORT_TERM]
                if m.timestamp > short_term_cutoff or m.importance > self.importance_threshold
            ]
            
            # Keep only most important long-term memories
            self.memories[MemoryType.LONG_TERM] = sorted(
                self.memories[MemoryType.LONG_TERM],
                key=lambda x: x.importance,
                reverse=True
            )[:100]  # Keep top 100 important memories
            
        except Exception as e:
            print(f"Error pruning memories: {e}")

    def clear(self):
        """Clear all memories and persistent storage."""
        self.memories = {memory_type: [] for memory_type in MemoryType}
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
