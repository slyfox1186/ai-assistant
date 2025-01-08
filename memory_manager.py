#!/usr/bin/env python3

import redis
import numpy as np
import json
from typing import List, Dict, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import (
    TextField,
    VectorField,
    TagField
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from custom_logger import CustomLogger, log_execution_time
import hashlib
import time
import re
import os

logger = CustomLogger.get_logger()

class MemoryManager:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis connection and memory structure"""
        try:
            # Load configuration
            with open('static/json/ai_configuration.json', 'r') as f:
                config = json.load(f)
            
            # Get memory limits from config
            self.memory_limits = config['system']['memory_settings']['limits']
            self.max_stored_memories = self.memory_limits['max_stored_memories']
            self.max_context_memories = self.memory_limits['max_context_memories']
            
            # Initialize Redis connection with optimized settings
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True
            )
            self.redis.ping()
            
            # Configure Redis for optimized persistence and memory usage
            self.redis.config_set('maxmemory', '2gb')
            self.redis.config_set('maxmemory-policy', 'allkeys-lru')
            self.redis.config_set('maxmemory-samples', '10')
            
            # Persistence settings
            self.redis.config_set('save', '1 1')
            self.redis.config_set('appendonly', 'yes')
            self.redis.config_set('appendfsync', 'everysec')
            self.redis.config_set('aof-use-rdb-preamble', 'yes')
            
            # Memory optimization settings
            self.redis.config_set('activedefrag', 'yes')
            self.redis.config_set('active-defrag-threshold-lower', '10')
            self.redis.config_set('active-defrag-threshold-upper', '100')
            
            # Initialize sentence transformer with proper downloading and verification
            logger.info("Initializing sentence transformer model...")
            model_id = 'sentence-transformers/all-mpnet-base-v2'  # Use full model ID
            cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers')
            os.makedirs(cache_dir, exist_ok=True)
            
            try:
                # First try to use huggingface_hub to verify model
                from huggingface_hub import snapshot_download, HfFolder, hf_hub_download
                
                # Enable faster downloads if available
                os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
                
                try:
                    # First try local cache
                    logger.info(f"Checking for model {model_id} in cache...")
                    snapshot_download(
                        repo_id=model_id,
                        cache_dir=cache_dir,
                        local_files_only=True
                    )
                    logger.info("Model found in cache")
                except Exception as cache_error:
                    # If not in cache, attempt to download
                    logger.info(f"Model not found in cache, downloading {model_id}...")
                    try:
                        snapshot_download(
                            repo_id=model_id,
                            cache_dir=cache_dir,
                            local_files_only=False  # Allow download
                        )
                        logger.info("Model downloaded successfully")
                    except Exception as download_error:
                        error_msg = (
                            f"Failed to download model '{model_id}'.\n"
                            "Please ensure you:\n"
                            "1. Have an internet connection\n"
                            "2. Have installed requirements:\n"
                            "   pip install -U sentence-transformers huggingface-hub[cli]\n"
                            "3. Optionally login for faster downloads:\n"
                            "   huggingface-cli login\n\n"
                            f"Download Error: {str(download_error)}"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                
                # Load model after verifying/downloading
                self.embedder = SentenceTransformer(model_id, cache_folder=cache_dir)
                logger.info(f"Model {model_id} loaded successfully")
                
                # Verify model is working
                test_text = "Test embedding generation"
                test_embedding = self.embedder.encode(test_text)
                self.vector_dim = len(test_embedding)
                logger.info(f"Model verification successful. Vector dimension: {self.vector_dim}")
                
            except Exception as e:
                error_msg = f"Failed to load or verify sentence transformer model: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Create search index if it doesn't exist
            self._create_search_index()
            
            logger.info("Redis connection and search index initialized with optimized settings")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _create_search_index(self):
        """Create Redis search index for vector similarity search and deduplication"""
        try:
            # Define schema for the index with optimized settings
            schema = (
                TextField("$.content", as_name="content", sortable=True, no_stem=True),
                TagField("$.type", as_name="type", sortable=True),
                TextField("$.hash", as_name="hash", sortable=True, no_stem=True),
                TagField("$.timestamp", as_name="timestamp", sortable=True),
                VectorField("$.embedding",
                    "HNSW", {  # Change to HNSW for better performance
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        "M": 40,  # Number of connections per node
                        "EF_CONSTRUCTION": 200  # Balance between build time and accuracy
                    },
                    as_name="embedding"
                )
            )
            
            # Create the index with optimized definition
            definition = IndexDefinition(
                prefix=["memory:"],
                index_type=IndexType.JSON,
                score_field="score",
                score=1.0
            )
            
            try:
                self.redis.ft("memory_idx").create_index(
                    fields=schema,
                    definition=definition
                )
                logger.info("Search index created with optimized settings")
            except Exception as e:
                if "Index already exists" in str(e):
                    logger.info("Search index already exists")
                else:
                    raise

        except Exception as e:
            logger.error(f"Failed to create search index: {str(e)}")
            raise

    @log_execution_time
    def add_memory(self, content: str, memory_type: str = 'long_term') -> bool:
        """Add a new memory with vector embedding and deduplication"""
        try:
            # Simple special memory check - trust the model's classification
            is_special = memory_type == 'special'
            
            if is_special:
                # Format special content
                special_content = content.strip()
                
                # Get existing special memories to append/update
                existing_special = self.get_special_memory()
                if existing_special and 'content' in existing_special:
                    # Append new content to existing special memories
                    existing_content = existing_special['content']
                    if special_content not in existing_content:  # Avoid duplicates
                        special_content = f"{existing_content}\n{special_content}"
                
                # Generate embedding for special memory
                embedding = self.embedder.encode(special_content).astype(np.float32).tolist()
                timestamp = datetime.now().isoformat()
                
                # Store special memory
                memory = {
                    'content': special_content,
                    'timestamp': timestamp,
                    'type': 'special',
                    'embedding': embedding,
                    'hash': hashlib.sha256(special_content.encode()).hexdigest()
                }
                
                # Use pipelining for better performance
                pipe = self.redis.pipeline(transaction=True)
                key = 'memory:special'
                pipe.json().set(key, '$', memory)
                pipe.persist(key)
                pipe.zadd('memory:timestamps', {key: float('inf')})
                pipe.persist('memory:timestamps')
                pipe.execute()
                
                logger.info(f"Special memory updated: {special_content[:100]}...")
                return True
            
            # For regular memories
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check for duplicates using vector similarity
            query = (
                Query(f"(@hash:{{{content_hash}}})")
                .return_fields("content", "hash")
                .dialect(2)
            )
            
            existing = self.redis.ft("memory_idx").search(query)
            if existing.total > 0:
                logger.info(f"Duplicate memory detected: {content[:100]}...")
                return False
            
            # Generate embedding
            embedding = self.embedder.encode(content).astype(np.float32).tolist()
            timestamp = datetime.now().isoformat()
            
            # Create memory object
            memory = {
                'content': content,
                'timestamp': timestamp,
                'type': memory_type,
                'hash': content_hash,
                'embedding': embedding
            }
            
            # Use timestamp as part of the key for proper ordering
            key = f'memory:{timestamp}'
            
            # Store memory and update indexes atomically
            pipe = self.redis.pipeline(transaction=True)
            
            # Store the memory
            pipe.json().set(key, '$', memory)
            pipe.persist(key)  # Make it permanent
            
            # Update timestamp index
            pipe.zadd('memory:timestamps', {key: time.time()})
            pipe.persist('memory:timestamps')
            
            # Maintain memory limit from configuration
            pipe.zremrangebyrank('memory:timestamps', 0, -(self.max_stored_memories + 1))
            
            # Execute all commands atomically
            pipe.execute()
            
            logger.info(f"Memory stored successfully: {content[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add memory: {str(e)}")
            return False

    @log_execution_time
    def find_similar_memories(self, query: str, k: int = 5) -> List[Dict]:
        """Find similar memories using vector similarity search"""
        try:
            # Generate embedding for query
            query_embedding = self.embedder.encode(query).astype(np.float32)
            
            # Prepare vector similarity search query with hybrid search
            q = (
                Query("*=>[KNN $K @embedding $query_vector AS vector_score]")
                .sort_by("vector_score")
                .return_fields("content", "type", "timestamp", "vector_score")
                .paging(0, k)
                .dialect(2)
            )
            
            # Execute search with parameters
            results = self.redis.ft("memory_idx").search(
                q, 
                query_params={
                    "query_vector": query_embedding.tobytes(),
                    "K": k
                }
            )
            
            # Format results
            memories = []
            for doc in results.docs:
                memories.append({
                    'content': doc.content,
                    'type': doc.type,
                    'timestamp': doc.timestamp,
                    'similarity': 1 - float(doc.vector_score)  # Convert distance to similarity
                })
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search memories: {str(e)}")
            return []

    @log_execution_time
    def get_special_memory(self) -> Optional[Dict]:
        """Retrieve the special memory containing important user information"""
        try:
            # Get the special memory from Redis
            special_memory = self.redis.json().get('memory:special')
            
            if special_memory:
                # Verify the memory is properly persisted
                if not self.redis.persist('memory:special'):
                    # If not persisted, make it persistent
                    self.redis.persist('memory:special')
                
                logger.debug(f"Retrieved special memory: {special_memory}")
                return special_memory
            else:
                logger.debug("No special memory found")
                return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve special memory: {str(e)}")
            return None

    @log_execution_time
    def get_recent_memories(self, limit: int = 5) -> List[Dict]:
        """Get the most recent memories"""
        try:
            # Get the most recent memory keys from the sorted set
            keys = self.redis.zrevrange('memory:timestamps', 0, limit-1)
            if not keys:
                return []
            
            # Use pipelining for efficient batch retrieval
            pipe = self.redis.pipeline(transaction=True)
            for key in keys:
                pipe.json().get(key)
            
            # Execute all gets at once
            results = pipe.execute()
            
            # Filter and process results
            memories = []
            for memory in results:
                if memory and memory.get('type') != 'special':  # Exclude special memories
                    memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Failed to get recent memories: {str(e)}")
            return []

    @log_execution_time
    def get_long_term_memories(self, limit: int = 20) -> List[Dict]:
        """Retrieve most recent long-term memories"""
        try:
            # Use RediSearch for efficient filtering
            q = (
                Query("@type:{long_term}")
                .sort_by("timestamp", asc=False)
                .paging(0, limit)
                .return_fields("content", "timestamp", "type")
                .dialect(2)
            )
            
            results = self.redis.ft("memory_idx").search(q)
            
            memories = []
            for doc in results.docs:
                memories.append({
                    'content': doc.content,
                    'timestamp': doc.timestamp,
                    'type': doc.type
                })
            
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve long-term memories: {str(e)}")
            return []

    def format_memories_for_prompt(self) -> str:
        """Format all memories for inclusion in the prompt"""
        try:
            # Get special memory first
            special = self.get_special_memory()
            formatted_memories = []
            
            # Special memories section - critical user information that must never be forgotten
            formatted_memories.append("\n=== 🔒 SPECIAL MEMORIES (Must Never Be Forgotten) ===")
            if special and 'content' in special:
                content = special.get('content', '')
                entries = []
                
                # Parse entries
                raw_entries = [e.strip() for e in content.split('\n') if e.strip()]
                
                # Group entries by type for better organization
                personal_info = []
                locations = []
                preferences = []
                other = []
                
                for entry in raw_entries:
                    if "Address:" in entry or "Location:" in entry:
                        locations.append(entry)
                    elif "Name:" in entry or "Phone:" in entry:
                        personal_info.append(entry)
                    elif "Preference:" in entry:
                        preferences.append(entry)
                    else:
                        other.append(entry)
                
                # Format each category
                if personal_info:
                    formatted_memories.append("Personal Information:")
                    for i, entry in enumerate(personal_info, 1):
                        formatted_memories.append(f"  {i}. {entry}")
                
                if locations:
                    formatted_memories.append("Important Locations:")
                    for i, entry in enumerate(locations, 1):
                        formatted_memories.append(f"  {i}. {entry}")
                
                if preferences:
                    formatted_memories.append("User Preferences:")
                    for i, entry in enumerate(preferences, 1):
                        formatted_memories.append(f"  {i}. {entry}")
                
                if other:
                    formatted_memories.append("Other Important Information:")
                    for i, entry in enumerate(other, 1):
                        formatted_memories.append(f"  {i}. {entry}")
            else:
                formatted_memories.append("No special memories stored yet.")
            
            # Get a mix of recent and relevant memories
            # First, get 5 most recent memories for immediate context
            recent_memories = self.get_recent_memories(limit=5)
            
            # Then get relevant memories if we have a recent memory to compare against
            relevant_memories = []
            if recent_memories:
                # Use the most recent memory as the query for finding relevant ones
                query = recent_memories[0].get('content', '')
                relevant_memories = self.find_similar_memories(query, k=5)
                
                # Filter out memories that are too similar (likely duplicates)
                seen_hashes = set()
                filtered_memories = []
                for memory in relevant_memories:
                    content = memory.get('content', '').strip()
                    memory_hash = hashlib.sha256(content.encode()).hexdigest()
                    if memory_hash not in seen_hashes and memory.get('similarity', 0) > 0.5:  # Only include if similarity > 0.5
                        seen_hashes.add(memory_hash)
                        filtered_memories.append(memory)
                relevant_memories = filtered_memories

            # Combine and deduplicate memories
            all_memories = []
            seen_contents = set()
            
            # Add recent memories first
            if recent_memories:
                formatted_memories.append("\n=== 📝 RECENT CONTEXT ===")
                for memory in recent_memories:
                    content = memory.get('content', '').strip()
                    if content and not content.startswith('SPECIAL_MEMORY:') and content not in seen_contents:
                        formatted_memories.append(f"- {content}")
                        seen_contents.add(content)
            
            # Add relevant memories
            if relevant_memories:
                formatted_memories.append("\n=== 🔍 RELEVANT CONTEXT ===")
                for memory in relevant_memories:
                    content = memory.get('content', '').strip()
                    if content and not content.startswith('SPECIAL_MEMORY:') and content not in seen_contents:
                        formatted_memories.append(f"- {content}")
                        seen_contents.add(content)
            
            return "\n".join(formatted_memories)
            
        except Exception as e:
            logger.error(f"Failed to format memories: {str(e)}")
            return "Error retrieving memories"

    def clear_memories(self):
        """Clear all memories from Redis"""
        try:
            logger.debug("Current memories before clearing:")
            logger.debug(self.format_memories_for_prompt())
            
            # Clear all keys in the current database
            self.redis.flushdb()
            
            # Also clear the sorted set used for tracking
            self.redis.delete('memory:timestamps')
            
            # Clear any special memories
            self.redis.delete('memory:special')
            
            # Force index rebuild
            try:
                self.redis.ft("memory_idx").dropindex()
                self._create_search_index()
            except:
                pass
            
            logger.debug("Memories after clearing:")
            logger.debug(self.format_memories_for_prompt())
            
            logger.info("All memories cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing memories: {str(e)}")
            raise 

    def clear_map_memories(self):
        """Clear only map/direction related memories with verification"""
        try:
            # Debug before state
            self.debug_redis_contents()
            
            # Start Redis transaction
            pipe = self.redis.pipeline()
            
            # Get all keys first
            keys = self.redis.keys('memory:20*')
            special_memory = self.get_special_memory()
            
            # Track what we're going to delete
            to_delete = []
            
            # Check each memory systematically
            for key in keys:
                memory = self.redis.json().get(key)
                if memory and self._is_map_related(memory.get('content', '')):
                    to_delete.append(key)
                    pipe.delete(key)
                    pipe.zrem('memory:timestamps', key)
            
            # Check special memory
            if special_memory and self._is_map_related(special_memory.get('content', '')):
                pipe.delete('memory:special')
            
            # Execute transaction
            pipe.execute()
            
            # Verify cleanup
            self.debug_redis_contents()
            
            # Return cleanup report
            return {
                "cleaned_keys": to_delete,
                "verification": "success" if all(not self.redis.exists(k) for k in to_delete) else "failed"
            }
            
        except Exception as e:
            logger.error(f"Error clearing map memories: {str(e)}")
            raise

    def _is_map_related(self, content: str) -> bool:
        """Systematic check if content is map related"""
        if not content or not isinstance(content, str):
            return False
        
        content = content.lower()
        
        # Core map indicators
        if any(x in content for x in ['directions from', 'google maps url', 'route from']):
            return True
        
        # Address pattern check
        if re.search(r'\d+.*(?:street|st|avenue|ave|road|rd|highway|hwy)', content, re.I):
            return True
        
        # Navigation pattern check
        if re.search(r'(?:turn|head|continue|take exit|miles)', content, re.I):
            return True
        
        return False

    def debug_redis_contents(self):
        """Debug helper to see what's actually in Redis"""
        try:
            logger.debug("=== REDIS CONTENTS DEBUG ===")
            
            # Check all keys
            all_keys = self.redis.keys('*')
            logger.debug(f"All Redis keys: {all_keys}")
            
            # Check memory timestamps
            timestamps = self.redis.zrange('memory:timestamps', 0, -1, withscores=True)
            logger.debug(f"Memory timestamps: {timestamps}")
            
            # Check special memory
            special = self.redis.json().get('memory:special')
            logger.debug(f"Special memory: {special}")
            
            # Check each memory content
            for key in self.redis.keys('memory:20*'):
                content = self.redis.json().get(key)
                logger.debug(f"Memory {key}: {content}")
            
            logger.debug("=== END REDIS DEBUG ===")
        except Exception as e:
            logger.error(f"Debug failed: {str(e)}") 