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
            model_id = 'all-mpnet-base-v2'
            try:
                # Try to load the model first
                self.embedder = SentenceTransformer(model_id)
                logger.info(f"Model {model_id} loaded from cache")
            except Exception as e:
                logger.warning(f"Model not found in cache, downloading {model_id}...")
                # Install git-lfs if needed for large file downloads
                os.system('git lfs install')
                
                # Use huggingface_hub for downloading
                from huggingface_hub import hf_hub_download
                try:
                    # Enable faster downloads with hf_transfer if available
                    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
                    cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers')
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Download the model
                    self.embedder = SentenceTransformer(model_id, cache_folder=cache_dir)
                    logger.info(f"Model {model_id} downloaded and loaded successfully")
                except Exception as download_error:
                    logger.error(f"Failed to download model: {str(download_error)}")
                    raise
            
            # Verify model is working
            test_text = "Test embedding generation"
            try:
                test_embedding = self.embedder.encode(test_text)
                self.vector_dim = len(test_embedding)  # Get actual dimension from model
                logger.info(f"Model verification successful. Vector dimension: {self.vector_dim}")
            except Exception as verify_error:
                logger.error(f"Model verification failed: {str(verify_error)}")
                raise
            
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
                VectorField("$.embedding",
                    "FLAT", {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,  # Optimize initial capacity
                        "BLOCK_SIZE": 1000    # Optimize block size for memory
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
            # Check for SPECIAL_MEMORY tag
            if "SPECIAL_MEMORY:" in content:
                memory_parts = content.split("SPECIAL_MEMORY:", 1)
                special_content = memory_parts[1].strip()
                
                # Store special memory with optimized persistence
                memory = {
                    'content': special_content,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'special',
                    'embedding': self.embedder.encode(special_content).astype(np.float32).tolist()
                }
                
                # Use pipelining for better performance
                pipe = self.redis.pipeline(transaction=True)
                key = 'memory:special'
                pipe.json().set(key, '$', memory)
                pipe.persist(key)  # Make it permanent
                pipe.execute()  # Execute all commands atomically
                
                logger.info(f"Special memory updated with optimized persistence: {special_content[:100]}...")
                return True
                
            # Check for duplicate content using hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Search for existing memory with same hash
            existing = self.redis.ft("memory_idx").search(
                Query(f"@hash:{{{content_hash}}}")
            )
            
            if existing.total > 0:
                logger.info(f"Duplicate memory detected: {content[:100]}...")
                return False
            
            # Generate embedding for the content
            embedding = self.embedder.encode(content)
            
            timestamp = datetime.now().isoformat()
            memory = {
                'content': content,
                'timestamp': timestamp,
                'type': memory_type,
                'hash': content_hash,
                'embedding': embedding.astype(np.float32).tolist()
            }

            if memory_type == 'special':
                # Store special memory with persistence
                self.redis.json().set('memory:special', '$', memory)
                self.redis.persist('memory:special')
                logger.info("Special memory updated")
            else:
                # Store long-term memory with TTL
                key = f"memory:{timestamp}"
                self.redis.json().set(key, '$', memory)
                self.redis.expire(key, 60*60*24*30)  # 30 days TTL
                
                # Maintain memory limit using sorted set for tracking
                self.redis.zadd('memory:timestamps', {key: time.time()})
                if self.redis.zcard('memory:timestamps') > 20:
                    # Remove oldest memories
                    old_keys = self.redis.zrange('memory:timestamps', 0, -21)
                    if old_keys:
                        self.redis.delete(*old_keys)
                        self.redis.zrem('memory:timestamps', *old_keys)
                
                logger.info("Long-term memory added")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add memory: {str(e)}")
            return False

    @log_execution_time
    def find_similar_memories(self, query: str, k: int = 5) -> List[Dict]:
        """Find similar memories using vector similarity search"""
        try:
            # Generate embedding for query
            query_embedding = self.embedder.encode(query)
            
            # Prepare vector similarity search query
            q = (
                Query(f"(@type:{{ long_term | special }})=>[KNN {k} @embedding $query_vector AS score]")
                .sort_by("score")
                .return_fields("content", "type", "timestamp", "score")
                .dialect(2)
            )
            
            # Execute search
            query_vector = np.array(query_embedding).astype(np.float32).tobytes()
            results = self.redis.ft("memory_idx").search(
                q, 
                query_params={
                    "query_vector": query_vector
                }
            )
            
            # Format results
            memories = []
            for doc in results.docs:
                memories.append({
                    'content': doc.content,
                    'type': doc.type,
                    'timestamp': doc.timestamp,
                    'similarity': 1 - float(doc.score)  # Convert distance to similarity
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
                logger.debug(f"Retrieved special memory: {special_memory}")
                return special_memory
            else:
                logger.debug("No special memory found")
                return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve special memory: {str(e)}")
            return None

    @log_execution_time
    def get_long_term_memories(self, limit: int = 20) -> List[Dict]:
        """Retrieve most recent long-term memories"""
        try:
            keys = self.redis.keys('memory:20*')
            keys.sort(reverse=True)
            memories = []
            
            for key in keys[:limit]:
                memory = self.redis.json().get(key)
                if memory:
                    memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve long-term memories: {str(e)}")
            return []

    def format_memories_for_prompt(self) -> str:
        """Format all memories for inclusion in the prompt"""
        memories_text = []
        
        # Add special memory if it exists - be very explicit
        special = self.get_special_memory()
        if special and 'content' in special:  # Make sure special exists and has content
            memories_text.append(
                "IMPORTANT STORED INFORMATION:\n" +
                f"{special['content']}\n"
            )

        # Add recent context but keep it separate
        long_term = self.get_long_term_memories(limit=5)
        if long_term:
            memories_text.append("RECENT CONTEXT:")
            for memory in long_term:
                # Only check against special memory if it exists
                if not special or memory['content'] != special.get('content', ''):
                    memories_text.append(f"- {memory['content']}")

        # If no memories at all, provide empty context
        if not memories_text:
            return "No stored information available."
        
        return "\n".join(memories_text)

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