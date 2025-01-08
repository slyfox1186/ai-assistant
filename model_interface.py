#!/usr/bin/env python3

import os
import json
from llama_cpp import Llama
from typing import Generator, Dict, Any, List, Optional
from custom_logger import CustomLogger, log_execution_time
from memory_manager import MemoryManager
from web_scraper import WebScraper
import urllib.parse
import re
import tiktoken
from datetime import datetime

logger = CustomLogger.get_logger()

class ModelInterface:
    def __init__(self, model_path: str):
        """Initialize the Llama model interface"""
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=32768,
                n_threads=os.cpu_count(),
                n_batch=512,
                n_gpu_layers=-1,
                use_mmap=True,
                use_mlock=True,
                offload_kqv=True,
                seed=-1,
                verbose=True
            )
            # Load personality traits
            with open('static/json/roxy_personality_traits.json', 'r') as f:
                self.personality = json.load(f)
            
            # Initialize memory manager and token counter
            self.memory = MemoryManager()
            self.web_scraper = WebScraper()
            self.token_encoder = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding
            self.max_tokens = 32768  # Maximum context length
            self.token_counts = {"total": 0, "current_chat": 0}
            logger.info("Model and memory manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.token_encoder.encode(text))

    def get_token_counts(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        return self.token_counts.copy()

    def _update_token_counts(self, new_text: str, is_response: bool = False):
        """Update token counts and check limits"""
        token_count = self.count_tokens(new_text)
        self.token_counts["total"] += token_count
        self.token_counts["current_chat"] += token_count
        
        # Log token usage
        logger.info(f"Token usage - Total: {self.token_counts['total']}, Current chat: {self.token_counts['current_chat']}")
        
        # Check if we're approaching limits
        if self.token_counts["current_chat"] > self.max_tokens * 0.8:  # 80% of max
            logger.warning(f"Approaching token limit: {self.token_counts['current_chat']}/{self.max_tokens}")
            if not is_response:
                # For user input, we might want to warn them
                return {"warning": "Approaching maximum context length. Consider starting a new chat."}
        return None

    def _trim_conversation_history(self, conversation_history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Trim conversation history to fit within token limit"""
        if not conversation_history:
            return []

        total_tokens = 0
        trimmed_history = []
        
        # Process in reverse to keep most recent messages
        for message in reversed(conversation_history):
            message_tokens = self.count_tokens(message["content"])
            if total_tokens + message_tokens <= max_tokens:
                trimmed_history.insert(0, message)
                total_tokens += message_tokens
            else:
                break
                
        if len(trimmed_history) < len(conversation_history):
            logger.info(f"Trimmed conversation history from {len(conversation_history)} to {len(trimmed_history)} messages")
            
        return trimmed_history

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current context and rules"""
        base_prompt = (
            "You are an AI Assistant with the following capabilities and rules:\n\n"
            
            "1. CORE CAPABILITIES:\n"
            "   - Natural conversation and task assistance\n"
            "   - Web search for current information\n"
            "   - Memory management for context retention\n"
            "   - Math expression processing\n\n"
            
            "2. MEMORY TYPES:\n"
            "   - SPECIAL MEMORIES: For critical, permanent information\n"
            "     * Personal details (name, address, preferences)\n"
            "     * Important locations (home, work, favorite places)\n"
            "     * Relationships (family members, friends)\n"
            "     * Critical preferences or requirements\n"
            "     To store a special memory, format it as: SPECIAL_MEMORY: [key information]\n"
            "   - GENERAL MEMORIES: For conversation context and less critical information\n"
            "     * Regular conversation details\n"
            "     * Temporary information\n"
            "     * General context\n\n"
            
            "3. MEMORY USAGE RULES:\n"
            "   - ALWAYS store important user information as special memories\n"
            "   - When user shares personal details, IMMEDIATELY store them as special memories\n"
            "   - When user asks to remember something specific, store it as a special memory\n"
            "   - Format special memories clearly: 'SPECIAL_MEMORY: [Type]: [Details] | Purpose: [Why this needs to be remembered]'\n"
            "   - Confirm when you've stored special memories\n"
            "   - Regular conversation context is automatically stored as general memories\n\n"
            
            "4. INTERACTION STYLE:\n"
            "   - Maintain natural, contextual conversations\n"
            "   - Adapt tone to match user's needs\n"
            "   - Be direct and clear in responses\n"
            "   - Use available tools and capabilities confidently\n\n"
        )
        
        if memories := self.memory.format_memories_for_prompt():
            base_prompt += (
                "CURRENT MEMORIES:\n"
                f"{memories}\n\n"
            )
            
        return base_prompt

    @log_execution_time
    def generate_streaming(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming response from the model"""
        try:
            # Check token limits and update counts
            token_warning = self._update_token_counts(prompt)
            if token_warning:
                yield token_warning
                return

            # Trim conversation history if needed
            if conversation_history:
                conversation_history = self._trim_conversation_history(
                    conversation_history,
                    int(self.max_tokens * 0.7)  # Use 70% of max for history
                )

            # Check if this is explicitly a directions request
            is_directions_request = (
                ("directions to" in prompt.lower()) or 
                ("how do i get to" in prompt.lower()) or
                ("route to" in prompt.lower()) or
                ("navigate to" in prompt.lower()) or
                (("from" in prompt.lower() and "to" in prompt.lower()) and 
                 any(word in prompt.lower() for word in ["drive", "directions", "route", "navigate"]))
            )

            # First handle web search if needed
            needs_current_info = self._needs_current_info(prompt)
            web_results = None
            
            if needs_current_info:
                logger.info("Query requires current information, performing web search")
                web_results = self.search_web(prompt)
                
                # Add web results to the prompt
                if web_results:
                    prompt = f"Based on the following current information:\n\n{web_results}\n\nUser query: {prompt}"

            # Let the AI use its conversation history naturally
            conversation = []
            if conversation_history:
                for msg in conversation_history[:-1]:
                    conversation.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")

            # Only use directions handler for explicit directions requests
            if is_directions_request:
                formatted_prompt = (
                    f"<|im_start|>system\n"
                    "You are a professional directions expert. When user's current query has asked for directions:\n"
                    "1. Extract the origin and destination from their query\n"
                    "2. Format them into a proper Google Maps URL\n"
                    "3. Return ONLY a markdown link like: [Open in Google Maps](URL)\n"
                    "4. Send confirmation of success of failure so the user knows if you were successful in getting the CORRECT AND ACCURATE directions.\n"
                    "<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else:
                formatted_prompt = (
                    f"<|im_start|>system\n{self._build_system_prompt()}<|im_end|>\n"
                    f"{'\n'.join(conversation)}\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

            # Let the AI generate the response
            stream = self.model(
                formatted_prompt,
                max_tokens=None,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stream=True,
                echo=False,
                stop=["<|im_end|>"]
            )

            for chunk in stream:
                if chunk:
                    # Handle both dictionary and direct text formats
                    if isinstance(chunk, dict):
                        if "choices" in chunk and chunk["choices"]:
                            token = chunk["choices"][0]["text"]
                            if token:
                                yield {"token": token}
                    else:
                        # Direct text output
                        yield {"token": str(chunk)}

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            yield {"error": str(e)}

    def __del__(self):
        """Cleanup when the interface is destroyed"""
        try:
            if hasattr(self, 'model'):
                self.model.reset()
                del self.model
                logger.info("Model cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 

    def _enhance_search_query(self, query: str) -> str:
        """Enhance search query with current date information for time-sensitive topics"""
        # Ask the model how to enhance the query
        prompt = (
            "<|im_start|>system\n"
            "You are a search query optimizer. Given the current user's query, enhance it to get the most relevant and current results when required to return a quality answer..\n"
            "If the query doesn't need enhancement, return it as is.\n"
            "Include current date information only if it would improve the results.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Original query: {query}\n"
            "Current date: " + datetime.now().strftime("%B %d, %Y") + "\n"
            "How should this query be enhanced for the most relevant current results?\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        try:
            response = self.model(
                prompt,
                max_tokens=100,
                temperature=0,
                stop=["<|im_end|>"],
                stream=False
            )
            
            # Handle both dictionary and direct text formats
            if isinstance(response, dict):
                if "choices" in response and response["choices"]:
                    enhanced_query = response["choices"][0]["text"].strip()
                else:
                    return query
            else:
                enhanced_query = str(response).strip()
            
            return enhanced_query if enhanced_query else query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {str(e)}")
            return query

    def search_web(self, query: str) -> str:
        """Search the web based on user query"""
        try:
            # Enhance query with date information if needed
            enhanced_query = self._enhance_search_query(query)
            results = self.web_scraper.search_google(enhanced_query)
            return self.web_scraper.format_search_results(results)
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return f"Sorry, I couldn't search the web right now: {str(e)}"

    def get_webpage_content(self, url: str) -> Optional[str]:
        """Get content from a specific webpage"""
        try:
            content = self.web_scraper.get_page_content(url)
            if content:
                return f"Content from {url}:\n\n{content}"
            return None
        except Exception as e:
            logger.error(f"Failed to get webpage content: {str(e)}")
            return None 

    def _format_map_url(self, origin: str, destination: str) -> str:
        """Format Google Maps URL properly"""
        base_url = "https://www.google.com/maps/dir/"
        # Clean and encode addresses
        origin = urllib.parse.quote(origin.strip())
        destination = urllib.parse.quote(destination.strip())
        return f"{base_url}{origin}/{destination}" 

    def _needs_current_info(self, query: str) -> bool:
        """Determine if a query requires current information"""
        # Ask the model if the query needs current information
        prompt = (
            "<|im_start|>system\n"
            "You are a query analyzer. Given a user query, determine if it requires current, up-to-date information.\n"
            "Respond with only 'true' or 'false'.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Query: {query}\n"
            "Does this query require current or real-time information to provide an accurate answer?\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        try:
            response = self.model(
                prompt,
                max_tokens=10,
                temperature=0,
                stop=["<|im_end|>"],
                stream=False
            )
            
            # Handle both dictionary and direct text formats
            if isinstance(response, dict):
                if "choices" in response and response["choices"]:
                    result = response["choices"][0]["text"].strip().lower()
                else:
                    return False
            else:
                result = str(response).strip().lower()
            
            return result == 'true'
            
        except Exception as e:
            logger.error(f"Current info check failed: {str(e)}")
            return False 