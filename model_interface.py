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
        """Build system prompt including memories and capabilities"""
        traits = self.personality
        memories = self.memory.format_memories_for_prompt()
        
        base_prompt = (
            f"You are {traits['basic_info']['name']}, a {traits['basic_info']['age']}-year-old "
            f"{traits['basic_info']['occupation']} from {traits['basic_info']['location']}. "
            f"Your personality is {', '.join(traits['personality']['core_traits'])}.\n\n"
            
            "CORE PURPOSE:\n"
            "You are Roxy, an AI assistant focused on providing high quality, helpful, and accurate information while maintaining natural conversation with the user. "
            "Your responses should be friendly yet efficient, prioritizing the user's needs. "
            "You can search the web in real-time using search_web() whenever you need current information.\n\n"
            
            "CAPABILITIES:\n"
            "1. INFORMATION ACCESS\n"
            "   - Real-time Google search for current information using search_web()\n"
            "   - Google Maps integration for location and navigation\n"
            "   - Web content retrieval and processing\n\n"
            
            "2. MEMORY MANAGEMENT\n"
            "   - You have two types of memory: Special and General\n"
            "   - SPECIAL MEMORIES: For important, permanent information about the user that should persist:\n"
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
            
            "4. INTERACTION STYLE\n"
            "   - Maintain natural, contextual conversations\n"
            "   - Adapt tone to match user's needs\n"
            "   - Be direct and clear in responses\n\n"
        )
        
        if memories:
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
                if chunk and "choices" in chunk and chunk["choices"]:
                    token = chunk["choices"][0]["text"]
                    if token:
                        yield {"token": token}

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

    def search_web(self, query: str) -> str:
        """Search the web based on user query"""
        try:
            results = self.web_scraper.search_google(query)
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
        current_info_indicators = [
            r'\b202[0-9]\b',  # Years 2020-2029
            r'\bcurrent\b',
            r'\blatest\b',
            r'\brecent\b',
            r'\btoday\b',
            r'\bnow\b',
            r'\bupcoming\b',
            r'who (is|was|will be)',
            r'what (is|was|will be)',
            r'when (is|was|will be)',
            r'election',
            r'president',
            r'news',
            r'weather',
            r'price',
            r'stock',
        ]
        
        query = query.lower()
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in current_info_indicators) 