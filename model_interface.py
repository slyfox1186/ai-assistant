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

logger = CustomLogger.get_logger()

class ModelInterface:
    def __init__(self, model_path: str):
        """Initialize the Llama model interface"""
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=16384,
                n_threads=os.cpu_count(),
                n_batch=512,
                n_gpu_layers=-1,
                use_mmap=True,
                use_mlock=True,
                offload_kqv=False,
                seed=-1,
                verbose=True
            )
            # Load personality traits
            with open('static/json/charlotte_personality_traits.json', 'r') as f:
                self.personality = json.load(f)
            
            # Initialize memory manager
            self.memory = MemoryManager()
            self.web_scraper = WebScraper()
            logger.info("Model and memory manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise

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
            
            "2. INTERACTION STYLE\n"
            "   - Maintain natural, contextual conversations\n"
            "   - Adapt tone to match user's needs\n"
            "   - Be direct and clear in responses\n\n"
            
            "3. INFORMATION HANDLING\n"
            "   - For any query not specifically defined: Use COMMON SENSE and apply the KISS principle (Keep It Simple Stupid) to provide an optimal response\n"
            "   - For directions: STOP GENERATING DIRECTIONS. Your ONLY job is to return: [Get Directions](URL). Nothing else.\n"
            "   - For searches: Use search_web() to get current information when needed\n"
            "   - For web content: Extract and summarize key points\n\n"
            
            "4. MEMORY UTILIZATION\n"
            "   - Remember user preferences and details\n"
            "   - Maintain conversation context\n"
            "   - Use past interactions to improve responses\n\n"
        )
        
        if memories:
            base_prompt += (
                "CONVERSATION CONTEXT:\n"
                f"{memories}\n\n"
            )
        
        base_prompt += (
            "MEMORY TRACKING:\n"
            "After responses, note important user information:\n"
            "SPECIAL_MEMORY: [relevant user details]\n"
        )
        
        return base_prompt

    @log_execution_time
    def generate_streaming(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming response from the model"""
        try:
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