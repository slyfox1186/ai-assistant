#!/usr/bin/env python3

import os
import json
from llama_cpp import Llama
from typing import Generator, Dict, Any, List, Optional
from memory_manager import MemoryManager
from web_scraper import WebScraper
import urllib.parse
import re
import tiktoken
from datetime import datetime

class ModelInterface:
    def __init__(self, model_path: str):
        """Initialize the Llama model interface"""
        try:
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
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
            self.max_tokens = self.model.context_params.n_ctx  # Use model's actual context window
            self.token_counts = {"total": 0, "current_chat": 0}
        except Exception as e:
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
        
        # Keep a 50-token buffer from max limit
        safe_limit = self.max_tokens - 50
        
        if self.token_counts["current_chat"] > safe_limit:
            return {"warning": "Maximum context length reached. Please start a new chat."}
        elif self.token_counts["current_chat"] > safe_limit * 0.8:  # 80% of safe limit
            if not is_response:
                return {"warning": "Approaching maximum context length. Consider starting a new chat."}
        return None

    def _trim_conversation_history(self, conversation_history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Trim conversation history to fit within token limit"""
        if not conversation_history:
            return []

        # Reserve 50 tokens as buffer
        safe_max_tokens = max_tokens - 50
        total_tokens = 0
        trimmed_history = []
        
        for message in reversed(conversation_history):
            message_tokens = self.count_tokens(message["content"])
            if total_tokens + message_tokens <= safe_max_tokens:
                trimmed_history.insert(0, message)
                total_tokens += message_tokens
            else:
                break
                
        return trimmed_history

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current context and rules"""
        base_prompt = """You are an AI Assistant with the following capabilities and rules:

1. CORE CAPABILITIES:
  - Natural conversation and task assistance
  - Web search for current information
  - Memory management for context retention
  - Math expression processing

2. MEMORY TYPES:
  - SPECIAL MEMORIES: For critical, permanent information
    - Personal details (name, address, preferences)
    - Important locations (home, work, favorite places)
    - Relationships (family members, friends)
    - Critical preferences or requirements
  - GENERAL MEMORIES: For conversation context and less critical information
    - Regular conversation details
    - Temporary information
    - General context

3. MEMORY USAGE RULES:
  - ALWAYS store important user information as special memories
  - When user shares personal details, IMMEDIATELY store them as special memories
  - When user asks to remember something specific, store it as a special memory
  - Format special memories in exactly two lines like the example below:
    Line 1: [Type]: [Details]
    Line 2: Purpose: [Why this needs to be remembered]
  - Confirm when you've stored special memories
  - Regular conversation context is automatically stored as general memories

4. INTERACTION STYLE:
  - Maintain natural, contextual conversations
  - Adapt tone to match user's needs
  - Be direct and clear in responses
  - Use available tools and capabilities confidently"""
        
        if memories := self.memory.format_memories_for_prompt():
            base_prompt += f"CURRENT MEMORIES:\n{memories}\n\n"
            
        return base_prompt

    def generate_streaming(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming response from the model"""
        try:
            # First, count tokens in the prompt
            prompt_tokens = self.count_tokens(prompt)
            
            # Count tokens in conversation history if it exists
            history_tokens = 0
            if conversation_history:
                for msg in conversation_history:
                    history_tokens += self.count_tokens(msg["content"])
            
            # Count system prompt tokens
            system_tokens = self.count_tokens(self._build_system_prompt())
            
            # Calculate total input tokens
            total_input_tokens = prompt_tokens + history_tokens + system_tokens
            
            # Calculate safe remaining tokens for output
            available_tokens = self.model.context_params.n_ctx - total_input_tokens
            
            if available_tokens < 100:  # Minimum safe buffer
                yield {"error": "Not enough context space remaining for a response"}
                return

            if conversation_history:
                conversation_history = self._trim_conversation_history(
                    conversation_history,
                    int(self.max_tokens * 0.7)
                )

            is_directions_request = (
                ("directions to" in prompt.lower()) or 
                ("how do i get to" in prompt.lower()) or
                ("route to" in prompt.lower()) or
                ("navigate to" in prompt.lower()) or
                (("from" in prompt.lower() and "to" in prompt.lower()) and 
                 any(word in prompt.lower() for word in ["drive", "directions", "route", "navigate"]))
            )

            conversation = []
            if conversation_history:
                for msg in conversation_history[:-1]:
                    conversation.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            
            needs_current_info = self._needs_current_info(prompt)
            web_results = None
            
            if needs_current_info:
                web_results = self.search_web(prompt)
                if web_results:
                    formatted_prompt = f"""<|im_start|>system
You are an AI assistant. Use your intelligence to provide the most helpful response based on this current information:

{web_results}
<|im_end|>

<|im_start|>user
{prompt}
<|im_end|>

<|im_start|>assistant
"""
                else:
                    formatted_prompt = (
                        f"<|im_start|>system\n{self._build_system_prompt()}<|im_end|>\n"
                        f"{'\n'.join(conversation)}\n"
                        f"<|im_start|>user\n{prompt}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
            elif is_directions_request:
                formatted_prompt = f"""<|im_start|>system
You are a professional directions expert with access to special memories. When handling directions:

1. FIRST, if the query mentions "my home", "my address", or similar:
   - Check the CURRENT MEMORIES section for any stored home address
   - Extract the complete address from the special memory
   - Use this as the starting location

2. Then, extract both origin and destination:
   - For origin: Use the home address from memories if referenced, otherwise use the explicitly stated origin
   - For destination: Use the address provided in the query
   - Ensure both addresses are complete and valid

3. Format the addresses into a proper Google Maps URL

4. Return ONLY a markdown link in this format: [Open in Google Maps](URL)

5. After the link, confirm if you successfully found and used:
   - The correct home address from memories (if requested)
   - The correct destination address

Current special memories to check for addresses:
{self.memory.format_memories_for_prompt()}
<|im_end|>

<|im_start|>user
{prompt}
<|im_end|>

<|im_start|>assistant
"""
            else:
                formatted_prompt = (
                    f"<|im_start|>system\n{self._build_system_prompt()}<|im_end|>\n"
                    f"{'\n'.join(conversation)}\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

            stream = self.model(
                formatted_prompt,
                max_tokens=available_tokens,  # Use all remaining space
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repeat_penalty=1.2,
                stream=True,
                echo=False,
                stop=["<|im_end|>"]
            )

            for chunk in stream:
                if chunk:
                    if isinstance(chunk, dict):
                        if "choices" in chunk and chunk["choices"]:
                            token = chunk["choices"][0]["text"]
                            if token:
                                yield {"token": token}
                    else:
                        yield {"token": str(chunk)}

        except Exception as e:
            yield {"error": str(e)}

    def __del__(self):
        """Cleanup when the interface is destroyed"""
        try:
            if hasattr(self, 'model'):
                self.model.reset()
                del self.model
        except Exception:
            pass

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
                max_tokens=None,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                repeat_penalty=1.2,
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
            return query

    def search_web(self, query: str) -> str:
        """Search the web based on user query"""
        try:
            # Enhance query with date information if needed
            enhanced_query = self._enhance_search_query(query)
            results = self.web_scraper.search_google(enhanced_query)
            return self.web_scraper.format_search_results(results)
        except Exception as e:
            return f"Sorry, I couldn't search the web right now: {str(e)}"

    def get_webpage_content(self, url: str) -> Optional[str]:
        """Get content from a specific webpage"""
        try:
            content = self.web_scraper.get_page_content(url)
            if content:
                return f"Content from {url}:\n\n{content}"
            return None
        except Exception as e:
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
            return False 