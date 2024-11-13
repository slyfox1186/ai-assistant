from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Dict
import os
from pathlib import Path
from .custom_logger import CustomLogger
from .memory import BrainMemory
from .system_clock import SystemClock
from .web_scraper import WebScraper
from .stocks import StockTracker
from .identity import IdentityManager
import json
from transformers.generation.streamers import TextIteratorStreamer
from threading import Thread
from .file_handler import FileHandler

class Llama3ChatQAssistant:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-14B-Instruct", device: str = "auto", verbose: bool = False):
        self.logger = CustomLogger.get_logger("Model", verbose)
        
        # Initialize model
        try:
            # Get HF_HOME or default cache location
            hf_cache = os.environ.get('HF_HOME', os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
            model_cache = os.path.join(hf_cache, "hub", model_id.replace("/", "--"))
            
            # Initialize CUDA with RTX 4090 optimizations
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info(f"CUDA initialized with device: {torch.cuda.get_device_name(0)}", "GPU")
                
            self.logger.info("Loading Qwen tokenizer from cache", "MODEL")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True if os.path.exists(model_cache) else False,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info("Loading Qwen model weights from cache", "MODEL")
            
            # Configure 4-bit quantization optimized for RTX 4090
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_has_fp16_weight=True
            )
            
            # Load the model with RTX 4090 optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="cuda:0",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_cache=True,
                local_files_only=True if os.path.exists(model_cache) else False,
                max_memory={
                    0: "23GB",
                    "cpu": "1GB"
                }
            )
            
            # Enable CUDA graph optimization for static shapes
            self.model.config.use_cache = True
            
            # Move model to GPU and optimize for inference
            self.model.to("cuda:0")
            self.model.eval()
            
            # Load system prompt
            system_prompt_path = "data/json/system_prompt.json"
            if os.path.exists(system_prompt_path):
                with open(system_prompt_path, 'r') as f:
                    self.system_prompt = json.load(f)["prompt"]
            else:
                self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            
            # Initialize components with proper integration
            self.memory = BrainMemory()
            self.clock = SystemClock()
            self.web_scraper = WebScraper()
            self.stock_tracker = StockTracker()
            self.identity = IdentityManager()
            self.file_handler = FileHandler(verbose=verbose)
            
            # Get assistant and user identities
            self.assistant_name = self.identity.get_identity("assistant") or "Charlotte"
            self.user_name = self.identity.get_identity("user") or "User"
            
            self.logger.success("Model and tools initialized", "SYSTEM")
            
        except Exception as e:
            self.logger.critical(f"Failed to initialize model: {e}", "MODEL")
            raise

    def _route_query(self, query: str) -> Dict:
        """Provide context-specific prompts and data for model to analyze."""
        
        # Get memory context for continuity
        memory_context = self.memory.get_context(max_turns=10)
        
        # Base context
        context = {
            "system": f"You are {self.assistant_name}, speaking with {self.user_name}.",
            "context": {
                "memory": memory_context,
                "current_time": self.clock.get_current_time()["formatted"],
                "query": query
            }
        }
        
        try:
            # Stock market queries
            if any(word in query.lower() for word in ["stock", "price", "shares"]):
                stock_data = self.stock_tracker.get_stock_data(query)
                if stock_data:
                    context["system"] += """
You have access to real-time stock market data from Yahoo Finance. When providing stock information:
- Present the current price clearly and concisely
- Include the company's full name with its ticker symbol
- If the data is older than 1 minute, mention when it was last updated
- If the stock doesn't exist or data is unavailable, explain why and suggest checking financial websites"""
                    context["context"]["stock_data"] = stock_data["context"]
                    context["context"]["data_source"] = "Yahoo Finance real-time data"
            
            # Location/address queries
            elif any(word in query.lower() for word in ["where", "address", "location"]):
                web_data = self.web_scraper.search(query)
                if web_data:
                    context["system"] += """
You have access to location and address information from web searches. When providing location information:
- Present the full address in a clear, standard format
- Include relevant details like business names if available
- If multiple locations exist, specify which one you're providing
- If the location can't be found, suggest checking business websites or maps"""
                    context["context"]["location_data"] = web_data["context"]
                    context["context"]["data_source"] = "Web search results"
            
            # File operations
            elif any(word in query.lower() for word in ["file", "read", "write", "save"]):
                file_result = self.file_handler.handle_file_operation(query, "")
                if file_result:
                    context["system"] += """
You have access to file system operations. When handling files:
- Confirm successful operations clearly
- If reading a file, present the content appropriately
- If writing/saving, confirm the action was completed
- If errors occur, explain the issue and suggest solutions"""
                    context["context"]["file_data"] = file_result
                
            # Identity/memory queries
            elif any(word in query.lower() for word in ["name", "remember", "recall", "who"]):
                context["system"] += """
You have access to conversation history and can remember user interactions. When handling identity/memory queries:
- Reference previous interactions accurately
- Maintain consistent use of names and identities
- If information isn't in memory, explain why
- Be confident about remembered information"""
            
            # Default conversation
            else:
                context["system"] += """
You are a helpful AI assistant. Provide clear, accurate, and natural responses while:
- Maintaining conversation context
- Being polite and professional
- Asking for clarification when needed
- Suggesting relevant follow-up topics"""
            
        except Exception as e:
            self.logger.error(f"Error gathering data: {e}", "MODEL")
            context["context"]["error"] = str(e)
        
        return context

    def generate_response(self, messages: List[Dict], max_new_tokens: int = 512) -> str:
        try:
            query = messages[-1]["content"]
            
            # Route query and get relevant data
            routed_data = self._route_query(query)
            
            # Build context
            context_parts = []
            context_parts.append(self.system_prompt)
            context_parts.append(f"You are {self.assistant_name}, speaking with {self.user_name}.")
            
            # Add routed data context if available
            if routed_data:
                if isinstance(routed_data.get("context"), dict):
                    # Handle dictionary context (like from stock tracker)
                    context_parts.append("\n".join(f"{k}: {v}" for k, v in routed_data["context"].items()))
                else:
                    # Handle string context
                    context_parts.append(routed_data.get("context", ""))
            
            # Add time context
            time_info = self.clock.get_current_time()
            context_parts.append(f"\nCurrent Time: {time_info['formatted']}")
            
            # Format messages with context
            messages = [
                {
                    "role": "system", 
                    "content": "\n".join(context_parts)
                },
                *messages
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            with torch.cuda.amp.autocast():
                model_inputs = self.tokenizer(
                    [text], 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_attention_mask=True
                )
                
                input_ids = model_inputs['input_ids'].to(self.model.device)
                attention_mask = model_inputs['attention_mask'].to(self.model.device)
                
                # Stream generation with optimizations
                with torch.inference_mode():
                    try:
                        # Initialize streamer with proper settings
                        streamer = TextIteratorStreamer(
                            tokenizer=self.tokenizer,
                            skip_special_tokens=True,
                            skip_prompt=True,
                            timeout=None,  # No timeout
                            decode_kwargs={"skip_special_tokens": True}
                        )
                        
                        # Start generation
                        generated_text = ""
                        print(f"\n{self.assistant_name}: ", end='', flush=True)
                        
                        # Set up generation parameters according to docs
                        generation_kwargs = dict(
                            inputs=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.95,
                            repetition_penalty=1.1,
                            streamer=streamer
                        )
                        
                        # Start generation in a separate thread
                        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                        thread.start()
                        
                        # Process streamed tokens in main thread
                        for new_text in streamer:
                            if new_text:  # Only process non-empty tokens
                                print(new_text, end='', flush=True)
                                generated_text += new_text
                        
                        # Wait for generation to complete
                        thread.join()
                        print()  # New line after generation
                        
                        # Store in memory with rich metadata
                        memory_data = {
                            "user_name": self.user_name,
                            "assistant_name": self.assistant_name,
                            "timestamp": self.clock.get_current_time()["timestamp"],
                            "formatted_time": self.clock.get_current_time()["formatted"],
                            "query": query,
                            "response": generated_text,
                            "importance_score": 1.0,
                            "emotional_weight": 0.0,
                            "topic_tags": [],
                            "references": []
                        }
                        
                        self.memory.add("user", query)
                        self.memory.add("assistant", generated_text)
                        
                        return generated_text
                        
                    except torch.cuda.OutOfMemoryError as e:
                        torch.cuda.empty_cache()
                        self.logger.error(f"CUDA out of memory during generation: {e}", "GPU")
                        return "I apologize, but I ran out of memory. Please try a shorter input or clear the conversation history."
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}", "MODEL")
            return f"I apologize, but I encountered an error: {str(e)}"

    def clear_history(self):
        """Clear memory and cache."""
        self.memory.memory_stream = []
        torch.cuda.empty_cache()