from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import sys
from typing import List, Dict, Optional
from .memory import BrainMemory
from .web_scraper import WebScraper

class Llama3ChatQAssistant:
    def __init__(self, model_id: str = "nvidia/Llama3-ChatQA-2-8B", device: str = "auto"):
        """Initialize model with memory system."""
        try:
            # Initialize support systems first
            self.memory = BrainMemory("data/memory/brain_memory.json")
            self.web_scraper = WebScraper()
            
            # Load system prompt
            self.system_prompt_file = "data/json/system_prompt.json"
            self.system_prompt = self._load_system_prompt()
            
            # Initialize tokenizer and model with 128K context
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Set model config for 128K context
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                low_cpu_mem_usage=True,
                max_position_embeddings=128000,  # Enable 128K context
                rope_scaling={"type": "dynamic", "factor": 2.0}  # Enable RoPE scaling
            )
            
        except Exception as e:
            print(f"Error initializing model: {e}", file=sys.stderr)
            raise

    def _load_system_prompt(self) -> str:
        """Load system prompt from JSON."""
        try:
            if os.path.exists(self.system_prompt_file):
                with open(self.system_prompt_file, 'r') as f:
                    data = json.load(f)
                    return data.get('prompt', self._get_default_prompt())
            return self._get_default_prompt()
        except Exception as e:
            print(f"Error loading system prompt: {e}", file=sys.stderr)
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default system prompt."""
        return ("This is a chat between a user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's "
                "questions based on the context.")

    def generate_response(self, messages: List[Dict], max_new_tokens: int = 512) -> str:
        """Generate response using NVIDIA's exact format."""
        try:
            # Store message in memory
            if not messages or messages[-1]["role"] != "user":
                return ""
                
            message = messages[-1]["content"]
            self.memory.add("user", message)
            
            # Check for identity updates
            message_lower = message.lower()
            if "my name is" in message_lower:
                name = message_lower.split("my name is")[-1].strip().split()[0].capitalize()
                self.memory.update_identity("user", name)
            
            # Build prompt following NVIDIA's format
            formatted_input = f"System: {self.system_prompt}\n\n"
            
            # Load current identities
            with open("data/json/identities.json", 'r') as f:
                identities = json.load(f)
                
            # Load memory context
            with open("data/memory/brain_memory.json", 'r') as f:
                memory_data = json.load(f)
                
            # Add identity context
            if identities["identities"]["assistant"]["name"]:
                formatted_input += f"Assistant's name: {identities['identities']['assistant']['name']}\n"
            if identities["identities"]["user"]["name"]:
                formatted_input += f"User's name: {identities['identities']['user']['name']}\n"
            formatted_input += "\n"
                
            # Add conversation history
            for msg in messages[-3:]:  # Keep last 3 turns
                if msg["role"] == "user":
                    formatted_input += f"User: {msg['content']}\n\n"
                else:
                    formatted_input += f"Assistant: {msg['content']}\n\n"
            formatted_input += "Assistant:"
            
            # Generate with NVIDIA's parameters
            inputs = self.tokenizer(
                self.tokenizer.bos_token + formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128000
            ).to(self.model.device)
            
            print("\nAssistant: ", end='', flush=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    use_cache=True
                )
                
                # Stream tokens
                response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
                for token in response_tokens:
                    token_text = self.tokenizer.decode([token], skip_special_tokens=True)
                    if token_text:
                        print(token_text, end='', flush=True)
            
            # Get complete response
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Store response
            if response:
                self.memory.add("assistant", response)
                print('', flush=True)
            
            return response
            
        except Exception as e:
            error_response = f"I apologize, but something went wrong: {str(e)}"
            print(f"\n{error_response}", flush=True)
            return error_response

    def clear_history(self):
        """Clear memory and CUDA cache thoroughly."""
        self.memory.clear()
        # Force complete CUDA cache clear
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()