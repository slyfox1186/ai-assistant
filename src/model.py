from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional
import time
from .memory import BrainMemory

class Llama3ChatQAssistant:
    def __init__(self, model_id: str = "nvidia/Llama3-ChatQA-2-8B", device: str = "auto"):
        """Initialize the model following NVIDIA's recommendations."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Set pad token to eos token if not set (required for Llama models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,  # Use float16 as recommended by NVIDIA
                device_map=device,
                low_cpu_mem_usage=True     # Optimize memory usage
            )
            
            # Initialize brain-like memory system with persistence
            self.memory = BrainMemory()
            
            # Default context that clearly distinguishes assistant from user
            self.default_context = """You are an AI assistant named Brenda. You engage in natural 
            conversation and maintain context well. You understand that your name is Brenda, and 
            you should never confuse your own identity with the user's identity. If asked about 
            someone's name, you should only confirm if they have explicitly told you their name 
            earlier in the conversation. If you haven't been told someone's name, you should say 
            that you don't know their name yet."""
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def _get_formatted_input(self, messages: List[Dict], context: Optional[str] = None) -> str:
        """Format input following NVIDIA's format but preserving natural conversation."""
        try:
            # System prompt
            system = ("This is a chat between a user and an artificial intelligence assistant. "
                     "The assistant gives helpful, detailed, and polite answers to the user's "
                     "questions based on the context.")

            formatted_input = f"System: {system}\n\n"
            
            # Add context, using default if none provided
            context_to_use = context if context else self.default_context
            if context_to_use.strip():
                formatted_input += f"{context_to_use}\n\n"
            
            # Get conversation history from brain memory
            memory_context = self.memory.get_context()
            
            # Add memory context and current messages
            for msg in memory_context + messages:
                if msg["role"] == "user":
                    formatted_input += f"User: {msg['content']}\n\n"
                else:
                    formatted_input += f"Assistant: {msg['content']}\n\n"
            
            # Add final prompt
            formatted_input += "Assistant:"
            
            return formatted_input
            
        except Exception as e:
            print(f"Error formatting input: {e}")
            return f"System: {system}\n\nAssistant:"

    def generate_response(self, messages: List[Dict], context: Optional[str] = None, max_new_tokens: int = 128) -> str:
        """Generate response letting the model handle the conversation naturally."""
        try:
            # Format input for model
            formatted_input = self._get_formatted_input(messages, context)
            
            # Generate response
            tokenized_input = self.tokenizer(
                self.tokenizer.bos_token + formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            outputs = self.model.generate(
                input_ids=tokenized_input.input_ids,
                attention_mask=tokenized_input.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract and clean response
            response_tokens = outputs[0][tokenized_input.input_ids.shape[-1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if response.startswith("Assistant:"):
                response = response[len("Assistant:"):].strip()
            
            # Store messages in brain memory
            for message in messages:
                self.memory.add(message["role"], message["content"])
            self.memory.add("assistant", response)
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            print("Error: CUDA out of memory")
            self.clear_history()
            return "I apologize, but I ran out of memory. Please try a shorter input or clear the conversation history."
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return "I apologize, but something went wrong. Please try again."

    def clear_history(self):
        """Clear conversation history and memory."""
        self.memory.clear()
        torch.cuda.empty_cache()