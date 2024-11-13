from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Dict, Optional
from .custom_logger import CustomLogger

class Llama3ChatQAssistant:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-14B-Instruct", device: str = "auto", verbose: bool = False):
        """Initialize model following Qwen's recommendations."""
        try:
            self.logger = CustomLogger.get_logger("ChatModel", verbose)
            self.logger.info("Initializing chat model", "MODEL")
            
            # Initialize tokenizer with proper settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                padding_side="left",  # Better for decoder-only models
                truncation_side="left",
                model_max_length=8192  # Conservative context length
            )
            
            if self.tokenizer.pad_token is None:
                self.logger.debug("Setting pad token to eos token", "MODEL")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info("Loading model weights", "MODEL")
            
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="cuda:0",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Initialize conversation history
            self.conversation_history = []
            self.logger.info("Conversation history initialized", "MEMORY")
            
        except Exception as e:
            self.logger.critical(f"Error initializing model: {e}", "MODEL")
            raise

    def generate_response(self, messages: List[Dict], max_new_tokens: int = 512) -> str:
        """Generate response using model's natural language capabilities."""
        try:
            self.logger.info("Generating response", "MODEL")
            
            # Format messages using Qwen's chat template
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                *messages
            ]
            
            # Apply chat template without tokenization first
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with proper settings
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_attention_mask=True,
                add_special_tokens=True
            )
            
            # Move to GPU efficiently
            input_ids = encoding.input_ids.to(self.model.device)
            attention_mask = encoding.attention_mask.to(self.model.device)
            
            # Generate response with optimizations
            with torch.inference_mode(), torch.cuda.amp.autocast():
                try:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    # Extract only the new tokens
                    new_tokens = outputs[:, input_ids.shape[1]:]
                    
                    # Decode with proper settings
                    response = self.tokenizer.decode(
                        new_tokens[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Update conversation history
                    self.conversation_history.extend(messages)
                    self.conversation_history.append({"role": "assistant", "content": response})
                    
                    # Keep reasonable history size
                    if len(self.conversation_history) > 20:
                        self.logger.info("Trimming conversation history", "MEMORY")
                        self.conversation_history = self.conversation_history[-20:]
                    
                    self.logger.success("Response generated successfully", "MODEL")
                    return response
                    
                except torch.cuda.OutOfMemoryError:
                    self.logger.error("CUDA out of memory", "GPU")
                    self.clear_history()
                    return "I apologize, but I ran out of memory. Please try a shorter input or clear the conversation history."
                    
        except Exception as e:
            self.logger.error(f"Error generating response: {e}", "MODEL")
            return "I apologize, but something went wrong. Please try again."

    def clear_history(self):
        """Clear conversation history and CUDA cache."""
        self.logger.info("Clearing history and cache", "MEMORY")
        self.conversation_history = []
        torch.cuda.empty_cache()
        self.logger.success("History cleared", "MEMORY")