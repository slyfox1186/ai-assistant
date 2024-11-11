from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional

class Llama3ChatQAssistant:
    def __init__(self, model_id: str = "nvidia/Llama3-ChatQA-2-8B", device: str = "auto"):
        """Initialize model following NVIDIA's recommendations."""
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

            # Initialize conversation history
            self.conversation_history = []
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def _format_conversation(self, messages: List[Dict], context: Optional[str] = None) -> str:
        """Format conversation history following NVIDIA's format."""
        try:
            # System prompt
            system = ("This is a chat between a user and an artificial intelligence assistant. "
                     "The assistant gives helpful, detailed, and polite answers to the user's "
                     "questions based on the context.")

            formatted_input = f"System: {system}\n\n"
            
            # Add context if available
            if context and context.strip():
                formatted_input += f"{context}\n\n"

            # Add conversation history
            for msg in self.conversation_history + messages:
                if msg["role"] == "user":
                    formatted_input += f"User: {msg['content']}\n\n"
                else:
                    formatted_input += f"Assistant: {msg['content']}\n\n"

            # Add final prompt
            formatted_input += "Assistant:"
            
            return formatted_input
            
        except Exception as e:
            print(f"Error formatting conversation: {e}")
            return f"System: {system}\n\nAssistant:"

    def generate_response(self, messages: List[Dict], context: Optional[str] = None, max_new_tokens: int = 128) -> str:
        """Generate response using model's natural language capabilities."""
        try:
            # Format conversation for model
            formatted_input = self._format_conversation(messages, context)
            
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
                temperature=0.7,  # Adjust for more natural responses
                top_p=0.9,
                repetition_penalty=1.1  # Prevent repetitive responses
            )
            
            # Extract and clean response
            response_tokens = outputs[0][tokenized_input.input_ids.shape[-1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if response.startswith("Assistant:"):
                response = response[len("Assistant:"):].strip()
            
            # Update conversation history
            self.conversation_history.extend(messages)
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep reasonable history size
            if len(self.conversation_history) > 20:  # Keep last 10 exchanges
                self.conversation_history = self.conversation_history[-20:]
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            print("Error: CUDA out of memory")
            self.clear_history()
            return "I apologize, but I ran out of memory. Please try a shorter input or clear the conversation history."
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but something went wrong. Please try again."

    def clear_history(self):
        """Clear conversation history and CUDA cache."""
        self.conversation_history = []
        torch.cuda.empty_cache()