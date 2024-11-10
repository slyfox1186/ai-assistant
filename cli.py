from src.model import Llama3ChatQAssistant
import sys
import torch

def main():
    try:
        print("Initializing model... (this may take a few minutes)")
        
        # Initialize model
        chat_model = Llama3ChatQAssistant()
        
        print("\nModel ready! Type 'quit' to exit.")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    break
                
                # Let model handle the conversation naturally
                message = {"role": "user", "content": user_input}
                print("\nAssistant: ", end='', flush=True)
                response = chat_model.generate_response([message])
                print(response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except torch.cuda.OutOfMemoryError:
                print("\nError: Out of memory. Please try a shorter input.")
                chat_model.clear_history()
                continue
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main() 