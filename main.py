from src.model import Llama3ChatQAssistant
import sys
import torch
import os

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    try:
        print("Initializing model... (this may take a few minutes)")
        
        # Initialize model with proper error handling
        try:
            chatqa = Llama3ChatQAssistant()
        except Exception as e:
            print(f"Error initializing model: {e}")
            return
        
        print("\nModel ready! Type 'quit' to exit, 'clear' to clear history.")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    chatqa.clear_history()
                    clear_screen()
                    print("Model ready! Type 'quit' to exit, 'clear' to clear history.")
                    continue
                
                # Let model handle the conversation naturally
                message = {"role": "user", "content": user_input}
                chatqa.generate_response([message])  # Response is printed during streaming
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except torch.cuda.OutOfMemoryError:
                print("\nError: Out of memory. Please try a shorter input.")
                chatqa.clear_history()
                continue
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main() 