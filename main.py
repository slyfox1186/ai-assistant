from src.model import Llama3ChatQAssistant
from train_calculator import TrainCalculator
import sys
import torch

def main():
    try:
        print("Initializing model... (this may take a few minutes)")
        
        # Initialize model with proper error handling
        try:
            chatqa = Llama3ChatQAssistant()
        except Exception as e:
            print(f"Error initializing model: {e}")
            return
            
        train_calc = TrainCalculator()
        
        print("\nModel ready! Type 'quit' to exit, 'clear' to clear history.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    chatqa.clear_history()
                    print("\nConversation history cleared!")
                    continue
                
                # Let the model handle all input naturally
                message = {"role": "user", "content": user_input}
                print("\nAssistant: ", end='', flush=True)
                
                # Generate response
                response = chatqa.generate_response([message])
                print(response)
                
                # If the response indicates a train calculation is needed,
                # handle it as additional information
                if train_calc.parse_question(user_input):
                    calculation = train_calc.get_formatted_result()
                    follow_up = {"role": "user", "content": f"Here are the calculated details: {calculation}"}
                    additional_response = chatqa.generate_response([follow_up])
                    print("\nAssistant:", additional_response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except torch.cuda.OutOfMemoryError:
                print("\nError: Out of memory. Please try a shorter input or clear the conversation history.")
                continue
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue

    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main() 