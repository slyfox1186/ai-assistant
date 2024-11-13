# Set ALL environment variables before ANY imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Disable ALL logging before imports
import logging
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').disabled = True
logging.getLogger('tensorboard').disabled = True
logging.getLogger('tensorflow_hub').disabled = True

# Disable CUDA warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore')

# Now import everything else
import argparse
from src.model import Llama3ChatQAssistant
import sys
import torch
from src.custom_logger import CustomLogger

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Llama3ChatQAssistant')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Initialize logger
    logger = CustomLogger.get_logger("Main", args.verbose)
    logger.info("Starting main application", "SYSTEM")

    try:
        logger.info("Initializing model (this may take a few minutes)", "MODEL")
        
        # Initialize model with proper error handling
        try:
            chatqa = Llama3ChatQAssistant(verbose=args.verbose)
            logger.success("Model initialized successfully", "MODEL")
        except Exception as e:
            logger.critical(f"Error initializing model: {e}", "MODEL")
            return
        
        logger.info("Model ready for interaction", "SYSTEM")
        print("\nModel ready! Type 'quit' to exit, 'clear' to clear history.")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    logger.info("User requested quit", "USER")
                    break
                elif user_input.lower() == 'clear':
                    logger.info("Clearing history", "MEMORY")
                    chatqa.clear_history()
                    clear_screen()
                    print("Model ready! Type 'quit' to exit, 'clear' to clear history.")
                    continue
                
                logger.info(f"Processing user input: {user_input[:50]}...", "USER")
                message = {"role": "user", "content": user_input}
                chatqa.generate_response([message])
                
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected", "SYSTEM")
                break
            except torch.cuda.OutOfMemoryError:
                logger.error("Out of memory error", "GPU")
                chatqa.clear_history()
                continue
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}", "SYSTEM")
                continue
                
    except Exception as e:
        logger.critical(f"Fatal error: {e}", "SYSTEM")

if __name__ == "__main__":
    main() 