from flask import Flask, render_template, request, Response, stream_with_context
import json
from model_interface import ModelInterface
from custom_logger import CustomLogger, log_execution_time
from typing import List, Dict
import threading

app = Flask(__name__)
logger = CustomLogger.get_logger()

# Load configuration
with open('static/json/ai_configuration.json', 'r') as f:
    config = json.load(f)

# Initialize model interface with model path from config
model = ModelInterface(config['system']['model_path'])
conversation_history: List[Dict[str, str]] = []
chat_lock = threading.Lock()  # Add lock for thread safety

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@log_execution_time
def chat():
    """Handle chat requests and stream responses"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return Response('data: {"error": "Empty message"}\n\n', 
                          mimetype='text/event-stream')

        # Add user message to history first
        conversation_history.append({"role": "user", "content": user_message})
        current_response = []  # Buffer for building the complete response

        def generate():
            try:
                for chunk in model.generate_streaming(user_message, conversation_history):
                    if "error" in chunk:
                        logger.error(f"Generation error: {chunk['error']}")
                        yield f'data: {json.dumps({"error": chunk["error"]})}\n\n'
                        return

                    if "token" in chunk:
                        # Add token to our buffer
                        current_response.append(chunk["token"])
                        
                        # Stream the token immediately
                        yield f'data: {json.dumps(chunk)}\n\n'
                        
                        # Flush to ensure real-time streaming
                        yield ' ' * 2048 + '\n'

                # After streaming complete, process the full response
                full_response = ''.join(current_response)
                if full_response:
                    # Handle special memories if present
                    if "SPECIAL_MEMORY:" in full_response:
                        memory_parts = full_response.split("SPECIAL_MEMORY:", 1)
                        # Store only the special memory part
                        model.memory.add_memory(memory_parts[1].strip(), memory_type='special')
                    else:
                        # Store as regular memory
                        model.memory.add_memory(full_response)

                    # Add to conversation history
                    conversation_history.append({
                        "role": "assistant",
                        "content": full_response
                    })

            except Exception as e:
                logger.error(f"Error in generate function: {str(e)}", exc_info=True)
                yield f'data: {{"error": "Failed to get response: {str(e)}"}}\n\n'

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return Response(
            f'data: {{"error": "Failed to get response: {str(e)}"}}\n\n',
            mimetype='text/event-stream'
        )

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Clear map-related memories with verification"""
    try:
        # Clear Redis memories with verification
        cleanup_report = model.memory.clear_map_memories()
        logger.debug(f"Cleanup report: {cleanup_report}")
        
        # Clear conversation history with tracking
        global conversation_history
        original_length = len(conversation_history)
        conversation_history = [
            msg for msg in conversation_history 
            if not model.memory._is_map_related(msg['content'])
        ]
        
        return {
            "status": "success",
            "cleanup_report": cleanup_report,
            "conversation_cleanup": f"Removed {original_length - len(conversation_history)} entries"
        }
    except Exception as e:
        logger.error(f"Error clearing memories: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.route('/flush_redis', methods=['POST'])
def flush_redis():
    """Flush entire Redis database"""
    try:
        # Flush all data from Redis
        model.memory.redis.flushall()
        
        # Clear conversation history
        global conversation_history
        conversation_history.clear()
        
        return {
            "status": "success",
            "message": "Redis database flushed successfully"
        }
    except Exception as e:
        logger.error(f"Error flushing Redis: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    # Disable debug mode to prevent double loading of the model
    app.run(host='0.0.0.0', port=5000, debug=False)
