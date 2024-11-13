from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Optional
import json
import os
from .custom_logger import CustomLogger

class SystemClock:
    """Provide time context for the model to interpret naturally."""
    
    def __init__(self, timezone_str: str = "America/New_York", verbose: bool = False):
        """Initialize with timezone."""
        self.timezone = ZoneInfo(timezone_str)
        self.logger = CustomLogger.get_logger("SystemClock", verbose)
        
        # Simple time prompt that lets model decide how to respond
        self.time_prompt = (
            "You are an AI assistant. The current time information is provided in the context. "
            "Use your understanding to interpret the user's question and provide an appropriate "
            "response about the time or date."
        )
        self.logger.info("SystemClock initialized", "TIME")

    def get_current_time(self) -> Dict[str, str]:
        """Provide comprehensive time information for model to interpret."""
        current_time = datetime.now(self.timezone)
        self.logger.debug(f"Getting current time in {self.timezone}", "TIME")
        
        time_info = {
            "formatted": current_time.strftime('%B %d, %Y at %I:%M %p %Z'),
            "full": current_time.strftime('%B %d, %Y at %I:%M %p %Z'),
            "date": current_time.strftime('%B %d, %Y'),
            "time": current_time.strftime('%I:%M %p %Z'),
            "day": current_time.strftime('%A'),
            "month": current_time.strftime('%B'),
            "year": current_time.strftime('%Y'),
            "hour": current_time.strftime('%I %p'),
            "minute": current_time.strftime('%M'),
            "timezone": str(self.timezone),
            "timestamp": current_time.timestamp(),
            "iso": current_time.isoformat()
        }
        
        self.logger.debug(f"Current time info: {time_info['formatted']}", "TIME")
        return time_info

    def get_time_prompt(self, query: str = "") -> str:
        """Let model handle time-related queries naturally."""
        self.logger.debug("Getting time prompt", "TIME")
        return self.time_prompt

    def format_time_context(self, query: str = "") -> str:
        """Provide time context for model to interpret."""
        self.logger.debug("Formatting time context", "TIME")
        time_info = self.get_current_time()
        
        context = (
            f"Current Date and Time Information:\n"
            f"Full: {time_info['formatted']}\n"
            f"Date: {time_info['date']}\n"
            f"Time: {time_info['time']}\n"
            f"Day: {time_info['day']}\n"
            f"Timezone: {time_info['timezone']}\n"
        )
        
        self.logger.debug("Time context formatted", "TIME")
        return context

    def is_time_query(self, query: str) -> bool:
        """Basic check for time-related queries - let model handle details."""
        time_related = ["time", "date", "day", "today", "now", "current", "moment"]
        is_time = any(word in query.lower() for word in time_related)
        
        if is_time:
            self.logger.debug(f"Time query detected: {query}", "TIME")
        
        return is_time

    def save_time_log(self, query: str, response: str, log_file: str = "data/logs/time_queries.json"):
        """Save raw time data for model to reference later."""
        try:
            self.logger.info("Saving time query log", "TIME")
            time_info = self.get_current_time()
            log_entry = {
                "timestamp": time_info["timestamp"],
                "query": query,
                "response": response,
                "time_info": time_info
            }
            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Load and append to existing logs
            logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            self.logger.success("Time query log saved", "TIME")
                
        except Exception as e:
            self.logger.error(f"Error saving time log: {e}", "TIME")