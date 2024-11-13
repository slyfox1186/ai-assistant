import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Add SUCCESS level to logging
logging.SUCCESS = 25  # Between INFO and WARNING
logging.addLevelName(logging.SUCCESS, 'SUCCESS')

class CustomLogger:
    """Enhanced logging system with rich emoji support."""
    
    # ANSI color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'SUCCESS': '\033[92m',    # Bright Green
        'RESET': '\033[0m'        # Reset
    }
    
    # Enhanced emoji mappings for different components and states
    EMOJIS = {
        # Log Levels
        'DEBUG': '🔍',
        'INFO': '📝',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨',
        'SUCCESS': '✅',  # Changed to checkmark
        
        # System Components
        'SYSTEM': '🤖',
        'MODEL': '🧠',
        'MEMORY': '💾',
        'TIME': '🕒',
        'FILE': '📁',
        'WEB': '🌐',
        'IDENTITY': '👤',
        
        # Actions
        'START': '🚀',
        'END': '🏁',
        'LOAD': '📥',
        'SAVE': '📤',
        'CLEAR': '🧹',
        'UPDATE': '🔄',
        'GENERATE': '⚡',
        'PROCESS': '⚙️',
        
        # States
        'READY': '✨',
        'BUSY': '⌛',
        'DONE': '✅',  # Changed to checkmark
        'FAIL': '💥',
        
        # Data Types
        'CONFIG': '⚙️',
        'PROMPT': '💭',
        'RESPONSE': '💬',
        'TOKEN': '🔤',
        'CACHE': '📦',
        
        # Performance
        'SPEED': '⚡',
        'MEMORY_USE': '📊',
        'CPU': '💻',
        'GPU': '🎮',
        
        # Network
        'API': '🔌',
        'REQUEST': '📡',
        'RESPONSE_NET': '📨',
        
        # User Interaction
        'INPUT': '⌨️',
        'OUTPUT': '🖥️',
        'USER': '👤',
        'ASSISTANT': '🤖',
        'CHAT': '💬',
        
        # Status
        'OK': '✅',  # Changed to checkmark
        'ERROR_STATUS': '👎',
        'WAIT': '⏳',
        'DONE_STATUS': '✅'  # Changed to checkmark
    }

    def __init__(self, name: str, verbose: bool = False, log_file: Optional[str] = None):
        """Initialize enhanced logger with emoji support."""
        self.verbose = verbose
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Add success method to logger
        def success(msg, *args, **kwargs):
            self.logger.log(logging.SUCCESS, msg, *args, **kwargs)
        self.logger.success = success
        
        # Create logs directory if needed
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler with enhanced formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.EnhancedFormatter(self.verbose))
        self.logger.addHandler(console_handler)
        
        # File handler with timestamp
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
            ))
            self.logger.addHandler(file_handler)

    class EnhancedFormatter(logging.Formatter):
        """Enhanced formatter with emoji and color support."""
        
        def __init__(self, verbose: bool = False):
            super().__init__()
            self.verbose = verbose

        def format(self, record):
            # Get base color and emoji
            color = CustomLogger.COLORS.get(record.levelname, CustomLogger.COLORS['RESET'])
            
            # Handle SUCCESS level specifically
            if record.levelno == logging.SUCCESS:
                level_emoji = '✅'  # Always use checkmark for success
            else:
                level_emoji = CustomLogger.EMOJIS.get(record.levelname, '')
            
            # Add timestamp and module in verbose mode
            time_str = f"{datetime.now().strftime('%H:%M:%S')} " if self.verbose else ""
            module_str = f"[{record.module}] " if self.verbose else ""
            
            # Format the message with multiple emojis if components are specified
            msg = record.getMessage()
            if hasattr(record, 'component'):
                component_emoji = CustomLogger.EMOJIS.get(record.component, '')
                if component_emoji and component_emoji != level_emoji:
                    msg = f"{component_emoji}  {msg}"
            
            # Build the final message with bright green for success
            if record.levelno == logging.SUCCESS:
                message = f"{CustomLogger.COLORS['SUCCESS']}{level_emoji} {time_str}{module_str}{msg}{CustomLogger.COLORS['RESET']}"
            else:
                message = f"{color}{level_emoji} {time_str}{module_str}{msg}{CustomLogger.COLORS['RESET']}"
            
            return message

    def _log(self, level: str, msg: str, component: str = None):
        """Enhanced logging with component support."""
        if not self.verbose and level == 'DEBUG':
            return
            
        if level == 'SUCCESS':
            self.logger.success(msg, extra={'component': component})
        else:
            log_method = getattr(self.logger, level.lower())
            record = logging.LogRecord(
                name=self.logger.name,
                level=getattr(logging, level),
                pathname='',
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None
            )
            if component:
                record.component = component
            log_method(msg, extra={'component': component})

    def debug(self, msg: str, component: str = None):
        """Debug level with component support."""
        self._log('DEBUG', msg, component)

    def info(self, msg: str, component: str = None):
        """Info level with component support."""
        self._log('INFO', msg, component)

    def warning(self, msg: str, component: str = None):
        """Warning level with component support."""
        self._log('WARNING', msg, component)

    def error(self, msg: str, component: str = None):
        """Error level with component support."""
        self._log('ERROR', msg, component)

    def critical(self, msg: str, component: str = None):
        """Critical level with component support."""
        self._log('CRITICAL', msg, component)

    def success(self, msg: str, component: str = None):
        """Success level with component support."""
        self._log('SUCCESS', msg, component)

    @staticmethod
    def get_logger(name: str, verbose: bool = False, log_file: Optional[str] = None):
        """Get or create an enhanced logger instance."""
        return CustomLogger(name, verbose, log_file)