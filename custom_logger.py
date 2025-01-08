#!/usr/bin/env python3

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from functools import wraps
import time
import psutil
import traceback

class CustomLogger:
    _instance = None
    _logger = None
    _initialized = False
    
    @classmethod
    def get_logger(cls):
        if not cls._initialized:
            cls._initialize_logger()
        return cls._logger
    
    @classmethod
    def _initialize_logger(cls):
        if cls._initialized:
            return
            
        # Create logger with a unique name
        cls._logger = logging.getLogger('ai_assistant_custom')
        
        # Force debug level and propagate=False to prevent interference
        cls._logger.setLevel(logging.DEBUG)
        cls._logger.propagate = False
        
        # Prevent adding handlers multiple times
        if not cls._logger.handlers:
            # Console handler with detailed formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            cls._logger.addHandler(console_handler)
            
            # File handler with even more detailed formatting
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, 'ai_assistant.log'),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            cls._logger.addHandler(file_handler)
        
        cls._initialized = True

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 / 1024, 1)

def get_system_metrics():
    """Get detailed system metrics"""
    process = psutil.Process(os.getpid())
    return {
        'memory_usage': f"{get_memory_usage()}MB",
        'cpu_percent': f"{process.cpu_percent(interval=0.1)}%",
        'thread_count': len(process.threads()),
        'open_files': len(process.open_files()),
        'connections': len(process.connections()),
        'context_switches': process.num_ctx_switches()
    }

def format_metrics(metrics):
    """Format system metrics for logging"""
    return (
        f"[Metrics: Mem={metrics['memory_usage']}, "
        f"CPU={metrics['cpu_percent']}, "
        f"Threads={metrics['thread_count']}, "
        f"Files={metrics['open_files']}, "
        f"Conns={metrics['connections']}]"
    )

def verbose(message, include_trace=False):
    """Verbose debug logging with detailed system metrics"""
    metrics = get_system_metrics()
    formatted_msg = f"{message} {format_metrics(metrics)}"
    if include_trace:
        formatted_msg += f"\nStack trace:\n{traceback.format_stack()}"
    CustomLogger.get_logger().debug(formatted_msg)

def info(message):
    """Enhanced info logging with system metrics"""
    metrics = get_system_metrics()
    CustomLogger.get_logger().info(f"{message} {format_metrics(metrics)}")

def error(message, exc_info=True):
    """Enhanced error logging with system metrics and stack trace"""
    metrics = get_system_metrics()
    CustomLogger.get_logger().error(
        f"{message} {format_metrics(metrics)}",
        exc_info=exc_info
    )

def debug(message):
    """Enhanced debug logging with system metrics"""
    metrics = get_system_metrics()
    CustomLogger.get_logger().debug(f"{message} {format_metrics(metrics)}")

def warning(message):
    """Enhanced warning logging with system metrics"""
    metrics = get_system_metrics()
    CustomLogger.get_logger().warning(f"{message} {format_metrics(metrics)}")

def critical(message, exc_info=True):
    """Enhanced critical logging with system metrics and stack trace"""
    metrics = get_system_metrics()
    CustomLogger.get_logger().critical(
        f"{message} {format_metrics(metrics)}",
        exc_info=exc_info
    )

def log_execution_time(func):
    """Enhanced decorator to log execution time and detailed metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_metrics = get_system_metrics()
        try:
            CustomLogger.get_logger().debug(
                f"⚡ Starting {func.__name__} {format_metrics(start_metrics)}"
            )
            result = func(*args, **kwargs)
            end_metrics = get_system_metrics()
            elapsed_time = time.time() - start_time
            CustomLogger.get_logger().info(
                f"✓ {func.__name__} completed in {elapsed_time:.3f}s {format_metrics(end_metrics)}"
            )
            return result
        except Exception as e:
            end_metrics = get_system_metrics()
            elapsed_time = time.time() - start_time
            CustomLogger.get_logger().error(
                f"❌ {func.__name__} failed after {elapsed_time:.3f}s: {str(e)} {format_metrics(end_metrics)}",
                exc_info=True
            )
            raise
    return wrapper 