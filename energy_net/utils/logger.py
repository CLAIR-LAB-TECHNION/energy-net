# utils/logger.py

import logging
import os
from typing import Optional

def setup_logger(name: str, log_file: str, level=logging.DEBUG) -> logging.Logger:
    """
    Sets up a logger with the specified name and log file.
    
    Ensures that each logger has only one handler to prevent duplicate logs
    and unclosed file handles.
    
    Args:
        name (str): The name of the logger.
        log_file (str): The path to the log file.
        level (int): Logging level (default: logging.DEBUG).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, do not add another one
    if not logger.handlers:
        logger.setLevel(level)
        
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create a file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        
        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(fh)
        
        # Optionally, prevent log messages from being propagated to the root logger
        logger.propagate = False
    
    return logger
