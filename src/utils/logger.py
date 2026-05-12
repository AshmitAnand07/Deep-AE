import logging
import os
import sys

def get_logger(name: str, config: dict = None) -> logging.Logger:
    """
    Creates and returns a configured logger instance.
    
    Args:
        name (str): Name of the logger (typically __name__).
        config (dict, optional): Configuration dictionary. Defaults to None.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Fallback default configuration if config is not provided
    log_level = logging.INFO
    log_format = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
    log_to_console = True
    log_to_file = True
    log_dir = "logs/"
    
    if config and "logging" in config:
        log_level = getattr(logging, config["logging"].get("level", "INFO").upper(), logging.INFO)
        log_format = config["logging"].get("format", log_format)
        log_to_console = config["logging"].get("log_to_console", True)
        log_to_file = config["logging"].get("log_to_file", True)
        
    if config and "paths" in config:
        log_dir = config["paths"].get("log_dir", log_dir)
        
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "deep_ae.log"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger
