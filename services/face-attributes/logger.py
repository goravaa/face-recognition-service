#frs/services/face-attributes/logger.py
import logging
import sys
from config import Config 

def setup_logger(name: str, level: int = Config.LOGGING_LEVEL) -> logging.Logger:
    """Setup centralized logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers(): 
        return logger
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation 
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('face-attributes-service.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger("face-attributes-service", Config.LOGGING_LEVEL)