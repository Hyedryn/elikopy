"""
ElikopyLogger class - Logging configuration
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


class ElikopyLogger:
    """Class for logging configuration"""
    
    def __init__(self, 
                 name: str = "elikopy", 
                 log_dir: Optional[Union[str, Path]] = None,
                 log_level: str = "INFO",
                 log_to_console: bool = True,
                 log_to_file: bool = True):
        """Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files (optional)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_level = self._get_log_level(log_level)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add handlers
        if log_to_console:
            self._add_console_handler()
        
        if log_to_file and log_dir:
            self._add_file_handler()
    
    def _get_log_level(self, level: str) -> int:
        """Get logging level
        
        Args:
            level: Logging level string
            
        Returns:
            Logging level constant
        """
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        return levels.get(level.upper(), logging.INFO)
    
    def _add_console_handler(self) -> None:
        """Add console handler to logger"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self) -> None:
        """Add file handler to logger"""
        if not self.log_dir:
            return
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get logger instance
        
        Returns:
            Logger instance
        """
        return self.logger
    
    def set_level(self, level: str) -> None:
        """Set logging level
        
        Args:
            level: Logging level string
        """
        log_level = self._get_log_level(level)
        self.logger.setLevel(log_level)
        
        for handler in self.logger.handlers:
            handler.setLevel(log_level)
    
    def add_file_handler(self, file_path: Union[str, Path]) -> None:
        """Add additional file handler
        
        Args:
            file_path: Path to log file
        """
        # Create file handler
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def log_system_info(self) -> None:
        """Log system information"""
        import platform
        import sys
        
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Platform: {platform.platform()}")
        
        # Log environment variables
        self.logger.debug("Environment variables:")
        for key, value in os.environ.items():
            if key.startswith("ELIKOPY_"):
                self.logger.debug(f"  {key}={value}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Configuration:")
        
        def log_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    self.logger.info(f"{prefix}{key}:")
                    log_dict(value, prefix + "  ")
                else:
                    self.logger.info(f"{prefix}{key}: {value}")
        
        log_dict(config)
    
    def log_progress(self, message: str, current: int, total: int) -> None:
        """Log progress
        
        Args:
            message: Progress message
            current: Current progress
            total: Total progress
        """
        percentage = int(100 * current / total)
        self.logger.info(f"{message}: {current}/{total} ({percentage}%)")
    
    def log_performance(self, operation: str, duration: float) -> None:
        """Log performance
        
        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        self.logger.debug(f"Performance: {operation} took {duration:.2f} seconds")
    
    def log_error_with_traceback(self, message: str) -> None:
        """Log error with traceback
        
        Args:
            message: Error message
        """
        import traceback
        self.logger.error(f"{message}\n{traceback.format_exc()}")
    
    def create_child_logger(self, name: str) -> logging.Logger:
        """Create child logger
        
        Args:
            name: Child logger name
            
        Returns:
            Child logger instance
        """
        child_name = f"{self.name}.{name}"
        child_logger = logging.getLogger(child_name)
        child_logger.parent = self.logger
        child_logger.propagate = True
        
        return child_logger