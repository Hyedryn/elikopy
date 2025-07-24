"""
Infrastructure module for ElikoPy
================================

This module handles infrastructure concerns:
- JobScheduler: HPC/SLURM job management
- FileManager: File operations and management
- ElikopyLogger: Logging configuration
- Exception hierarchy: Comprehensive error handling
"""

from elikopy.infrastructure.scheduler import JobScheduler
from elikopy.infrastructure.file_manager import FileManager
from elikopy.infrastructure.logging import ElikopyLogger, LoggingConfig, get_logger, configure_logging
from elikopy.infrastructure.exceptions import (
    ElikopyError, DataValidationError, ProcessingError, BIDSError,
    ConfigurationError, FileOperationError, HPCError, DependencyError,
    MemoryError, ErrorContext, ErrorHandler, create_error_context,
    handle_exceptions
)

__all__ = [
    # Job scheduling
    'JobScheduler',
    
    # File management
    'FileManager',
    
    # Logging
    'ElikopyLogger',
    'LoggingConfig',
    'get_logger',
    'configure_logging',
    
    # Exception handling
    'ElikopyError',
    'DataValidationError',
    'ProcessingError',
    'BIDSError',
    'ConfigurationError',
    'FileOperationError',
    'HPCError',
    'DependencyError',
    'MemoryError',
    'ErrorContext',
    'ErrorHandler',
    'create_error_context',
    'handle_exceptions'
]