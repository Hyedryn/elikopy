"""
Exception hierarchy and error handling for elikopy.
"""

import traceback
import sys
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ErrorContext:
    """Context information for errors"""
    operation: str
    file_path: Optional[Path] = None
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    processing_step: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class ElikopyError(Exception):
    """Base exception class for all elikopy errors"""
    
    def __init__(self, 
                 message: str, 
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None,
                 suggestions: Optional[List[str]] = None):
        """Initialize elikopy error
        
        Args:
            message: Error message
            context: Error context information
            cause: Original exception that caused this error
            suggestions: List of suggested solutions
        """
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext(operation="unknown")
        self.cause = cause
        self.suggestions = suggestions or []
        
        # Store traceback information
        self.traceback_info = traceback.format_exc() if cause else None
    
    def __str__(self) -> str:
        """String representation of the error"""
        parts = [f"ElikopyError: {self.message}"]
        
        if self.context:
            parts.append(f"Operation: {self.context.operation}")
            if self.context.subject_id:
                parts.append(f"Subject: {self.context.subject_id}")
            if self.context.session_id:
                parts.append(f"Session: {self.context.session_id}")
            if self.context.file_path:
                parts.append(f"File: {self.context.file_path}")
            if self.context.processing_step:
                parts.append(f"Processing step: {self.context.processing_step}")
        
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")
        
        if self.suggestions:
            parts.append("Suggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  - {suggestion}")
        
        return "\n".join(parts)
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """Get detailed error information as dictionary"""
        info = {
            "error_type": type(self).__name__,
            "message": self.message,
            "context": {
                "operation": self.context.operation,
                "subject_id": self.context.subject_id,
                "session_id": self.context.session_id,
                "file_path": str(self.context.file_path) if self.context.file_path else None,
                "processing_step": self.context.processing_step,
                "parameters": self.context.parameters,
                "metadata": self.context.metadata
            },
            "suggestions": self.suggestions
        }
        
        if self.cause:
            info["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
                "traceback": self.traceback_info
            }
        
        return info


class DataValidationError(ElikopyError):
    """Raised when data validation fails"""
    
    def __init__(self, 
                 message: str,
                 validation_errors: Optional[List[str]] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize data validation error
        
        Args:
            message: Error message
            validation_errors: List of specific validation errors
            context: Error context
            cause: Original exception
        """
        self.validation_errors = validation_errors or []
        
        # Add validation-specific suggestions
        suggestions = [
            "Check input data format and integrity",
            "Verify BIDS compliance if applicable",
            "Ensure all required files are present"
        ]
        
        if validation_errors:
            suggestions.extend([f"Fix validation error: {error}" for error in validation_errors])
        
        super().__init__(message, context, cause, suggestions)


class ProcessingError(ElikopyError):
    """Raised during processing failures"""
    
    def __init__(self, 
                 message: str,
                 processing_step: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize processing error
        
        Args:
            message: Error message
            processing_step: Name of the processing step that failed
            context: Error context
            cause: Original exception
        """
        if context and processing_step:
            context.processing_step = processing_step
        elif processing_step:
            context = ErrorContext(operation="processing", processing_step=processing_step)
        
        suggestions = [
            "Check processing parameters",
            "Verify input data quality",
            "Review log files for detailed error information"
        ]
        
        if processing_step:
            suggestions.append(f"Review {processing_step} configuration")
        
        super().__init__(message, context, cause, suggestions)


class BIDSError(ElikopyError):
    """Raised for BIDS-related issues"""
    
    def __init__(self, 
                 message: str,
                 bids_path: Optional[Path] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize BIDS error
        
        Args:
            message: Error message
            bids_path: Path to BIDS dataset or file
            context: Error context
            cause: Original exception
        """
        if context and bids_path:
            context.file_path = bids_path
        elif bids_path:
            context = ErrorContext(operation="bids_validation", file_path=bids_path)
        
        suggestions = [
            "Validate BIDS dataset structure using bids-validator",
            "Check file naming conventions",
            "Ensure required metadata files are present",
            "Verify dataset_description.json is valid"
        ]
        
        super().__init__(message, context, cause, suggestions)


class ConfigurationError(ElikopyError):
    """Raised for configuration-related issues"""
    
    def __init__(self, 
                 message: str,
                 config_parameter: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize configuration error
        
        Args:
            message: Error message
            config_parameter: Name of the problematic configuration parameter
            context: Error context
            cause: Original exception
        """
        suggestions = [
            "Check configuration file syntax",
            "Verify parameter values are within valid ranges",
            "Review configuration documentation"
        ]
        
        if config_parameter:
            suggestions.append(f"Check parameter: {config_parameter}")
        
        super().__init__(message, context, cause, suggestions)


class FileOperationError(ElikopyError):
    """Raised for file operation failures"""
    
    def __init__(self, 
                 message: str,
                 file_path: Optional[Path] = None,
                 operation: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize file operation error
        
        Args:
            message: Error message
            file_path: Path to the problematic file
            operation: Type of file operation (read, write, copy, etc.)
            context: Error context
            cause: Original exception
        """
        if context and file_path:
            context.file_path = file_path
        elif file_path:
            context = ErrorContext(operation=operation or "file_operation", file_path=file_path)
        
        suggestions = [
            "Check file permissions",
            "Verify file path exists",
            "Ensure sufficient disk space",
            "Check for file locks or concurrent access"
        ]
        
        if file_path:
            suggestions.append(f"Verify file: {file_path}")
        
        super().__init__(message, context, cause, suggestions)


class HPCError(ElikopyError):
    """Raised for HPC/SLURM-related issues"""
    
    def __init__(self, 
                 message: str,
                 job_id: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize HPC error
        
        Args:
            message: Error message
            job_id: SLURM job ID if applicable
            context: Error context
            cause: Original exception
        """
        suggestions = [
            "Check SLURM queue status",
            "Verify resource allocation",
            "Review job submission parameters",
            "Check HPC system status"
        ]
        
        if job_id:
            suggestions.append(f"Check job status: squeue -j {job_id}")
            suggestions.append(f"Review job logs: scontrol show job {job_id}")
        
        super().__init__(message, context, cause, suggestions)


class DependencyError(ElikopyError):
    """Raised when required dependencies are missing or incompatible"""
    
    def __init__(self, 
                 message: str,
                 dependency: Optional[str] = None,
                 required_version: Optional[str] = None,
                 found_version: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize dependency error
        
        Args:
            message: Error message
            dependency: Name of the missing/incompatible dependency
            required_version: Required version
            found_version: Found version (if any)
            context: Error context
            cause: Original exception
        """
        suggestions = [
            "Check package installation",
            "Verify package versions",
            "Update packages if necessary"
        ]
        
        if dependency:
            suggestions.append(f"Install/update package: {dependency}")
            if required_version:
                suggestions.append(f"Required version: {required_version}")
            if found_version:
                suggestions.append(f"Found version: {found_version}")
        
        super().__init__(message, context, cause, suggestions)


class MemoryError(ElikopyError):
    """Raised when memory-related issues occur"""
    
    def __init__(self, 
                 message: str,
                 memory_required: Optional[str] = None,
                 memory_available: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """Initialize memory error
        
        Args:
            message: Error message
            memory_required: Required memory amount
            memory_available: Available memory amount
            context: Error context
            cause: Original exception
        """
        suggestions = [
            "Reduce processing batch size",
            "Use memory-efficient processing options",
            "Consider processing on a system with more RAM",
            "Close other applications to free memory"
        ]
        
        if memory_required:
            suggestions.append(f"Required memory: {memory_required}")
        if memory_available:
            suggestions.append(f"Available memory: {memory_available}")
        
        super().__init__(message, context, cause, suggestions)


class ErrorHandler:
    """Centralized error handling and recovery system"""
    
    def __init__(self, logger=None):
        """Initialize error handler
        
        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger
        self.error_history: List[ElikopyError] = []
    
    def handle_error(self, 
                    error: Union[Exception, ElikopyError],
                    context: Optional[ErrorContext] = None,
                    reraise: bool = True) -> Optional[ElikopyError]:
        """Handle an error with proper logging and context
        
        Args:
            error: Exception to handle
            context: Error context information
            reraise: Whether to reraise the error after handling
            
        Returns:
            ElikopyError instance if not reraising
        """
        # Convert to ElikopyError if needed
        if isinstance(error, ElikopyError):
            elikopy_error = error
        else:
            elikopy_error = ElikopyError(
                message=str(error),
                context=context,
                cause=error
            )
        
        # Add to error history
        self.error_history.append(elikopy_error)
        
        # Log the error
        if self.logger:
            self.logger.log_error_with_traceback(
                f"Error in {elikopy_error.context.operation}: {elikopy_error.message}",
                elikopy_error.cause
            )
            
            # Log detailed error information
            error_info = elikopy_error.get_detailed_info()
            self.logger.log_structured("ERROR", "Detailed error information", error_details=error_info)
        
        if reraise:
            raise elikopy_error
        
        return elikopy_error
    
    def attempt_recovery(self, 
                        error: ElikopyError,
                        recovery_strategies: Optional[List[callable]] = None) -> bool:
        """Attempt to recover from an error
        
        Args:
            error: Error to recover from
            recovery_strategies: List of recovery functions to try
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if not recovery_strategies:
            return False
        
        for strategy in recovery_strategies:
            try:
                if self.logger:
                    self.logger.logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                
                result = strategy(error)
                if result:
                    if self.logger:
                        self.logger.logger.info(f"Recovery successful using: {strategy.__name__}")
                    return True
                    
            except Exception as recovery_error:
                if self.logger:
                    self.logger.log_error_with_traceback(
                        f"Recovery strategy {strategy.__name__} failed",
                        recovery_error
                    )
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all handled errors
        
        Returns:
            Dictionary with error statistics and information
        """
        if not self.error_history:
            return {"total_errors": 0, "error_types": {}}
        
        error_types = {}
        for error in self.error_history:
            error_type = type(error).__name__
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": [error.get_detailed_info() for error in self.error_history[-5:]]
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history"""
        self.error_history.clear()


def create_error_context(operation: str, **kwargs) -> ErrorContext:
    """Convenience function to create error context
    
    Args:
        operation: Operation name
        **kwargs: Additional context parameters
        
    Returns:
        ErrorContext instance
    """
    return ErrorContext(operation=operation, **kwargs)


def handle_exceptions(operation: str, 
                     logger=None,
                     reraise: bool = True):
    """Decorator for automatic exception handling
    
    Args:
        operation: Name of the operation being performed
        logger: Logger instance
        reraise: Whether to reraise exceptions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler(logger)
            context = ErrorContext(operation=operation)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return error_handler.handle_error(e, context, reraise)
        
        return wrapper
    return decorator