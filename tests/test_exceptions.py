"""
Unit tests for elikopy exception hierarchy and error handling
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from elikopy.infrastructure.exceptions import (
    ElikopyError, DataValidationError, ProcessingError, BIDSError,
    ConfigurationError, FileOperationError, HPCError, DependencyError,
    MemoryError, ErrorContext, ErrorHandler, create_error_context,
    handle_exceptions
)


class TestErrorContext:
    """Test ErrorContext dataclass"""
    
    def test_error_context_creation(self):
        """Test error context creation"""
        context = ErrorContext(
            operation="test_operation",
            subject_id="sub-01",
            session_id="ses-01",
            processing_step="preprocessing"
        )
        
        assert context.operation == "test_operation"
        assert context.subject_id == "sub-01"
        assert context.session_id == "ses-01"
        assert context.processing_step == "preprocessing"
        assert context.file_path is None
        assert context.parameters is None
        assert context.metadata == {}
    
    def test_create_error_context_function(self):
        """Test create_error_context convenience function"""
        context = create_error_context(
            "test_op",
            subject_id="sub-01",
            file_path=Path("/test/path")
        )
        
        assert context.operation == "test_op"
        assert context.subject_id == "sub-01"
        assert context.file_path == Path("/test/path")


class TestElikopyError:
    """Test base ElikopyError class"""
    
    def test_basic_error_creation(self):
        """Test basic error creation"""
        error = ElikopyError("Test error message")
        
        assert str(error).startswith("ElikopyError: Test error message")
        assert error.message == "Test error message"
        assert error.context.operation == "unknown"
        assert error.cause is None
        assert error.suggestions == []
    
    def test_error_with_context(self):
        """Test error with context"""
        context = ErrorContext(
            operation="test_operation",
            subject_id="sub-01",
            file_path=Path("/test/file.nii")
        )
        error = ElikopyError("Test error", context=context)
        
        error_str = str(error)
        assert "Operation: test_operation" in error_str
        assert "Subject: sub-01" in error_str
        assert "File:" in error_str and "file.nii" in error_str
    
    def test_error_with_cause(self):
        """Test error with underlying cause"""
        original_error = ValueError("Original error")
        error = ElikopyError("Wrapper error", cause=original_error)
        
        error_str = str(error)
        assert "Caused by: ValueError: Original error" in error_str
        assert error.cause == original_error
    
    def test_error_with_suggestions(self):
        """Test error with suggestions"""
        suggestions = ["Try this", "Or try that"]
        error = ElikopyError("Test error", suggestions=suggestions)
        
        error_str = str(error)
        assert "Suggestions:" in error_str
        assert "- Try this" in error_str
        assert "- Or try that" in error_str
    
    def test_get_detailed_info(self):
        """Test detailed error information"""
        context = ErrorContext(
            operation="test_op",
            subject_id="sub-01",
            parameters={"param1": "value1"}
        )
        original_error = ValueError("Original")
        error = ElikopyError(
            "Test error",
            context=context,
            cause=original_error,
            suggestions=["Fix it"]
        )
        
        info = error.get_detailed_info()
        
        assert info["error_type"] == "ElikopyError"
        assert info["message"] == "Test error"
        assert info["context"]["operation"] == "test_op"
        assert info["context"]["subject_id"] == "sub-01"
        assert info["context"]["parameters"] == {"param1": "value1"}
        assert info["suggestions"] == ["Fix it"]
        assert info["cause"]["type"] == "ValueError"
        assert info["cause"]["message"] == "Original"


class TestSpecificErrors:
    """Test specific error types"""
    
    def test_data_validation_error(self):
        """Test DataValidationError"""
        validation_errors = ["Missing bvals file", "Invalid dimensions"]
        error = DataValidationError(
            "Validation failed",
            validation_errors=validation_errors
        )
        
        assert isinstance(error, ElikopyError)
        assert error.validation_errors == validation_errors
        assert any("Check input data format" in s for s in error.suggestions)
        assert "Fix validation error: Missing bvals file" in error.suggestions
    
    def test_processing_error(self):
        """Test ProcessingError"""
        error = ProcessingError(
            "Processing failed",
            processing_step="tensor_fitting"
        )
        
        assert isinstance(error, ElikopyError)
        assert error.context.processing_step == "tensor_fitting"
        assert "Check processing parameters" in error.suggestions
        assert "Review tensor_fitting configuration" in error.suggestions
    
    def test_bids_error(self):
        """Test BIDSError"""
        bids_path = Path("/data/bids_dataset")
        error = BIDSError(
            "BIDS validation failed",
            bids_path=bids_path
        )
        
        assert isinstance(error, ElikopyError)
        assert error.context.file_path == bids_path
        assert any("Validate BIDS dataset structure" in s for s in error.suggestions)
        assert "Check file naming conventions" in error.suggestions
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError(
            "Invalid configuration",
            config_parameter="max_iterations"
        )
        
        assert isinstance(error, ElikopyError)
        assert "Check configuration file syntax" in error.suggestions
        assert "Check parameter: max_iterations" in error.suggestions
    
    def test_file_operation_error(self):
        """Test FileOperationError"""
        file_path = Path("/test/file.nii")
        error = FileOperationError(
            "File operation failed",
            file_path=file_path,
            operation="read"
        )
        
        assert isinstance(error, ElikopyError)
        assert error.context.file_path == file_path
        assert error.context.operation == "read"
        assert "Check file permissions" in error.suggestions
        assert f"Verify file: {file_path}" in error.suggestions
    
    def test_hpc_error(self):
        """Test HPCError"""
        job_id = "12345"
        error = HPCError(
            "Job failed",
            job_id=job_id
        )
        
        assert isinstance(error, ElikopyError)
        assert "Check SLURM queue status" in error.suggestions
        assert f"Check job status: squeue -j {job_id}" in error.suggestions
    
    def test_dependency_error(self):
        """Test DependencyError"""
        error = DependencyError(
            "Package not found",
            dependency="dipy",
            required_version="1.5.0",
            found_version="1.4.0"
        )
        
        assert isinstance(error, ElikopyError)
        assert "Install/update package: dipy" in error.suggestions
        assert "Required version: 1.5.0" in error.suggestions
        assert "Found version: 1.4.0" in error.suggestions
    
    def test_memory_error(self):
        """Test MemoryError"""
        error = MemoryError(
            "Insufficient memory",
            memory_required="16GB",
            memory_available="8GB"
        )
        
        assert isinstance(error, ElikopyError)
        assert "Reduce processing batch size" in error.suggestions
        assert "Required memory: 16GB" in error.suggestions
        assert "Available memory: 8GB" in error.suggestions


class TestErrorHandler:
    """Test ErrorHandler class"""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization"""
        mock_logger = MagicMock()
        handler = ErrorHandler(logger=mock_logger)
        
        assert handler.logger == mock_logger
        assert handler.error_history == []
    
    def test_handle_elikopy_error(self):
        """Test handling ElikopyError"""
        mock_logger = MagicMock()
        handler = ErrorHandler(logger=mock_logger)
        
        error = ElikopyError("Test error")
        
        with pytest.raises(ElikopyError):
            handler.handle_error(error)
        
        assert len(handler.error_history) == 1
        assert handler.error_history[0] == error
        mock_logger.log_error_with_traceback.assert_called_once()
    
    def test_handle_regular_exception(self):
        """Test handling regular exception"""
        mock_logger = MagicMock()
        handler = ErrorHandler(logger=mock_logger)
        
        context = ErrorContext(operation="test_op")
        original_error = ValueError("Original error")
        
        with pytest.raises(ElikopyError) as exc_info:
            handler.handle_error(original_error, context=context)
        
        elikopy_error = exc_info.value
        assert isinstance(elikopy_error, ElikopyError)
        assert elikopy_error.cause == original_error
        assert elikopy_error.context == context
        assert len(handler.error_history) == 1
    
    def test_handle_error_no_reraise(self):
        """Test handling error without reraising"""
        handler = ErrorHandler()
        
        error = ValueError("Test error")
        result = handler.handle_error(error, reraise=False)
        
        assert isinstance(result, ElikopyError)
        assert result.cause == error
        assert len(handler.error_history) == 1
    
    def test_attempt_recovery_success(self):
        """Test successful error recovery"""
        mock_logger = MagicMock()
        handler = ErrorHandler(logger=mock_logger)
        
        error = ElikopyError("Test error")
        
        def successful_recovery(err):
            return True
        
        def failed_recovery(err):
            return False
        
        strategies = [failed_recovery, successful_recovery]
        result = handler.attempt_recovery(error, strategies)
        
        assert result is True
        mock_logger.logger.info.assert_called()
    
    def test_attempt_recovery_failure(self):
        """Test failed error recovery"""
        mock_logger = MagicMock()
        handler = ErrorHandler(logger=mock_logger)
        
        error = ElikopyError("Test error")
        
        def failed_recovery(err):
            return False
        
        strategies = [failed_recovery]
        result = handler.attempt_recovery(error, strategies)
        
        assert result is False
    
    def test_attempt_recovery_exception(self):
        """Test recovery strategy that raises exception"""
        mock_logger = MagicMock()
        handler = ErrorHandler(logger=mock_logger)
        
        error = ElikopyError("Test error")
        
        def failing_recovery(err):
            raise RuntimeError("Recovery failed")
        
        strategies = [failing_recovery]
        result = handler.attempt_recovery(error, strategies)
        
        assert result is False
        mock_logger.log_error_with_traceback.assert_called()
    
    def test_get_error_summary_empty(self):
        """Test error summary with no errors"""
        handler = ErrorHandler()
        
        summary = handler.get_error_summary()
        
        assert summary["total_errors"] == 0
        assert summary["error_types"] == {}
    
    def test_get_error_summary_with_errors(self):
        """Test error summary with errors"""
        handler = ErrorHandler()
        
        # Add some errors
        error1 = ElikopyError("Error 1")
        error2 = DataValidationError("Error 2")
        error3 = ElikopyError("Error 3")
        
        handler.error_history = [error1, error2, error3]
        
        summary = handler.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert summary["error_types"]["ElikopyError"] == 2
        assert summary["error_types"]["DataValidationError"] == 1
        assert len(summary["recent_errors"]) == 3
    
    def test_clear_error_history(self):
        """Test clearing error history"""
        handler = ErrorHandler()
        handler.error_history = [ElikopyError("Test")]
        
        handler.clear_error_history()
        
        assert handler.error_history == []


class TestHandleExceptionsDecorator:
    """Test handle_exceptions decorator"""
    
    def test_decorator_success(self):
        """Test decorator with successful function"""
        mock_logger = MagicMock()
        
        @handle_exceptions("test_operation", logger=mock_logger)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_decorator_exception_reraise(self):
        """Test decorator with exception (reraise=True)"""
        mock_logger = MagicMock()
        
        @handle_exceptions("test_operation", logger=mock_logger, reraise=True)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ElikopyError):
            failing_function()
        
        mock_logger.log_error_with_traceback.assert_called()
    
    def test_decorator_exception_no_reraise(self):
        """Test decorator with exception (reraise=False)"""
        mock_logger = MagicMock()
        
        @handle_exceptions("test_operation", logger=mock_logger, reraise=False)
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        
        assert isinstance(result, ElikopyError)
        assert result.cause.__class__ == ValueError
        mock_logger.log_error_with_traceback.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])