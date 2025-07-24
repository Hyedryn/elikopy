"""
Unit tests for elikopy logging infrastructure
"""

import pytest
import tempfile
import json
import time
import threading
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from elikopy.infrastructure.logging import (
    ElikopyLogger, LoggingConfig, PerformanceMetric, ProgressInfo,
    StructuredFormatter, HPCCompatibleFileHandler, get_logger, configure_logging
)


class TestLoggingConfig:
    """Test LoggingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.console_enabled is True
        assert config.file_enabled is True
        assert config.structured_format is False
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.backup_count == 5
        assert config.hpc_compatible is False
        assert config.performance_tracking is True
        assert config.progress_tracking is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = LoggingConfig(
            level="DEBUG",
            console_enabled=False,
            structured_format=True,
            hpc_compatible=True
        )
        assert config.level == "DEBUG"
        assert config.console_enabled is False
        assert config.structured_format is True
        assert config.hpc_compatible is True


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass"""
    
    def test_performance_metric_creation(self):
        """Test performance metric creation"""
        start_time = time.time()
        metric = PerformanceMetric(
            operation="test_operation",
            start_time=start_time,
            metadata={"key": "value"}
        )
        
        assert metric.operation == "test_operation"
        assert metric.start_time == start_time
        assert metric.end_time is None
        assert metric.duration is None
        assert metric.metadata == {"key": "value"}


class TestProgressInfo:
    """Test ProgressInfo dataclass"""
    
    def test_progress_info_creation(self):
        """Test progress info creation"""
        start_time = time.time()
        progress = ProgressInfo(
            task_name="test_task",
            current=50,
            total=100,
            start_time=start_time,
            metadata={"stage": "processing"}
        )
        
        assert progress.task_name == "test_task"
        assert progress.current == 50
        assert progress.total == 100
        assert progress.start_time == start_time
        assert progress.metadata == {"stage": "processing"}


class TestStructuredFormatter:
    """Test StructuredFormatter"""
    
    def test_basic_formatting(self):
        """Test basic log record formatting"""
        formatter = StructuredFormatter()
        
        # Create a mock log record
        record = MagicMock()
        record.created = time.time()
        record.levelname = "INFO"
        record.name = "test_logger"
        record.getMessage.return_value = "Test message"
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 42
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
    
    def test_formatting_with_extra_fields(self):
        """Test formatting with extra fields"""
        formatter = StructuredFormatter()
        
        record = MagicMock()
        record.created = time.time()
        record.levelname = "INFO"
        record.name = "test_logger"
        record.getMessage.return_value = "Test message"
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 42
        record.extra_fields = {"custom_field": "custom_value"}
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["custom_field"] == "custom_value"


class TestElikopyLogger:
    """Test ElikopyLogger class"""
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(level="DEBUG")
            logger = ElikopyLogger(
                name="test_logger",
                log_dir=temp_dir,
                config=config
            )
            
            try:
                assert logger.name == "test_logger"
                assert logger.log_dir == Path(temp_dir)
                assert logger.config == config
                assert logger.logger.name == "test_logger"
            finally:
                # Close all handlers to release file locks
                for handler in logger.logger.handlers[:]:
                    handler.close()
                    logger.logger.removeHandler(handler)
    
    def test_console_only_logging(self):
        """Test console-only logging"""
        config = LoggingConfig(console_enabled=True, file_enabled=False)
        logger = ElikopyLogger(name="test_logger", config=config)
        
        # Should have only console handler
        assert len(logger.logger.handlers) == 1
        assert logger.logger.handlers[0].__class__.__name__ == "StreamHandler"
    
    def test_file_logging(self):
        """Test file logging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(console_enabled=False, file_enabled=True)
            logger = ElikopyLogger(
                name="test_logger",
                log_dir=temp_dir,
                config=config
            )
            
            try:
                # Should have file handler
                assert len(logger.logger.handlers) == 1
                handler = logger.logger.handlers[0]
                assert "FileHandler" in handler.__class__.__name__
            finally:
                # Close all handlers to release file locks
                for handler in logger.logger.handlers[:]:
                    handler.close()
                    logger.logger.removeHandler(handler)
    
    def test_hpc_compatible_logging(self):
        """Test HPC-compatible logging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(
                console_enabled=False,
                file_enabled=True,
                hpc_compatible=True
            )
            logger = ElikopyLogger(
                name="test_logger",
                log_dir=temp_dir,
                config=config
            )
            
            try:
                # Should have HPC-compatible file handler
                assert len(logger.logger.handlers) == 1
                handler = logger.logger.handlers[0]
                assert isinstance(handler, HPCCompatibleFileHandler)
            finally:
                # Close all handlers to release file locks
                for handler in logger.logger.handlers[:]:
                    handler.close()
                    logger.logger.removeHandler(handler)
    
    def test_structured_logging(self):
        """Test structured logging"""
        config = LoggingConfig(structured_format=True, file_enabled=False)
        logger = ElikopyLogger(name="test_logger", config=config)
        
        # Should use structured formatter
        handler = logger.logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)
    
    def test_performance_timer(self):
        """Test performance timer context manager"""
        config = LoggingConfig(performance_tracking=True)
        logger = ElikopyLogger(name="test_logger", config=config)
        
        with logger.performance_timer("test_operation") as metric:
            time.sleep(0.01)  # Small delay
            assert metric.operation == "test_operation"
            assert metric.start_time > 0
        
        # Check that metric was recorded
        metrics = logger.get_performance_metrics()
        assert len(metrics) == 1
        assert metrics[0].operation == "test_operation"
        assert metrics[0].duration > 0
    
    def test_progress_tracking(self):
        """Test progress tracking"""
        config = LoggingConfig(progress_tracking=True)
        logger = ElikopyLogger(name="test_logger", config=config)
        
        # Log progress
        logger.log_progress("test_task", 25, 100)
        logger.log_progress("test_task", 50, 100)
        logger.log_progress("test_task", 100, 100)
        
        # Check progress info
        progress = logger.get_progress_info("test_task")
        assert progress is not None
        assert progress.task_name == "test_task"
        assert progress.current == 100
        assert progress.total == 100
    
    def test_structured_logging_method(self):
        """Test structured logging method"""
        config = LoggingConfig(file_enabled=False)
        logger = ElikopyLogger(name="test_logger", config=config)
        
        # Mock the logger to capture the record
        with patch.object(logger.logger, 'handle') as mock_handle:
            logger.log_structured("INFO", "Test message", custom_field="value")
            
            # Verify that handle was called with structured data
            mock_handle.assert_called_once()
            record = mock_handle.call_args[0][0]
            assert hasattr(record, 'extra_fields')
            assert record.extra_fields == {"custom_field": "value"}
    
    def test_error_logging_with_traceback(self):
        """Test error logging with traceback"""
        logger = ElikopyLogger(name="test_logger", config=LoggingConfig(file_enabled=False))
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            with patch.object(logger.logger, 'error') as mock_error:
                logger.log_error_with_traceback("Test error occurred", e)
                
                # Verify error was logged with traceback
                mock_error.assert_called_once()
                logged_message = mock_error.call_args[0][0]
                assert "Test error occurred" in logged_message
                assert "ValueError: Test error" in logged_message
                assert "Traceback:" in logged_message
    
    def test_metrics_export(self):
        """Test metrics export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(performance_tracking=True, progress_tracking=True)
            logger = ElikopyLogger(name="test_logger", config=config)
            
            # Add some metrics
            with logger.performance_timer("test_op"):
                time.sleep(0.01)
            
            logger.log_progress("test_task", 50, 100)
            
            # Export metrics
            export_path = Path(temp_dir) / "metrics.json"
            logger.export_metrics(export_path)
            
            # Verify export
            assert export_path.exists()
            with open(export_path) as f:
                data = json.load(f)
            
            assert "performance_metrics" in data
            assert "progress_trackers" in data
            assert "export_timestamp" in data
            assert len(data["performance_metrics"]) == 1
            assert len(data["progress_trackers"]) == 1
    
    def test_child_logger_creation(self):
        """Test child logger creation"""
        parent_logger = ElikopyLogger(name="parent", config=LoggingConfig(file_enabled=False))
        child_logger = parent_logger.create_child_logger("child")
        
        assert child_logger.name == "parent.child"
        assert child_logger.logger.parent == parent_logger.logger
        assert child_logger.logger.propagate is True
    
    def test_metrics_clearing(self):
        """Test metrics clearing"""
        config = LoggingConfig(performance_tracking=True, progress_tracking=True)
        logger = ElikopyLogger(name="test_logger", config=config)
        
        # Add metrics
        with logger.performance_timer("test_op"):
            pass
        logger.log_progress("test_task", 50, 100)
        
        # Verify metrics exist
        assert len(logger.get_performance_metrics()) == 1
        assert len(logger.get_progress_info()) == 1
        
        # Clear metrics
        logger.clear_metrics()
        
        # Verify metrics are cleared
        assert len(logger.get_performance_metrics()) == 0
        assert len(logger.get_progress_info()) == 0


class TestGlobalLoggerFunctions:
    """Test global logger functions"""
    
    def test_get_logger(self):
        """Test get_logger function"""
        logger = get_logger("test_global")
        assert isinstance(logger, ElikopyLogger)
        assert logger.name == "test_global"
        
        # Should return same instance for same name
        logger2 = get_logger("test_global")
        assert logger is logger2
    
    def test_configure_logging(self):
        """Test configure_logging function"""
        config = LoggingConfig(level="DEBUG", structured_format=True)
        logger = configure_logging(config)
        
        assert isinstance(logger, ElikopyLogger)
        assert logger.config == config
        assert logger.name == "elikopy"


class TestHPCCompatibleFileHandler:
    """Test HPC-compatible file handler"""
    
    def test_thread_safe_logging(self):
        """Test thread-safe logging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            handler = HPCCompatibleFileHandler(str(log_file))
            
            try:
                # Set a simple formatter
                formatter = logging.Formatter('%(message)s')
                handler.setFormatter(formatter)
                
                # Create multiple threads that log simultaneously
                def log_messages(thread_id):
                    for i in range(10):
                        # Create a proper LogRecord
                        record = logging.LogRecord(
                            name="test",
                            level=logging.INFO,
                            pathname="test.py",
                            lineno=1,
                            msg=f"Thread {thread_id} message {i}",
                            args=(),
                            exc_info=None
                        )
                        handler.emit(record)
                
                threads = []
                for i in range(5):
                    thread = threading.Thread(target=log_messages, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                # Verify all messages were logged
                with open(log_file) as f:
                    lines = f.readlines()
                
                # Should have 50 lines (5 threads * 10 messages)
                assert len(lines) == 50
            finally:
                # Ensure handler is closed
                handler.close()


if __name__ == "__main__":
    pytest.main([__file__])