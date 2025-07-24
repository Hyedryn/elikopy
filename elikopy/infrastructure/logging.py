"""
Enhanced logging infrastructure for elikopy with HPC compatibility and structured logging.
"""

import logging
import logging.handlers
import sys
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class LoggingConfig:
    """Configuration for logging system"""
    level: str = "INFO"
    console_enabled: bool = True
    file_enabled: bool = True
    structured_format: bool = False
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    hpc_compatible: bool = False
    performance_tracking: bool = True
    progress_tracking: bool = True


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProgressInfo:
    """Progress tracking information"""
    task_name: str
    current: int
    total: int
    start_time: float
    estimated_completion: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add performance metrics if present
        if hasattr(record, 'performance_metric') and record.performance_metric is not None:
            try:
                log_entry['performance'] = asdict(record.performance_metric)
            except (TypeError, AttributeError):
                # Handle case where performance_metric is not a dataclass
                log_entry['performance'] = str(record.performance_metric)
        
        # Add progress info if present
        if hasattr(record, 'progress_info') and record.progress_info is not None:
            try:
                log_entry['progress'] = asdict(record.progress_info)
            except (TypeError, AttributeError):
                # Handle case where progress_info is not a dataclass
                log_entry['progress'] = str(record.progress_info)
        
        return json.dumps(log_entry)


class HPCCompatibleFileHandler(logging.handlers.RotatingFileHandler):
    """File handler optimized for HPC environments"""
    
    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 0, 
                 backupCount: int = 0, encoding: Optional[str] = None, 
                 delay: bool = False, errors: Optional[str] = None):
        """Initialize HPC-compatible file handler"""
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay, errors)
        self._lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Thread-safe emit for HPC environments"""
        with self._lock:
            try:
                super().emit(record)
                # Force flush for HPC environments where buffering can cause issues
                if self.stream:
                    self.stream.flush()
                    os.fsync(self.stream.fileno())
            except Exception:
                self.handleError(record)


class ElikopyLogger:
    """Enhanced logging system with HPC compatibility and structured logging"""
    
    def __init__(self, 
                 name: str = "elikopy", 
                 log_dir: Optional[Union[str, Path]] = None,
                 config: Optional[LoggingConfig] = None):
        """Initialize enhanced logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files (optional)
            config: Logging configuration
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.config = config or LoggingConfig()
        self.log_level = self._get_log_level(self.config.level)
        
        # Performance and progress tracking
        self._performance_metrics: List[PerformanceMetric] = []
        self._active_operations: Dict[str, PerformanceMetric] = {}
        self._progress_trackers: Dict[str, ProgressInfo] = {}
        self._metrics_lock = threading.Lock()
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add handlers
        if self.config.console_enabled:
            self._add_console_handler()
        
        if self.config.file_enabled and self.log_dir:
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
        
        # Create formatter based on configuration
        if self.config.structured_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self) -> None:
        """Add file handler to logger with HPC compatibility"""
        if not self.log_dir:
            return
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        
        # Create appropriate file handler
        if self.config.hpc_compatible:
            file_handler = HPCCompatibleFileHandler(
                str(log_file),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                str(log_file),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
        
        file_handler.setLevel(self.log_level)
        
        # Create formatter based on configuration
        if self.config.structured_format:
            formatter = StructuredFormatter()
        else:
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
    
    def log_progress(self, task_name: str, current: int, total: int, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Enhanced progress logging with tracking
        
        Args:
            task_name: Name of the task
            current: Current progress
            total: Total progress
            metadata: Additional metadata
        """
        if not self.config.progress_tracking:
            return
        
        with self._metrics_lock:
            # Update or create progress tracker
            if task_name not in self._progress_trackers:
                self._progress_trackers[task_name] = ProgressInfo(
                    task_name=task_name,
                    current=current,
                    total=total,
                    start_time=time.time(),
                    metadata=metadata
                )
            else:
                tracker = self._progress_trackers[task_name]
                tracker.current = current
                tracker.total = total
                if metadata:
                    tracker.metadata = {**(tracker.metadata or {}), **metadata}
                
                # Estimate completion time
                elapsed = time.time() - tracker.start_time
                if current > 0:
                    estimated_total_time = elapsed * total / current
                    tracker.estimated_completion = tracker.start_time + estimated_total_time
        
        percentage = int(100 * current / total) if total > 0 else 0
        message = f"{task_name}: {current}/{total} ({percentage}%)"
        
        # Add ETA if available
        tracker = self._progress_trackers.get(task_name)
        if tracker and tracker.estimated_completion:
            eta_seconds = tracker.estimated_completion - time.time()
            if eta_seconds > 0:
                eta_str = f"{eta_seconds:.0f}s"
                message += f" ETA: {eta_str}"
        
        # Log with progress info
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, __file__, 0, message, (), None
        )
        record.progress_info = self._progress_trackers[task_name]
        self.logger.handle(record)
    
    @contextmanager
    def performance_timer(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for performance timing
        
        Args:
            operation: Operation name
            metadata: Additional metadata
        """
        if not self.config.performance_tracking:
            yield
            return
        
        metric = PerformanceMetric(
            operation=operation,
            start_time=time.time(),
            metadata=metadata
        )
        
        with self._metrics_lock:
            self._active_operations[operation] = metric
        
        try:
            yield metric
        finally:
            metric.end_time = time.time()
            metric.duration = metric.end_time - metric.start_time
            
            with self._metrics_lock:
                self._performance_metrics.append(metric)
                self._active_operations.pop(operation, None)
            
            # Log performance
            message = f"Performance: {operation} completed in {metric.duration:.2f}s"
            record = self.logger.makeRecord(
                self.logger.name, logging.DEBUG, __file__, 0, message, (), None
            )
            record.performance_metric = metric
            self.logger.handle(record)
    
    def log_performance(self, operation: str, duration: float, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metric
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Additional metadata
        """
        if not self.config.performance_tracking:
            return
        
        metric = PerformanceMetric(
            operation=operation,
            start_time=time.time() - duration,
            end_time=time.time(),
            duration=duration,
            metadata=metadata
        )
        
        with self._metrics_lock:
            self._performance_metrics.append(metric)
        
        message = f"Performance: {operation} took {duration:.2f} seconds"
        record = self.logger.makeRecord(
            self.logger.name, logging.DEBUG, __file__, 0, message, (), None
        )
        record.performance_metric = metric
        self.logger.handle(record)
    
    def log_structured(self, level: str, message: str, **extra_fields) -> None:
        """Log with structured extra fields
        
        Args:
            level: Log level
            message: Log message
            **extra_fields: Additional structured fields
        """
        log_level = self._get_log_level(level)
        record = self.logger.makeRecord(
            self.logger.name, log_level, __file__, 0, message, (), None
        )
        record.extra_fields = extra_fields
        self.logger.handle(record)
    
    def log_error_with_traceback(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log error with traceback
        
        Args:
            message: Error message
            exception: Exception instance (optional)
        """
        import traceback
        
        if exception:
            tb_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            full_message = f"{message}\nException: {str(exception)}\nTraceback:\n{tb_str}"
        else:
            tb_str = traceback.format_exc()
            full_message = f"{message}\nTraceback:\n{tb_str}"
        
        self.logger.error(full_message)
    
    def get_performance_metrics(self) -> List[PerformanceMetric]:
        """Get collected performance metrics
        
        Returns:
            List of performance metrics
        """
        with self._metrics_lock:
            return self._performance_metrics.copy()
    
    def get_progress_info(self, task_name: Optional[str] = None) -> Union[ProgressInfo, Dict[str, ProgressInfo]]:
        """Get progress information
        
        Args:
            task_name: Specific task name (optional)
            
        Returns:
            Progress info for specific task or all tasks
        """
        with self._metrics_lock:
            if task_name:
                return self._progress_trackers.get(task_name)
            return self._progress_trackers.copy()
    
    def clear_metrics(self) -> None:
        """Clear collected metrics and progress trackers"""
        with self._metrics_lock:
            self._performance_metrics.clear()
            self._progress_trackers.clear()
    
    def export_metrics(self, output_path: Union[str, Path]) -> None:
        """Export metrics to JSON file
        
        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        
        with self._metrics_lock:
            metrics_data = {
                "performance_metrics": [asdict(m) for m in self._performance_metrics],
                "progress_trackers": {k: asdict(v) for k, v in self._progress_trackers.items()},
                "export_timestamp": datetime.now().isoformat()
            }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_path}")
    
    def create_child_logger(self, name: str) -> 'ElikopyLogger':
        """Create child logger
        
        Args:
            name: Child logger name
            
        Returns:
            Child logger instance
        """
        child_name = f"{self.name}.{name}"
        child_logger = ElikopyLogger(
            name=child_name,
            log_dir=self.log_dir,
            config=self.config
        )
        child_logger.logger.parent = self.logger
        child_logger.logger.propagate = True
        
        return child_logger


# Global logger instance
_global_logger: Optional[ElikopyLogger] = None


def get_logger(name: str = "elikopy", 
               log_dir: Optional[Union[str, Path]] = None,
               config: Optional[LoggingConfig] = None) -> ElikopyLogger:
    """Get or create logger instance
    
    Args:
        name: Logger name
        log_dir: Log directory
        config: Logging configuration
        
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None or _global_logger.name != name:
        _global_logger = ElikopyLogger(name=name, log_dir=log_dir, config=config)
    
    return _global_logger


def configure_logging(config: LoggingConfig, 
                     log_dir: Optional[Union[str, Path]] = None) -> ElikopyLogger:
    """Configure global logging
    
    Args:
        config: Logging configuration
        log_dir: Log directory
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    _global_logger = ElikopyLogger(name="elikopy", log_dir=log_dir, config=config)
    return _global_logger