#!/usr/bin/env python3
"""
Demo script showing the enhanced logging and error handling infrastructure.
"""

import tempfile
import time
from pathlib import Path

from elikopy.infrastructure import (
    ElikopyLogger, LoggingConfig, 
    ElikopyError, DataValidationError, ProcessingError,
    ErrorContext, ErrorHandler, create_error_context,
    handle_exceptions
)


def demo_basic_logging():
    """Demonstrate basic logging functionality"""
    print("=== Basic Logging Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create logger with different configurations
        config = LoggingConfig(
            level="DEBUG",
            structured_format=False,
            performance_tracking=True,
            progress_tracking=True
        )
        
        logger = ElikopyLogger(
            name="demo_logger",
            log_dir=temp_dir,
            config=config
        )
        
        # Basic logging
        logger.logger.info("Starting demo")
        logger.logger.debug("Debug information")
        logger.logger.warning("This is a warning")
        
        # Performance tracking
        with logger.performance_timer("demo_operation") as metric:
            time.sleep(0.1)  # Simulate work
            logger.logger.info("Doing some work...")
        
        # Progress tracking
        for i in range(5):
            logger.log_progress("demo_task", i+1, 5)
            time.sleep(0.05)
        
        # Structured logging
        logger.log_structured("INFO", "Processing subject", 
                             subject_id="sub-01", 
                             session="ses-01",
                             processing_step="preprocessing")
        
        # Get performance metrics
        metrics = logger.get_performance_metrics()
        print(f"Recorded {len(metrics)} performance metrics")
        
        # Close handlers to release file locks
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)


def demo_error_handling():
    """Demonstrate error handling functionality"""
    print("\n=== Error Handling Demo ===")
    
    # Create error handler with logger
    config = LoggingConfig(console_enabled=True, file_enabled=False)
    logger = ElikopyLogger(name="error_demo", config=config)
    error_handler = ErrorHandler(logger=logger)
    
    # Demo 1: Basic error with context
    try:
        context = create_error_context(
            "data_validation",
            subject_id="sub-01",
            file_path=Path("/data/sub-01/dwi.nii.gz")
        )
        
        raise DataValidationError(
            "DWI file has incorrect dimensions",
            validation_errors=["Expected 4D, got 3D", "Missing bvals file"],
            context=context
        )
    except ElikopyError as e:
        print("Caught DataValidationError:")
        print(str(e))
        print()
    
    # Demo 2: Error with recovery
    def recovery_strategy(error):
        print(f"Attempting to recover from: {error.message}")
        # Simulate successful recovery
        return True
    
    try:
        raise ProcessingError(
            "Tensor fitting failed",
            processing_step="dti_fitting"
        )
    except ElikopyError as e:
        print("Attempting error recovery...")
        recovered = error_handler.attempt_recovery(e, [recovery_strategy])
        print(f"Recovery successful: {recovered}")
        print()
    
    # Demo 3: Using decorator
    @handle_exceptions("demo_function", logger=logger, reraise=False)
    def failing_function():
        raise ValueError("Something went wrong")
    
    result = failing_function()
    print(f"Decorator handled error: {type(result).__name__}")
    print()
    
    # Show error summary
    summary = error_handler.get_error_summary()
    print("Error Summary:")
    print(f"Total errors: {summary['total_errors']}")
    print(f"Error types: {summary['error_types']}")


def demo_structured_logging():
    """Demonstrate structured logging"""
    print("\n=== Structured Logging Demo ===")
    
    config = LoggingConfig(
        structured_format=True,
        console_enabled=True,
        file_enabled=False
    )
    
    logger = ElikopyLogger(name="structured_demo", config=config)
    
    # Log structured data
    logger.log_structured("INFO", "Subject processing started",
                         subject_id="sub-01",
                         session_id="ses-01",
                         processing_pipeline="dti",
                         input_files=["dwi.nii.gz", "bvals", "bvecs"])
    
    # Log with performance data
    with logger.performance_timer("structured_operation", 
                                 metadata={"algorithm": "RESTORE"}) as metric:
        time.sleep(0.05)
    
    print("Structured logging completed (check console output above)")


def demo_hpc_logging():
    """Demonstrate HPC-compatible logging"""
    print("\n=== HPC-Compatible Logging Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LoggingConfig(
            hpc_compatible=True,
            max_file_size=1024,  # Small size for demo
            backup_count=3,
            console_enabled=False,
            file_enabled=True
        )
        
        logger = ElikopyLogger(
            name="hpc_demo",
            log_dir=temp_dir,
            config=config
        )
        
        # Generate enough logs to trigger rotation
        for i in range(50):
            logger.logger.info(f"HPC log message {i:03d} - simulating processing step")
        
        # Check created files
        log_files = list(Path(temp_dir).glob("*.log*"))
        print(f"Created {len(log_files)} log files in HPC mode")
        
        # Close handlers
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)


if __name__ == "__main__":
    print("ElikoPy Logging and Error Handling Demo")
    print("=" * 50)
    
    demo_basic_logging()
    demo_error_handling()
    demo_structured_logging()
    demo_hpc_logging()
    
    print("\n=== Demo Complete ===")