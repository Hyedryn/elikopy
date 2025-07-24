# Implementation Plan

- [x] 1. Set up new modular project structure and core interfaces
  - Create new directory structure (core/, data/, processing/, infrastructure/, utils/, external/)
  - Define base interfaces and abstract classes for all major components
  - Set up proper __init__.py files with clean imports
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement configuration management system
  - Create ElikopyConfig dataclass with all configuration sections
  - Implement configuration loading from YAML/JSON files
  - Add basic configuration validation with clear error messages
  - Create default configuration templates
  - _Requirements: 6.4_

- [x] 3. Create QSIPrep data handling foundation
- [x] 3.1 Implement BIDSHandler class for qsiprep derivatives access using pybids

  - Write BIDSHandler class with pybids integration for qsiprep derivatives
  - Implement flexible query methods for preprocessed DWI files with support for all BIDS entities
  - Add support for multi-session, multi-run, and other BIDS entities
  - Create unit tests for pybids-based derivatives parsing
  - _Requirements: 2.1, 2.2, 2.3_
- [x] 3.2 Implement BIDS derivatives management
  - Create derivatives.py module for BIDS-compliant output structure
  - Implement automatic creation of derivatives directory structure
  - Add metadata generation for dataset_description.json and provenance
  - Write unit tests for derivatives structure creation
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. Implement comprehensive data validation system
- [x] 4.1 Create DataValidator class for input validation
  - Implement DWI data integrity validation (file formats, dimensions, gradients)
  - Add bvals/bvecs validation with gradient table checks
  - Create BIDS compliance validation with detailed error reporting
  - Write unit tests for all validation scenarios
  - _Requirements: 7.1, 7.3, 7.4_

- [x] 4.2 Implement parameter validation utilities
  - Create parameter validation functions for all processing steps
  - Add range checking and compatibility validation for processing parameters
  - Implement validation result reporting with actionable error messages
  - Enhance configuration validation with comprehensive parameter checks
  - Write unit tests for parameter validation
  - _Requirements: 7.2, 7.4_

- [x] 5. Create logging and error handling infrastructure
- [x] 5.1 Implement structured logging system
  - Create logging.py module with configurable log levels and formats
  - Add HPC-compatible logging with proper file handling
  - Implement progress tracking and performance metrics logging
  - Write unit tests for logging functionality
  - _Requirements: 4.1, 4.2, 4.4_
- [x] 5.2 Create exception hierarchy and error handling
  - Define ElikopyError base class and specific exception types
  - Implement graceful error recovery with informative messages
  - Add error context tracking and debugging support
  - Write unit tests for error handling scenarios
  - _Requirements: 4.1, 4.3_

- [ ] 6. Implement file management utilities
  - Create FileManager class for safe file operations
  - Add directory structure creation with proper permissions
  - Implement file copying with integrity validation
  - Add temporary file cleanup utilities
  - Write unit tests for file operations
  - _Requirements: 4.3, 1.3_

- [ ] 7. Create HPC job scheduling infrastructure
  - Implement JobScheduler class with SLURM integration
  - Add job submission, monitoring, and cancellation functionality
  - Create job dependency management and resource allocation
  - Implement checkpoint and resume capabilities for failed jobs
  - Write unit tests for job scheduling (with mocking)
  - _Requirements: 5.1, 5.2, 5.4_

- [ ] 8. Implement core ElikopyStudy class
- [ ] 8.1 Create main study management class
  - Implement ElikopyStudy class with qsiprep derivatives support
  - Add subject discovery and metadata management
  - Create study initialization from qsiprep outputs
  - Write unit tests for study setup and subject management
  - _Requirements: 2.1, 2.2, 1.4_

- [ ] 9. Create QSIPrep adapter implementation
- [ ] 9.1 Implement QsiPrepAdapter class
  - Create adapter for loading qsiprep preprocessed data
  - Implement methods to extract preprocessed DWI and anatomical data
  - Add metadata extraction and validation
  - Write unit tests for qsiprep adapter
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 9.2 Implement quality control for qsiprep outputs
  - Create validation methods for qsiprep outputs
  - Add checks for required files and metadata
  - Implement reporting for preprocessing quality metrics
  - Write unit tests for quality control
  - _Requirements: 2.4, 7.1_

- [ ] 10. Implement DTI processing module
  - Create DTIProcessor class with tensor fitting and metrics computation
  - Add support for different fitting algorithms and masking
  - Implement BIDS-compliant output generation for DTI metrics
  - Write unit tests for DTI processing
  - _Requirements: 1.1, 3.5_

- [ ] 11. Implement CSD and MSMT-CSD processing
- [ ] 11.1 Create CSDProcessor class
  - Implement single-shell CSD fitting with response function estimation
  - Add peak extraction and ODF computation
  - Create BIDS-compliant output for CSD results
  - Write unit tests for CSD processing
  - _Requirements: 1.1, 3.5_

- [ ] 11.2 Implement MSMT-CSD processing
  - Add multi-shell multi-tissue CSD fitting
  - Implement tissue-specific response function estimation
  - Create separate outputs for WM, GM, and CSF signals
  - Write unit tests for MSMT-CSD processing
  - _Requirements: 1.1, 3.5_

- [ ] 12. Create microstructure modeling modules
- [ ] 12.1 Implement NODDI processing
  - Create NODDIProcessor class with model fitting
  - Add parameter estimation for ICVF, ODI, and ISOVF
  - Implement BIDS-compliant output generation
  - Write unit tests for NODDI processing
  - _Requirements: 1.1, 3.5_

- [ ] 12.2 Implement microstructure fingerprinting
  - Create MicrostructureFingerprintingProcessor class
  - Add dictionary loading and signal matching algorithms
  - Implement parameter map generation and quality metrics
  - Write unit tests for fingerprinting processing
  - _Requirements: 1.1, 3.5_

- [ ] 13. Implement tractography and connectivity analysis
- [ ] 13.1 Create TrackingProcessor class
  - Implement streamline generation with different algorithms
  - Add SIFT filtering and streamline optimization
  - Create BIDS-compliant tractography output
  - Write unit tests for tractography processing
  - _Requirements: 1.1, 3.5_

- [ ] 13.2 Implement connectivity matrix extraction
  - Create ConnectivityProcessor class with atlas registration
  - Add streamline-atlas intersection computation
  - Implement connectivity matrix generation and export
  - Write unit tests for connectivity analysis
  - _Requirements: 1.1, 3.5_

- [ ] 14. Create processing orchestration system
- [ ] 14.1 Implement ElikopyProcessor class
  - Create processing orchestration with pipeline management
  - Add input validation and processing configuration
  - Implement parallel processing and job submission
  - Write unit tests for processing orchestration
  - _Requirements: 1.1, 5.1, 5.3_

- [ ] 14.2 Add checkpoint and resume functionality
  - Implement processing state persistence
  - Add resume capabilities for interrupted processing
  - Create progress tracking and status reporting
  - Write unit tests for checkpoint/resume functionality
  - _Requirements: 5.4, 4.2_

- [ ] 15. Refactor and integrate existing utilities
- [ ] 15.1 Refactor image processing utilities
  - Move relevant functions from utils.py into utils/image_utils.py
  - Clean up and document image processing functions
  - Add input validation and error handling
  - Write unit tests for image utilities
  - _Requirements: 1.1, 1.5_

- [ ] 16. Create comprehensive documentation
  - Create API reference documentation with examples
  - Add tutorials for using qsiprep outputs with elikopy
  - Document workflow for microstructural modeling and tractography
  - Create usage examples for common analysis pipelines
  - _Requirements: 1.4_

- [ ] 17. Implement comprehensive testing suite
- [ ] 17.1 Create unit tests for all modules
  - Write comprehensive unit tests for each module and class
  - Add test data fixtures and mocking for external dependencies
  - Implement test coverage reporting and validation
  - Create continuous integration test configuration
  - _Requirements: 1.3, 6.1, 6.2_

- [ ] 17.2 Create integration and end-to-end tests
  - Implement integration tests for component interactions
  - Create end-to-end tests for complete processing pipelines
  - Add BIDS compliance validation tests
  - Write performance and regression tests
  - _Requirements: 1.3, 2.1, 3.1_

- [ ] 18. Update package configuration and documentation
- [ ] 18.1 Update package structure and dependencies
  - Update pyproject.toml with new module structure and dependencies
  - Add proper entry points and CLI configuration
  - Update package metadata and version information
  - Create installation and setup documentation
  - _Requirements: 6.5, 1.4_

- [ ] 18.2 Create comprehensive API documentation
  - Add docstrings with type hints for all public methods
  - Create API reference documentation
  - Add usage examples and tutorials
  - Create migration guide from old to new API
  - _Requirements: 1.4, 6.2_

- [ ] 19. Final integration and validation
- [ ] 19.1 Integrate all modules and test complete system
  - Wire together all implemented modules into cohesive system
  - Run comprehensive integration tests on complete pipeline
  - Validate BIDS compliance across all processing steps
  - Test HPC integration and job scheduling functionality
  - _Requirements: 1.1, 2.1, 3.1, 5.1_

- [ ] 19.2 Performance optimization and validation
  - Profile processing performance and identify bottlenecks
  - Optimize memory usage and processing efficiency
  - Validate HPC scaling and resource utilization
  - Create performance benchmarks and regression tests
  - _Requirements: 5.3, 5.5_