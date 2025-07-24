# Requirements Document

## Introduction

This feature involves comprehensive refactoring of the entire elikopy diffusion MRI processing library to improve code maintainability, clean up all modules (core.py, utils.py, individual_subject_processing.py, registration.py, utilsSynb0Disco.py, etc.), and implement full BIDS (Brain Imaging Data Structure) standard compliance for data access and derivative storage. The refactored library will use qsiprep preprocessing outputs as input rather than implementing its own preprocessing pipeline, focusing on microstructural modeling and tractography. The refactoring will modernize the entire codebase architecture while ensuring compatibility with HPC/SLURM environments.

## Requirements

### Requirement 1

**User Story:** As a developer maintaining the elikopy library, I want the entire codebase to be modularized into smaller, focused components, so that the codebase is easier to understand, test, and maintain.

#### Acceptance Criteria

1. WHEN the refactoring is complete THEN all modules (core.py, utils.py, individual_subject_processing.py, registration.py, utilsSynb0Disco.py, etc.) SHALL be reorganized with clear separation of concerns
2. WHEN each module is created THEN it SHALL have a single, well-defined responsibility (e.g., data validation, preprocessing orchestration, BIDS handling, job scheduling)
3. WHEN the modularization is complete THEN each module SHALL be independently testable with minimal dependencies
4. WHEN the refactoring is done THEN the new API SHALL be well-documented and provide clear migration guidance for existing users
5. WHEN code is reorganized THEN duplicate functionality across modules SHALL be consolidated into shared utilities

### Requirement 2

**User Story:** As a researcher using elikopy, I want the library to fully support qsiprep preprocessing outputs as input, so that I can leverage standardized preprocessing without duplicating effort.

#### Acceptance Criteria

1. WHEN qsiprep derivatives are provided THEN the library SHALL automatically detect and parse the BIDS derivatives structure
2. WHEN processing qsiprep outputs THEN the library SHALL correctly identify preprocessed diffusion MRI files, associated metadata (JSON), and gradient information (bval/bvec)
3. WHEN qsiprep outputs contain multiple sessions or runs THEN the library SHALL handle them appropriately
4. WHEN qsiprep outputs are missing required files THEN the library SHALL provide clear, actionable error messages
5. WHEN qsiprep outputs include anatomical data (T1w) THEN the library SHALL automatically use it for registration and segmentation

### Requirement 3

**User Story:** As a researcher using elikopy, I want the library to store processed derivatives in BIDS-compliant format, so that outputs are standardized and interoperable with other neuroimaging tools.

#### Acceptance Criteria

1. WHEN processing is complete THEN derivatives SHALL be stored following BIDS derivatives specification
2. WHEN derivatives are created THEN they SHALL include proper BIDS metadata files (dataset_description.json, etc.)
3. WHEN processing steps are performed THEN the library SHALL generate BIDS-compliant provenance information
4. WHEN multiple processing pipelines are run THEN derivatives SHALL be organized in separate, clearly named pipeline directories
5. WHEN derivatives are created THEN file naming SHALL follow BIDS conventions with appropriate suffixes and entities

### Requirement 4

**User Story:** As a developer working with the elikopy codebase, I want improved error handling and logging throughout the library, so that issues can be diagnosed and resolved more efficiently.

#### Acceptance Criteria

1. WHEN errors occur THEN the library SHALL provide informative error messages with context about what went wrong
2. WHEN processing steps are executed THEN the library SHALL log progress and status information at appropriate levels
3. WHEN file operations fail THEN the library SHALL handle exceptions gracefully and provide recovery suggestions
4. WHEN running in HPC environments THEN logging SHALL be compatible with job scheduling systems
5. WHEN debugging is needed THEN the library SHALL support verbose logging modes

### Requirement 5

**User Story:** As a researcher using elikopy on HPC systems, I want the refactored library to maintain full compatibility with SLURM job scheduling.

#### Acceptance Criteria

1. WHEN running on HPC systems THEN the library SHALL maintain all existing SLURM integration functionality
2. WHEN jobs are submitted THEN the library SHALL properly handle job dependencies and resource allocation
3. WHEN processing large datasets THEN the library SHALL efficiently manage parallel processing across compute nodes
4. WHEN jobs fail THEN the library SHALL provide mechanisms for resuming processing from checkpoints
5. WHEN using CUDA acceleration THEN the library SHALL maintain compatibility with GPU-enabled processing

### Requirement 6

**User Story:** As a developer contributing to elikopy, I want the refactored code to follow modern Python best practices, so that the codebase is more professional and easier to contribute to.

#### Acceptance Criteria

1. WHEN code is refactored THEN it SHALL follow PEP 8 style guidelines
2. WHEN functions are created THEN they SHALL have proper type hints and docstrings
3. WHEN modules are organized THEN they SHALL have clear import structures and minimal circular dependencies
4. WHEN configuration is needed THEN it SHALL use modern configuration management approaches
5. WHEN the refactoring is complete THEN the code SHALL be compatible with modern Python versions (3.8+)

### Requirement 7

**User Story:** As a user of elikopy, I want comprehensive validation of input data and processing parameters, so that I can catch configuration errors early and avoid wasted computation time.

#### Acceptance Criteria

1. WHEN input data is provided THEN the library SHALL validate file formats, required metadata, and data integrity
2. WHEN processing parameters are specified THEN the library SHALL validate parameter ranges and compatibility
3. WHEN BIDS datasets are processed THEN the library SHALL validate BIDS compliance and report any issues
4. WHEN validation fails THEN the library SHALL provide specific, actionable error messages
5. WHEN optional files are missing THEN the library SHALL warn users about potential limitations in processing