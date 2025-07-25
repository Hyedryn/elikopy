# Project Structure and Organization

## Main Package Structure
```
elikopy/
├── core/           # Main API classes and interfaces
├── data/           # BIDS data handling and validation
├── processing/     # Processing algorithms and models
├── infrastructure/ # Job scheduling, logging, file management
├── utils/          # Utility functions and helpers
├── external/       # External tool integrations
├── cli/            # Command-line interface
└── templates/      # Configuration templates
```

## Core Module (`elikopy/core/`)
- `base.py` - Abstract base classes and interfaces
- `config.py` - Configuration management (ElikopyConfig)
- `study.py` - Main study management (ElikopyStudy)
- `processor.py` - Processing orchestration (ElikopyProcessor)

## Data Module (`elikopy/data/`)
- `bids_handler.py` - BIDS data access using pybids
- `derivatives.py` - BIDS derivatives management
- `validator.py` - Data validation and integrity checks

## Processing Module (`elikopy/processing/`)
- `dti.py` - DTI tensor fitting and metrics
- `noddi.py` - NODDI microstructural modeling
- `csd.py` - Constrained spherical deconvolution
- `fingerprinting.py` - Microstructure fingerprinting
- `tracking.py` - Tractography and streamline generation
- `connectivity.py` - Connectivity matrix analysis
- `qsiprep_adapter.py` - QSIPrep output integration

## Infrastructure Module (`elikopy/infrastructure/`)
- `logging.py` - Structured logging system
- `exceptions.py` - Custom exception hierarchy
- `scheduler.py` - HPC/SLURM job scheduling
- `file_manager.py` - File operations and management

## Utils Module (`elikopy/utils/`)
- `image_utils.py` - Image processing utilities
- `validation.py` - Parameter and data validation helpers

## Test Structure (`tests/`)
- Unit tests for each module following `test_<module>.py` naming
- Integration tests for component interactions
- Demo scripts for validation and debugging

## Configuration Templates (`elikopy/templates/`)
- `default_config.yaml` - Standard configuration
- `hpc_config.yaml` - HPC-specific settings
- `minimal_dti_config.yaml` - Minimal DTI processing config

## Key Files
- `pyproject.toml` - Poetry package configuration
- `README.md` - Project documentation
- BIDS example data in `bids_example/`

## Naming Conventions
- Classes use PascalCase (e.g., `ElikopyStudy`, `BIDSHandler`)
- Functions and variables use snake_case
- Constants use UPPER_SNAKE_CASE
- Private methods/attributes prefixed with underscore
- Test files prefixed with `test_`
- Demo/debug scripts descriptively named