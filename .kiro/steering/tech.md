# Technology Stack and Build System

## Core Dependencies
- **Scientific Computing**: numpy, scipy, scikit-learn, scikit-image
- **Neuroimaging**: dipy, nibabel, pybids
- **Machine Learning**: torch, torchvision, numba
- **Data Processing**: pandas, matplotlib
- **External Integrations**: 
  - microstructure-fingerprinting-rensonnetg (git dependency)
  - dmipy (git dependency from Hyedryn fork)
  - unravel-python

## Common Commands

### Testing
```bash
# Run tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_data_validator.py
```

### Development
```bash
# Run validation demos
python tests/test_validator_demo.py
python tests/test_parameter_validation_demo.py

# Debug BIDS entities
python tests/debug_pybids_entities.py

# Run demo scripts
python tests/demo_checkpoint_resume.py
python tests/demo_connectivity_processor.py
python tests/demo_logging_error_handling.py
```

## Architecture Patterns
- **Modular Design**: Clear separation into core/, data/, processing/, infrastructure/, utils/, cli/, external/
- **BIDS Compliance**: All data handling follows BIDS specification with pybids integration
- **Configuration Management**: YAML-based configuration with dataclass validation and templates
- **Error Handling**: Structured exception hierarchy with ElikopyError base class
- **HPC Integration**: Native SLURM job scheduling support via scheduler module
- **Type Safety**: Type hints throughout codebase (Python 3.8+ compatible)
- **CLI Support**: Command-line interface for configuration management
- **External Tool Integration**: Modular approach for integrating external neuroimaging tools
- **Quality Control**: Built-in validation and quality control reporting
- **Checkpoint/Resume**: Support for resuming interrupted processing workflows