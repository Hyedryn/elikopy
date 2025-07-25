# Technology Stack and Build System

## Build System
- **Python Version**: >=3.8,<4.0

## Core Dependencies
- **Scientific Computing**: numpy, scipy, scikit-learn, scikit-image
- **Neuroimaging**: dipy, nibabel, pybids
- **External Integrations**: 
  - microstructure-fingerprinting-rensonnetg (git dependency)
  - unravel-python

## Development Dependencies
- **Testing**: pytest (inferred from test structure)

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
python test_validator_demo.py
python test_parameter_validation_demo.py

# Debug BIDS entities
python debug_pybids_entities.py
```

## Architecture Patterns
- **Modular Design**: Clear separation into core/, data/, processing/, infrastructure/, utils/
- **BIDS Compliance**: All data handling follows BIDS specification
- **Configuration Management**: YAML-based configuration with dataclass validation
- **Error Handling**: Structured exception hierarchy with ElikopyError base class
- **HPC Integration**: Native SLURM job scheduling support
- **Type Safety**: Type hints throughout codebase (Python 3.8+ compatible)