"""
ElikoPy - A Python library for diffusion MRI processing and analysis
====================================================================

ElikoPy is a Python library designed to simplify the processing of diffusion 
imaging for microstructural analysis. This refactored version focuses on 
modularity, BIDS compliance, and integration with qsiprep outputs.

Main Components:
---------------
- core: Main API classes (ElikopyStudy, ElikopyProcessor, ElikopyConfig)
- data: BIDS data handling and validation (BIDSHandler, DataValidator, DerivativesManager)
- processing: Processing algorithms (DTI, NODDI, CSD, Tractography, etc.)
- infrastructure: Job scheduling, file management, logging (JobScheduler, FileManager, Logger)
- utils: Utility functions (ImageUtils, ValidationUtils)
- external: External integrations

For more information, see the documentation at:
https://elikopy.readthedocs.io/
"""

# Import core components for easy access
from elikopy.core.study import ElikopyStudy
from elikopy.core.processor import ElikopyProcessor
from elikopy.core.config import ElikopyConfig

# Import base classes for extension
from elikopy.core.base import (
    ProcessingComponent, 
    ConfigurableComponent, 
    DataHandler, 
    BIDSComponent,
    JobManager,
    ModelProcessor,
    ProcessingResult,
    ProcessingStatus,
    ValidationResult,
    Subject,
    DWIFile
)

# Define version
__version__ = "0.5.0"

# Define public API
__all__ = [
    # Core classes
    'ElikopyStudy',
    'ElikopyProcessor', 
    'ElikopyConfig',
    
    # Base classes
    'ProcessingComponent',
    'ConfigurableComponent',
    'DataHandler',
    'BIDSComponent', 
    'JobManager',
    'ModelProcessor',
    
    # Data structures
    'ProcessingResult',
    'ProcessingStatus',
    'ValidationResult',
    'Subject',
    'DWIFile',
    
    # Version
    '__version__'
]