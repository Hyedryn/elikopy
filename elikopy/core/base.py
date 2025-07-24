"""
Base interfaces and abstract classes for ElikoPy core components
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ProcessingStatus(Enum):
    """Processing status enum"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of a processing operation"""
    status: ProcessingStatus
    output_files: List[Path]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class Subject:
    """Represents a study subject"""
    id: str
    sessions: List[str]
    dwi_files: List[Path]
    anatomical_files: List[Path]
    metadata: Dict[str, Any]


@dataclass
class DWIFile:
    """Represents a DWI file with associated metadata"""
    path: Path
    bval_path: Path
    bvec_path: Path
    json_path: Optional[Path]
    acquisition_params: Optional[Dict[str, Any]] = None


@dataclass
class ValidationError:
    """Represents a validation error"""
    message: str
    field: Optional[str] = None
    severity: str = "error"


@dataclass
class ValidationWarning:
    """Represents a validation warning"""
    message: str
    field: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    suggestions: List[str]


class ProcessingComponent(ABC):
    """Base class for all processing components"""
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate inputs before processing"""
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> ProcessingResult:
        """Execute processing"""
        pass


class ConfigurableComponent(ABC):
    """Base class for components that can be configured"""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component"""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        pass


class DataHandler(ABC):
    """Base class for data handling components"""
    
    @abstractmethod
    def load_data(self, path: Path) -> Any:
        """Load data from path"""
        pass
    
    @abstractmethod
    def save_data(self, data: Any, path: Path) -> None:
        """Save data to path"""
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> ValidationResult:
        """Validate data integrity"""
        pass


class BIDSComponent(ABC):
    """Base class for BIDS-related components"""
    
    @abstractmethod
    def validate_bids_structure(self, bids_root: Path) -> ValidationResult:
        """Validate BIDS directory structure"""
        pass
    
    @abstractmethod
    def create_bids_derivatives(self, output_dir: Path, 
                              pipeline_name: str) -> Path:
        """Create BIDS derivatives structure"""
        pass


class JobManager(ABC):
    """Base class for job management components"""
    
    @abstractmethod
    def submit_job(self, job_config: Dict[str, Any]) -> str:
        """Submit a job and return job ID"""
        pass
    
    @abstractmethod
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor job status"""
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        pass


class ModelProcessor(ProcessingComponent, ConfigurableComponent):
    """Base class for diffusion model processors"""
    
    @abstractmethod
    def fit_model(self, dwi_data: np.ndarray, 
                  bvals: np.ndarray, 
                  bvecs: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Fit diffusion model to data"""
        pass
    
    @abstractmethod
    def compute_metrics(self, model_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute derived metrics from model parameters"""
        pass