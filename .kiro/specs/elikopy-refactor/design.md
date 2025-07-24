# Design Document

## Overview

This design outlines the comprehensive refactoring of the elikopy diffusion MRI processing library to transform it from a monolithic architecture into a modern, modular, and maintainable codebase with full BIDS compliance. The refactored library will use qsiprep preprocessing outputs as input rather than implementing its own preprocessing pipeline, focusing on microstructural modeling and tractography. The refactoring will significantly improve code organization, error handling, and user experience.

The current elikopy library consists of a single large `Elikopy` class in `core.py` (~2000 lines) with tightly coupled functionality. The refactored architecture will separate concerns into focused modules, implement proper BIDS derivatives handling, and provide a clean, extensible API that builds upon qsiprep's preprocessing capabilities.

## Architecture

### High-Level Architecture

The refactored elikopy will follow a layered architecture pattern:

```
┌─────────────────────────────────────────┐
│           User API Layer                │
│  (ElikopyStudy, ElikopyProcessor)       │
├─────────────────────────────────────────┤
│         Processing Layer                │
│  (Preprocessing, DTI, NODDI, etc.)      │
├─────────────────────────────────────────┤
│         Data Management Layer           │
│  (BIDS Handler, Data Validator)         │
├─────────────────────────────────────────┤
│         Infrastructure Layer            │
│  (Job Scheduler, File Manager, Logger)  │
└─────────────────────────────────────────┘
```

### Module Organization

The refactored codebase will be organized into the following modules:

```
elikopy/
├── __init__.py                 # Main API exports
├── core/
│   ├── __init__.py
│   ├── study.py               # Main ElikopyStudy class
│   ├── processor.py           # Processing orchestration
│   └── config.py              # Configuration management
├── data/
│   ├── __init__.py
│   ├── bids_handler.py        # BIDS data access and validation
│   ├── validator.py           # Data validation utilities
│   └── derivatives.py         # BIDS derivatives management
├── processing/
│   ├── __init__.py
│   ├── qsiprep_adapter.py    # QSIPrep output handling
│   ├── dti.py                # DTI processing
│   ├── noddi.py              # NODDI processing
│   ├── csd.py                # CSD and MSMT-CSD processing
│   ├── tracking.py           # Tractography
│   ├── microstructure.py     # NODDI, DIAMOND, IVIM models
│   ├── fingerprinting.py     # Microstructure fingerprinting
│   └── connectivity.py       # Connectivity matrix extraction
├── infrastructure/
│   ├── __init__.py
│   ├── scheduler.py          # HPC/SLURM job management
│   ├── file_manager.py       # File operations and management
│   └── logging.py            # Logging configuration
└── utils/
    ├── __init__.py
    ├── image_utils.py        # Image processing utilities
    └── validation.py         # Parameter validation
```

## Components and Interfaces

### 1. Core Components

#### ElikopyStudy Class
The main entry point that replaces the current monolithic `Elikopy` class:

```python
class ElikopyStudy:
    def __init__(self, 
                 study_path: Path,
                 bids_root: Optional[Path] = None,
                 derivatives_name: str = "elikopy",
                 config: Optional[ElikopyConfig] = None):
        """Initialize study with BIDS-compliant data organization"""
        
    def setup_from_qsiprep(self, qsiprep_dir: Path) -> None:
        """Setup study from qsiprep derivatives"""
        
    def get_subjects(self) -> List[Subject]:
        """Get list of available subjects"""
        
    def create_processor(self, 
                        processing_type: str,
                        **kwargs) -> ElikopyProcessor:
        """Create processor for specific analysis type"""
```

#### ElikopyProcessor Class
Handles processing orchestration and job management:

```python
class ElikopyProcessor:
    def __init__(self, 
                 study: ElikopyStudy,
                 processing_type: str,
                 scheduler: Optional[JobScheduler] = None):
        """Initialize processor for specific analysis"""
        
    def validate_inputs(self) -> ValidationResult:
        """Validate all inputs before processing"""
        
    def run(self, 
            subjects: Optional[List[str]] = None,
            parallel: bool = True) -> ProcessingResult:
        """Execute processing pipeline"""
        
    def resume(self, checkpoint_path: Path) -> ProcessingResult:
        """Resume processing from checkpoint"""
```

### 2. Data Management Components

#### BIDSHandler Class
Manages BIDS data access and organization using pybids:

```python
class BIDSHandler:
    def __init__(self, bids_root: Path, qsiprep_dir: Path = None):
        """Initialize BIDS dataset handler with qsiprep derivatives using pybids"""
        
    def validate_qsiprep_structure(self) -> BIDSValidationResult:
        """Validate qsiprep derivatives structure"""
        
    def get_preprocessed_dwi_files(self, subject: str, session: Optional[str] = None, 
                                  run: Optional[str] = None, task: Optional[str] = None, 
                                  **filters) -> List[DWIFile]:
        """Get preprocessed DWI files from qsiprep for subject with flexible filtering"""
        
    def get_preprocessed_anatomical_files(self, subject: str, session: Optional[str] = None, 
                                        **filters) -> List[AnatomicalFile]:
        """Get preprocessed anatomical files from qsiprep for subject with flexible filtering"""
        
    def create_derivatives_structure(self, pipeline_name: str) -> Path:
        """Create BIDS derivatives structure for elikopy outputs"""
        
    def get_bids_layout(self) -> BIDSLayout:
        """Get pybids BIDSLayout object for advanced queries"""
```

#### DataValidator Class
Provides comprehensive data validation:

```python
class DataValidator:
    def validate_dwi_data(self, dwi_file: DWIFile) -> ValidationResult:
        """Validate DWI data integrity"""
        
    def validate_bvals_bvecs(self, bval_file: Path, bvec_file: Path) -> ValidationResult:
        """Validate gradient information"""
        
    def validate_processing_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate processing parameters"""
```

### 3. Processing Components

#### QsiPrepAdapter Class
Handles qsiprep output adaptation:

```python
class QsiPrepAdapter:
    def __init__(self, qsiprep_dir: Path):
        """Initialize qsiprep adapter"""
        
    def load_preprocessed_data(self, subject: str, session: Optional[str] = None) -> DWIData:
        """Load preprocessed DWI data from qsiprep outputs"""
        
    def load_anatomical_data(self, subject: str, session: Optional[str] = None) -> AnatomicalData:
        """Load preprocessed anatomical data from qsiprep outputs"""
        
    def get_preprocessing_metadata(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """Get preprocessing metadata from qsiprep outputs"""
        
    def validate_qsiprep_outputs(self, subject: str, session: Optional[str] = None) -> ValidationResult:
        """Validate qsiprep outputs for completeness and quality"""
```

#### DTIProcessor Class
Handles DTI-specific processing:

```python
class DTIProcessor:
    def __init__(self, config: DTIConfig):
        """Initialize DTI processor"""
        
    def fit_tensor(self, dwi_data: DWIData, mask: np.ndarray) -> DTIResult:
        """Fit diffusion tensor model"""
        
    def compute_metrics(self, tensor_data: DTIResult) -> DTIMetrics:
        """Compute DTI scalar metrics"""
```

#### CSDProcessor Class
Handles CSD and MSMT-CSD processing:

```python
class CSDProcessor:
    def __init__(self, config: CSDConfig):
        """Initialize CSD processor"""
        
    def fit_csd(self, dwi_data: DWIData, mask: np.ndarray) -> CSDResult:
        """Fit Constrained Spherical Deconvolution model"""
        
    def fit_msmt_csd(self, dwi_data: DWIData, mask: np.ndarray) -> MSMTCSDResult:
        """Fit Multi-Shell Multi-Tissue CSD model"""
        
    def extract_peaks(self, odf_data: Union[CSDResult, MSMTCSDResult]) -> PeaksResult:
        """Extract fiber orientation peaks"""
```

#### MicrostructureFingerprintingProcessor Class
Handles microstructure fingerprinting analysis:

```python
class MicrostructureFingerprintingProcessor:
    def __init__(self, dictionary_path: Path, config: FingerprintingConfig):
        """Initialize microstructure fingerprinting processor"""
        
    def load_dictionary(self, dictionary_path: Path) -> FingerprintingDictionary:
        """Load fingerprinting dictionary"""
        
    def fit_fingerprinting(self, dwi_data: DWIData, mask: np.ndarray) -> FingerprintingResult:
        """Perform microstructure fingerprinting analysis"""
        
    def compute_metrics(self, fingerprinting_data: FingerprintingResult) -> FingerprintingMetrics:
        """Compute fingerprinting-derived metrics"""
```

#### TrackingProcessor Class
Handles tractography and connectivity analysis:

```python
class TrackingProcessor:
    def __init__(self, config: TrackingConfig):
        """Initialize tracking processor"""
        
    def generate_streamlines(self, odf_data: Union[CSDResult, MSMTCSDResult], 
                           mask: np.ndarray) -> StreamlinesResult:
        """Generate tractography streamlines"""
        
    def apply_sift(self, streamlines: StreamlinesResult, 
                   odf_data: Union[CSDResult, MSMTCSDResult]) -> StreamlinesResult:
        """Apply SIFT filtering to streamlines"""
        
    def extract_connectivity_matrix(self, streamlines: StreamlinesResult,
                                  atlas: AtlasData) -> ConnectivityMatrix:
        """Extract connectivity matrix from streamlines and atlas"""
```

#### ConnectivityProcessor Class
Handles connectivity matrix extraction and analysis:

```python
class ConnectivityProcessor:
    def __init__(self, config: ConnectivityConfig):
        """Initialize connectivity processor"""
        
    def register_to_mni(self, subject_data: SubjectData) -> RegistrationResult:
        """Register subject data to MNI space"""
        
    def apply_atlas(self, registered_data: RegistrationResult, 
                   atlas: AtlasData) -> AtlasResult:
        """Apply atlas to registered data"""
        
    def compute_connectivity_matrix(self, streamlines: StreamlinesResult,
                                  atlas_result: AtlasResult) -> ConnectivityMatrix:
        """Compute connectivity matrix from streamlines and atlas"""
        
    def export_connectivity_matrices(self, matrices: List[ConnectivityMatrix],
                                   output_path: Path) -> None:
        """Export connectivity matrices in various formats"""
```



### 4. Infrastructure Components

#### JobScheduler Class
Manages HPC job submission and monitoring:

```python
class JobScheduler:
    def __init__(self, scheduler_type: str = "slurm", config: SchedulerConfig = None):
        """Initialize job scheduler"""
        
    def submit_job(self, job: ProcessingJob) -> JobID:
        """Submit processing job"""
        
    def monitor_jobs(self, job_ids: List[JobID]) -> List[JobStatus]:
        """Monitor job status"""
        
    def cancel_job(self, job_id: JobID) -> bool:
        """Cancel running job"""
```

#### FileManager Class
Handles file operations with proper error handling:

```python
class FileManager:
    def __init__(self, base_path: Path):
        """Initialize file manager"""
        
    def create_directory_structure(self, structure: Dict[str, Any]) -> None:
        """Create directory structure safely"""
        
    def copy_with_validation(self, src: Path, dst: Path) -> bool:
        """Copy files with integrity validation"""
        
    def cleanup_temporary_files(self, temp_dir: Path) -> None:
        """Clean up temporary files"""
```

## Data Models

### Core Data Structures

```python
@dataclass
class Subject:
    id: str
    sessions: List[str]
    dwi_files: List[DWIFile]
    anatomical_files: List[AnatomicalFile]
    metadata: Dict[str, Any]

@dataclass
class DWIFile:
    path: Path
    bval_path: Path
    bvec_path: Path
    json_path: Optional[Path]
    acquisition_params: Optional[AcquisitionParams]

@dataclass
class ProcessingConfig:
    preprocessing: PreprocessingConfig
    dti: Optional[DTIConfig]
    noddi: Optional[NODDIConfig]
    csd: Optional[CSDConfig]
    msmt_csd: Optional[MSMTCSDConfig]
    fingerprinting: Optional[FingerprintingConfig]
    tracking: Optional[TrackingConfig]
    connectivity: Optional[ConnectivityConfig]
    scheduler: SchedulerConfig
    output: OutputConfig

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    suggestions: List[str]

@dataclass
class CSDResult:
    odf_data: np.ndarray
    peaks: np.ndarray
    peak_values: np.ndarray
    peak_indices: np.ndarray

@dataclass
class MSMTCSDResult:
    wm_odf: np.ndarray
    gm_signal: np.ndarray
    csf_signal: np.ndarray
    peaks: np.ndarray
    peak_values: np.ndarray

@dataclass
class FingerprintingResult:
    parameter_maps: Dict[str, np.ndarray]
    fit_quality: np.ndarray
    dictionary_indices: np.ndarray

@dataclass
class StreamlinesResult:
    streamlines: List[np.ndarray]
    streamline_count: int
    length_stats: Dict[str, float]
    
@dataclass
class ConnectivityMatrix:
    matrix: np.ndarray
    atlas_labels: List[str]
    subject_id: str
    processing_info: Dict[str, Any]

@dataclass
class AtlasData:
    atlas_image: np.ndarray
    labels: List[str]
    label_indices: List[int]
    mni_space: bool
```

### BIDS Derivatives Structure

The refactored elikopy will create BIDS-compliant derivatives following this structure:

```
derivatives/
└── elikopy/
    ├── dataset_description.json
    ├── sub-<subject>/
    │   └── ses-<session>/
    │       ├── dwi/
    │       │   ├── sub-<subject>_ses-<session>_desc-preproc_dwi.nii.gz
    │       │   ├── sub-<subject>_ses-<session>_desc-preproc_dwi.bval
    │       │   ├── sub-<subject>_ses-<session>_desc-preproc_dwi.bvec
    │       │   └── sub-<subject>_ses-<session>_desc-preproc_dwi.json
    │       ├── dti/
    │       │   ├── sub-<subject>_ses-<session>_model-DTI_parameter-FA.nii.gz
    │       │   ├── sub-<subject>_ses-<session>_model-DTI_parameter-MD.nii.gz
    │       │   └── sub-<subject>_ses-<session>_model-DTI_parameter-RD.nii.gz
    │       ├── noddi/
    │       │   ├── sub-<subject>_ses-<session>_model-NODDI_parameter-ICVF.nii.gz
    │       │   └── sub-<subject>_ses-<session>_model-NODDI_parameter-ODI.nii.gz
    │       ├── csd/
    │       │   ├── sub-<subject>_ses-<session>_model-CSD_odf.nii.gz
    │       │   └── sub-<subject>_ses-<session>_model-CSD_peaks.nii.gz
    │       ├── msmtcsd/
    │       │   ├── sub-<subject>_ses-<session>_model-MSMTCSD_wm-odf.nii.gz
    │       │   ├── sub-<subject>_ses-<session>_model-MSMTCSD_gm-signal.nii.gz
    │       │   └── sub-<subject>_ses-<session>_model-MSMTCSD_peaks.nii.gz
    │       ├── fingerprinting/
    │       │   ├── sub-<subject>_ses-<session>_model-MF_parameter-<param>.nii.gz
    │       │   └── sub-<subject>_ses-<session>_model-MF_fit-quality.nii.gz
    │       ├── tractography/
    │       │   ├── sub-<subject>_ses-<session>_tractography.trk
    │       │   └── sub-<subject>_ses-<session>_tractography-sift.trk
    │       └── connectivity/
    │           ├── sub-<subject>_ses-<session>_atlas-<atlas>_connectivity.csv
    │           └── sub-<subject>_ses-<session>_atlas-<atlas>_connectivity.json
    └── logs/
        └── processing_log_<timestamp>.json
```

## Error Handling

### Error Handling Strategy

1. **Structured Exception Hierarchy**:
   ```python
   class ElikopyError(Exception):
       """Base exception for elikopy"""
   
   class DataValidationError(ElikopyError):
       """Raised when data validation fails"""
   
   class ProcessingError(ElikopyError):
       """Raised during processing failures"""
   
   class BIDSError(ElikopyError):
       """Raised for BIDS-related issues"""
   ```

2. **Graceful Error Recovery**:
   - Checkpoint system for long-running processes
   - Automatic retry mechanisms for transient failures
   - Clear error messages with suggested solutions

3. **Comprehensive Logging**:
   - Structured logging with different levels (DEBUG, INFO, WARNING, ERROR)
   - Processing provenance tracking
   - Performance metrics collection

### Validation Framework

```python
class ValidationFramework:
    def __init__(self):
        self.validators = []
        
    def add_validator(self, validator: Validator) -> None:
        """Add custom validator"""
        
    def validate_all(self, data: Any) -> ValidationResult:
        """Run all validators"""
        
    def generate_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report"""
```

## Testing Strategy

### Testing Architecture

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete processing pipelines
4. **Performance Tests**: Validate processing performance
5. **BIDS Compliance Tests**: Ensure BIDS standard compliance

### Test Data Management

- Synthetic test datasets for different scenarios
- BIDS-compliant test datasets
- Legacy format test datasets for migration testing
- Performance benchmarking datasets



## Performance Considerations

### Optimization Strategies

1. **Lazy Loading**: Load data only when needed
2. **Memory Management**: Efficient memory usage for large datasets
3. **Parallel Processing**: Leverage multiprocessing and HPC resources
4. **Caching**: Cache intermediate results to avoid recomputation
5. **Progress Tracking**: Real-time progress monitoring

### HPC Integration

- Maintain full SLURM compatibility
- Support for GPU acceleration where applicable
- Efficient resource utilization
- Job dependency management
- Automatic resource estimation

## Configuration Management

### Configuration System

```python
@dataclass
class ElikopyConfig:
    study_name: str
    processing: ProcessingConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig
    output: OutputConfig
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'ElikopyConfig':
        """Load configuration from file"""
        
    def to_file(self, config_path: Path) -> None:
        """Save configuration to file"""
        
    def validate(self) -> ValidationResult:
        """Validate configuration"""
```

### Configuration Sources

1. **Default Configuration**: Sensible defaults for all parameters
2. **File Configuration**: YAML/JSON configuration files
3. **Environment Variables**: Override via environment
4. **Command Line Arguments**: Runtime parameter override
5. **Programmatic Configuration**: Direct API configuration

This design provides a solid foundation for the elikopy refactoring while maintaining all existing functionality and significantly improving maintainability, usability, and BIDS compliance.