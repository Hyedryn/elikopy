"""
BIDSHandler class - BIDS data access and organization using pybids
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

try:
    from bids.layout import BIDSLayout
    from bids.exceptions import BIDSValidationError
    PYBIDS_AVAILABLE = True
except ImportError:
    PYBIDS_AVAILABLE = False
    BIDSLayout = None
    BIDSValidationError = Exception

from ..core.base import BIDSComponent, ValidationResult, ValidationError, ValidationWarning, DWIFile


class BIDSValidationResult:
    """Result of BIDS validation"""
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


class AnatomicalFile:
    """Represents an anatomical file with metadata"""
    def __init__(self, path: Path, json_path: Optional[Path] = None, 
                 acquisition_params: Optional[Dict[str, Any]] = None):
        self.path = path
        self.json_path = json_path
        self.acquisition_params = acquisition_params or {}


class BIDSHandler(BIDSComponent):
    """Class for handling BIDS data access and organization using pybids"""
    
    def __init__(self, bids_root: Path, qsiprep_dir: Optional[Path] = None):
        """Initialize BIDS handler with pybids integration for qsiprep derivatives
        
        Args:
            bids_root: Path to BIDS root directory
            qsiprep_dir: Path to qsiprep derivatives directory (optional)
        """
        if not PYBIDS_AVAILABLE:
            raise ImportError("pybids is required for BIDSHandler. Install with 'pip install pybids'.")
        
        self.bids_root = Path(bids_root)
        self.qsiprep_dir = Path(qsiprep_dir) if qsiprep_dir else None
        self._layout = None
        self._qsiprep_layout = None
        
        # Validate BIDS structure
        validation_result = self.validate_bids_structure(self.bids_root)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid BIDS structure: {validation_result.errors}")
        
        # Initialize pybids layout
        self._initialize_layouts()
        
    def _initialize_layouts(self) -> None:
        """Initialize pybids layouts for BIDS root and qsiprep derivatives"""
        try:
            # Initialize main BIDS layout
            self._layout = BIDSLayout(self.bids_root, derivatives=True, validate=False)
            
            # Initialize qsiprep layout if directory is provided
            if self.qsiprep_dir and self.qsiprep_dir.exists():
                self._qsiprep_layout = BIDSLayout(self.qsiprep_dir, validate=False)
                
        except Exception as e:
            raise ValueError(f"Failed to initialize BIDS layouts: {e}")
    
    def validate_bids_structure(self, bids_root: Path) -> ValidationResult:
        """Validate BIDS directory structure
        
        Args:
            bids_root: Path to BIDS root directory
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check if BIDS root exists
        if not bids_root.exists():
            errors.append(f"BIDS root directory not found: {bids_root}")
            return ValidationResult(False, [ValidationError(msg) for msg in errors], 
                                  [ValidationWarning(msg) for msg in warnings], [])
        
        # Check for dataset_description.json
        dataset_desc = bids_root / "dataset_description.json"
        if not dataset_desc.exists():
            errors.append(f"dataset_description.json not found in BIDS root: {dataset_desc}")
        
        # Try to validate with pybids if available
        if PYBIDS_AVAILABLE:
            try:
                layout = BIDSLayout(bids_root, validate=True)
                # If we get here, basic BIDS validation passed
            except BIDSValidationError as e:
                warnings.append(f"BIDS validation warnings: {str(e)}")
            except Exception as e:
                errors.append(f"BIDS validation failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, [ValidationError(msg) for msg in errors], 
                              [ValidationWarning(msg) for msg in warnings], [])
    
    def validate_qsiprep_structure(self) -> BIDSValidationResult:
        """Validate qsiprep derivatives structure
        
        Returns:
            BIDSValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        if not self.qsiprep_dir:
            errors.append("QSIPrep directory not set")
            return BIDSValidationResult(False, errors, warnings)
        
        if not self.qsiprep_dir.exists():
            errors.append(f"QSIPrep directory not found: {self.qsiprep_dir}")
            return BIDSValidationResult(False, errors, warnings)
        
        # Check for dataset_description.json
        dataset_desc = self.qsiprep_dir / "dataset_description.json"
        if not dataset_desc.exists():
            errors.append(f"dataset_description.json not found in qsiprep directory: {dataset_desc}")
        
        # Check for at least one subject directory
        subject_dirs = list(self.qsiprep_dir.glob("sub-*"))
        if not subject_dirs:
            errors.append("No subject directories found in qsiprep directory")
        else:
            # Check if subjects have DWI data
            dwi_found = False
            for subject_dir in subject_dirs:
                # Check for DWI files in sessions or directly in subject
                session_dirs = list(subject_dir.glob("ses-*"))
                if session_dirs:
                    for session_dir in session_dirs:
                        dwi_dir = session_dir / "dwi"
                        if dwi_dir.exists() and list(dwi_dir.glob("*_dwi.nii.gz")):
                            dwi_found = True
                            break
                else:
                    dwi_dir = subject_dir / "dwi"
                    if dwi_dir.exists() and list(dwi_dir.glob("*_dwi.nii.gz")):
                        dwi_found = True
                        break
                
                if dwi_found:
                    break
            
            if not dwi_found:
                warnings.append("No DWI files found in any subject directory")
        
        is_valid = len(errors) == 0
        return BIDSValidationResult(is_valid, errors, warnings)
    
    def get_subjects_info(self) -> Dict[str, List[str]]:
        """Get information about subjects in the dataset using pybids
        
        Returns:
            Dictionary mapping subject IDs to lists of session IDs
        """
        if not self._qsiprep_layout:
            if not self.qsiprep_dir:
                raise ValueError("QSIPrep directory not set")
            # Try to initialize layout if not done yet
            self._initialize_layouts()
        
        subjects_info = {}
        
        # Get all subjects from pybids layout
        subjects = self._qsiprep_layout.get_subjects()
        
        for subject in subjects:
            # Get sessions for this subject
            sessions = self._qsiprep_layout.get_sessions(subject=subject)
            
            # Store subject info
            subject_key = f"sub-{subject}"
            if sessions:
                subjects_info[subject_key] = [f"ses-{session}" for session in sessions]
            else:
                subjects_info[subject_key] = [""]
        
        return subjects_info
    
    def get_preprocessed_dwi_files(
        self, 
        subject: str, 
        session: Optional[str] = None, 
        run: Optional[str] = None, 
        task: Optional[str] = None, 
        **filters
    ) -> List[DWIFile]:
        """Get preprocessed DWI files from qsiprep for subject with flexible filtering
        
        Args:
            subject: Subject ID (with or without 'sub-' prefix)
            session: Session ID (with or without 'ses-' prefix, optional)
            run: Run ID (with or without 'run-' prefix, optional)
            task: Task ID (with or without 'task-' prefix, optional)
            **filters: Additional BIDS entities for filtering
            
        Returns:
            List of DWIFile objects with preprocessed DWI data
        """
        if not self._qsiprep_layout:
            if not self.qsiprep_dir:
                raise ValueError("QSIPrep directory not set")
            self._initialize_layouts()
        
        # Normalize subject ID (remove 'sub-' prefix for pybids)
        subject_id = subject.replace("sub-", "") if subject.startswith("sub-") else subject
        
        # Build query parameters
        query_params = {
            'subject': subject_id,
            'datatype': 'dwi',
            'suffix': 'dwi',
            'space': 'ACPC',  # QSIPrep outputs are in ACPC space
            'extension': '.nii.gz'
        }
        
        # Add optional filters
        if session:
            session_id = session.replace("ses-", "") if session.startswith("ses-") else session
            query_params['session'] = session_id
        
        if run:
            run_id = run.replace("run-", "") if run.startswith("run-") else run
            query_params['run'] = run_id
        
        if task:
            task_id = task.replace("task-", "") if task.startswith("task-") else task
            query_params['task'] = task_id
        
        # Add any additional filters
        for key, value in filters.items():
            query_params[key] = value
        
        # Query DWI files using pybids
        try:
            dwi_files_bids = self._qsiprep_layout.get(**query_params)
        except Exception as e:
            raise ValueError(f"Failed to query DWI files: {e}")
        
        dwi_files = []
        
        for dwi_file_bids in dwi_files_bids:
            dwi_path = Path(dwi_file_bids.path)
            
            # Get associated bval/bvec files
            bval_query = query_params.copy()
            bval_query.update({'extension': '.bval'})
            bval_files = self._qsiprep_layout.get(**bval_query)
            
            bvec_query = query_params.copy()
            bvec_query.update({'extension': '.bvec'})
            bvec_files = self._qsiprep_layout.get(**bvec_query)
            
            # Get JSON sidecar
            json_query = query_params.copy()
            json_query.update({'extension': '.json'})
            json_files = self._qsiprep_layout.get(**json_query)
            
            # Find matching bval/bvec files (same entities)
            bval_path = None
            bvec_path = None
            json_path = None
            
            for bval_file in bval_files:
                if self._files_match_entities(dwi_file_bids, bval_file):
                    bval_path = Path(bval_file.path)
                    break
            
            for bvec_file in bvec_files:
                if self._files_match_entities(dwi_file_bids, bvec_file):
                    bvec_path = Path(bvec_file.path)
                    break
            
            for json_file in json_files:
                if self._files_match_entities(dwi_file_bids, json_file):
                    json_path = Path(json_file.path)
                    break
            
            # Only include if we have both bval and bvec files
            if bval_path and bvec_path:
                # Load acquisition parameters from JSON if available
                acquisition_params = None
                if json_path and json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            acquisition_params = json.load(f)
                    except Exception:
                        pass  # Continue without acquisition params if JSON loading fails
                
                dwi_file = DWIFile(
                    path=dwi_path,
                    bval_path=bval_path,
                    bvec_path=bvec_path,
                    json_path=json_path,
                    acquisition_params=acquisition_params
                )
                dwi_files.append(dwi_file)
        
        return dwi_files
    
    def _files_match_entities(self, file1, file2) -> bool:
        """Check if two BIDS files have matching entities (except extension and suffix)"""
        # Get entities for both files
        entities1 = file1.get_entities()
        entities2 = file2.get_entities()
        
        # Remove extension and suffix for comparison
        entities1.pop('extension', None)
        entities1.pop('suffix', None)
        entities2.pop('extension', None)
        entities2.pop('suffix', None)
        
        return entities1 == entities2
    
    def get_preprocessed_anatomical_files(
        self, 
        subject: str, 
        session: Optional[str] = None, 
        **filters
    ) -> List[AnatomicalFile]:
        """Get preprocessed anatomical files from qsiprep for subject with flexible filtering
        
        Args:
            subject: Subject ID (with or without 'sub-' prefix)
            session: Session ID (with or without 'ses-' prefix, optional)
            **filters: Additional BIDS entities for filtering
            
        Returns:
            List of AnatomicalFile objects with preprocessed anatomical data
        """
        if not self._qsiprep_layout:
            if not self.qsiprep_dir:
                raise ValueError("QSIPrep directory not set")
            self._initialize_layouts()
        
        # Normalize subject ID (remove 'sub-' prefix for pybids)
        subject_id = subject.replace("sub-", "") if subject.startswith("sub-") else subject
        
        # Build query parameters for T1w files
        query_params = {
            'subject': subject_id,
            'datatype': 'anat',
            'suffix': 'T1w',
            'space': 'ACPC',  # QSIPrep outputs are in ACPC space
            'extension': '.nii.gz'
        }
        
        # Add optional filters
        if session:
            session_id = session.replace("ses-", "") if session.startswith("ses-") else session
            query_params['session'] = session_id
        
        # Add any additional filters
        for key, value in filters.items():
            query_params[key] = value
        
        # Query anatomical files using pybids
        try:
            anat_files_bids = self._qsiprep_layout.get(**query_params)
        except Exception as e:
            raise ValueError(f"Failed to query anatomical files: {e}")
        
        anat_files = []
        
        for anat_file_bids in anat_files_bids:
            anat_path = Path(anat_file_bids.path)
            
            # Get JSON sidecar
            json_query = query_params.copy()
            json_query.update({'extension': '.json'})
            json_files = self._qsiprep_layout.get(**json_query)
            
            # Find matching JSON file
            json_path = None
            for json_file in json_files:
                if self._files_match_entities(anat_file_bids, json_file):
                    json_path = Path(json_file.path)
                    break
            
            # Load acquisition parameters from JSON if available
            acquisition_params = None
            if json_path and json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        acquisition_params = json.load(f)
                except Exception:
                    pass  # Continue without acquisition params if JSON loading fails
            
            anat_file = AnatomicalFile(
                path=anat_path,
                json_path=json_path,
                acquisition_params=acquisition_params
            )
            anat_files.append(anat_file)
        
        return anat_files
    
    def create_bids_derivatives(self, output_dir: Path, pipeline_name: str) -> Path:
        """Create BIDS derivatives structure for elikopy outputs
        
        Args:
            output_dir: Base output directory
            pipeline_name: Name of the pipeline
            
        Returns:
            Path to derivatives directory
        """
        # Create derivatives directory
        derivatives_dir = output_dir / "derivatives" / pipeline_name
        derivatives_dir.mkdir(exist_ok=True, parents=True)
        
        # Create dataset_description.json
        dataset_desc = {
            "Name": f"{pipeline_name} outputs",
            "BIDSVersion": "1.4.0",
            "PipelineDescription": {
                "Name": pipeline_name,
                "Version": "0.5.0",
                "Description": f"Outputs of {pipeline_name} processing pipeline"
            },
            "GeneratedBy": [
                {
                    "Name": "elikopy",
                    "Version": "0.5.0",
                    "CodeURL": "https://github.com/Hyedryn/elikopy"
                }
            ]
        }
        
        # Write dataset_description.json
        with open(derivatives_dir / "dataset_description.json", "w") as f:
            json.dump(dataset_desc, f, indent=2)
        
        return derivatives_dir
    
    def create_derivatives_structure(self, pipeline_name: str) -> Path:
        """Create BIDS derivatives structure for elikopy outputs in BIDS root
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            Path to derivatives directory
        """
        return self.create_bids_derivatives(self.bids_root, pipeline_name)
    
    def get_bids_layout(self) -> BIDSLayout:
        """Get pybids BIDSLayout object for advanced queries
        
        Returns:
            BIDSLayout object for the main BIDS dataset
        """
        if not self._layout:
            self._initialize_layouts()
        return self._layout
    
    def get_qsiprep_layout(self) -> Optional[BIDSLayout]:
        """Get pybids BIDSLayout object for qsiprep derivatives
        
        Returns:
            BIDSLayout object for qsiprep derivatives, or None if not available
        """
        if not self._qsiprep_layout and self.qsiprep_dir:
            self._initialize_layouts()
        return self._qsiprep_layout