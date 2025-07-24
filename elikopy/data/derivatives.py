"""
DerivativesManager class - BIDS derivatives management
"""

import json
import datetime
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ..core.base import BIDSComponent, ValidationResult, ValidationError, ValidationWarning


class DerivativesManager(BIDSComponent):
    """Class for managing BIDS derivatives with automatic structure creation and metadata generation"""
    
    def __init__(self, bids_root: Path, pipeline_name: str = "elikopy", 
                 pipeline_version: str = "0.5.0"):
        """Initialize derivatives manager
        
        Args:
            bids_root: Path to BIDS root directory
            pipeline_name: Name of the pipeline
            pipeline_version: Version of the pipeline
        """
        self.bids_root = Path(bids_root)
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.derivatives_dir = self.bids_root / "derivatives" / pipeline_name
        
        # Validate BIDS root structure
        validation_result = self.validate_bids_structure(self.bids_root)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid BIDS structure: {validation_result.errors}")
        
        # Create derivatives directory structure
        self.create_bids_derivatives(self.bids_root, pipeline_name)
    
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
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, [ValidationError(msg) for msg in errors], 
                              [ValidationWarning(msg) for msg in warnings], [])
    
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
        
        # Create logs directory
        logs_dir = derivatives_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create dataset_description.json
        self._create_dataset_description(derivatives_dir)
        
        # Create README.md
        self._create_readme(derivatives_dir)
        
        return derivatives_dir
    
    def _create_dataset_description(self, derivatives_dir: Path) -> None:
        """Create dataset_description.json with comprehensive metadata"""
        dataset_desc_path = derivatives_dir / "dataset_description.json"
        
        if not dataset_desc_path.exists():
            dataset_desc = {
                "Name": f"{self.pipeline_name} outputs",
                "BIDSVersion": "1.4.0",
                "DatasetType": "derivative",
                "PipelineDescription": {
                    "Name": self.pipeline_name,
                    "Version": self.pipeline_version,
                    "Description": f"Diffusion MRI processing outputs from {self.pipeline_name}",
                    "CodeURL": "https://github.com/Hyedryn/elikopy"
                },
                "GeneratedBy": [
                    {
                        "Name": "elikopy",
                        "Version": self.pipeline_version,
                        "Description": "A Python library for diffusion MRI processing and analysis",
                        "CodeURL": "https://github.com/Hyedryn/elikopy"
                    }
                ],
                "SourceDatasets": [],
                "HowToAcknowledge": "Please cite elikopy when using these outputs",
                "DatasetDOI": "",
                "License": "AGPL-3.0"
            }
            
            with open(dataset_desc_path, "w") as f:
                json.dump(dataset_desc, f, indent=2)
    
    def _create_readme(self, derivatives_dir: Path) -> None:
        """Create README.md with processing information"""
        readme_path = derivatives_dir / "README.md"
        
        if not readme_path.exists():
            readme_content = f"""# {self.pipeline_name} Derivatives

This directory contains diffusion MRI processing outputs generated by elikopy v{self.pipeline_version}.

## Contents

This derivatives dataset contains the following processing outputs:

- **DTI**: Diffusion tensor imaging metrics (FA, MD, RD, AD)
- **NODDI**: Neurite Orientation Dispersion and Density Imaging parameters
- **CSD**: Constrained Spherical Deconvolution fiber orientation distributions
- **MSMT-CSD**: Multi-shell multi-tissue CSD outputs
- **Microstructure Fingerprinting**: Microstructural parameter maps
- **Tractography**: White matter tractography streamlines
- **Connectivity**: Structural connectivity matrices

## File Organization

Files are organized following BIDS derivatives conventions:

```
derivatives/{self.pipeline_name}/
├── dataset_description.json
├── README.md
├── logs/
└── sub-<subject>/
    └── [ses-<session>/]
        ├── dwi/
        ├── dti/
        ├── noddi/
        ├── csd/
        ├── msmtcsd/
        ├── fingerprinting/
        ├── tractography/
        └── connectivity/
```

## Processing Information

- **Pipeline**: {self.pipeline_name} v{self.pipeline_version}
- **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Platform**: {platform.system()} {platform.release()}
- **Python**: {platform.python_version()}

## Citation

If you use these outputs in your research, please cite:

```
[Citation information for elikopy]
```

For more information, visit: https://github.com/Hyedryn/elikopy
"""
            
            with open(readme_path, "w") as f:
                f.write(readme_content)
    
    def get_subject_dir(self, subject_id: str, session_id: Optional[str] = None, 
                       create: bool = True) -> Path:
        """Get subject directory in derivatives with automatic creation
        
        Args:
            subject_id: Subject ID (with or without 'sub-' prefix)
            session_id: Session ID (with or without 'ses-' prefix, optional)
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Path to subject directory
        """
        # Normalize subject ID
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"
        
        # Create subject directory
        subject_dir = self.derivatives_dir / subject_id
        if create:
            subject_dir.mkdir(exist_ok=True, parents=True)
        
        # Handle session if provided
        if session_id:
            # Normalize session ID
            if not session_id.startswith("ses-"):
                session_id = f"ses-{session_id}"
            
            # Create session directory
            session_dir = subject_dir / session_id
            if create:
                session_dir.mkdir(exist_ok=True, parents=True)
            
            return session_dir
        
        return subject_dir
    
    def get_modality_dir(self, subject_id: str, modality: str, 
                        session_id: Optional[str] = None, create: bool = True) -> Path:
        """Get modality directory in derivatives with automatic creation
        
        Args:
            subject_id: Subject ID (with or without 'sub-' prefix)
            modality: Modality name (e.g., dwi, anat, dti, noddi, csd, etc.)
            session_id: Session ID (with or without 'ses-' prefix, optional)
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Path to modality directory
        """
        # Get subject directory
        subject_dir = self.get_subject_dir(subject_id, session_id, create=create)
        
        # Create modality directory
        modality_dir = subject_dir / modality
        if create:
            modality_dir.mkdir(exist_ok=True, parents=True)
        
        return modality_dir
    
    def get_output_path(self, subject_id: str, modality: str, suffix: str,
                       extension: str = ".nii.gz", session_id: Optional[str] = None,
                       create_dir: bool = True, **entities) -> Path:
        """Get output file path following BIDS conventions with automatic directory creation
        
        Args:
            subject_id: Subject ID (with or without 'sub-' prefix)
            modality: Modality name (e.g., dwi, anat, dti, noddi, csd, etc.)
            suffix: File suffix (e.g., FA, MD, dwi, T1w)
            extension: File extension (default: .nii.gz)
            session_id: Session ID (with or without 'ses-' prefix, optional)
            create_dir: Whether to create directory structure
            **entities: Additional BIDS entities (e.g., run, task, model, desc, parameter)
            
        Returns:
            Path to output file following BIDS naming conventions
        """
        # Get modality directory
        modality_dir = self.get_modality_dir(subject_id, modality, session_id, create=create_dir)
        
        # Normalize subject ID
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"
        
        # Build filename following BIDS entity order
        filename_parts = [subject_id]
        
        # Add session if provided
        if session_id:
            if not session_id.startswith("ses-"):
                session_id = f"ses-{session_id}"
            filename_parts.append(session_id)
        
        # Define BIDS entity order for consistent naming
        entity_order = [
            'task', 'acq', 'ce', 'dir', 'rec', 'run', 'mod', 'echo', 'flip', 
            'inv', 'mt', 'part', 'proc', 'space', 'desc', 'model', 'parameter'
        ]
        
        # Add entities in BIDS order
        for entity in entity_order:
            if entity in entities and entities[entity] is not None:
                filename_parts.append(f"{entity}-{entities[entity]}")
        
        # Add any remaining entities not in standard order
        for key, value in sorted(entities.items()):
            if key not in entity_order and value is not None:
                filename_parts.append(f"{key}-{value}")
        
        # Build filename
        filename = "_".join(filename_parts) + f"_{suffix}{extension}"
        
        return modality_dir / filename
    
    def create_provenance(self, output_file: Path, inputs: List[Path], 
                         parameters: Dict[str, Any], processing_info: Optional[Dict[str, Any]] = None) -> Path:
        """Create comprehensive provenance sidecar JSON file
        
        Args:
            output_file: Path to output file
            inputs: List of input file paths
            parameters: Processing parameters
            processing_info: Additional processing information
            
        Returns:
            Path to created JSON sidecar file
        """
        # Create comprehensive provenance data
        provenance = {
            "Sources": [],
            "Parameters": parameters,
            "ProcessedBy": {
                "Name": "elikopy",
                "Version": self.pipeline_version,
                "Description": "A Python library for diffusion MRI processing and analysis",
                "CodeURL": "https://github.com/Hyedryn/elikopy"
            },
            "ProcessingDate": datetime.datetime.now().isoformat(),
            "Environment": {
                "Platform": platform.system(),
                "PlatformVersion": platform.release(),
                "PythonVersion": platform.python_version(),
                "WorkingDirectory": str(Path.cwd())
            }
        }
        
        # Add source files (relative to BIDS root if possible)
        for input_path in inputs:
            try:
                # Try to make path relative to BIDS root
                relative_path = input_path.relative_to(self.bids_root)
                provenance["Sources"].append(str(relative_path))
            except ValueError:
                # If not under BIDS root, use absolute path
                provenance["Sources"].append(str(input_path))
        
        # Add processing information if provided
        if processing_info:
            provenance.update(processing_info)
        
        # Create JSON sidecar file
        json_file = output_file.with_suffix(".json")
        json_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(json_file, "w") as f:
            json.dump(provenance, f, indent=2)
        
        return json_file
    
    def create_readme(self, content: str) -> Path:
        """Create README file in derivatives directory
        
        Args:
            content: README content
            
        Returns:
            Path to created README file
        """
        readme_path = self.derivatives_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(content)
        return readme_path
    
    def update_dataset_description(self, updates: Dict[str, Any]) -> None:
        """Update dataset_description.json with new information
        
        Args:
            updates: Dictionary of updates to apply
        """
        dataset_desc_path = self.derivatives_dir / "dataset_description.json"
        
        # Read existing dataset_description.json
        if dataset_desc_path.exists():
            with open(dataset_desc_path, "r") as f:
                dataset_desc = json.load(f)
        else:
            dataset_desc = {}
        
        # Apply updates
        dataset_desc.update(updates)
        
        # Write updated dataset_description.json
        with open(dataset_desc_path, "w") as f:
            json.dump(dataset_desc, f, indent=2)
    
    def create_pipeline_description(self, name: str, version: str, 
                                  description: str, code_url: Optional[str] = None) -> None:
        """Update pipeline description in dataset_description.json
        
        Args:
            name: Pipeline name
            version: Pipeline version
            description: Pipeline description
            code_url: URL to pipeline code (optional)
        """
        pipeline_desc = {
            "Name": name,
            "Version": version,
            "Description": description
        }
        
        if code_url:
            pipeline_desc["CodeURL"] = code_url
        
        updates = {"PipelineDescription": pipeline_desc}
        self.update_dataset_description(updates)
    
    def add_source_dataset(self, dataset_name: str, dataset_url: Optional[str] = None,
                          dataset_version: Optional[str] = None) -> None:
        """Add source dataset information to dataset_description.json
        
        Args:
            dataset_name: Name of source dataset
            dataset_url: URL to source dataset (optional)
            dataset_version: Version of source dataset (optional)
        """
        dataset_desc_path = self.derivatives_dir / "dataset_description.json"
        
        # Read existing dataset_description.json
        with open(dataset_desc_path, "r") as f:
            dataset_desc = json.load(f)
        
        # Initialize SourceDatasets if not present
        if "SourceDatasets" not in dataset_desc:
            dataset_desc["SourceDatasets"] = []
        
        # Create source dataset entry
        source_dataset = {"Name": dataset_name}
        if dataset_url:
            source_dataset["URL"] = dataset_url
        if dataset_version:
            source_dataset["Version"] = dataset_version
        
        # Add to SourceDatasets if not already present
        if source_dataset not in dataset_desc["SourceDatasets"]:
            dataset_desc["SourceDatasets"].append(source_dataset)
        
        # Write updated dataset_description.json
        with open(dataset_desc_path, "w") as f:
            json.dump(dataset_desc, f, indent=2)
    
    def create_processing_log(self, log_data: Dict[str, Any]) -> Path:
        """Create processing log file in logs directory
        
        Args:
            log_data: Processing log data
            
        Returns:
            Path to created log file
        """
        logs_dir = self.derivatives_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"processing_log_{timestamp}.json"
        
        # Add timestamp to log data
        log_data["timestamp"] = datetime.datetime.now().isoformat()
        log_data["pipeline"] = {
            "name": self.pipeline_name,
            "version": self.pipeline_version
        }
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        return log_file
    
    def get_derivatives_structure(self) -> Dict[str, Any]:
        """Get overview of derivatives directory structure
        
        Returns:
            Dictionary describing the derivatives structure
        """
        structure = {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "derivatives_dir": str(self.derivatives_dir),
            "subjects": [],
            "modalities": set(),
            "total_files": 0
        }
        
        if not self.derivatives_dir.exists():
            return structure
        
        # Scan subjects
        for subject_dir in self.derivatives_dir.glob("sub-*"):
            if subject_dir.is_dir():
                subject_info = {
                    "subject_id": subject_dir.name,
                    "sessions": [],
                    "modalities": set()
                }
                
                # Check for sessions
                session_dirs = list(subject_dir.glob("ses-*"))
                if session_dirs:
                    for session_dir in session_dirs:
                        session_info = {
                            "session_id": session_dir.name,
                            "modalities": []
                        }
                        
                        # Get modalities in session
                        for modality_dir in session_dir.iterdir():
                            if modality_dir.is_dir():
                                session_info["modalities"].append(modality_dir.name)
                                subject_info["modalities"].add(modality_dir.name)
                                structure["modalities"].add(modality_dir.name)
                                
                                # Count files
                                structure["total_files"] += len(list(modality_dir.glob("*")))
                        
                        subject_info["sessions"].append(session_info)
                
                # Also check for modalities directly under subject (no sessions)
                for item in subject_dir.iterdir():
                    if item.is_dir() and not item.name.startswith("ses-"):
                        # This is a modality directory
                        subject_info["modalities"].add(item.name)
                        structure["modalities"].add(item.name)
                        
                        # Count files
                        structure["total_files"] += len(list(item.glob("*")))
                
                # Convert sets to lists for JSON serialization
                subject_info["modalities"] = list(subject_info["modalities"])
                structure["subjects"].append(subject_info)
        
        # Convert set to list
        structure["modalities"] = list(structure["modalities"])
        
        return structure