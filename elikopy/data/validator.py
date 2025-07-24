"""
DataValidator class - Data validation utilities for elikopy

This module provides comprehensive validation for DWI data, gradient tables,
BIDS compliance, and processing parameters.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np


class ValidationError:
    """Class representing a validation error"""
    
    def __init__(self, message: str, severity: str = "error"):
        """Initialize a validation error
        
        Args:
            message: Error message
            severity: Error severity (error, warning, info)
        """
        self.message = message
        self.severity = severity
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.severity.upper()}: {self.message}"


class ValidationResult:
    """Class representing a validation result"""
    
    def __init__(self):
        """Initialize a validation result"""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.info: List[ValidationError] = []
    
    def add_error(self, message: str) -> None:
        """Add an error
        
        Args:
            message: Error message
        """
        self.errors.append(ValidationError(message, "error"))
    
    def add_warning(self, message: str) -> None:
        """Add a warning
        
        Args:
            message: Warning message
        """
        self.warnings.append(ValidationError(message, "warning"))
    
    def add_info(self, message: str) -> None:
        """Add an info message
        
        Args:
            message: Info message
        """
        self.info.append(ValidationError(message, "info"))
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed
        
        Returns:
            True if no errors
        """
        return len(self.errors) == 0
    
    def __str__(self) -> str:
        """String representation"""
        result = []
        
        if self.errors:
            result.append("Errors:")
            for error in self.errors:
                result.append(f"  - {error}")
        
        if self.warnings:
            result.append("Warnings:")
            for warning in self.warnings:
                result.append(f"  - {warning}")
        
        if self.info:
            result.append("Info:")
            for info in self.info:
                result.append(f"  - {info}")
        
        if not result:
            result.append("Validation passed with no issues.")
        
        return "\n".join(result)


class DataValidator:
    """Comprehensive data validation for DWI processing
    
    This class provides validation for:
    - DWI data integrity (file formats, dimensions, gradients)
    - bvals/bvecs validation with gradient table checks
    - BIDS compliance validation with detailed error reporting
    - Processing parameter validation
    """
    
    # BIDS-compliant file extensions
    VALID_DWI_EXTENSIONS = ['.nii', '.nii.gz']
    VALID_BVAL_EXTENSIONS = ['.bval']
    VALID_BVEC_EXTENSIONS = ['.bvec']
    VALID_JSON_EXTENSIONS = ['.json']
    
    # Required BIDS metadata fields for DWI
    REQUIRED_DWI_METADATA = [
        'EffectiveEchoSpacing',
        'PhaseEncodingDirection'
    ]
    
    # Recommended BIDS metadata fields for DWI
    RECOMMENDED_DWI_METADATA = [
        'TotalReadoutTime',
        'SliceTiming',
        'RepetitionTime',
        'EchoTime',
        'FlipAngle'
    ]
    
    # BIDS entity patterns
    BIDS_ENTITIES = {
        'sub': r'sub-[a-zA-Z0-9]+',
        'ses': r'ses-[a-zA-Z0-9]+',
        'task': r'task-[a-zA-Z0-9]+',
        'acq': r'acq-[a-zA-Z0-9]+',
        'ce': r'ce-[a-zA-Z0-9]+',
        'rec': r'rec-[a-zA-Z0-9]+',
        'dir': r'dir-[a-zA-Z0-9]+',
        'run': r'run-[0-9]+',
        'mod': r'mod-[a-zA-Z0-9]+',
        'echo': r'echo-[0-9]+',
        'flip': r'flip-[0-9]+',
        'inv': r'inv-[0-9]+',
        'mt': r'mt-[a-zA-Z0-9]+',
        'part': r'part-[a-zA-Z0-9]+',
        'recording': r'recording-[a-zA-Z0-9]+'
    }
    
    # Valid BIDS suffixes for DWI
    VALID_DWI_SUFFIXES = ['dwi']
    
    # Valid phase encoding directions
    VALID_PHASE_ENCODING_DIRECTIONS = ['i', 'j', 'k', 'i-', 'j-', 'k-']
    
    def validate_dwi_data(self, dwi_file: Path, bval_file: Path, bvec_file: Path, 
                         json_file: Optional[Path] = None) -> ValidationResult:
        """Validate DWI data integrity
        
        Args:
            dwi_file: Path to DWI file
            bval_file: Path to bval file
            bvec_file: Path to bvec file
            json_file: Path to JSON file (optional)
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult()
        
        # Check if files exist
        files_missing = False
        if not dwi_file.exists():
            result.add_error(f"DWI file not found: {dwi_file}")
            files_missing = True
        
        if not bval_file.exists():
            result.add_error(f"bval file not found: {bval_file}")
            files_missing = True
        
        if not bvec_file.exists():
            result.add_error(f"bvec file not found: {bvec_file}")
            files_missing = True
        
        if json_file and not json_file.exists():
            result.add_warning(f"JSON file not found: {json_file}")
        
        # If any required files are missing, return early
        if files_missing:
            return result
        
        # Validate DWI file
        try:
            import nibabel as nib
            img = nib.load(str(dwi_file))
            
            # Check dimensions
            if len(img.shape) < 4:
                result.add_error(f"DWI file has incorrect dimensions: {img.shape}")
            
            # Check datatype
            if img.get_data_dtype() not in ['int16', 'float32', 'float64']:
                result.add_warning(f"Unusual datatype for DWI: {img.get_data_dtype()}")
        except Exception as e:
            result.add_error(f"Error reading DWI file: {e}")
        
        # Validate bval/bvec files
        try:
            # Read bval file
            with open(bval_file, 'r') as f:
                bvals = f.read().strip().split()
            
            # Read bvec file
            with open(bvec_file, 'r') as f:
                bvec_lines = f.readlines()
                if len(bvec_lines) != 3:
                    result.add_error(f"bvec file should have 3 lines, found {len(bvec_lines)}")
                    return result
                
                bvecs = [line.strip().split() for line in bvec_lines]
            
            # Check lengths
            if len(bvals) != len(bvecs[0]) or len(bvals) != len(bvecs[1]) or len(bvals) != len(bvecs[2]):
                result.add_error(f"Mismatch in bval/bvec lengths: bval={len(bvals)}, bvec=({len(bvecs[0])}, {len(bvecs[1])}, {len(bvecs[2])})")
            
            # Check if DWI dimensions match bvals/bvecs
            try:
                if img.shape[3] != len(bvals):
                    result.add_error(f"Mismatch between DWI volumes ({img.shape[3]}) and bvals/bvecs ({len(bvals)})")
            except:
                pass
            
            # Check for b0 volumes
            b0_count = sum(1 for b in bvals if float(b) < 50)
            if b0_count == 0:
                result.add_error("No b0 volumes found in bvals")
            elif b0_count < 3:
                result.add_warning(f"Only {b0_count} b0 volumes found, recommended at least 3")
            
            # Check for non-unit vectors in bvecs
            for i, (x, y, z) in enumerate(zip(bvecs[0], bvecs[1], bvecs[2])):
                try:
                    x, y, z = float(x), float(y), float(z)
                    norm = (x**2 + y**2 + z**2)**0.5
                    
                    # Skip b0 volumes
                    if float(bvals[i]) < 50:
                        continue
                    
                    if abs(norm - 1.0) > 0.01:
                        result.add_warning(f"Non-unit vector at index {i}: norm = {norm}")
                except:
                    result.add_error(f"Invalid bvec value at index {i}")
        except Exception as e:
            result.add_error(f"Error validating bval/bvec files: {e}")
        
        # Validate JSON file if provided
        if json_file and json_file.exists():
            try:
                import json
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check for required fields
                required_fields = ['EffectiveEchoSpacing', 'PhaseEncodingDirection']
                for field in required_fields:
                    if field not in metadata:
                        result.add_warning(f"Missing field in JSON metadata: {field}")
            except Exception as e:
                result.add_warning(f"Error reading JSON file: {e}")
        
        return result
    
    def validate_bvals_bvecs(self, bval_file: Path, bvec_file: Path) -> ValidationResult:
        """Validate gradient information
        
        Args:
            bval_file: Path to bval file
            bvec_file: Path to bvec file
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult()
        
        # Check if files exist
        if not bval_file.exists():
            result.add_error(f"bval file not found: {bval_file}")
            return result
        
        if not bvec_file.exists():
            result.add_error(f"bvec file not found: {bvec_file}")
            return result
        
        # Validate bval/bvec files
        try:
            # Read bval file
            with open(bval_file, 'r') as f:
                bvals = f.read().strip().split()
            
            # Read bvec file
            with open(bvec_file, 'r') as f:
                bvec_lines = f.readlines()
                if len(bvec_lines) != 3:
                    result.add_error(f"bvec file should have 3 lines, found {len(bvec_lines)}")
                    return result
                
                bvecs = [line.strip().split() for line in bvec_lines]
            
            # Check lengths
            if len(bvals) != len(bvecs[0]) or len(bvals) != len(bvecs[1]) or len(bvals) != len(bvecs[2]):
                result.add_error(f"Mismatch in bval/bvec lengths: bval={len(bvals)}, bvec=({len(bvecs[0])}, {len(bvecs[1])}, {len(bvecs[2])})")
            
            # Check for b0 volumes
            b0_count = sum(1 for b in bvals if float(b) < 50)
            if b0_count == 0:
                result.add_error("No b0 volumes found in bvals")
            elif b0_count < 3:
                result.add_warning(f"Only {b0_count} b0 volumes found, recommended at least 3")
            
            # Check for non-unit vectors in bvecs
            for i, (x, y, z) in enumerate(zip(bvecs[0], bvecs[1], bvecs[2])):
                try:
                    x, y, z = float(x), float(y), float(z)
                    norm = (x**2 + y**2 + z**2)**0.5
                    
                    # Skip b0 volumes
                    if float(bvals[i]) < 50:
                        continue
                    
                    if abs(norm - 1.0) > 0.01:
                        result.add_warning(f"Non-unit vector at index {i}: norm = {norm}")
                except:
                    result.add_error(f"Invalid bvec value at index {i}")
            
            # Check for shell structure
            shells = {}
            for b in bvals:
                b_val = int(float(b) / 100) * 100  # Round to nearest 100
                shells[b_val] = shells.get(b_val, 0) + 1
            
            # Report shell structure
            shell_info = ", ".join([f"b={shell}: {count} volumes" for shell, count in shells.items()])
            result.add_info(f"Shell structure: {shell_info}")
            
            # Check for sufficient directions per shell
            for shell, count in shells.items():
                if shell > 0 and count < 6:
                    result.add_warning(f"Shell b={shell} has only {count} volumes, recommended at least 6")
        except Exception as e:
            result.add_error(f"Error validating bval/bvec files: {e}")
        
        return result
    
    def validate_processing_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate processing parameters
        
        Args:
            params: Processing parameters
            
        Returns:
            ValidationResult object
        """
        # Import here to avoid circular imports
        from elikopy.utils.validation import ParameterValidator
        
        # Use the new comprehensive parameter validator
        param_validator = ParameterValidator()
        
        # Validate based on processing type
        if 'processing_type' not in params:
            result = ValidationResult()
            result.add_error("Missing required parameter: processing_type")
            return result
        
        processing_type = params['processing_type']
        
        # Remove processing_type from params for validation
        validation_params = {k: v for k, v in params.items() if k != 'processing_type'}
        
        # Use the comprehensive parameter validator
        validation_result = param_validator.validate_processing_parameters(
            processing_type, validation_params
        )
        
        # Convert to our ValidationResult format
        result = ValidationResult()
        
        for error in validation_result.errors:
            result.add_error(error.message)
        
        for warning in validation_result.warnings:
            result.add_warning(warning.message)
        
        for suggestion in validation_result.suggestions:
            result.add_info(suggestion)
        
        return result
    
    def validate_bids_compliance(self, file_path: Path, dataset_root: Optional[Path] = None) -> ValidationResult:
        """Validate BIDS compliance with detailed error reporting
        
        Args:
            file_path: Path to the file to validate
            dataset_root: Path to the BIDS dataset root (optional)
            
        Returns:
            ValidationResult object with detailed BIDS compliance information
        """
        result = ValidationResult()
        
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            result.add_error(f"File not found: {file_path}")
            return result
        
        # Get filename and extension
        filename = file_path.name
        stem = file_path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]  # Remove .nii from .nii.gz files
        
        # Validate file extension
        valid_extensions = (self.VALID_DWI_EXTENSIONS + self.VALID_BVAL_EXTENSIONS + 
                          self.VALID_BVEC_EXTENSIONS + self.VALID_JSON_EXTENSIONS)
        
        if not any(filename.endswith(ext) for ext in valid_extensions):
            result.add_error(f"Invalid file extension: {filename}")
        
        # Parse BIDS filename
        bids_info = self._parse_bids_filename(filename)
        
        # Validate subject entity (required)
        if 'sub' not in bids_info:
            result.add_error(f"Missing required 'sub' entity in filename: {filename}")
        else:
            if not re.match(self.BIDS_ENTITIES['sub'], bids_info['sub']):
                result.add_error(f"Invalid subject entity format: {bids_info['sub']}")
        
        # Validate other entities if present
        for entity, value in bids_info.items():
            if entity in self.BIDS_ENTITIES:
                if not re.match(self.BIDS_ENTITIES[entity], value):
                    result.add_error(f"Invalid {entity} entity format: {value}")
        
        # Validate suffix for DWI files
        if filename.endswith(tuple(self.VALID_DWI_EXTENSIONS)):
            if 'suffix' not in bids_info or bids_info['suffix'] not in self.VALID_DWI_SUFFIXES:
                result.add_error(f"Invalid or missing suffix for DWI file: {filename}")
        
        # Check for associated files (bval, bvec, json) for DWI files
        if filename.endswith(tuple(self.VALID_DWI_EXTENSIONS)) and bids_info.get('suffix') == 'dwi':
            base_name = filename
            for ext in self.VALID_DWI_EXTENSIONS:
                if base_name.endswith(ext):
                    base_name = base_name[:-len(ext)]
                    break
            
            # Check for bval file
            bval_file = file_path.parent / f"{base_name}.bval"
            if not bval_file.exists():
                result.add_error(f"Missing associated bval file: {bval_file}")
            
            # Check for bvec file
            bvec_file = file_path.parent / f"{base_name}.bvec"
            if not bvec_file.exists():
                result.add_error(f"Missing associated bvec file: {bvec_file}")
            
            # Check for json file (recommended)
            json_file = file_path.parent / f"{base_name}.json"
            if not json_file.exists():
                result.add_warning(f"Missing recommended JSON file: {json_file}")
            else:
                # Validate JSON metadata
                json_result = self._validate_dwi_json_metadata(json_file)
                result.errors.extend(json_result.errors)
                result.warnings.extend(json_result.warnings)
                result.info.extend(json_result.info)
        
        # Validate directory structure if dataset_root is provided
        if dataset_root:
            structure_result = self._validate_bids_directory_structure(file_path, dataset_root)
            result.errors.extend(structure_result.errors)
            result.warnings.extend(structure_result.warnings)
            result.info.extend(structure_result.info)
        
        return result
    
    def _parse_bids_filename(self, filename: str) -> Dict[str, str]:
        """Parse BIDS filename into entities
        
        Args:
            filename: BIDS filename
            
        Returns:
            Dictionary of BIDS entities
        """
        entities = {}
        
        # Remove extension
        name = filename
        for ext in ['.nii.gz', '.nii', '.bval', '.bvec', '.json']:
            if name.endswith(ext):
                name = name[:-len(ext)]
                break
        
        # Split by underscores
        parts = name.split('_')
        
        # Last part is the suffix
        if parts:
            entities['suffix'] = parts[-1]
            parts = parts[:-1]
        
        # Parse entities
        for part in parts:
            if '-' in part:
                key, value = part.split('-', 1)
                entities[key] = f"{key}-{value}"
        
        return entities
    
    def _validate_dwi_json_metadata(self, json_file: Path) -> ValidationResult:
        """Validate DWI JSON metadata
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult()
        
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Check required fields
            for field in self.REQUIRED_DWI_METADATA:
                if field not in metadata:
                    result.add_error(f"Missing required metadata field: {field}")
                else:
                    # Validate specific fields
                    if field == 'PhaseEncodingDirection':
                        if metadata[field] not in self.VALID_PHASE_ENCODING_DIRECTIONS:
                            result.add_error(f"Invalid PhaseEncodingDirection: {metadata[field]}")
                    elif field == 'EffectiveEchoSpacing':
                        try:
                            ees = float(metadata[field])
                            if ees <= 0:
                                result.add_error(f"EffectiveEchoSpacing must be positive: {ees}")
                        except (ValueError, TypeError):
                            result.add_error(f"EffectiveEchoSpacing must be a number: {metadata[field]}")
            
            # Check recommended fields
            for field in self.RECOMMENDED_DWI_METADATA:
                if field not in metadata:
                    result.add_warning(f"Missing recommended metadata field: {field}")
            
            # Validate specific field values
            if 'RepetitionTime' in metadata:
                try:
                    tr = float(metadata['RepetitionTime'])
                    if tr <= 0:
                        result.add_error(f"RepetitionTime must be positive: {tr}")
                except (ValueError, TypeError):
                    result.add_error(f"RepetitionTime must be a number: {metadata['RepetitionTime']}")
            
            if 'EchoTime' in metadata:
                try:
                    te = float(metadata['EchoTime'])
                    if te <= 0:
                        result.add_error(f"EchoTime must be positive: {te}")
                except (ValueError, TypeError):
                    result.add_error(f"EchoTime must be a number: {metadata['EchoTime']}")
            
            if 'FlipAngle' in metadata:
                try:
                    fa = float(metadata['FlipAngle'])
                    if not 0 <= fa <= 180:
                        result.add_warning(f"FlipAngle outside typical range [0, 180]: {fa}")
                except (ValueError, TypeError):
                    result.add_error(f"FlipAngle must be a number: {metadata['FlipAngle']}")
        
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {e}")
        except Exception as e:
            result.add_error(f"Error reading JSON file: {e}")
        
        return result
    
    def _validate_bids_directory_structure(self, file_path: Path, dataset_root: Path) -> ValidationResult:
        """Validate BIDS directory structure
        
        Args:
            file_path: Path to the file
            dataset_root: Path to BIDS dataset root
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult()
        
        try:
            # Get relative path from dataset root
            rel_path = file_path.relative_to(dataset_root)
            path_parts = rel_path.parts
            
            # Check if file is in derivatives
            if len(path_parts) > 0 and path_parts[0] == 'derivatives':
                # Derivatives structure validation
                if len(path_parts) < 3:
                    result.add_error(f"Invalid derivatives structure: {rel_path}")
                else:
                    pipeline_name = path_parts[1]
                    result.add_info(f"File in derivatives pipeline: {pipeline_name}")
            else:
                # Raw data structure validation
                if len(path_parts) < 2:
                    result.add_error(f"File not in proper BIDS structure: {rel_path}")
                else:
                    # Check subject directory
                    subject_dir = path_parts[0]
                    if not subject_dir.startswith('sub-'):
                        result.add_error(f"Invalid subject directory: {subject_dir}")
                    
                    # Check datatype directory (if present)
                    if len(path_parts) > 2:
                        if path_parts[1].startswith('ses-'):
                            # Session-based structure
                            if len(path_parts) > 3:
                                datatype = path_parts[2]
                                if datatype not in ['anat', 'func', 'dwi', 'fmap', 'perf', 'meg', 'eeg', 'ieeg']:
                                    result.add_warning(f"Unknown datatype directory: {datatype}")
                        else:
                            # No session structure
                            datatype = path_parts[1]
                            if datatype not in ['anat', 'func', 'dwi', 'fmap', 'perf', 'meg', 'eeg', 'ieeg']:
                                result.add_warning(f"Unknown datatype directory: {datatype}")
            
            # Check for dataset_description.json at root
            dataset_desc = dataset_root / 'dataset_description.json'
            if not dataset_desc.exists():
                result.add_warning("Missing dataset_description.json at dataset root")
        
        except ValueError:
            result.add_error(f"File is not within the specified dataset root: {file_path}")
        except Exception as e:
            result.add_error(f"Error validating directory structure: {e}")
        
        return result