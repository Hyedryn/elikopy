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

from ..core.base import ValidationResult, ValidationError, ValidationWarning


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
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []
        
        # Check if files exist
        files_missing = False
        if not dwi_file.exists():
            errors.append(ValidationError(f"DWI file not found: {dwi_file}"))
            files_missing = True
        
        if not bval_file.exists():
            errors.append(ValidationError(f"bval file not found: {bval_file}"))
            files_missing = True
        
        if not bvec_file.exists():
            errors.append(ValidationError(f"bvec file not found: {bvec_file}"))
            files_missing = True
        
        if json_file and not json_file.exists():
            warnings.append(ValidationWarning(f"JSON file not found: {json_file}"))
        
        # If any required files are missing, return early
        if files_missing:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, suggestions=[])
        
        # Validate DWI file
        try:
            import nibabel as nib
            img = nib.load(str(dwi_file))
            
            # Check dimensions
            if len(img.shape) < 4:
                errors.append(ValidationError(f"DWI file has incorrect dimensions: {img.shape}"))
            
            # Check datatype
            if img.get_data_dtype() not in ['int16', 'float32', 'float64']:
                warnings.append(ValidationWarning(f"Unusual datatype for DWI: {img.get_data_dtype()}"))
        except Exception as e:
            errors.append(ValidationError(f"Error reading DWI file: {e}"))
        
        # Validate bval/bvec files
        try:
            # Read bval file
            with open(bval_file, 'r') as f:
                bvals = f.read().strip().split()
            
            # Read bvec file
            with open(bvec_file, 'r') as f:
                bvec_lines = f.readlines()
                if len(bvec_lines) != 3:
                    errors.append(ValidationError(f"bvec file should have 3 lines, found {len(bvec_lines)}"))
                    return ValidationResult(is_valid=False, errors=errors, warnings=warnings, suggestions=[])
                
                bvecs = [line.strip().split() for line in bvec_lines]
            
            # Check lengths
            if len(bvals) != len(bvecs[0]) or len(bvals) != len(bvecs[1]) or len(bvals) != len(bvecs[2]):
                errors.append(ValidationError(f"Mismatch in bval/bvec lengths: bval={len(bvals)}, bvec=({len(bvecs[0])}, {len(bvecs[1])}, {len(bvecs[2])})"))
            
            # Check if DWI dimensions match bvals/bvecs
            try:
                if 'img' in locals() and img.shape[3] != len(bvals):
                    errors.append(ValidationError(f"Mismatch between DWI volumes ({img.shape[3]}) and bvals/bvecs ({len(bvals)})"))
            except:
                pass
            
            # Check for b0 volumes
            b0_count = sum(1 for b in bvals if float(b) < 50)
            if b0_count == 0:
                errors.append(ValidationError("No b0 volumes found in bvals"))
            elif b0_count < 3:
                warnings.append(ValidationWarning(f"Only {b0_count} b0 volumes found, recommended at least 3"))
            
            # Check for non-unit vectors in bvecs
            for i, (x, y, z) in enumerate(zip(bvecs[0], bvecs[1], bvecs[2])):
                try:
                    x, y, z = float(x), float(y), float(z)
                    norm = (x**2 + y**2 + z**2)**0.5
                    
                    # Skip b0 volumes
                    if float(bvals[i]) < 50:
                        continue
                    
                    if abs(norm - 1.0) > 0.01:
                        warnings.append(ValidationWarning(f"Non-unit vector at index {i}: norm = {norm}"))
                except:
                    errors.append(ValidationError(f"Invalid bvec value at index {i}"))
        except Exception as e:
            errors.append(ValidationError(f"Error validating bval/bvec files: {e}"))
        
        # Validate JSON file if provided
        if json_file and json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check for required fields
                for field in self.REQUIRED_DWI_METADATA:
                    if field not in metadata:
                        warnings.append(ValidationWarning(f"Missing field in JSON metadata: {field}"))
            except Exception as e:
                warnings.append(ValidationWarning(f"Error reading JSON file: {e}"))
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, suggestions=[])
    
    def validate_bvals_bvecs(self, bval_file: Path, bvec_file: Path) -> ValidationResult:
        """Validate gradient information
        
        Args:
            bval_file: Path to bval file
            bvec_file: Path to bvec file
            
        Returns:
            ValidationResult object
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []
        suggestions: List[str] = []

        # Check if files exist
        if not bval_file.exists():
            errors.append(ValidationError(f"bval file not found: {bval_file}"))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, suggestions=suggestions)
        
        if not bvec_file.exists():
            errors.append(ValidationError(f"bvec file not found: {bvec_file}"))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, suggestions=suggestions)
        
        # Validate bval/bvec files
        try:
            # Read bval file
            with open(bval_file, 'r') as f:
                bvals = f.read().strip().split()
            
            # Read bvec file
            with open(bvec_file, 'r') as f:
                bvec_lines = f.readlines()
                if len(bvec_lines) != 3:
                    errors.append(ValidationError(f"bvec file should have 3 lines, found {len(bvec_lines)}"))
                    return ValidationResult(is_valid=False, errors=errors, warnings=warnings, suggestions=suggestions)
                
                bvecs = [line.strip().split() for line in bvec_lines]
            
            # Check lengths
            if len(bvals) != len(bvecs[0]) or len(bvals) != len(bvecs[1]) or len(bvals) != len(bvecs[2]):
                errors.append(ValidationError(f"Mismatch in bval/bvec lengths: bval={len(bvals)}, bvec=({len(bvecs[0])}, {len(bvecs[1])}, {len(bvecs[2])})"))
            
            # Check for b0 volumes
            b0_count = sum(1 for b in bvals if float(b) < 50)
            if b0_count == 0:
                errors.append(ValidationError("No b0 volumes found in bvals"))
            elif b0_count < 3:
                warnings.append(ValidationWarning(f"Only {b0_count} b0 volumes found, recommended at least 3"))
            
            # Check for non-unit vectors in bvecs
            for i, (x, y, z) in enumerate(zip(bvecs[0], bvecs[1], bvecs[2])):
                try:
                    x, y, z = float(x), float(y), float(z)
                    norm = (x**2 + y**2 + z**2)**0.5
                    
                    # Skip b0 volumes
                    if float(bvals[i]) < 50:
                        continue
                    
                    if abs(norm - 1.0) > 0.01:
                        warnings.append(ValidationWarning(f"Non-unit vector at index {i}: norm = {norm}"))
                except:
                    errors.append(ValidationError(f"Invalid bvec value at index {i}"))
            
            # Check for shell structure
            shells = {}
            for b in bvals:
                b_val = int(float(b) / 100) * 100  # Round to nearest 100
                shells[b_val] = shells.get(b_val, 0) + 1
            
            # Report shell structure
            shell_info = ", ".join([f"b={shell}: {count} volumes" for shell, count in shells.items()])
            suggestions.append(f"Shell structure: {shell_info}")
            
            # Check for sufficient directions per shell
            for shell, count in shells.items():
                if shell > 0 and count < 6:
                    warnings.append(ValidationWarning(f"Shell b={shell} has only {count} volumes, recommended at least 6"))
        except Exception as e:
            errors.append(ValidationError(f"Error validating bval/bvec files: {e}"))
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, suggestions=suggestions)
    
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
            return ValidationResult(is_valid=False, errors=[ValidationError("Missing required parameter: processing_type")], warnings=[], suggestions=[])
        
        processing_type = params['processing_type']
        
        # Remove processing_type from params for validation
        validation_params = {k: v for k, v in params.items() if k != 'processing_type'}
        
        # Use the comprehensive parameter validator
        return param_validator.validate_processing_parameters(
            processing_type, validation_params
        )
    
    def validate_bids_compliance(self, file_path: Path, dataset_root: Optional[Path] = None) -> ValidationResult:
        """Validate BIDS compliance with detailed error reporting
        
        Args:
            file_path: Path to the file to validate
            dataset_root: Path to the BIDS dataset root (optional)
            
        Returns:
            ValidationResult object with detailed BIDS compliance information
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []
        suggestions: List[str] = []
        
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            errors.append(ValidationError(f"File not found: {file_path}"))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, suggestions=suggestions)
        
        # Get filename and extension
        filename = file_path.name
        
        # Validate file extension
        valid_extensions = (self.VALID_DWI_EXTENSIONS + self.VALID_BVAL_EXTENSIONS + 
                          self.VALID_BVEC_EXTENSIONS + self.VALID_JSON_EXTENSIONS)
        
        if not any(filename.endswith(ext) for ext in valid_extensions):
            errors.append(ValidationError(f"Invalid file extension: {filename}"))
        
        # Parse BIDS filename
        bids_info = self._parse_bids_filename(filename)
        
        # Validate subject entity (required)
        if 'sub' not in bids_info:
            errors.append(ValidationError(f"Missing required 'sub' entity in filename: {filename}"))
        else:
            if not re.match(self.BIDS_ENTITIES['sub'], f"sub-{bids_info['sub']}"):
                errors.append(ValidationError(f"Invalid subject entity format: {bids_info['sub']}"))
        
        # Validate other entities if present
        for entity, value in bids_info.items():
            if entity in self.BIDS_ENTITIES and entity != 'sub':
                if not re.match(self.BIDS_ENTITIES[entity], f"{entity}-{value}"):
                    errors.append(ValidationError(f"Invalid {entity} entity format: {value}"))
        
        # Validate suffix for DWI files
        if filename.endswith(tuple(self.VALID_DWI_EXTENSIONS)):
            if 'suffix' not in bids_info or bids_info['suffix'] not in self.VALID_DWI_SUFFIXES:
                errors.append(ValidationError(f"Invalid or missing suffix for DWI file: {filename}"))
        
        # Check for associated files (bval, bvec, json) for DWI files
        if filename.endswith(tuple(self.VALID_DWI_EXTENSIONS)) and bids_info.get('suffix') == 'dwi':
            base_name = filename.split('.')[0]
            
            # Check for bval file
            bval_file = file_path.parent / f"{base_name}.bval"
            if not bval_file.exists():
                errors.append(ValidationError(f"Missing associated bval file: {bval_file}"))
            
            # Check for bvec file
            bvec_file = file_path.parent / f"{base_name}.bvec"
            if not bvec_file.exists():
                errors.append(ValidationError(f"Missing associated bvec file: {bvec_file}"))
            
            # Check for json file (recommended)
            json_file = file_path.parent / f"{base_name}.json"
            if not json_file.exists():
                warnings.append(ValidationWarning(f"Missing recommended JSON file: {json_file}"))
            else:
                # Validate JSON metadata
                json_result = self._validate_dwi_json_metadata(json_file)
                errors.extend(json_result.errors)
                warnings.extend(json_result.warnings)
        
        # Validate directory structure if dataset_root is provided
        if dataset_root:
            structure_result = self._validate_bids_directory_structure(file_path, dataset_root)
            errors.extend(structure_result.errors)
            warnings.extend(structure_result.warnings)
            suggestions.extend(structure_result.suggestions)
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, suggestions=suggestions)
    
    def _parse_bids_filename(self, filename: str) -> Dict[str, str]:
        """Parse BIDS filename into entities
        
        Args:
            filename: BIDS filename
            
        Returns:
            Dictionary of BIDS entities
        """
        entities = {}
        
        # Remove extension
        name = filename.split('.')[0]
        
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
                entities[key] = value
        
        return entities
    
    def _validate_dwi_json_metadata(self, json_file: Path) -> ValidationResult:
        """Validate DWI JSON metadata
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            ValidationResult object
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []
        
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Check required fields
            for field in self.REQUIRED_DWI_METADATA:
                if field not in metadata:
                    warnings.append(ValidationWarning(f"Missing required metadata field: {field} in {json_file}"))
                else:
                    # Validate specific fields
                    if field == 'PhaseEncodingDirection':
                        if metadata[field] not in self.VALID_PHASE_ENCODING_DIRECTIONS:
                            errors.append(ValidationError(f"Invalid PhaseEncodingDirection '{metadata[field]}' in {json_file}"))
                    elif field == 'EffectiveEchoSpacing':
                        try:
                            ees = float(metadata[field])
                            if ees <= 0:
                                errors.append(ValidationError(f"EffectiveEchoSpacing must be positive in {json_file}, but got {ees}"))
                        except (ValueError, TypeError):
                            errors.append(ValidationError(f"EffectiveEchoSpacing must be a number in {json_file}, but got {metadata[field]}"))
            
            # Check recommended fields
            for field in self.RECOMMENDED_DWI_METADATA:
                if field not in metadata:
                    warnings.append(ValidationWarning(f"Missing recommended metadata field: {field} in {json_file}"))
            
            # Validate specific field values
            if 'RepetitionTime' in metadata:
                try:
                    tr = float(metadata['RepetitionTime'])
                    if tr <= 0:
                        errors.append(ValidationError(f"RepetitionTime must be positive in {json_file}, but got {tr}"))
                except (ValueError, TypeError):
                    errors.append(ValidationError(f"RepetitionTime must be a number in {json_file}, but got {metadata['RepetitionTime']}"))
            
            if 'EchoTime' in metadata:
                try:
                    te = float(metadata['EchoTime'])
                    if te <= 0:
                        errors.append(ValidationError(f"EchoTime must be positive in {json_file}, but got {te}"))
                except (ValueError, TypeError):
                    errors.append(ValidationError(f"EchoTime must be a number in {json_file}, but got {metadata['EchoTime']}"))
            
            if 'FlipAngle' in metadata:
                try:
                    fa = float(metadata['FlipAngle'])
                    if not 0 <= fa <= 180:
                        warnings.append(ValidationWarning(f"FlipAngle {fa} in {json_file} is outside the typical range of [0, 180]"))
                except (ValueError, TypeError):
                    errors.append(ValidationError(f"FlipAngle must be a number in {json_file}, but got {metadata['FlipAngle']}"))
        
        except json.JSONDecodeError as e:
            errors.append(ValidationError(f"Invalid JSON format in {json_file}: {e}"))
        except Exception as e:
            errors.append(ValidationError(f"Error reading or validating JSON file {json_file}: {e}"))
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, suggestions=[])
    
    def _validate_bids_directory_structure(self, file_path: Path, dataset_root: Path) -> ValidationResult:
        """Validate BIDS directory structure
        
        Args:
            file_path: Path to the file
            dataset_root: Path to the BIDS dataset root
            
        Returns:
            ValidationResult object
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []
        suggestions: List[str] = []
        
        try:
            # Get relative path from dataset root
            rel_path = file_path.relative_to(dataset_root)
            path_parts = rel_path.parts
            
            # Check if file is in derivatives
            if len(path_parts) > 0 and path_parts[0] == 'derivatives':
                # Derivatives structure validation
                if len(path_parts) < 3:
                    errors.append(ValidationError(f"Invalid derivatives structure: {rel_path}"))
                else:
                    pipeline_name = path_parts[1]
                    suggestions.append(f"File in derivatives pipeline: {pipeline_name}")
            else:
                # Raw data structure validation
                if len(path_parts) < 2:
                    errors.append(ValidationError(f"File not in a valid BIDS directory structure (e.g., sub-XXX/anat/): {rel_path}"))
                else:
                    # Check subject directory
                    subject_dir = path_parts[0]
                    if not re.match(r'sub-[a-zA-Z0-9]+', subject_dir):
                        errors.append(ValidationError(f"Invalid subject directory name: {subject_dir}"))
                    
                    # Check datatype directory (if present)
                    if len(path_parts) > 2:
                        if path_parts[1].startswith('ses-'):
                            # Session-based structure
                            if len(path_parts) > 3:
                                datatype = path_parts[2]
                                if datatype not in ['anat', 'func', 'dwi', 'fmap', 'perf', 'meg', 'eeg', 'ieeg']:
                                    warnings.append(ValidationWarning(f"Unknown datatype directory: {datatype}"))
                        else:
                            # No session structure
                            datatype = path_parts[1]
                            if datatype not in ['anat', 'func', 'dwi', 'fmap', 'perf', 'meg', 'eeg', 'ieeg']:
                                warnings.append(ValidationWarning(f"Unknown datatype directory: {datatype}"))
            
            # Check for dataset_description.json at root
            dataset_desc = dataset_root / 'dataset_description.json'
            if not dataset_desc.exists():
                warnings.append(ValidationWarning("Missing dataset_description.json at dataset root"))
        
        except ValueError:
            errors.append(ValidationError(f"File is not within the specified dataset root: {file_path}"))
        except Exception as e:
            errors.append(ValidationError(f"Error validating directory structure: {e}"))
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, suggestions=suggestions)
