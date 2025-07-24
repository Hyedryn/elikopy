"""
DataValidator class - Data validation utilities
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple


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
    """Class for data validation"""
    
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
        if not dwi_file.exists():
            result.add_error(f"DWI file not found: {dwi_file}")
            return result
        
        if not bval_file.exists():
            result.add_error(f"bval file not found: {bval_file}")
            return result
        
        if not bvec_file.exists():
            result.add_error(f"bvec file not found: {bvec_file}")
            return result
        
        if json_file and not json_file.exists():
            result.add_warning(f"JSON file not found: {json_file}")
        
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
        result = ValidationResult()
        
        # Validate based on processing type
        if 'processing_type' not in params:
            result.add_error("Missing required parameter: processing_type")
            return result
        
        processing_type = params['processing_type']
        
        if processing_type == 'dti':
            # Validate DTI parameters
            if 'fit_method' in params:
                if params['fit_method'] not in ['WLS', 'OLS', 'NLLS']:
                    result.add_error(f"Invalid fit_method: {params['fit_method']}")
            
            if 'mask_threshold' in params:
                if not 0 <= params['mask_threshold'] <= 1:
                    result.add_error(f"mask_threshold must be between 0 and 1, got {params['mask_threshold']}")
        
        elif processing_type == 'noddi':
            # Validate NODDI parameters
            if 'fit_method' in params:
                if params['fit_method'] not in ['amico', 'noddi-python']:
                    result.add_error(f"Invalid fit_method: {params['fit_method']}")
        
        elif processing_type == 'csd':
            # Validate CSD parameters
            if 'response_method' in params:
                if params['response_method'] not in ['tournier', 'tax', 'dhollander']:
                    result.add_error(f"Invalid response_method: {params['response_method']}")
            
            if 'sh_order' in params:
                if not 2 <= params['sh_order'] <= 12 or params['sh_order'] % 2 != 0:
                    result.add_error(f"sh_order must be an even number between 2 and 12, got {params['sh_order']}")
        
        elif processing_type == 'tracking':
            # Validate tracking parameters
            if 'algorithm' in params:
                if params['algorithm'] not in ['deterministic', 'probabilistic']:
                    result.add_error(f"Invalid algorithm: {params['algorithm']}")
            
            if 'step_size' in params:
                if params['step_size'] <= 0:
                    result.add_error(f"step_size must be positive, got {params['step_size']}")
            
            if 'max_angle' in params:
                if not 0 <= params['max_angle'] <= 90:
                    result.add_error(f"max_angle must be between 0 and 90, got {params['max_angle']}")
            
            if 'min_length' in params and 'max_length' in params:
                if params['min_length'] >= params['max_length']:
                    result.add_error(f"min_length ({params['min_length']}) must be less than max_length ({params['max_length']})")
        
        return result