"""
Validation utilities for ElikoPy
===============================

This module contains utility functions for parameter and data validation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import numpy as np


@dataclass
class ValidationError:
    """Represents a validation error"""
    message: str
    field: Optional[str] = None
    severity: str = "error"
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.severity.upper()}: {self.message}"


@dataclass
class ValidationWarning:
    """Represents a validation warning"""
    message: str
    field: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation"""
        return f"WARNING: {self.message}"


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    suggestions: List[str]
    
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
        
        if self.suggestions:
            result.append("Suggestions:")
            for suggestion in self.suggestions:
                result.append(f"  - {suggestion}")
        
        if not result:
            result.append("Validation passed with no issues.")
        
        return "\n".join(result)


class ParameterValidator:
    """Comprehensive parameter validation utilities for all processing steps"""
    
    # Valid parameter ranges and options for different processing types
    PARAMETER_SPECS = {
        'dti': {
            'fit_method': {
                'type': str,
                'valid_values': ['WLS', 'OLS', 'NLLS'],
                'default': 'WLS',
                'description': 'Tensor fitting method'
            },
            'mask_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'default': 0.2,
                'description': 'Threshold for brain mask creation'
            },
            'compute_metrics': {
                'type': list,
                'valid_values': ['FA', 'MD', 'AD', 'RD', 'MO', 'RGB'],
                'default': ['FA', 'MD', 'AD', 'RD'],
                'description': 'DTI metrics to compute'
            }
        },
        'noddi': {
            'fit_method': {
                'type': str,
                'valid_values': ['amico', 'noddi-python'],
                'default': 'amico',
                'description': 'NODDI fitting method'
            },
            'mask_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'default': 0.2,
                'description': 'Threshold for brain mask creation'
            },
            'compute_metrics': {
                'type': list,
                'valid_values': ['ICVF', 'ODI', 'ISOVF', 'FISO'],
                'default': ['ICVF', 'ODI', 'ISOVF'],
                'description': 'NODDI metrics to compute'
            }
        },
        'csd': {
            'response_method': {
                'type': str,
                'valid_values': ['tournier', 'tax', 'dhollander'],
                'default': 'tournier',
                'description': 'Response function estimation method'
            },
            'sh_order': {
                'type': int,
                'min_value': 2,
                'max_value': 12,
                'must_be_even': True,
                'default': 8,
                'description': 'Spherical harmonics order'
            },
            'mask_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'default': 0.2,
                'description': 'Threshold for brain mask creation'
            }
        },
        'msmt_csd': {
            'response_method': {
                'type': str,
                'valid_values': ['dhollander', 'msmt-5tt'],
                'default': 'dhollander',
                'description': 'Multi-tissue response function estimation method'
            },
            'sh_order': {
                'type': int,
                'min_value': 2,
                'max_value': 12,
                'must_be_even': True,
                'default': 8,
                'description': 'Spherical harmonics order'
            },
            'mask_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'default': 0.2,
                'description': 'Threshold for brain mask creation'
            }
        },
        'tracking': {
            'algorithm': {
                'type': str,
                'valid_values': ['deterministic', 'probabilistic'],
                'default': 'deterministic',
                'description': 'Tracking algorithm'
            },
            'step_size': {
                'type': float,
                'min_value': 0.1,
                'max_value': 2.0,
                'default': 0.5,
                'description': 'Step size in mm'
            },
            'max_angle': {
                'type': float,
                'min_value': 5.0,
                'max_value': 90.0,
                'default': 30.0,
                'description': 'Maximum turning angle in degrees'
            },
            'min_length': {
                'type': float,
                'min_value': 0.1,
                'default': 20.0,
                'description': 'Minimum streamline length in mm'
            },
            'max_length': {
                'type': float,
                'min_value': 1.0,
                'default': 200.0,
                'description': 'Maximum streamline length in mm'
            },
            'num_seeds': {
                'type': int,
                'min_value': 1,
                'default': 10000,
                'description': 'Number of seeds for tracking'
            },
            'seed_mask': {
                'type': str,
                'valid_values': ['wm', 'interface', 'gmwmi'],
                'default': 'wm',
                'description': 'Seeding mask type'
            },
            'sift_term_count': {
                'type': int,
                'min_value': 1,
                'default': 5000,
                'description': 'Target streamline count after SIFT'
            },
            'apply_sift': {
                'type': bool,
                'default': True,
                'description': 'Whether to apply SIFT filtering'
            }
        },
        'connectivity': {
            'atlas': {
                'type': str,
                'valid_values': ['aal', 'desikan', 'schaefer100', 'schaefer200', 'schaefer400'],
                'default': 'aal',
                'description': 'Atlas to use for connectivity analysis'
            },
            'measure': {
                'type': str,
                'valid_values': ['count', 'density', 'length', 'mean_length'],
                'default': 'count',
                'description': 'Connectivity measure to compute'
            }
        },
        'fingerprinting': {
            'dictionary_path': {
                'type': (str, type(None)),
                'file_must_exist': True,
                'optional': True,
                'description': 'Path to fingerprinting dictionary'
            },
            'mask_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'default': 0.2,
                'description': 'Threshold for brain mask creation'
            },
            'compute_metrics': {
                'type': list,
                'valid_values': ['fvf', 'diameter', 'orientation', 'dispersion', 'kappa'],
                'default': ['fvf', 'diameter', 'orientation'],
                'description': 'Fingerprinting metrics to compute'
            }
        },
        'scheduler': {
            'type': {
                'type': str,
                'valid_values': ['slurm', 'local'],
                'default': 'slurm',
                'description': 'Scheduler type'
            },
            'cpus_per_task': {
                'type': int,
                'min_value': 1,
                'max_value': 128,
                'default': 4,
                'description': 'Number of CPUs per task'
            },
            'mem_per_cpu': {
                'type': int,
                'min_value': 1,
                'max_value': 256,
                'default': 4,
                'description': 'Memory per CPU in GB'
            },
            'time_limit': {
                'type': str,
                'pattern': r'^\d{1,2}:[0-5]\d:[0-5]\d$',
                'default': '24:00:00',
                'description': 'Time limit in HH:MM:SS format'
            },
            'gpu_count': {
                'type': int,
                'min_value': 0,
                'max_value': 8,
                'default': 0,
                'description': 'Number of GPUs to request'
            }
        }
    }
    
    def __init__(self):
        """Initialize ParameterValidator"""
        pass
    
    def validate_processing_parameters(self, processing_type: str, 
                                     parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate processing parameters for a specific processing type
        
        Parameters
        ----------
        processing_type : str
            Type of processing (dti, noddi, csd, etc.)
        parameters : Dict[str, Any]
            Parameters to validate
            
        Returns
        -------
        ValidationResult
            Validation result with errors, warnings, and suggestions
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Check if processing type is supported
        if processing_type not in self.PARAMETER_SPECS:
            result.errors.append(ValidationError(
                f"Unsupported processing type: {processing_type}",
                field="processing_type"
            ))
            result.is_valid = False
            return result
        
        spec = self.PARAMETER_SPECS[processing_type]
        
        # Validate each parameter
        for param_name, param_value in parameters.items():
            if param_name in spec:
                param_result = self._validate_single_parameter(
                    param_name, param_value, spec[param_name], processing_type
                )
                result.errors.extend(param_result.errors)
                result.warnings.extend(param_result.warnings)
                result.suggestions.extend(param_result.suggestions)
        
        # Check for unknown parameters
        unknown_params = set(parameters.keys()) - set(spec.keys())
        for param in unknown_params:
            result.warnings.append(ValidationWarning(
                f"Unknown parameter '{param}' for {processing_type} processing",
                field=param
            ))
            result.suggestions.append(
                f"Valid parameters for {processing_type}: {list(spec.keys())}"
            )
        
        # Check parameter compatibility
        compatibility_result = self._validate_parameter_compatibility(
            processing_type, parameters
        )
        result.errors.extend(compatibility_result.errors)
        result.warnings.extend(compatibility_result.warnings)
        result.suggestions.extend(compatibility_result.suggestions)
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _validate_single_parameter(self, param_name: str, param_value: Any, 
                                 param_spec: Dict[str, Any], 
                                 processing_type: str) -> ValidationResult:
        """
        Validate a single parameter against its specification
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
        param_value : Any
            Value of the parameter
        param_spec : Dict[str, Any]
            Parameter specification
        processing_type : str
            Type of processing
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Check type
        expected_type = param_spec['type']
        
        # Handle optional parameters that can be None
        if param_spec.get('optional', False) and param_value is None:
            return result  # None is valid for optional parameters
        
        # Handle tuple of types (e.g., (str, type(None)))
        if isinstance(expected_type, tuple):
            if not isinstance(param_value, expected_type):
                type_names = [t.__name__ for t in expected_type]
                result.errors.append(ValidationError(
                    f"Parameter '{param_name}' must be of type {' or '.join(type_names)}, "
                    f"got {type(param_value).__name__}",
                    field=param_name
                ))
                result.is_valid = False
                return result
        else:
            if not isinstance(param_value, expected_type):
                result.errors.append(ValidationError(
                    f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                    f"got {type(param_value).__name__}",
                    field=param_name
                ))
                result.is_valid = False
                return result
        
        # Check valid values (for string and list parameters)
        if 'valid_values' in param_spec:
            if expected_type == list:
                invalid_values = [v for v in param_value if v not in param_spec['valid_values']]
                if invalid_values:
                    result.errors.append(ValidationError(
                        f"Parameter '{param_name}' contains invalid values: {invalid_values}. "
                        f"Valid values: {param_spec['valid_values']}",
                        field=param_name
                    ))
            else:
                if param_value not in param_spec['valid_values']:
                    result.errors.append(ValidationError(
                        f"Parameter '{param_name}' must be one of {param_spec['valid_values']}, "
                        f"got '{param_value}'",
                        field=param_name
                    ))
                    result.suggestions.append(
                        f"Try using one of: {param_spec['valid_values']}"
                    )
        
        # Check numeric ranges
        if expected_type in [int, float]:
            if 'min_value' in param_spec and param_value < param_spec['min_value']:
                result.errors.append(ValidationError(
                    f"Parameter '{param_name}' value {param_value} is below minimum "
                    f"{param_spec['min_value']}",
                    field=param_name
                ))
            
            if 'max_value' in param_spec and param_value > param_spec['max_value']:
                result.errors.append(ValidationError(
                    f"Parameter '{param_name}' value {param_value} is above maximum "
                    f"{param_spec['max_value']}",
                    field=param_name
                ))
        
        # Check even number requirement
        if 'must_be_even' in param_spec and param_spec['must_be_even']:
            if param_value % 2 != 0:
                result.errors.append(ValidationError(
                    f"Parameter '{param_name}' must be an even number, got {param_value}",
                    field=param_name
                ))
        
        # Check file existence (skip if value is None and parameter is optional)
        if 'file_must_exist' in param_spec and param_spec['file_must_exist']:
            if param_value is not None:  # Only check if not None
                file_path = Path(param_value)
                if not file_path.exists():
                    result.errors.append(ValidationError(
                        f"File specified in parameter '{param_name}' does not exist: {param_value}",
                        field=param_name
                    ))
        
        # Check string patterns
        if 'pattern' in param_spec:
            if not re.match(param_spec['pattern'], param_value):
                result.errors.append(ValidationError(
                    f"Parameter '{param_name}' does not match required pattern. "
                    f"Got '{param_value}', expected pattern: {param_spec['pattern']}",
                    field=param_name
                ))
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _validate_parameter_compatibility(self, processing_type: str, 
                                        parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameter compatibility and cross-parameter constraints
        
        Parameters
        ----------
        processing_type : str
            Type of processing
        parameters : Dict[str, Any]
            Parameters to validate
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Tracking-specific compatibility checks
        if processing_type == 'tracking':
            # Check min_length vs max_length
            if 'min_length' in parameters and 'max_length' in parameters:
                if parameters['min_length'] >= parameters['max_length']:
                    result.errors.append(ValidationError(
                        f"min_length ({parameters['min_length']}) must be less than "
                        f"max_length ({parameters['max_length']})",
                        field="min_length,max_length"
                    ))
            
            # Check SIFT parameters
            if parameters.get('apply_sift', True):
                if 'sift_term_count' in parameters and 'num_seeds' in parameters:
                    if parameters['sift_term_count'] > parameters['num_seeds']:
                        result.warnings.append(ValidationWarning(
                            f"sift_term_count ({parameters['sift_term_count']}) is greater than "
                            f"num_seeds ({parameters['num_seeds']}). SIFT may not reduce streamlines.",
                            field="sift_term_count"
                        ))
        
        # Scheduler-specific compatibility checks
        elif processing_type == 'scheduler':
            # Check GPU settings
            if parameters.get('use_gpu', False):
                if parameters.get('gpu_count', 0) <= 0:
                    result.errors.append(ValidationError(
                        "gpu_count must be positive when use_gpu is True",
                        field="gpu_count"
                    ))
            
            # Check memory vs CPU ratio
            if 'mem_per_cpu' in parameters and 'cpus_per_task' in parameters:
                total_mem = parameters['mem_per_cpu'] * parameters['cpus_per_task']
                if total_mem > 512:  # Warn if total memory > 512GB
                    result.warnings.append(ValidationWarning(
                        f"Total memory request ({total_mem}GB) is very high. "
                        f"Consider reducing mem_per_cpu or cpus_per_task.",
                        field="mem_per_cpu,cpus_per_task"
                    ))
        
        # CSD/MSMT-CSD compatibility checks
        elif processing_type in ['csd', 'msmt_csd']:
            # Check sh_order vs data requirements
            if 'sh_order' in parameters:
                sh_order = parameters['sh_order']
                min_dirs = (sh_order + 1) * (sh_order + 2) // 2
                result.suggestions.append(
                    f"For sh_order={sh_order}, you need at least {min_dirs} "
                    f"gradient directions per shell"
                )
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def validate_configuration_object(self, config_obj) -> ValidationResult:
        """
        Validate an entire configuration object
        
        Parameters
        ----------
        config_obj : ElikopyConfig
            Configuration object to validate
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Validate each processing section
        processing_sections = [
            'dti', 'noddi', 'csd', 'msmt_csd', 'tracking', 
            'connectivity', 'fingerprinting', 'scheduler'
        ]
        
        for section in processing_sections:
            if hasattr(config_obj, section):
                section_obj = getattr(config_obj, section)
                section_dict = section_obj.__dict__ if hasattr(section_obj, '__dict__') else {}
                
                section_result = self.validate_processing_parameters(section, section_dict)
                
                # Prefix field names with section name
                for error in section_result.errors:
                    if error.field:
                        error.field = f"{section}.{error.field}"
                for warning in section_result.warnings:
                    if warning.field:
                        warning.field = f"{section}.{warning.field}"
                
                result.errors.extend(section_result.errors)
                result.warnings.extend(section_result.warnings)
                result.suggestions.extend(section_result.suggestions)
        
        # Global configuration checks
        global_result = self._validate_global_configuration(config_obj)
        result.errors.extend(global_result.errors)
        result.warnings.extend(global_result.warnings)
        result.suggestions.extend(global_result.suggestions)
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _validate_global_configuration(self, config_obj) -> ValidationResult:
        """
        Validate global configuration constraints
        
        Parameters
        ----------
        config_obj : ElikopyConfig
            Configuration object
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Check study name
        if hasattr(config_obj, 'study_name'):
            if not config_obj.study_name or not config_obj.study_name.strip():
                result.errors.append(ValidationError(
                    "study_name cannot be empty",
                    field="study_name"
                ))
            elif not re.match(r'^[a-zA-Z0-9_-]+$', config_obj.study_name):
                result.errors.append(ValidationError(
                    f"study_name '{config_obj.study_name}' contains invalid characters. "
                    f"Use only letters, numbers, underscores, and hyphens.",
                    field="study_name"
                ))
        
        # Check output configuration
        if hasattr(config_obj, 'output'):
            output_obj = config_obj.output
            if hasattr(output_obj, 'derivatives_name'):
                if not output_obj.derivatives_name or not output_obj.derivatives_name.strip():
                    result.errors.append(ValidationError(
                        "output.derivatives_name cannot be empty",
                        field="output.derivatives_name"
                    ))
                elif not re.match(r'^[a-zA-Z0-9_-]+$', output_obj.derivatives_name):
                    result.errors.append(ValidationError(
                        f"output.derivatives_name '{output_obj.derivatives_name}' contains "
                        f"invalid characters. Use only letters, numbers, underscores, and hyphens.",
                        field="output.derivatives_name"
                    ))
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def get_parameter_suggestions(self, processing_type: str, 
                                parameter_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameter suggestions for a processing type
        
        Parameters
        ----------
        processing_type : str
            Type of processing
        parameter_name : Optional[str]
            Specific parameter name (if None, returns all parameters)
            
        Returns
        -------
        Dict[str, Any]
            Parameter suggestions with descriptions and valid ranges
        """
        if processing_type not in self.PARAMETER_SPECS:
            return {}
        
        spec = self.PARAMETER_SPECS[processing_type]
        
        if parameter_name:
            if parameter_name in spec:
                return {parameter_name: spec[parameter_name]}
            else:
                return {}
        
        return spec
    
    def generate_parameter_documentation(self, processing_type: str) -> str:
        """
        Generate human-readable parameter documentation
        
        Parameters
        ----------
        processing_type : str
            Type of processing
            
        Returns
        -------
        str
            Formatted parameter documentation
        """
        if processing_type not in self.PARAMETER_SPECS:
            return f"No parameters defined for processing type: {processing_type}"
        
        spec = self.PARAMETER_SPECS[processing_type]
        doc_lines = [f"Parameters for {processing_type.upper()} processing:", ""]
        
        for param_name, param_spec in spec.items():
            doc_lines.append(f"  {param_name}:")
            doc_lines.append(f"    Description: {param_spec.get('description', 'No description')}")
            doc_lines.append(f"    Type: {param_spec['type'].__name__}")
            
            if 'default' in param_spec:
                doc_lines.append(f"    Default: {param_spec['default']}")
            
            if 'valid_values' in param_spec:
                doc_lines.append(f"    Valid values: {param_spec['valid_values']}")
            
            if 'min_value' in param_spec or 'max_value' in param_spec:
                range_str = "    Range: "
                if 'min_value' in param_spec:
                    range_str += f"{param_spec['min_value']} <= value"
                if 'max_value' in param_spec:
                    if 'min_value' in param_spec:
                        range_str += f" <= {param_spec['max_value']}"
                    else:
                        range_str += f"value <= {param_spec['max_value']}"
                doc_lines.append(range_str)
            
            if param_spec.get('must_be_even'):
                doc_lines.append("    Constraint: Must be even number")
            
            if param_spec.get('file_must_exist'):
                doc_lines.append("    Constraint: File must exist")
            
            if 'pattern' in param_spec:
                doc_lines.append(f"    Pattern: {param_spec['pattern']}")
            
            doc_lines.append("")
        
        return "\n".join(doc_lines)
    
    @staticmethod
    def validate_file_exists(file_path: Path, 
                           required: bool = True) -> ValidationResult:
        """
        Validate that a file exists
        
        Parameters
        ----------
        file_path : Path
            Path to file to check
        required : bool
            Whether the file is required
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []
        
        if not file_path.exists():
            if required:
                errors.append(ValidationError(
                    f"Required file does not exist: {file_path}",
                    field="file_path"
                ))
            else:
                warnings.append(ValidationWarning(
                    f"Optional file does not exist: {file_path}",
                    field="file_path"
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_parameter_range(value: Union[int, float], 
                               min_val: Optional[Union[int, float]] = None,
                               max_val: Optional[Union[int, float]] = None,
                               parameter_name: str = "parameter") -> ValidationResult:
        """
        Validate that a parameter is within specified range
        
        Parameters
        ----------
        value : Union[int, float]
            Value to validate
        min_val : Optional[Union[int, float]]
            Minimum allowed value
        max_val : Optional[Union[int, float]]
            Maximum allowed value
        parameter_name : str
            Name of the parameter for error messages
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []
        
        if min_val is not None and value < min_val:
            errors.append(ValidationError(
                f"{parameter_name} value {value} is below minimum {min_val}",
                field=parameter_name
            ))
        
        if max_val is not None and value > max_val:
            errors.append(ValidationError(
                f"{parameter_name} value {value} is above maximum {max_val}",
                field=parameter_name
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_gradient_table(bvals: np.ndarray, bvecs: np.ndarray) -> ValidationResult:
        """
        Validate gradient table (bvals/bvecs) for processing compatibility
        
        Parameters
        ----------
        bvals : np.ndarray
            B-values array
        bvecs : np.ndarray
            B-vectors array (3 x N)
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Check array shapes
        if bvals.ndim != 1:
            result.errors.append(ValidationError(
                f"bvals must be 1D array, got shape {bvals.shape}",
                field="bvals"
            ))
        
        if bvecs.ndim != 2 or bvecs.shape[0] != 3:
            result.errors.append(ValidationError(
                f"bvecs must be 3xN array, got shape {bvecs.shape}",
                field="bvecs"
            ))
        
        if len(result.errors) > 0:
            result.is_valid = False
            return result
        
        # Check length consistency
        if len(bvals) != bvecs.shape[1]:
            result.errors.append(ValidationError(
                f"bvals length ({len(bvals)}) must match bvecs columns ({bvecs.shape[1]})",
                field="bvals,bvecs"
            ))
            result.is_valid = False
            return result
        
        # Check for b0 volumes
        b0_mask = bvals < 50
        b0_count = np.sum(b0_mask)
        
        if b0_count == 0:
            result.errors.append(ValidationError(
                "No b0 volumes found (b < 50 s/mm²)",
                field="bvals"
            ))
        elif b0_count < 3:
            result.warnings.append(ValidationWarning(
                f"Only {b0_count} b0 volumes found, recommended at least 3",
                field="bvals"
            ))
        
        # Check gradient vector normalization
        dwi_mask = ~b0_mask
        if np.any(dwi_mask):
            norms = np.linalg.norm(bvecs[:, dwi_mask], axis=0)
            non_unit_mask = np.abs(norms - 1.0) > 0.01
            
            if np.any(non_unit_mask):
                non_unit_count = np.sum(non_unit_mask)
                result.warnings.append(ValidationWarning(
                    f"{non_unit_count} gradient vectors are not unit vectors",
                    field="bvecs"
                ))
        
        # Analyze shell structure
        unique_bvals = np.unique(np.round(bvals / 100) * 100)
        shell_info = []
        
        for bval in unique_bvals:
            shell_mask = np.abs(bvals - bval) < 50
            count = np.sum(shell_mask)
            shell_info.append(f"b={int(bval)}: {count} volumes")
            
            # Check minimum directions per shell
            if bval > 0 and count < 6:
                result.warnings.append(ValidationWarning(
                    f"Shell b={int(bval)} has only {count} directions, recommended at least 6",
                    field="bvals"
                ))
        
        result.suggestions.append(f"Shell structure: {', '.join(shell_info)}")
        
        result.is_valid = len(result.errors) == 0
        return result
    
    @staticmethod
    def validate_processing_compatibility(processing_types: List[str], 
                                        gradient_info: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate compatibility between different processing types and data requirements
        
        Parameters
        ----------
        processing_types : List[str]
            List of processing types to validate
        gradient_info : Optional[Dict[str, Any]]
            Information about gradient table (shells, directions, etc.)
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Check for incompatible combinations
        if 'csd' in processing_types and 'msmt_csd' in processing_types:
            result.warnings.append(ValidationWarning(
                "Both CSD and MSMT-CSD are enabled. Consider using only one.",
                field="processing_types"
            ))
        
        # Check data requirements
        if gradient_info:
            shells = gradient_info.get('shells', [])
            
            # Single-shell requirements
            if 'csd' in processing_types and not gradient_info.get('multi_shell', False):
                if len(shells) > 1:
                    result.suggestions.append(
                        "Multi-shell data detected. Consider using MSMT-CSD instead of CSD."
                    )
            
            # Multi-shell requirements
            if 'msmt_csd' in processing_types:
                non_zero_shells = [s for s in shells if s > 0]
                if len(non_zero_shells) < 2:
                    result.warnings.append(ValidationWarning(
                        "MSMT-CSD requires multi-shell data, but only single shell detected",
                        field="msmt_csd"
                    ))
            
            # NODDI requirements
            if 'noddi' in processing_types:
                non_zero_shells = [s for s in shells if s > 0]
                if len(non_zero_shells) < 2:
                    result.warnings.append(ValidationWarning(
                        "NODDI typically requires multi-shell data for optimal results",
                        field="noddi"
                    ))
                
                # Check for high b-value shell
                max_bval = max(shells) if shells else 0
                if max_bval < 2000:
                    result.warnings.append(ValidationWarning(
                        f"NODDI benefits from high b-value shells (>2000 s/mm²), "
                        f"maximum b-value is {max_bval}",
                        field="noddi"
                    ))
        
        # Check processing order dependencies
        if 'tracking' in processing_types:
            if not any(proc in processing_types for proc in ['csd', 'msmt_csd']):
                result.warnings.append(ValidationWarning(
                    "Tractography typically requires CSD or MSMT-CSD for fiber orientation",
                    field="tracking"
                ))
        
        if 'connectivity' in processing_types:
            if 'tracking' not in processing_types:
                result.errors.append(ValidationError(
                    "Connectivity analysis requires tractography to be enabled",
                    field="connectivity"
                ))
        
        result.is_valid = len(result.errors) == 0
        return result


class ValidationUtils:
    """Legacy validation utilities for backward compatibility"""
    
    def __init__(self):
        """Initialize ValidationUtils"""
        self.parameter_validator = ParameterValidator()
    
    def validate_inputs(self) -> bool:
        """Validate inputs before processing"""
        return True
    
    def process(self, **kwargs):
        """Execute processing - placeholder for base class requirement"""
        pass
    
    @staticmethod
    def validate_config_dict(config: Dict[str, Any], 
                           required_keys: List[str],
                           optional_keys: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate configuration dictionary
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate
        required_keys : List[str]
            List of required keys
        optional_keys : Optional[List[str]]
            List of optional keys
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                errors.append(ValidationError(
                    f"Required configuration key missing: {key}",
                    field=key
                ))
        
        # Check for unknown keys
        all_valid_keys = set(required_keys)
        if optional_keys:
            all_valid_keys.update(optional_keys)
        
        for key in config.keys():
            if key not in all_valid_keys:
                warnings.append(ValidationWarning(
                    f"Unknown configuration key: {key}",
                    field=key
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )