"""
Validation utilities for ElikoPy
===============================

This module contains utility functions for parameter and data validation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from elikopy.core.base import ProcessingComponent


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


class ValidationUtils(ProcessingComponent):
    """Utility class for validation operations"""
    
    def __init__(self):
        """Initialize ValidationUtils"""
        pass
    
    def validate_inputs(self) -> bool:
        """Validate inputs before processing"""
        return True
    
    def process(self, **kwargs):
        """Execute processing - placeholder for base class requirement"""
        pass
    
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