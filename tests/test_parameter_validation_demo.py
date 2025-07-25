#!/usr/bin/env python3
"""
Demo script to test the comprehensive parameter validation functionality
"""

import numpy as np
from pathlib import Path
from elikopy.utils.validation import ParameterValidator
from elikopy.core.config import ElikopyConfig

def main():
    """Demo the comprehensive parameter validation functionality"""
    validator = ParameterValidator()
    
    print("=== Comprehensive Parameter Validation Demo ===\n")
    
    # Demo 1: DTI parameter validation
    print("1. Testing DTI parameter validation:")
    
    # Valid DTI parameters
    dti_params_valid = {
        'fit_method': 'WLS',
        'mask_threshold': 0.2,
        'compute_metrics': ['FA', 'MD', 'AD', 'RD']
    }
    result = validator.validate_processing_parameters('dti', dti_params_valid)
    print(f"   Valid DTI params: {'PASSED' if result.is_valid else 'FAILED'}")
    
    # Invalid DTI parameters
    dti_params_invalid = {
        'fit_method': 'INVALID_METHOD',
        'mask_threshold': 1.5,  # Invalid: > 1.0
        'compute_metrics': ['FA', 'INVALID_METRIC']
    }
    result = validator.validate_processing_parameters('dti', dti_params_invalid)
    print(f"   Invalid DTI params: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
    if result.errors:
        print(f"   Errors found: {len(result.errors)}")
        for error in result.errors[:2]:  # Show first 2 errors
            print(f"     - {error.message}")
    
    print()
    
    # Demo 2: CSD parameter validation
    print("2. Testing CSD parameter validation:")
    
    # Valid CSD parameters
    csd_params_valid = {
        'response_method': 'tournier',
        'sh_order': 8,
        'mask_threshold': 0.2
    }
    result = validator.validate_processing_parameters('csd', csd_params_valid)
    print(f"   Valid CSD params: {'PASSED' if result.is_valid else 'FAILED'}")
    
    # Invalid CSD parameters (odd sh_order)
    csd_params_invalid = {
        'response_method': 'invalid_method',
        'sh_order': 7  # Must be even
    }
    result = validator.validate_processing_parameters('csd', csd_params_invalid)
    print(f"   Invalid CSD params: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
    if result.errors:
        print(f"   Errors found: {len(result.errors)}")
        for error in result.errors:
            print(f"     - {error.message}")
    
    print()
    
    # Demo 3: Tracking parameter validation with compatibility checks
    print("3. Testing tracking parameter validation with compatibility:")
    
    # Parameters with compatibility warning
    tracking_params = {
        'algorithm': 'deterministic',
        'step_size': 0.5,
        'max_angle': 30.0,
        'min_length': 20.0,
        'max_length': 200.0,
        'num_seeds': 1000,
        'sift_term_count': 5000,  # Warning: > num_seeds
        'apply_sift': True
    }
    result = validator.validate_processing_parameters('tracking', tracking_params)
    print(f"   Tracking params with warnings: {'PASSED' if result.is_valid else 'FAILED'}")
    if result.warnings:
        print(f"   Warnings found: {len(result.warnings)}")
        for warning in result.warnings:
            print(f"     - {warning.message}")
    
    print()
    
    # Demo 4: Scheduler parameter validation
    print("4. Testing scheduler parameter validation:")
    
    # Valid scheduler parameters
    scheduler_params_valid = {
        'type': 'slurm',
        'cpus_per_task': 8,
        'mem_per_cpu': 4,
        'time_limit': '12:30:45',
        'gpu_count': 2
    }
    result = validator.validate_processing_parameters('scheduler', scheduler_params_valid)
    print(f"   Valid scheduler params: {'PASSED' if result.is_valid else 'FAILED'}")
    
    # Invalid time format
    scheduler_params_invalid = {
        'time_limit': '25:70:90'  # Invalid format
    }
    result = validator.validate_processing_parameters('scheduler', scheduler_params_invalid)
    print(f"   Invalid time format: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
    if result.errors:
        print(f"   Error: {result.errors[0].message}")
    
    print()
    
    # Demo 5: Gradient table validation
    print("5. Testing gradient table validation:")
    
    # Valid gradient table
    bvals = np.array([0, 0, 0, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000])
    bvecs = np.random.randn(3, len(bvals))
    # Normalize non-b0 vectors
    b0_mask = bvals < 50
    bvecs[:, ~b0_mask] = bvecs[:, ~b0_mask] / np.linalg.norm(bvecs[:, ~b0_mask], axis=0)
    bvecs[:, b0_mask] = 0  # Set b0 vectors to zero
    
    result = validator.validate_gradient_table(bvals, bvecs)
    print(f"   Valid gradient table: {'PASSED' if result.is_valid else 'FAILED'}")
    if result.suggestions:
        print(f"   Info: {result.suggestions[0]}")
    
    # Invalid gradient table (shape mismatch)
    bvals_invalid = np.array([0, 1000, 1000])  # 3 values
    bvecs_invalid = np.array([
        [0, 1, 0, 1, 0],  # 5 values
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0]
    ])
    result = validator.validate_gradient_table(bvals_invalid, bvecs_invalid)
    print(f"   Invalid gradient table: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
    if result.errors:
        print(f"   Error: {result.errors[0].message}")
    
    print()
    
    # Demo 6: Processing compatibility validation
    print("6. Testing processing compatibility validation:")
    
    # Valid combination
    processing_types = ['dti', 'csd', 'tracking', 'connectivity']
    result = validator.validate_processing_compatibility(processing_types)
    print(f"   Valid processing combination: {'PASSED' if result.is_valid else 'FAILED'}")
    
    # Invalid combination (connectivity without tracking)
    processing_types_invalid = ['dti', 'connectivity']
    result = validator.validate_processing_compatibility(processing_types_invalid)
    print(f"   Invalid combination: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
    if result.errors:
        print(f"   Error: {result.errors[0].message}")
    
    # Combination with warnings
    processing_types_warning = ['csd', 'msmt_csd']
    result = validator.validate_processing_compatibility(processing_types_warning)
    print(f"   Combination with warnings: {'PASSED' if result.is_valid else 'FAILED'}")
    if result.warnings:
        print(f"   Warning: {result.warnings[0].message}")
    
    print()
    
    # Demo 7: Configuration object validation
    print("7. Testing configuration object validation:")
    
    try:
        # Create a default configuration
        config = ElikopyConfig()
        config.study_name = "test_study"
        
        result = validator.validate_configuration_object(config)
        print(f"   Default config validation: {'PASSED' if result.is_valid else 'FAILED'}")
        if result.errors:
            print(f"   Errors found: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"     - {error.message}")
        
        # Invalid study name
        config.study_name = "invalid name!"  # Contains invalid characters
        result = validator.validate_configuration_object(config)
        print(f"   Invalid study name: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
        if result.errors:
            print(f"   Error: {result.errors[0].message}")
    
    except Exception as e:
        print(f"   Configuration validation failed: {e}")
    
    print()
    
    # Demo 8: Parameter suggestions and documentation
    print("8. Testing parameter suggestions and documentation:")
    
    # Get parameter suggestions for DTI
    suggestions = validator.get_parameter_suggestions('dti')
    print(f"   DTI parameter suggestions: {len(suggestions)} parameters")
    print(f"   Available parameters: {list(suggestions.keys())}")
    
    # Get specific parameter info
    fit_method_info = validator.get_parameter_suggestions('dti', 'fit_method')
    if 'fit_method' in fit_method_info:
        spec = fit_method_info['fit_method']
        print(f"   fit_method valid values: {spec.get('valid_values', 'N/A')}")
        print(f"   fit_method default: {spec.get('default', 'N/A')}")
    
    print()
    
    # Demo 9: Generate parameter documentation
    print("9. Testing parameter documentation generation:")
    
    doc = validator.generate_parameter_documentation('tracking')
    print("   Generated tracking parameter documentation:")
    print("   " + "\\n   ".join(doc.split("\\n")[:10]))  # Show first 10 lines
    print("   ... (truncated)")
    
    print("\\n=== Demo Complete ===")

if __name__ == "__main__":
    main()