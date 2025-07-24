#!/usr/bin/env python3
"""
Demo script to test the DataValidator functionality
"""

import tempfile
import json
from pathlib import Path
from elikopy.data import DataValidator

def main():
    """Demo the DataValidator functionality"""
    validator = DataValidator()
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("=== DataValidator Demo ===\n")
        
        # Demo 1: Valid DWI data validation
        print("1. Testing valid DWI data validation:")
        
        # Create mock files
        dwi_file = temp_path / "sub-01_dwi.nii.gz"
        bval_file = temp_path / "sub-01_dwi.bval"
        bvec_file = temp_path / "sub-01_dwi.bvec"
        json_file = temp_path / "sub-01_dwi.json"
        
        dwi_file.touch()
        bval_file.write_text("0 1000 1000 2000 2000")
        bvec_file.write_text("0 1 0 1 0\n0 0 1 0 1\n0 0 0 0 0")
        
        # Create valid JSON metadata
        json_data = {
            "EffectiveEchoSpacing": 0.00069,
            "PhaseEncodingDirection": "j-",
            "RepetitionTime": 3.0,
            "EchoTime": 0.089
        }
        json_file.write_text(json.dumps(json_data, indent=2))
        
        # Test bvals/bvecs validation
        result = validator.validate_bvals_bvecs(bval_file, bvec_file)
        print(f"   bvals/bvecs validation: {'PASSED' if result.is_valid else 'FAILED'}")
        if result.info:
            print(f"   Info: {result.info[0].message}")
        
        # Test BIDS compliance
        result = validator.validate_bids_compliance(dwi_file)
        print(f"   BIDS compliance: {'PASSED' if result.is_valid else 'FAILED'}")
        
        print()
        
        # Demo 2: Invalid data validation
        print("2. Testing invalid data validation:")
        
        # Create invalid bvals/bvecs
        invalid_bval_file = temp_path / "sub-02_dwi.bval"
        invalid_bvec_file = temp_path / "sub-02_dwi.bvec"
        
        invalid_bval_file.write_text("1000 1000 2000")  # 3 values
        invalid_bvec_file.write_text("1 0 1 0 0\n0 1 0 1 0\n0 0 0 0 0")  # 5 values
        
        result = validator.validate_bvals_bvecs(invalid_bval_file, invalid_bvec_file)
        print(f"   Invalid bvals/bvecs: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
        if result.errors:
            print(f"   Error: {result.errors[0].message}")
        
        print()
        
        # Demo 3: Processing parameter validation
        print("3. Testing processing parameter validation:")
        
        # Valid DTI parameters
        dti_params = {
            'processing_type': 'dti',
            'fit_method': 'WLS',
            'mask_threshold': 0.2
        }
        result = validator.validate_processing_parameters(dti_params)
        print(f"   Valid DTI params: {'PASSED' if result.is_valid else 'FAILED'}")
        
        # Invalid CSD parameters
        csd_params = {
            'processing_type': 'csd',
            'response_method': 'invalid_method',
            'sh_order': 7  # Should be even
        }
        result = validator.validate_processing_parameters(csd_params)
        print(f"   Invalid CSD params: {'PASSED' if result.is_valid else 'FAILED (as expected)'}")
        if result.errors:
            print(f"   Errors found: {len(result.errors)}")
        
        print()
        
        # Demo 4: BIDS filename parsing
        print("4. Testing BIDS filename parsing:")
        
        filename = "sub-01_ses-baseline_task-rest_acq-multiband_run-01_dwi.nii.gz"
        entities = validator._parse_bids_filename(filename)
        print(f"   Filename: {filename}")
        print(f"   Parsed entities: {entities}")
        
        print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()