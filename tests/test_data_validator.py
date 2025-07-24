"""
Unit tests for DataValidator class

Tests for comprehensive data validation including:
- DWI data integrity validation
- bvals/bvecs validation with gradient table checks
- BIDS compliance validation with detailed error reporting
- Processing parameter validation
"""

import json
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from elikopy.data.validator import DataValidator, ValidationResult, ValidationError


class TestValidationError:
    """Test ValidationError class"""
    
    def test_init(self):
        """Test ValidationError initialization"""
        error = ValidationError("Test message", "error")
        assert error.message == "Test message"
        assert error.severity == "error"
    
    def test_default_severity(self):
        """Test default severity"""
        error = ValidationError("Test message")
        assert error.severity == "error"
    
    def test_str_representation(self):
        """Test string representation"""
        error = ValidationError("Test message", "warning")
        assert str(error) == "WARNING: Test message"


class TestValidationResult:
    """Test ValidationResult class"""
    
    def test_init(self):
        """Test ValidationResult initialization"""
        result = ValidationResult()
        assert result.errors == []
        assert result.warnings == []
        assert result.info == []
        assert result.is_valid is True
    
    def test_add_error(self):
        """Test adding errors"""
        result = ValidationResult()
        result.add_error("Test error")
        assert len(result.errors) == 1
        assert result.errors[0].message == "Test error"
        assert result.errors[0].severity == "error"
        assert result.is_valid is False
    
    def test_add_warning(self):
        """Test adding warnings"""
        result = ValidationResult()
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert result.warnings[0].message == "Test warning"
        assert result.warnings[0].severity == "warning"
        assert result.is_valid is True  # Warnings don't affect validity
    
    def test_add_info(self):
        """Test adding info messages"""
        result = ValidationResult()
        result.add_info("Test info")
        assert len(result.info) == 1
        assert result.info[0].message == "Test info"
        assert result.info[0].severity == "info"
    
    def test_str_representation(self):
        """Test string representation"""
        result = ValidationResult()
        result.add_error("Error message")
        result.add_warning("Warning message")
        result.add_info("Info message")
        
        str_repr = str(result)
        assert "Errors:" in str_repr
        assert "Error message" in str_repr
        assert "Warnings:" in str_repr
        assert "Warning message" in str_repr
        assert "Info:" in str_repr
        assert "Info message" in str_repr
    
    def test_str_representation_empty(self):
        """Test string representation when empty"""
        result = ValidationResult()
        assert str(result) == "Validation passed with no issues."


class TestDataValidator:
    """Test DataValidator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self, dwi_data=None, bvals=None, bvecs=None, json_data=None):
        """Create test files for validation"""
        files = {}
        
        # Create DWI file
        dwi_file = self.temp_dir / "sub-01_dwi.nii.gz"
        dwi_file.touch()
        files['dwi'] = dwi_file
        
        # Create bval file
        bval_file = self.temp_dir / "sub-01_dwi.bval"
        if bvals is None:
            bvals = "0 1000 1000 2000 2000"
        bval_file.write_text(bvals)
        files['bval'] = bval_file
        
        # Create bvec file
        bvec_file = self.temp_dir / "sub-01_dwi.bvec"
        if bvecs is None:
            bvecs = "0 1 0 1 0\n0 0 1 0 1\n0 0 0 0 0"
        bvec_file.write_text(bvecs)
        files['bvec'] = bvec_file
        
        # Create JSON file
        if json_data is not None:
            json_file = self.temp_dir / "sub-01_dwi.json"
            json_file.write_text(json.dumps(json_data))
            files['json'] = json_file
        
        return files
    
    def test_validate_dwi_data_missing_files(self):
        """Test validation with missing files"""
        dwi_file = self.temp_dir / "nonexistent_dwi.nii.gz"
        bval_file = self.temp_dir / "nonexistent.bval"
        bvec_file = self.temp_dir / "nonexistent.bvec"
        
        result = self.validator.validate_dwi_data(dwi_file, bval_file, bvec_file)
        
        assert not result.is_valid
        assert len(result.errors) >= 3  # Missing DWI, bval, and bvec files
        assert any("DWI file not found" in error.message for error in result.errors)
        assert any("bval file not found" in error.message for error in result.errors)
        assert any("bvec file not found" in error.message for error in result.errors)
    
    @patch('nibabel.load')
    def test_validate_dwi_data_valid(self, mock_nib_load):
        """Test validation with valid DWI data"""
        # Mock nibabel image
        mock_img = Mock()
        mock_img.shape = (64, 64, 40, 5)
        mock_img.get_data_dtype.return_value = 'float32'
        mock_nib_load.return_value = mock_img
        
        files = self.create_test_files()
        
        result = self.validator.validate_dwi_data(
            files['dwi'], files['bval'], files['bvec']
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    @patch('nibabel.load')
    def test_validate_dwi_data_dimension_mismatch(self, mock_nib_load):
        """Test validation with dimension mismatch"""
        # Mock nibabel image with wrong number of volumes
        mock_img = Mock()
        mock_img.shape = (64, 64, 40, 3)  # 3 volumes but 5 bvals
        mock_img.get_data_dtype.return_value = 'float32'
        mock_nib_load.return_value = mock_img
        
        files = self.create_test_files()
        
        result = self.validator.validate_dwi_data(
            files['dwi'], files['bval'], files['bvec']
        )
        
        assert not result.is_valid
        assert any("Mismatch between DWI volumes" in error.message for error in result.errors)
    
    def test_validate_dwi_data_invalid_bvecs(self):
        """Test validation with invalid bvecs"""
        files = self.create_test_files(
            bvecs="0 2 0 1 0\n0 0 2 0 1\n0 0 0 0 0"  # Non-unit vectors
        )
        
        with patch('nibabel.load') as mock_nib_load:
            mock_img = Mock()
            mock_img.shape = (64, 64, 40, 5)
            mock_img.get_data_dtype.return_value = 'float32'
            mock_nib_load.return_value = mock_img
            
            result = self.validator.validate_dwi_data(
                files['dwi'], files['bval'], files['bvec']
            )
        
        assert len(result.warnings) > 0
        assert any("Non-unit vector" in warning.message for warning in result.warnings)
    
    def test_validate_dwi_data_no_b0(self):
        """Test validation with no b0 volumes"""
        files = self.create_test_files(
            bvals="1000 1000 2000 2000 3000"  # No b0 volumes
        )
        
        with patch('nibabel.load') as mock_nib_load:
            mock_img = Mock()
            mock_img.shape = (64, 64, 40, 5)
            mock_img.get_data_dtype.return_value = 'float32'
            mock_nib_load.return_value = mock_img
            
            result = self.validator.validate_dwi_data(
                files['dwi'], files['bval'], files['bvec']
            )
        
        assert not result.is_valid
        assert any("No b0 volumes found" in error.message for error in result.errors)
    
    def test_validate_bvals_bvecs_valid(self):
        """Test bvals/bvecs validation with valid data"""
        files = self.create_test_files()
        
        result = self.validator.validate_bvals_bvecs(files['bval'], files['bvec'])
        
        assert result.is_valid
        assert len(result.info) > 0  # Should report shell structure
        assert any("Shell structure" in info.message for info in result.info)
    
    def test_validate_bvals_bvecs_length_mismatch(self):
        """Test bvals/bvecs validation with length mismatch"""
        files = self.create_test_files(
            bvals="0 1000 1000",  # 3 values
            bvecs="0 1 0 1 0\n0 0 1 0 1\n0 0 0 0 0"  # 5 values
        )
        
        result = self.validator.validate_bvals_bvecs(files['bval'], files['bvec'])
        
        assert not result.is_valid
        assert any("Mismatch in bval/bvec lengths" in error.message for error in result.errors)
    
    def test_validate_bvals_bvecs_wrong_bvec_format(self):
        """Test bvals/bvecs validation with wrong bvec format"""
        files = self.create_test_files(
            bvecs="0 1 0 1 0\n0 0 1 0 1"  # Only 2 lines instead of 3
        )
        
        result = self.validator.validate_bvals_bvecs(files['bval'], files['bvec'])
        
        assert not result.is_valid
        assert any("bvec file should have 3 lines" in error.message for error in result.errors)
    
    def test_validate_processing_parameters_dti_valid(self):
        """Test processing parameter validation for DTI"""
        params = {
            'processing_type': 'dti',
            'fit_method': 'WLS',
            'mask_threshold': 0.2
        }
        
        result = self.validator.validate_processing_parameters(params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_processing_parameters_dti_invalid(self):
        """Test processing parameter validation for DTI with invalid parameters"""
        params = {
            'processing_type': 'dti',
            'fit_method': 'INVALID',
            'mask_threshold': 1.5
        }
        
        result = self.validator.validate_processing_parameters(params)
        
        assert not result.is_valid
        assert any("fit_method" in error.message and "INVALID" in error.message for error in result.errors)
        assert any("mask_threshold" in error.message and "1.5" in error.message for error in result.errors)
    
    def test_validate_processing_parameters_csd_valid(self):
        """Test processing parameter validation for CSD"""
        params = {
            'processing_type': 'csd',
            'response_method': 'tournier',
            'sh_order': 8
        }
        
        result = self.validator.validate_processing_parameters(params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_processing_parameters_csd_invalid(self):
        """Test processing parameter validation for CSD with invalid parameters"""
        params = {
            'processing_type': 'csd',
            'response_method': 'invalid',
            'sh_order': 7  # Odd number
        }
        
        result = self.validator.validate_processing_parameters(params)
        
        assert not result.is_valid
        assert any("response_method" in error.message and "invalid" in error.message for error in result.errors)
        assert any("sh_order" in error.message and "even" in error.message for error in result.errors)
    
    def test_validate_processing_parameters_tracking_valid(self):
        """Test processing parameter validation for tracking"""
        params = {
            'processing_type': 'tracking',
            'algorithm': 'probabilistic',
            'step_size': 0.5,
            'max_angle': 30.0,  # Use float to match parameter spec
            'min_length': 10.0,  # Use float to match parameter spec
            'max_length': 200.0  # Use float to match parameter spec
        }
        
        result = self.validator.validate_processing_parameters(params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_processing_parameters_tracking_invalid(self):
        """Test processing parameter validation for tracking with invalid parameters"""
        params = {
            'processing_type': 'tracking',
            'algorithm': 'invalid',
            'step_size': -0.5,
            'max_angle': 100.0,
            'min_length': 200.0,
            'max_length': 100.0
        }
        
        result = self.validator.validate_processing_parameters(params)
        
        assert not result.is_valid
        assert any("algorithm" in error.message and "invalid" in error.message for error in result.errors)
        assert any("step_size" in error.message and "below minimum" in error.message for error in result.errors)
        assert any("max_angle" in error.message and "above maximum" in error.message for error in result.errors)
        assert any("min_length" in error.message and "max_length" in error.message 
                  for error in result.errors)
    
    def test_validate_processing_parameters_missing_type(self):
        """Test processing parameter validation with missing processing type"""
        params = {
            'fit_method': 'WLS'
        }
        
        result = self.validator.validate_processing_parameters(params)
        
        assert not result.is_valid
        assert any("Missing required parameter: processing_type" in error.message for error in result.errors)
    
    def test_validate_bids_compliance_valid_filename(self):
        """Test BIDS compliance validation with valid filename"""
        dwi_file = self.temp_dir / "sub-01_ses-01_dwi.nii.gz"
        dwi_file.touch()
        
        # Create associated files
        (self.temp_dir / "sub-01_ses-01_dwi.bval").touch()
        (self.temp_dir / "sub-01_ses-01_dwi.bvec").touch()
        
        result = self.validator.validate_bids_compliance(dwi_file)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_bids_compliance_invalid_filename(self):
        """Test BIDS compliance validation with invalid filename"""
        dwi_file = self.temp_dir / "invalid_filename.nii.gz"
        dwi_file.touch()
        
        result = self.validator.validate_bids_compliance(dwi_file)
        
        assert not result.is_valid
        assert any("Missing required 'sub' entity" in error.message for error in result.errors)
    
    def test_validate_bids_compliance_missing_associated_files(self):
        """Test BIDS compliance validation with missing associated files"""
        dwi_file = self.temp_dir / "sub-01_dwi.nii.gz"
        dwi_file.touch()
        
        result = self.validator.validate_bids_compliance(dwi_file)
        
        assert not result.is_valid
        assert any("Missing associated bval file" in error.message for error in result.errors)
        assert any("Missing associated bvec file" in error.message for error in result.errors)
    
    def test_validate_bids_compliance_with_json_metadata(self):
        """Test BIDS compliance validation with JSON metadata"""
        dwi_file = self.temp_dir / "sub-01_dwi.nii.gz"
        dwi_file.touch()
        
        # Create associated files
        (self.temp_dir / "sub-01_dwi.bval").touch()
        (self.temp_dir / "sub-01_dwi.bvec").touch()
        
        # Create JSON with valid metadata
        json_data = {
            "EffectiveEchoSpacing": 0.00069,
            "PhaseEncodingDirection": "j-",
            "RepetitionTime": 3.0,
            "EchoTime": 0.089
        }
        json_file = self.temp_dir / "sub-01_dwi.json"
        json_file.write_text(json.dumps(json_data))
        
        result = self.validator.validate_bids_compliance(dwi_file)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_bids_compliance_invalid_json_metadata(self):
        """Test BIDS compliance validation with invalid JSON metadata"""
        dwi_file = self.temp_dir / "sub-01_dwi.nii.gz"
        dwi_file.touch()
        
        # Create associated files
        (self.temp_dir / "sub-01_dwi.bval").touch()
        (self.temp_dir / "sub-01_dwi.bvec").touch()
        
        # Create JSON with invalid metadata
        json_data = {
            "EffectiveEchoSpacing": -0.00069,  # Invalid negative value
            "PhaseEncodingDirection": "invalid",  # Invalid direction
            "RepetitionTime": "invalid"  # Invalid type
        }
        json_file = self.temp_dir / "sub-01_dwi.json"
        json_file.write_text(json.dumps(json_data))
        
        result = self.validator.validate_bids_compliance(dwi_file)
        
        assert not result.is_valid
        assert any("EffectiveEchoSpacing must be positive" in error.message for error in result.errors)
        assert any("Invalid PhaseEncodingDirection" in error.message for error in result.errors)
        assert any("RepetitionTime must be a number" in error.message for error in result.errors)
    
    def test_parse_bids_filename(self):
        """Test BIDS filename parsing"""
        filename = "sub-01_ses-baseline_task-rest_acq-multiband_run-01_dwi.nii.gz"
        
        entities = self.validator._parse_bids_filename(filename)
        
        assert entities['sub'] == 'sub-01'
        assert entities['ses'] == 'ses-baseline'
        assert entities['task'] == 'task-rest'
        assert entities['acq'] == 'acq-multiband'
        assert entities['run'] == 'run-01'
        assert entities['suffix'] == 'dwi'
    
    def test_validate_bids_directory_structure(self):
        """Test BIDS directory structure validation"""
        # Create a mock BIDS structure
        dataset_root = self.temp_dir / "bids_dataset"
        dataset_root.mkdir()
        
        # Create dataset_description.json
        dataset_desc = dataset_root / "dataset_description.json"
        dataset_desc.write_text('{"Name": "Test Dataset", "BIDSVersion": "1.6.0"}')
        
        # Create subject directory structure
        subject_dir = dataset_root / "sub-01" / "dwi"
        subject_dir.mkdir(parents=True)
        
        dwi_file = subject_dir / "sub-01_dwi.nii.gz"
        dwi_file.touch()
        
        result = self.validator._validate_bids_directory_structure(dwi_file, dataset_root)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_bids_directory_structure_invalid(self):
        """Test BIDS directory structure validation with invalid structure"""
        # Create invalid structure
        dataset_root = self.temp_dir / "bids_dataset"
        dataset_root.mkdir()
        
        # Create file in wrong location
        dwi_file = dataset_root / "invalid_dwi.nii.gz"
        dwi_file.touch()
        
        result = self.validator._validate_bids_directory_structure(dwi_file, dataset_root)
        
        assert not result.is_valid
        assert any("File not in proper BIDS structure" in error.message for error in result.errors)
    
    def test_validate_bids_directory_structure_derivatives(self):
        """Test BIDS directory structure validation for derivatives"""
        # Create derivatives structure
        dataset_root = self.temp_dir / "bids_dataset"
        dataset_root.mkdir()
        
        derivatives_dir = dataset_root / "derivatives" / "elikopy" / "sub-01" / "dwi"
        derivatives_dir.mkdir(parents=True)
        
        dwi_file = derivatives_dir / "sub-01_desc-preproc_dwi.nii.gz"
        dwi_file.touch()
        
        result = self.validator._validate_bids_directory_structure(dwi_file, dataset_root)
        
        assert result.is_valid
        assert any("File in derivatives pipeline: elikopy" in info.message for info in result.info)


if __name__ == "__main__":
    pytest.main([__file__])