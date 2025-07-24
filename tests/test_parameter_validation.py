"""
Unit tests for parameter validation utilities

Tests for comprehensive parameter validation including:
- Processing parameter validation for all processing steps
- Range checking and compatibility validation
- Configuration object validation
- Gradient table validation
- Processing compatibility validation
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from elikopy.utils.validation import (
    ParameterValidator, ValidationResult, ValidationError, ValidationWarning
)


class TestParameterValidator:
    """Test ParameterValidator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = ParameterValidator()
    
    def test_validate_dti_parameters_valid(self):
        """Test DTI parameter validation with valid parameters"""
        params = {
            'fit_method': 'WLS',
            'mask_threshold': 0.2,
            'compute_metrics': ['FA', 'MD', 'AD', 'RD']
        }
        
        result = self.validator.validate_processing_parameters('dti', params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_dti_parameters_invalid_fit_method(self):
        """Test DTI parameter validation with invalid fit method"""
        params = {
            'fit_method': 'INVALID_METHOD',
            'mask_threshold': 0.2
        }
        
        result = self.validator.validate_processing_parameters('dti', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "fit_method" in result.errors[0].message
        assert "INVALID_METHOD" in result.errors[0].message
    
    def test_validate_dti_parameters_invalid_threshold(self):
        """Test DTI parameter validation with invalid mask threshold"""
        params = {
            'fit_method': 'WLS',
            'mask_threshold': 1.5  # Invalid: > 1.0
        }
        
        result = self.validator.validate_processing_parameters('dti', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "mask_threshold" in result.errors[0].message
        assert "1.5" in result.errors[0].message
    
    def test_validate_dti_parameters_invalid_metrics(self):
        """Test DTI parameter validation with invalid metrics"""
        params = {
            'fit_method': 'WLS',
            'compute_metrics': ['FA', 'INVALID_METRIC']
        }
        
        result = self.validator.validate_processing_parameters('dti', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "compute_metrics" in result.errors[0].message
        assert "INVALID_METRIC" in result.errors[0].message
    
    def test_validate_noddi_parameters_valid(self):
        """Test NODDI parameter validation with valid parameters"""
        params = {
            'fit_method': 'amico',
            'mask_threshold': 0.3,
            'compute_metrics': ['ICVF', 'ODI']
        }
        
        result = self.validator.validate_processing_parameters('noddi', params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_noddi_parameters_invalid(self):
        """Test NODDI parameter validation with invalid parameters"""
        params = {
            'fit_method': 'invalid_method',
            'mask_threshold': -0.1,
            'compute_metrics': ['INVALID']
        }
        
        result = self.validator.validate_processing_parameters('noddi', params)
        
        assert not result.is_valid
        assert len(result.errors) == 3  # All three parameters are invalid
    
    def test_validate_csd_parameters_valid(self):
        """Test CSD parameter validation with valid parameters"""
        params = {
            'response_method': 'tournier',
            'sh_order': 8,
            'mask_threshold': 0.2
        }
        
        result = self.validator.validate_processing_parameters('csd', params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_csd_parameters_invalid_sh_order(self):
        """Test CSD parameter validation with invalid spherical harmonics order"""
        params = {
            'response_method': 'tournier',
            'sh_order': 7  # Invalid: must be even
        }
        
        result = self.validator.validate_processing_parameters('csd', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "sh_order" in result.errors[0].message
        assert "even" in result.errors[0].message
    
    def test_validate_csd_parameters_sh_order_out_of_range(self):
        """Test CSD parameter validation with sh_order out of range"""
        params = {
            'sh_order': 14  # Invalid: > 12
        }
        
        result = self.validator.validate_processing_parameters('csd', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "sh_order" in result.errors[0].message
        assert "14" in result.errors[0].message
    
    def test_validate_tracking_parameters_valid(self):
        """Test tracking parameter validation with valid parameters"""
        params = {
            'algorithm': 'deterministic',
            'step_size': 0.5,
            'max_angle': 30.0,
            'min_length': 20.0,
            'max_length': 200.0,
            'num_seeds': 10000,
            'seed_mask': 'wm'
        }
        
        result = self.validator.validate_processing_parameters('tracking', params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_tracking_parameters_length_compatibility(self):
        """Test tracking parameter validation with incompatible lengths"""
        params = {
            'min_length': 200.0,
            'max_length': 100.0  # Invalid: min > max
        }
        
        result = self.validator.validate_processing_parameters('tracking', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "min_length" in result.errors[0].message
        assert "max_length" in result.errors[0].message
    
    def test_validate_tracking_parameters_invalid_step_size(self):
        """Test tracking parameter validation with invalid step size"""
        params = {
            'step_size': 0.05  # Invalid: < 0.1
        }
        
        result = self.validator.validate_processing_parameters('tracking', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "step_size" in result.errors[0].message
    
    def test_validate_tracking_parameters_sift_compatibility(self):
        """Test tracking parameter validation with SIFT compatibility warning"""
        params = {
            'num_seeds': 1000,
            'sift_term_count': 5000,  # Warning: sift_term_count > num_seeds
            'apply_sift': True
        }
        
        result = self.validator.validate_processing_parameters('tracking', params)
        
        assert result.is_valid  # Should be valid but with warning
        assert len(result.warnings) >= 1
        assert any("sift_term_count" in w.message for w in result.warnings)
    
    def test_validate_scheduler_parameters_valid(self):
        """Test scheduler parameter validation with valid parameters"""
        params = {
            'type': 'slurm',
            'cpus_per_task': 8,
            'mem_per_cpu': 4,
            'time_limit': '12:30:45',
            'gpu_count': 2
        }
        
        result = self.validator.validate_processing_parameters('scheduler', params)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_scheduler_parameters_invalid_time_format(self):
        """Test scheduler parameter validation with invalid time format"""
        params = {
            'time_limit': '25:70:90'  # Invalid format (minutes and seconds > 59)
        }
        
        result = self.validator.validate_processing_parameters('scheduler', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "time_limit" in result.errors[0].message
        assert "pattern" in result.errors[0].message
    
    def test_validate_scheduler_parameters_gpu_compatibility(self):
        """Test scheduler parameter validation with GPU compatibility"""
        params = {
            'use_gpu': True,
            'gpu_count': 0  # Invalid: use_gpu=True but gpu_count=0
        }
        
        result = self.validator.validate_processing_parameters('scheduler', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "gpu_count" in result.errors[0].message
    
    def test_validate_scheduler_parameters_memory_warning(self):
        """Test scheduler parameter validation with high memory warning"""
        params = {
            'cpus_per_task': 64,
            'mem_per_cpu': 16  # Total: 1024GB - should trigger warning
        }
        
        result = self.validator.validate_processing_parameters('scheduler', params)
        
        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) == 1
        assert "memory request" in result.warnings[0].message
    
    def test_validate_fingerprinting_parameters_valid(self):
        """Test fingerprinting parameter validation with valid parameters"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            params = {
                'dictionary_path': tmp_path,
                'mask_threshold': 0.2,
                'compute_metrics': ['fvf', 'diameter']
            }
            
            result = self.validator.validate_processing_parameters('fingerprinting', params)
            
            assert result.is_valid
            assert len(result.errors) == 0
        finally:
            Path(tmp_path).unlink()
    
    def test_validate_fingerprinting_parameters_missing_file(self):
        """Test fingerprinting parameter validation with missing dictionary file"""
        params = {
            'dictionary_path': '/nonexistent/path/dictionary.mat'
        }
        
        result = self.validator.validate_processing_parameters('fingerprinting', params)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "dictionary_path" in result.errors[0].message
        assert "does not exist" in result.errors[0].message
    
    def test_validate_unknown_processing_type(self):
        """Test validation with unknown processing type"""
        result = self.validator.validate_processing_parameters('unknown_type', {})
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Unsupported processing type" in result.errors[0].message
    
    def test_validate_unknown_parameters(self):
        """Test validation with unknown parameters"""
        params = {
            'fit_method': 'WLS',
            'unknown_param': 'value'
        }
        
        result = self.validator.validate_processing_parameters('dti', params)
        
        assert result.is_valid  # Should be valid but with warning
        assert len(result.warnings) == 1
        assert "Unknown parameter" in result.warnings[0].message
        assert "unknown_param" in result.warnings[0].message
    
    def test_validate_configuration_object(self):
        """Test validation of entire configuration object"""
        # Create a simple mock configuration object
        class MockConfig:
            def __init__(self):
                self.study_name = "test_study"
                self.dti = MockDTIConfig()
                self.scheduler = MockSchedulerConfig()
                self.output = MockOutputConfig()
        
        class MockDTIConfig:
            def __init__(self):
                self.fit_method = 'WLS'
                self.mask_threshold = 0.2
                self.compute_metrics = ['FA', 'MD']
        
        class MockSchedulerConfig:
            def __init__(self):
                self.type = 'slurm'
                self.cpus_per_task = 4
                self.mem_per_cpu = 4
                self.time_limit = '24:00:00'
        
        class MockOutputConfig:
            def __init__(self):
                self.derivatives_name = "elikopy"
        
        config = MockConfig()
        result = self.validator.validate_configuration_object(config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_configuration_object_invalid_study_name(self):
        """Test configuration validation with invalid study name"""
        config = Mock()
        config.study_name = "invalid name!"  # Contains invalid characters
        config.output = Mock()
        config.output.derivatives_name = "elikopy"
        
        result = self.validator.validate_configuration_object(config)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "study_name" in result.errors[0].message
        assert "invalid characters" in result.errors[0].message
    
    def test_get_parameter_suggestions(self):
        """Test getting parameter suggestions"""
        suggestions = self.validator.get_parameter_suggestions('dti')
        
        assert 'fit_method' in suggestions
        assert 'mask_threshold' in suggestions
        assert 'compute_metrics' in suggestions
        
        # Check specific parameter
        fit_method_spec = self.validator.get_parameter_suggestions('dti', 'fit_method')
        assert 'fit_method' in fit_method_spec
        assert fit_method_spec['fit_method']['valid_values'] == ['WLS', 'OLS', 'NLLS']
    
    def test_get_parameter_suggestions_unknown_type(self):
        """Test getting parameter suggestions for unknown processing type"""
        suggestions = self.validator.get_parameter_suggestions('unknown')
        assert suggestions == {}
    
    def test_generate_parameter_documentation(self):
        """Test generating parameter documentation"""
        doc = self.validator.generate_parameter_documentation('dti')
        
        assert "DTI processing" in doc
        assert "fit_method" in doc
        assert "mask_threshold" in doc
        assert "Valid values" in doc
        assert "Range" in doc
    
    def test_generate_parameter_documentation_unknown_type(self):
        """Test generating documentation for unknown processing type"""
        doc = self.validator.generate_parameter_documentation('unknown')
        assert "No parameters defined" in doc


class TestGradientTableValidation:
    """Test gradient table validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = ParameterValidator()
    
    def test_validate_gradient_table_valid(self):
        """Test gradient table validation with valid data"""
        bvals = np.array([0, 1000, 1000, 2000, 2000])
        bvecs = np.array([
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0]
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.suggestions) > 0  # Should report shell structure
    
    def test_validate_gradient_table_shape_mismatch(self):
        """Test gradient table validation with shape mismatch"""
        bvals = np.array([0, 1000, 1000])  # 3 values
        bvecs = np.array([
            [0, 1, 0, 1, 0],  # 5 values
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0]
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "length" in result.errors[0].message
        assert "match" in result.errors[0].message
    
    def test_validate_gradient_table_wrong_bvals_shape(self):
        """Test gradient table validation with wrong bvals shape"""
        bvals = np.array([[0, 1000], [1000, 2000]])  # 2D array
        bvecs = np.array([
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "bvals must be 1D" in result.errors[0].message
    
    def test_validate_gradient_table_wrong_bvecs_shape(self):
        """Test gradient table validation with wrong bvecs shape"""
        bvals = np.array([0, 1000, 1000, 2000])
        bvecs = np.array([
            [0, 1, 0, 1],
            [0, 0, 1, 0]  # Only 2 rows instead of 3
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "bvecs must be 3xN" in result.errors[0].message
    
    def test_validate_gradient_table_no_b0(self):
        """Test gradient table validation with no b0 volumes"""
        bvals = np.array([1000, 1000, 2000, 2000])  # No b0
        bvecs = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "No b0 volumes" in result.errors[0].message
    
    def test_validate_gradient_table_few_b0(self):
        """Test gradient table validation with few b0 volumes"""
        bvals = np.array([0, 1000, 1000, 2000, 2000])  # Only 1 b0
        bvecs = np.array([
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0]
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert result.is_valid  # Valid but with warnings
        assert len(result.warnings) >= 1
        assert any("Only 1 b0 volumes" in w.message for w in result.warnings)
    
    def test_validate_gradient_table_non_unit_vectors(self):
        """Test gradient table validation with non-unit vectors"""
        bvals = np.array([0, 1000, 1000, 2000])
        bvecs = np.array([
            [0, 2, 0, 1],  # Non-unit vector
            [0, 0, 2, 0],  # Non-unit vector
            [0, 0, 0, 0]
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert result.is_valid  # Valid but with warnings
        assert len(result.warnings) >= 1
        assert any("not unit vectors" in w.message for w in result.warnings)
    
    def test_validate_gradient_table_insufficient_directions(self):
        """Test gradient table validation with insufficient directions per shell"""
        bvals = np.array([0, 0, 0, 1000, 1000, 2000])  # Only 1 direction for b=2000
        bvecs = np.array([
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        result = self.validator.validate_gradient_table(bvals, bvecs)
        
        assert result.is_valid  # Valid but with warnings
        assert len(result.warnings) == 2  # One for each shell with few directions
        assert any("only 1 directions" in w.message for w in result.warnings)


class TestProcessingCompatibility:
    """Test processing compatibility validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = ParameterValidator()
    
    def test_validate_processing_compatibility_valid(self):
        """Test processing compatibility with valid combination"""
        processing_types = ['dti', 'csd', 'tracking']
        
        result = self.validator.validate_processing_compatibility(processing_types)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_processing_compatibility_csd_msmt_csd(self):
        """Test processing compatibility with both CSD and MSMT-CSD"""
        processing_types = ['csd', 'msmt_csd']
        
        result = self.validator.validate_processing_compatibility(processing_types)
        
        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) == 1
        assert "Both CSD and MSMT-CSD" in result.warnings[0].message
    
    def test_validate_processing_compatibility_connectivity_without_tracking(self):
        """Test processing compatibility with connectivity but no tracking"""
        processing_types = ['dti', 'connectivity']
        
        result = self.validator.validate_processing_compatibility(processing_types)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Connectivity analysis requires tractography" in result.errors[0].message
    
    def test_validate_processing_compatibility_tracking_without_csd(self):
        """Test processing compatibility with tracking but no CSD"""
        processing_types = ['dti', 'tracking']
        
        result = self.validator.validate_processing_compatibility(processing_types)
        
        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) == 1
        assert "Tractography typically requires CSD" in result.warnings[0].message
    
    def test_validate_processing_compatibility_with_gradient_info(self):
        """Test processing compatibility with gradient information"""
        processing_types = ['msmt_csd']
        gradient_info = {
            'shells': [0, 1000],  # Only single shell (excluding b0)
            'multi_shell': False
        }
        
        result = self.validator.validate_processing_compatibility(
            processing_types, gradient_info
        )
        
        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) >= 1
        assert any("MSMT-CSD requires multi-shell data" in w.message for w in result.warnings)
    
    def test_validate_processing_compatibility_noddi_single_shell(self):
        """Test processing compatibility with NODDI and single shell"""
        processing_types = ['noddi']
        gradient_info = {
            'shells': [0, 1000],  # Only single shell (excluding b0)
            'multi_shell': False
        }
        
        result = self.validator.validate_processing_compatibility(
            processing_types, gradient_info
        )
        
        assert result.is_valid  # Valid but with warnings
        assert len(result.warnings) >= 1
        # Should have warnings about both multi-shell requirement and low b-value
        warning_messages = [w.message for w in result.warnings]
        assert any("NODDI typically requires multi-shell" in msg for msg in warning_messages) or \
               any("NODDI benefits from high b-value" in msg for msg in warning_messages)
    
    def test_validate_processing_compatibility_noddi_low_bvalue(self):
        """Test processing compatibility with NODDI and low b-values"""
        processing_types = ['noddi']
        gradient_info = {
            'shells': [0, 1000, 1500],  # Max b-value < 2000
            'multi_shell': True
        }
        
        result = self.validator.validate_processing_compatibility(
            processing_types, gradient_info
        )
        
        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) == 1
        assert "NODDI benefits from high b-value shells" in result.warnings[0].message


if __name__ == "__main__":
    pytest.main([__file__])