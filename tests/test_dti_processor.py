"""
Unit tests for DTI processing module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import nibabel as nib
import pytest

from elikopy.processing.dti import DTIProcessor, DTIConfig, DTIResult, DTIMetrics
from elikopy.core.base import ProcessingStatus


class TestDTIConfig:
    """Test DTI configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DTIConfig()
        
        assert config.fit_method == "WLS"
        assert config.mask_threshold == 0.0
        assert config.fa_threshold == 0.2
        assert config.min_signal == 1e-6
        assert config.output_metrics == ["FA", "MD", "AD", "RD", "GA", "RGB"]
        assert config.auto_mask is True
        assert config.mask_median_radius == 4
        assert config.mask_numpass == 1
        assert config.save_tensor is True
        assert config.save_eigenvalues is False
        assert config.save_eigenvectors is False
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DTIConfig(
            fit_method="OLS",
            output_metrics=["FA", "MD"],
            auto_mask=False,
            save_eigenvalues=True
        )
        
        assert config.fit_method == "OLS"
        assert config.output_metrics == ["FA", "MD"]
        assert config.auto_mask is False
        assert config.save_eigenvalues is True


class TestDTIProcessor:
    """Test DTI processor functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = DTIConfig(output_metrics=["FA", "MD"], save_tensor=False)
        self.processor = DTIProcessor(self.config)
        
        # Create synthetic test data
        self.shape = (10, 10, 10, 30)  # Small test volume
        self.dwi_data = np.random.rand(*self.shape).astype(np.float32)
        self.bvals = np.concatenate([np.zeros(3), np.ones(27) * 1000])  # 3 b0, 27 DWI
        self.bvecs = np.random.rand(3, 30)
        self.bvecs[:, :3] = 0  # b0 directions should be zero
        self.mask = np.ones(self.shape[:3], dtype=np.uint8)
        self.affine = np.eye(4)
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = DTIProcessor()
        assert processor.config.fit_method == "WLS"
        assert processor._model is None
        
        # Test with custom config
        config = DTIConfig(fit_method="OLS")
        processor = DTIProcessor(config)
        assert processor.config.fit_method == "OLS"
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid configuration"""
        config = DTIConfig(fit_method="INVALID")
        
        with pytest.raises(ValueError, match="Invalid DTI configuration"):
            DTIProcessor(config)
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid data"""
        assert self.processor.validate_inputs(
            self.dwi_data, self.bvals, self.bvecs, self.mask
        ) is True
    
    def test_validate_inputs_invalid_dwi_dimensions(self):
        """Test input validation with invalid DWI dimensions"""
        invalid_dwi = np.random.rand(10, 10, 10)  # 3D instead of 4D
        
        assert self.processor.validate_inputs(
            invalid_dwi, self.bvals, self.bvecs, self.mask
        ) is False
    
    def test_validate_inputs_invalid_bvals(self):
        """Test input validation with invalid b-values"""
        invalid_bvals = np.random.rand(10, 10)  # 2D instead of 1D
        
        assert self.processor.validate_inputs(
            self.dwi_data, invalid_bvals, self.bvecs, self.mask
        ) is False
    
    def test_validate_inputs_mismatched_volumes(self):
        """Test input validation with mismatched volumes"""
        invalid_bvals = np.ones(20)  # Wrong number of volumes
        
        assert self.processor.validate_inputs(
            self.dwi_data, invalid_bvals, self.bvecs, self.mask
        ) is False
    
    def test_validate_inputs_invalid_bvecs(self):
        """Test input validation with invalid b-vectors"""
        invalid_bvecs = np.random.rand(30)  # 1D instead of 2D
        
        assert self.processor.validate_inputs(
            self.dwi_data, self.bvals, invalid_bvecs, self.mask
        ) is False
    
    def test_validate_inputs_invalid_mask_shape(self):
        """Test input validation with invalid mask shape"""
        invalid_mask = np.ones((5, 5, 5), dtype=np.uint8)  # Wrong shape
        
        assert self.processor.validate_inputs(
            self.dwi_data, self.bvals, self.bvecs, invalid_mask
        ) is False
    
    def test_validate_inputs_insufficient_bvals(self):
        """Test input validation with insufficient b-values"""
        invalid_bvals = np.ones(30) * 1000  # All same b-value
        
        assert self.processor.validate_inputs(
            self.dwi_data, invalid_bvals, self.bvecs, self.mask
        ) is False
    
    def test_validate_inputs_transposed_bvecs(self):
        """Test input validation with transposed b-vectors"""
        transposed_bvecs = self.bvecs.T  # (30, 3) instead of (3, 30)
        
        assert self.processor.validate_inputs(
            self.dwi_data, self.bvals, transposed_bvecs, self.mask
        ) is True
    
    @patch('elikopy.processing.dti.gradient_table')
    @patch('elikopy.processing.dti.TensorModel')
    def test_fit_model(self, mock_tensor_model, mock_gradient_table):
        """Test tensor model fitting"""
        # Mock gradient table
        mock_gtab = Mock()
        mock_gradient_table.return_value = mock_gtab
        
        # Mock tensor model and fit
        mock_model = Mock()
        mock_fit = Mock()
        mock_fit.quadratic_form = np.random.rand(*self.shape[:3], 6)
        mock_fit.evals = np.random.rand(*self.shape[:3], 3)
        mock_fit.evecs = np.random.rand(*self.shape[:3], 3, 3)
        mock_model.fit.return_value = mock_fit
        mock_tensor_model.return_value = mock_model
        
        # Test fitting
        result = self.processor.fit_model(
            self.dwi_data, self.bvals, self.bvecs, self.mask
        )
        
        # Verify calls
        mock_gradient_table.assert_called_once_with(self.bvals, self.bvecs)
        mock_tensor_model.assert_called_once_with(mock_gtab, fit_method="WLS")
        mock_model.fit.assert_called_once()
        
        # Verify result
        assert isinstance(result, DTIResult)
        assert result.tensor_data is not None
        assert result.eigenvalues is not None
        assert result.eigenvectors is not None
        assert result.mask is self.mask
    
    @patch('elikopy.processing.dti.fractional_anisotropy')
    @patch('elikopy.processing.dti.mean_diffusivity')
    def test_compute_metrics(self, mock_md, mock_fa):
        """Test DTI metrics computation"""
        # Create mock DTI result
        dti_result = DTIResult(
            tensor_data=np.random.rand(*self.shape[:3], 6),
            eigenvalues=np.random.rand(*self.shape[:3], 3),
            eigenvectors=np.random.rand(*self.shape[:3], 3, 3),
            mask=self.mask
        )
        
        # Mock metric functions
        mock_fa.return_value = np.random.rand(*self.shape[:3])
        mock_md.return_value = np.random.rand(*self.shape[:3])
        
        # Compute metrics
        metrics = self.processor.compute_metrics(dti_result)
        
        # Verify calls
        mock_fa.assert_called_once_with(dti_result.eigenvalues)
        mock_md.assert_called_once_with(dti_result.eigenvalues)
        
        # Verify result
        assert isinstance(metrics, DTIMetrics)
        assert metrics.fa is not None
        assert metrics.md is not None
    
    @patch('elikopy.processing.dti.median_otsu')
    def test_process_with_auto_mask(self, mock_median_otsu):
        """Test processing with automatic masking"""
        # Mock median_otsu
        mock_mask = np.ones(self.shape[:3], dtype=np.uint8)
        mock_median_otsu.return_value = (None, mock_mask)
        
        # Configure for auto masking
        self.processor.config.auto_mask = True
        
        with patch.object(self.processor, 'fit_model') as mock_fit, \
             patch.object(self.processor, 'compute_metrics') as mock_compute:
            
            # Mock returns
            mock_dti_result = Mock()
            mock_dti_metrics = Mock()
            mock_fit.return_value = mock_dti_result
            mock_compute.return_value = mock_dti_metrics
            
            # Process without mask
            result = self.processor.process(
                self.dwi_data, self.bvals, self.bvecs
            )
            
            # Verify auto masking was called
            mock_median_otsu.assert_called_once()
            
            # Verify processing completed
            assert result.status == ProcessingStatus.COMPLETED
    
    def test_process_validation_failure(self):
        """Test processing with validation failure"""
        # Use invalid data
        invalid_dwi = np.random.rand(10, 10, 10)  # 3D instead of 4D
        
        result = self.processor.process(
            invalid_dwi, self.bvals, self.bvecs
        )
        
        assert result.status == ProcessingStatus.FAILED
        assert "Input validation failed" in result.error_message
    
    def test_process_with_output_dir(self):
        """Test processing with output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with patch.object(self.processor, 'fit_model') as mock_fit, \
                 patch.object(self.processor, 'compute_metrics') as mock_compute, \
                 patch.object(self.processor, '_save_outputs') as mock_save:
                
                # Mock returns
                mock_dti_result = Mock()
                mock_dti_metrics = Mock()
                mock_fit.return_value = mock_dti_result
                mock_compute.return_value = mock_dti_metrics
                mock_save.return_value = [Path("test_output.nii.gz")]
                
                # Process with output directory
                result = self.processor.process(
                    self.dwi_data, self.bvals, self.bvecs, self.mask,
                    output_dir=output_dir, subject_id="test01", session_id="ses01"
                )
                
                # Verify save was called
                mock_save.assert_called_once_with(
                    mock_dti_result, mock_dti_metrics, output_dir,
                    "test01", "ses01", None
                )
                
                # Verify result
                assert result.status == ProcessingStatus.COMPLETED
                assert len(result.output_files) == 1
    
    def test_save_outputs(self):
        """Test BIDS-compliant output saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create mock DTI result and metrics
            dti_result = DTIResult(
                tensor_data=np.random.rand(*self.shape[:3], 6),
                eigenvalues=np.random.rand(*self.shape[:3], 3),
                eigenvectors=np.random.rand(*self.shape[:3], 3, 3),
                mask=self.mask
            )
            
            dti_metrics = DTIMetrics(
                fa=np.random.rand(*self.shape[:3]),
                md=np.random.rand(*self.shape[:3])
            )
            
            # Save outputs
            output_files = self.processor._save_outputs(
                dti_result, dti_metrics, output_dir,
                "sub-test01", "ses-01", self.affine
            )
            
            # Verify directory structure
            dti_dir = output_dir / "dti"
            assert dti_dir.exists()
            
            # Verify files were created
            assert len(output_files) > 0
            
            # Check for expected files (based on config: FA, MD metrics, no tensor saving)
            expected_files = [
                "sub-test01_ses-01_model-DTI_parameter-FA.nii.gz",
                "sub-test01_ses-01_model-DTI_parameter-FA.json",
                "sub-test01_ses-01_model-DTI_parameter-MD.nii.gz",
                "sub-test01_ses-01_model-DTI_parameter-MD.json",
                "sub-test01_ses-01_model-DTI_mask.nii.gz"
            ]
            
            for expected_file in expected_files:
                file_path = dti_dir / expected_file
                assert file_path.exists(), f"Expected file not found: {expected_file}"
                
                # Verify JSON files have valid content
                if expected_file.endswith('.json'):
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        assert "Description" in json_data
                        assert "Model" in json_data
                        assert json_data["Model"] == "DTI"
    
    def test_configure(self):
        """Test processor configuration"""
        new_config = {
            "fit_method": "OLS",
            "output_metrics": ["FA", "MD", "AD"],
            "auto_mask": False
        }
        
        self.processor.configure(new_config)
        
        assert self.processor.config.fit_method == "OLS"
        assert self.processor.config.output_metrics == ["FA", "MD", "AD"]
        assert self.processor.config.auto_mask is False
    
    def test_configure_invalid(self):
        """Test configuration with invalid parameters"""
        invalid_config = {
            "fit_method": "INVALID_METHOD"
        }
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            self.processor.configure(invalid_config)
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        default_config = self.processor.get_default_config()
        
        assert default_config["fit_method"] == "WLS"
        assert default_config["output_metrics"] == ["FA", "MD", "AD", "RD", "GA", "RGB"]
        assert default_config["auto_mask"] is True
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid parameters"""
        valid_config = {
            "fit_method": "WLS",
            "output_metrics": ["FA", "MD"],
            "mask_threshold": 0.1,
            "auto_mask": True
        }
        
        assert self.processor.validate_config(valid_config) is True
    
    def test_validate_config_invalid_fit_method(self):
        """Test configuration validation with invalid fit method"""
        invalid_config = {
            "fit_method": "INVALID_METHOD"
        }
        
        assert self.processor.validate_config(invalid_config) is False
    
    def test_validate_config_invalid_metrics(self):
        """Test configuration validation with invalid metrics"""
        invalid_config = {
            "output_metrics": ["FA", "INVALID_METRIC"]
        }
        
        assert self.processor.validate_config(invalid_config) is False
    
    def test_validate_config_invalid_numeric_params(self):
        """Test configuration validation with invalid numeric parameters"""
        invalid_configs = [
            {"mask_threshold": -1.0},  # Negative threshold
            {"fa_threshold": 2.0},     # FA > 1
            {"mask_median_radius": 0}, # Radius < 1
            {"mask_numpass": 0}        # Numpass < 1
        ]
        
        for invalid_config in invalid_configs:
            assert self.processor.validate_config(invalid_config) is False
    
    def test_validate_config_invalid_boolean_params(self):
        """Test configuration validation with invalid boolean parameters"""
        invalid_config = {
            "auto_mask": "true"  # String instead of boolean
        }
        
        assert self.processor.validate_config(invalid_config) is False


class TestDTIResult:
    """Test DTI result data structure"""
    
    def test_dti_result_creation(self):
        """Test DTI result creation"""
        shape = (10, 10, 10)
        
        result = DTIResult(
            tensor_data=np.random.rand(*shape, 6),
            eigenvalues=np.random.rand(*shape, 3),
            eigenvectors=np.random.rand(*shape, 3, 3),
            mask=np.ones(shape, dtype=np.uint8)
        )
        
        assert result.tensor_data.shape == (*shape, 6)
        assert result.eigenvalues.shape == (*shape, 3)
        assert result.eigenvectors.shape == (*shape, 3, 3)
        assert result.mask.shape == shape
        assert result.fit_quality is None


class TestDTIMetrics:
    """Test DTI metrics data structure"""
    
    def test_dti_metrics_creation(self):
        """Test DTI metrics creation"""
        shape = (10, 10, 10)
        
        metrics = DTIMetrics(
            fa=np.random.rand(*shape),
            md=np.random.rand(*shape),
            ad=np.random.rand(*shape),
            rd=np.random.rand(*shape)
        )
        
        assert metrics.fa.shape == shape
        assert metrics.md.shape == shape
        assert metrics.ad.shape == shape
        assert metrics.rd.shape == shape
        assert metrics.ga is None
        assert metrics.rgb is None
    
    def test_dti_metrics_default(self):
        """Test default DTI metrics"""
        metrics = DTIMetrics()
        
        assert metrics.fa is None
        assert metrics.md is None
        assert metrics.ad is None
        assert metrics.rd is None
        assert metrics.ga is None
        assert metrics.rgb is None


if __name__ == "__main__":
    pytest.main([__file__])