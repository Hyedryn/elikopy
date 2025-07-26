"""
Unit tests for CSD processing module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import nibabel as nib
import pytest

from elikopy.processing.csd import CSDProcessor, CSDConfig, CSDResult, MSMTCSDResult
from elikopy.core.base import ProcessingStatus


class TestCSDConfig:
    """Test CSDConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CSDConfig()
        assert config.response_algorithm == "auto"
        assert config.sh_order == 8
        assert config.relative_peak_threshold == 0.5
        assert config.min_separation_angle == 25
        assert config.output_types == ["odf", "peaks", "peak_indices", "peak_values"]
        assert config.auto_mask is True
        assert config.fa_threshold == 0.7
        assert config.roi_radii == 10
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CSDConfig(
            response_algorithm="tournier",
            sh_order=6,
            relative_peak_threshold=0.3,
            min_separation_angle=30,
            output_types=["odf", "peaks"]
        )
        assert config.response_algorithm == "tournier"
        assert config.sh_order == 6
        assert config.relative_peak_threshold == 0.3
        assert config.min_separation_angle == 30
        assert config.output_types == ["odf", "peaks"]


class TestCSDResult:
    """Test CSDResult dataclass"""
    
    def test_csd_result_creation(self):
        """Test CSDResult creation"""
        odf_data = np.random.rand(10, 10, 10, 362)
        peaks = np.random.rand(10, 10, 10, 5, 3)
        peak_values = np.random.rand(10, 10, 10, 5)
        peak_indices = np.random.randint(0, 362, (10, 10, 10, 5))
        
        result = CSDResult(
            odf_data=odf_data,
            peaks=peaks,
            peak_values=peak_values,
            peak_indices=peak_indices
        )
        
        assert result.odf_data.shape == (10, 10, 10, 362)
        assert result.peaks.shape == (10, 10, 10, 5, 3)
        assert result.peak_values.shape == (10, 10, 10, 5)
        assert result.peak_indices.shape == (10, 10, 10, 5)
        assert result.response_function is None
        assert result.mask is None


class TestCSDProcessor:
    """Test CSDProcessor class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = CSDConfig(sh_order=6, auto_mask=False)
        self.processor = CSDProcessor(self.config)
        
        # Create synthetic test data
        self.dwi_data = np.random.rand(10, 10, 10, 64) * 1000 + 100
        self.bvals = np.concatenate([np.zeros(1), np.ones(63) * 1000])
        self.bvecs = np.random.randn(64, 3)
        self.bvecs[0] = [0, 0, 0]  # b0 direction
        # Normalize bvecs
        for i in range(1, 64):
            self.bvecs[i] = self.bvecs[i] / np.linalg.norm(self.bvecs[i])
        
        self.mask = np.ones((10, 10, 10), dtype=np.uint8)
        self.affine = np.eye(4)
    
    def test_initialization_default_config(self):
        """Test processor initialization with default config"""
        processor = CSDProcessor()
        assert processor.config.response_algorithm == "auto"
        assert processor.config.sh_order == 8
        assert processor._model is None
    
    def test_initialization_custom_config(self):
        """Test processor initialization with custom config"""
        config = CSDConfig(sh_order=6, response_algorithm="tournier")
        processor = CSDProcessor(config)
        assert processor.config.sh_order == 6
        assert processor.config.response_algorithm == "tournier"
    
    def test_initialization_invalid_config(self):
        """Test processor initialization with invalid config"""
        config = CSDConfig(sh_order=7)  # Invalid odd SH order
        with pytest.raises(ValueError, match="Invalid CSD configuration"):
            CSDProcessor(config)
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid data"""
        assert self.processor.validate_inputs(
            self.dwi_data, self.bvals, self.bvecs, self.mask
        ) is True
    
    def test_validate_inputs_invalid_dwi_dimensions(self):
        """Test input validation with invalid DWI dimensions"""
        invalid_dwi = np.random.rand(10, 10, 64)  # 3D instead of 4D
        assert self.processor.validate_inputs(
            invalid_dwi, self.bvals, self.bvecs, self.mask
        ) is False
    
    def test_validate_inputs_mismatched_bvals(self):
        """Test input validation with mismatched b-values"""
        invalid_bvals = np.ones(32)  # Wrong number of b-values
        assert self.processor.validate_inputs(
            self.dwi_data, invalid_bvals, self.bvecs, self.mask
        ) is False
    
    def test_validate_inputs_invalid_bvecs(self):
        """Test input validation with invalid b-vectors"""
        invalid_bvecs = np.random.rand(64)  # 1D instead of 2D
        assert self.processor.validate_inputs(
            self.dwi_data, self.bvals, invalid_bvecs, self.mask
        ) is False
    
    def test_validate_inputs_mismatched_mask(self):
        """Test input validation with mismatched mask"""
        invalid_mask = np.ones((5, 5, 5))  # Wrong dimensions
        assert self.processor.validate_inputs(
            self.dwi_data, self.bvals, self.bvecs, invalid_mask
        ) is False
    
    def test_validate_inputs_insufficient_bvals(self):
        """Test input validation with insufficient high b-values"""
        low_bvals = np.ones(64) * 500  # All low b-values
        assert self.processor.validate_inputs(
            self.dwi_data, low_bvals, self.bvecs, self.mask
        ) is False
    
    @patch('elikopy.processing.csd.auto_response_ssst')
    @patch('elikopy.processing.csd.ConstrainedSphericalDeconvModel')
    @patch('elikopy.processing.csd.peaks_from_model')
    def test_fit_model(self, mock_peaks, mock_model_class, mock_response):
        """Test CSD model fitting"""
        # Mock response function estimation
        mock_response.return_value = (np.array([1.0, 0.5, 0.1]), 0.8)
        
        # Mock CSD model
        mock_model = Mock()
        mock_fit = Mock()
        mock_fit.odf.return_value = np.random.rand(10, 10, 10, 362)
        mock_fit.predict.return_value = self.dwi_data
        mock_model.fit.return_value = mock_fit
        mock_model.sphere = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock peaks extraction
        mock_peaks_result = Mock()
        mock_peaks_result.peak_dirs = np.random.rand(10, 10, 10, 5, 3)
        mock_peaks_result.peak_values = np.random.rand(10, 10, 10, 5)
        mock_peaks_result.peak_indices = np.random.randint(0, 362, (10, 10, 10, 5))
        mock_peaks.return_value = mock_peaks_result
        
        result = self.processor.fit_model(
            self.dwi_data, self.bvals, self.bvecs, self.mask
        )
        
        assert isinstance(result, CSDResult)
        assert result.odf_data.shape == (10, 10, 10, 362)
        assert result.peaks.shape == (10, 10, 10, 5, 3)
        assert result.response_function is not None
        
        # Verify method calls
        mock_response.assert_called_once()
        mock_model_class.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_peaks.assert_called_once()
    
    def test_compute_metrics(self):
        """Test metrics computation from CSD results"""
        # Create mock CSD result
        odf_data = np.random.rand(5, 5, 5, 362) + 0.1
        peaks = np.random.rand(5, 5, 5, 5, 3)
        peak_values = np.random.rand(5, 5, 5, 5)
        peak_indices = np.random.randint(0, 362, (5, 5, 5, 5))
        mask = np.ones((5, 5, 5))
        
        csd_result = CSDResult(
            odf_data=odf_data,
            peaks=peaks,
            peak_values=peak_values,
            peak_indices=peak_indices,
            mask=mask
        )
        
        metrics = self.processor.compute_metrics(csd_result)
        
        assert "gfa" in metrics
        assert "num_peaks" in metrics
        assert "primary_peak_amplitude" in metrics
        
        # Check shapes
        assert metrics["gfa"].shape == (5, 5, 5)
        assert metrics["num_peaks"].shape == (5, 5, 5)
        assert metrics["primary_peak_amplitude"].shape == (5, 5, 5)
        
        # Check value ranges
        assert np.all(metrics["gfa"] >= 0)
        assert np.all(metrics["gfa"] <= 1)
        assert np.all(metrics["num_peaks"] >= 0)
    
    @patch('elikopy.processing.csd.CSDProcessor.fit_model')
    @patch('elikopy.processing.csd.median_otsu')
    def test_process_success(self, mock_median_otsu, mock_fit_model):
        """Test successful processing"""
        # Mock auto masking
        mock_median_otsu.return_value = (self.dwi_data, self.mask)
        
        # Mock model fitting
        mock_result = CSDResult(
            odf_data=np.random.rand(10, 10, 10, 362),
            peaks=np.random.rand(10, 10, 10, 5, 3),
            peak_values=np.random.rand(10, 10, 10, 5),
            peak_indices=np.random.randint(0, 362, (10, 10, 10, 5))
        )
        mock_fit_model.return_value = mock_result
        
        # Enable auto masking
        self.processor.config.auto_mask = True
        
        result = self.processor.process(
            dwi_data=self.dwi_data,
            bvals=self.bvals,
            bvecs=self.bvecs,
            affine=self.affine
        )
        
        assert result.status == ProcessingStatus.COMPLETED
        assert isinstance(result.metadata, dict)
        assert result.metadata["processing_method"] == "CSD"
        assert result.error_message is None
    
    def test_process_invalid_inputs(self):
        """Test processing with invalid inputs"""
        invalid_dwi = np.random.rand(10, 10, 64)  # 3D instead of 4D
        
        result = self.processor.process(
            dwi_data=invalid_dwi,
            bvals=self.bvals,
            bvecs=self.bvecs
        )
        
        assert result.status == ProcessingStatus.FAILED
        assert result.error_message == "Input validation failed"
        assert len(result.output_files) == 0
    
    @patch('elikopy.processing.csd.CSDProcessor.fit_model')
    def test_process_with_output_dir(self, mock_fit_model):
        """Test processing with output directory"""
        # Mock model fitting
        mock_result = CSDResult(
            odf_data=np.random.rand(10, 10, 10, 362),
            peaks=np.random.rand(10, 10, 10, 5, 3),
            peak_values=np.random.rand(10, 10, 10, 5),
            peak_indices=np.random.randint(0, 362, (10, 10, 10, 5)),
            response_function=np.array([1.0, 0.5, 0.1])
        )
        mock_fit_model.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            result = self.processor.process(
                dwi_data=self.dwi_data,
                bvals=self.bvals,
                bvecs=self.bvecs,
                mask=self.mask,
                affine=self.affine,
                output_dir=output_dir,
                subject_id="01",
                session_id="01"
            )
            
            assert result.status == ProcessingStatus.COMPLETED
            assert len(result.output_files) > 0
            
            # Check that CSD directory was created
            csd_dir = output_dir / "csd"
            assert csd_dir.exists()
            
            # Check for expected output files
            expected_files = [
                "sub-01_ses-01_model-CSD_odf.nii.gz",
                "sub-01_ses-01_model-CSD_peaks.nii.gz",
                "sub-01_ses-01_model-CSD_response.npy"
            ]
            
            for expected_file in expected_files:
                assert (csd_dir / expected_file).exists()
    
    @patch('elikopy.processing.csd.multi_shell_fiber_response')
    @patch('elikopy.processing.csd.MultiShellDeconvModel')
    @patch('elikopy.processing.csd.peaks_from_model')
    def test_fit_msmt_csd(self, mock_peaks, mock_model_class, mock_response):
        """Test MSMT-CSD model fitting"""
        # Create multi-shell data
        multi_bvals = np.concatenate([
            np.zeros(10),  # b0
            np.ones(30) * 1000,  # Shell 1
            np.ones(30) * 2000   # Shell 2
        ])
        multi_bvecs = np.random.randn(70, 3)
        multi_bvecs[:10] = [0, 0, 0]  # b0 directions
        # Normalize bvecs
        for i in range(10, 70):
            multi_bvecs[i] = multi_bvecs[i] / np.linalg.norm(multi_bvecs[i])
        
        multi_dwi = np.random.rand(5, 5, 5, 70) * 1000 + 100
        
        # Mock response functions
        mock_response.return_value = (
            np.array([1.0, 0.5, 0.1]),  # WM
            np.array([0.8]),            # GM
            np.array([3.0])             # CSF
        )
        
        # Mock MSMT-CSD model
        mock_model = Mock()
        mock_fit = Mock()
        mock_fit.odf.return_value = np.random.rand(5, 5, 5, 362)
        mock_fit.volume_fractions = np.random.rand(5, 5, 5, 3)
        mock_model.fit.return_value = mock_fit
        mock_model.sphere = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock peaks extraction
        mock_peaks_result = Mock()
        mock_peaks_result.peak_dirs = np.random.rand(5, 5, 5, 5, 3)
        mock_peaks_result.peak_values = np.random.rand(5, 5, 5, 5)
        mock_peaks_result.peak_indices = np.random.randint(0, 362, (5, 5, 5, 5))
        mock_peaks.return_value = mock_peaks_result
        
        result = self.processor.fit_msmt_csd(
            multi_dwi, multi_bvals, multi_bvecs, self.mask[:5, :5, :5]
        )
        
        assert isinstance(result, MSMTCSDResult)
        assert result.wm_odf.shape == (5, 5, 5, 362)
        assert result.gm_signal.shape == (5, 5, 5)
        assert result.csf_signal.shape == (5, 5, 5)
        assert "wm" in result.response_functions
        assert "gm" in result.response_functions
        assert "csf" in result.response_functions
    
    def test_fit_msmt_csd_insufficient_shells(self):
        """Test MSMT-CSD with insufficient shells"""
        with pytest.raises(ValueError, match="MSMT-CSD requires multi-shell data"):
            self.processor.fit_msmt_csd(
                self.dwi_data, self.bvals, self.bvecs, self.mask
            )
    
    def test_extract_peaks_no_model(self):
        """Test peak extraction without fitted model"""
        odf_data = np.random.rand(5, 5, 5, 362)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            self.processor.extract_peaks(odf_data)
    
    def test_configure(self):
        """Test processor configuration"""
        new_config = {
            "sh_order": 6,
            "response_algorithm": "tournier",
            "relative_peak_threshold": 0.3
        }
        
        self.processor.configure(new_config)
        
        assert self.processor.config.sh_order == 6
        assert self.processor.config.response_algorithm == "tournier"
        assert self.processor.config.relative_peak_threshold == 0.3
    
    def test_configure_invalid(self):
        """Test configuration with invalid parameters"""
        invalid_config = {"sh_order": 7}  # Invalid odd SH order
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            self.processor.configure(invalid_config)
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        default_config = self.processor.get_default_config()
        
        assert default_config["response_algorithm"] == "auto"
        assert default_config["sh_order"] == 8
        assert default_config["relative_peak_threshold"] == 0.5
        assert default_config["min_separation_angle"] == 25
        assert isinstance(default_config["output_types"], list)
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config"""
        valid_config = {
            "response_algorithm": "auto",
            "sh_order": 8,
            "relative_peak_threshold": 0.5,
            "min_separation_angle": 25,
            "output_types": ["odf", "peaks"]
        }
        
        assert self.processor.validate_config(valid_config) is True
    
    def test_validate_config_invalid_response_algorithm(self):
        """Test configuration validation with invalid response algorithm"""
        invalid_config = {"response_algorithm": "invalid"}
        
        assert self.processor.validate_config(invalid_config) is False
    
    def test_validate_config_invalid_sh_order(self):
        """Test configuration validation with invalid SH order"""
        invalid_configs = [
            {"sh_order": 7},   # Odd
            {"sh_order": 1},   # Too low
            {"sh_order": 14},  # Too high
            {"sh_order": "8"}  # Wrong type
        ]
        
        for config in invalid_configs:
            assert self.processor.validate_config(config) is False
    
    def test_validate_config_invalid_output_types(self):
        """Test configuration validation with invalid output types"""
        invalid_configs = [
            {"output_types": "odf"},  # Not a list
            {"output_types": ["invalid"]},  # Invalid type
            {"output_types": ["odf", "invalid"]}  # Mixed valid/invalid
        ]
        
        for config in invalid_configs:
            assert self.processor.validate_config(config) is False
    
    def test_validate_config_invalid_numeric_params(self):
        """Test configuration validation with invalid numeric parameters"""
        invalid_configs = [
            {"relative_peak_threshold": -0.1},  # Below range
            {"relative_peak_threshold": 1.1},   # Above range
            {"min_separation_angle": -1},       # Below range
            {"min_separation_angle": 91},       # Above range
            {"fa_threshold": 1.5},              # Above range
            {"roi_radii": 0},                   # Below range
            {"roi_radii": 100}                  # Above range
        ]
        
        for config in invalid_configs:
            assert self.processor.validate_config(config) is False
    
    def test_validate_config_invalid_bool_params(self):
        """Test configuration validation with invalid boolean parameters"""
        invalid_configs = [
            {"auto_mask": "true"},     # String instead of bool
            {"save_response": 1},      # Int instead of bool
            {"save_odf": "false"},     # String instead of bool
            {"save_peaks": 0}          # Int instead of bool
        ]
        
        for config in invalid_configs:
            assert self.processor.validate_config(config) is False


class TestCSDProcessorIntegration:
    """Integration tests for CSD processor"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.processor = CSDProcessor()
        
        # Create more realistic synthetic data
        np.random.seed(42)  # For reproducible tests
        self.dwi_data = self._create_synthetic_dwi_data()
        self.bvals, self.bvecs = self._create_gradient_table()
        self.mask = np.ones((8, 8, 8), dtype=np.uint8)
        self.affine = np.eye(4)
    
    def _create_synthetic_dwi_data(self):
        """Create synthetic DWI data with realistic properties"""
        # Simple synthetic data with some structure
        data = np.random.rand(8, 8, 8, 64) * 500 + 500
        
        # Add some signal decay for high b-values
        for i in range(1, 64):  # Skip b0
            data[..., i] *= np.exp(-0.001 * 1000)  # Simple exponential decay
        
        return data
    
    def _create_gradient_table(self):
        """Create realistic gradient table"""
        # 1 b0 + 63 diffusion directions at b=1000
        bvals = np.concatenate([np.zeros(1), np.ones(63) * 1000])
        
        # Random but normalized b-vectors
        bvecs = np.random.randn(64, 3)
        bvecs[0] = [0, 0, 0]  # b0 direction
        
        # Normalize non-b0 directions
        for i in range(1, 64):
            bvecs[i] = bvecs[i] / np.linalg.norm(bvecs[i])
        
        return bvals, bvecs
    
    @patch('elikopy.processing.csd.auto_response_ssst')
    @patch('elikopy.processing.csd.ConstrainedSphericalDeconvModel')
    @patch('elikopy.processing.csd.peaks_from_model')
    def test_full_processing_pipeline(self, mock_peaks, mock_model_class, mock_response):
        """Test complete processing pipeline"""
        # Mock all DIPY components for integration test
        mock_response.return_value = (np.array([1.0, 0.5, 0.1]), 0.8)
        
        mock_model = Mock()
        mock_fit = Mock()
        mock_fit.odf.return_value = np.random.rand(8, 8, 8, 362)
        mock_fit.predict.return_value = self.dwi_data
        mock_model.fit.return_value = mock_fit
        mock_model.sphere = Mock()
        mock_model_class.return_value = mock_model
        
        mock_peaks_result = Mock()
        mock_peaks_result.peak_dirs = np.random.rand(8, 8, 8, 5, 3)
        mock_peaks_result.peak_values = np.random.rand(8, 8, 8, 5)
        mock_peaks_result.peak_indices = np.random.randint(0, 362, (8, 8, 8, 5))
        mock_peaks.return_value = mock_peaks_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            result = self.processor.process(
                dwi_data=self.dwi_data,
                bvals=self.bvals,
                bvecs=self.bvecs,
                mask=self.mask,
                affine=self.affine,
                output_dir=output_dir,
                subject_id="test01",
                session_id="baseline"
            )
            
            # Verify successful processing
            assert result.status == ProcessingStatus.COMPLETED
            assert len(result.output_files) > 0
            assert result.error_message is None
            
            # Verify output directory structure
            csd_dir = output_dir / "csd"
            assert csd_dir.exists()
            
            # Verify specific output files exist
            prefix = "sub-test01_ses-baseline"
            expected_files = [
                f"{prefix}_model-CSD_odf.nii.gz",
                f"{prefix}_model-CSD_odf.json",
                f"{prefix}_model-CSD_peaks.nii.gz",
                f"{prefix}_model-CSD_peaks.json"
            ]
            
            for expected_file in expected_files:
                file_path = csd_dir / expected_file
                assert file_path.exists(), f"Expected file {expected_file} not found"
                
                # Verify JSON files have valid content
                if expected_file.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        assert "Description" in json_data
                        assert "Model" in json_data
                        assert json_data["Model"] == "CSD"