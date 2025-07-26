"""
Unit tests for microstructure fingerprinting processing module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import nibabel as nib

from elikopy.processing.fingerprinting import (
    MicrostructureFingerprintingProcessor, 
    FingerprintingConfig, 
    FingerprintingResult,
    FingerprintingDictionary
)
from elikopy.core.base import ProcessingStatus


class TestFingerprintingConfig:
    """Test fingerprinting configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FingerprintingConfig()
        
        assert config.dictionary_path is None
        assert config.peaks_type == "MSMT-CSD"
        assert config.use_parallel_processing is True
        assert config.core_count == 4
        assert config.csf_mask is True
        assert config.ear_mask is False
        assert config.verbose == 3
        assert "frac_f0" in config.output_metrics
        assert "fvf_tot" in config.output_metrics
        assert config.save_quality_control is True
        assert config.save_rgb_maps is True
        assert config.color_order == "rgb"
    
    def test_custom_config(self):
        """Test custom configuration"""
        dict_path = Path("/path/to/dictionary")
        config = FingerprintingConfig(
            dictionary_path=dict_path,
            peaks_type="CSD",
            use_parallel_processing=False,
            core_count=8,
            output_metrics=["frac_f0", "fvf_tot"],
            color_order="brg"
        )
        
        assert config.dictionary_path == dict_path
        assert config.peaks_type == "CSD"
        assert config.use_parallel_processing is False
        assert config.core_count == 8
        assert config.output_metrics == ["frac_f0", "fvf_tot"]
        assert config.color_order == "brg"


class TestFingerprintingResult:
    """Test fingerprinting result data structure"""
    
    def test_fingerprinting_result_creation(self):
        """Test FingerprintingResult creation"""
        shape = (64, 64, 32)
        
        result = FingerprintingResult(
            frac_f0=np.random.rand(*shape),
            fvf_tot=np.random.rand(*shape),
            mse=np.random.rand(*shape),
            r2=np.random.rand(*shape),
            peaks={"f0": np.random.rand(*shape, 3)},
            fractions={"f0": np.random.rand(*shape)},
            numfasc=np.ones(shape, dtype=int),
            mask=np.ones(shape, dtype=bool)
        )
        
        assert result.frac_f0.shape == shape
        assert result.fvf_tot.shape == shape
        assert result.mse.shape == shape
        assert result.r2.shape == shape
        assert result.peaks["f0"].shape == (*shape, 3)
        assert result.numfasc.shape == shape
        assert result.mask.shape == shape


class TestFingerprintingDictionary:
    """Test fingerprinting dictionary data structure"""
    
    def test_dictionary_creation(self):
        """Test FingerprintingDictionary creation"""
        mock_model = Mock()
        dict_path = Path("/path/to/dictionary")
        
        dictionary = FingerprintingDictionary(
            model=mock_model,
            path=dict_path,
            metadata={"test": "value"}
        )
        
        assert dictionary.model == mock_model
        assert dictionary.path == dict_path
        assert dictionary.metadata["test"] == "value"


class TestMicrostructureFingerprintingProcessor:
    """Test microstructure fingerprinting processor"""
    
    @pytest.fixture
    def processor(self):
        """Create fingerprinting processor for testing"""
        return MicrostructureFingerprintingProcessor()
    
    @pytest.fixture
    def sample_dwi_data(self):
        """Create sample DWI data for testing"""
        shape = (32, 32, 16, 60)  # Small volume with 60 directions
        dwi_data = np.random.rand(*shape) * 1000 + 100  # Realistic signal range
        
        # Create realistic b-values and b-vectors
        bvals = np.concatenate([
            np.zeros(6),  # b0 volumes
            np.ones(30) * 1000,  # b=1000 shell
            np.ones(24) * 2000   # b=2000 shell
        ])
        
        bvecs = np.random.randn(3, 60)
        bvecs[:, :6] = 0  # b0 directions
        # Normalize non-b0 directions
        for i in range(6, 60):
            bvecs[:, i] = bvecs[:, i] / np.linalg.norm(bvecs[:, i])
        
        mask = np.ones(shape[:3], dtype=bool)
        # Create a brain-like mask
        center = np.array(shape[:3]) // 2
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    if dist > min(shape[:3]) // 3:
                        mask[i, j, k] = False
        
        # Create sample peaks (2 fascicles)
        peaks = np.random.randn(*shape[:3], 3, 2)
        # Normalize peaks
        for i in range(2):
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        if mask[x, y, z]:
                            norm = np.linalg.norm(peaks[x, y, z, :, i])
                            if norm > 0:
                                peaks[x, y, z, :, i] /= norm
        
        numfasc = np.ones(shape[:3], dtype=int)
        numfasc[~mask] = 0
        
        return dwi_data, bvals, bvecs, mask, peaks, numfasc
    
    @pytest.fixture
    def mock_dictionary_path(self):
        """Create mock dictionary path"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            dict_path = Path(f.name)
        return dict_path
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert isinstance(processor.config, FingerprintingConfig)
        assert processor.dictionary is None
    
    def test_processor_initialization_with_config(self, mock_dictionary_path):
        """Test processor initialization with custom config"""
        config = FingerprintingConfig(
            dictionary_path=mock_dictionary_path,
            peaks_type="CSD",
            use_parallel_processing=False
        )
        processor = MicrostructureFingerprintingProcessor(config)
        
        assert processor.config.dictionary_path == mock_dictionary_path
        assert processor.config.peaks_type == "CSD"
        assert processor.config.use_parallel_processing is False
    
    def test_validate_inputs_valid(self, processor, sample_dwi_data, mock_dictionary_path):
        """Test input validation with valid data"""
        processor.config.dictionary_path = mock_dictionary_path
        dwi_data, bvals, bvecs, mask, peaks, numfasc = sample_dwi_data
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs, mask, peaks, numfasc)
        assert result is True
    
    def test_validate_inputs_no_dictionary(self, processor, sample_dwi_data):
        """Test input validation without dictionary path"""
        dwi_data, bvals, bvecs, mask, peaks, numfasc = sample_dwi_data
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs, mask, peaks, numfasc)
        assert result is False
    
    def test_validate_inputs_invalid_dwi_shape(self, processor, mock_dictionary_path):
        """Test input validation with invalid DWI shape"""
        processor.config.dictionary_path = mock_dictionary_path
        dwi_data = np.random.rand(32, 32, 16)  # 3D instead of 4D
        bvals = np.ones(60)
        bvecs = np.random.randn(3, 60)
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs)
        assert result is False
    
    def test_validate_inputs_mismatched_volumes(self, processor, mock_dictionary_path):
        """Test input validation with mismatched volumes"""
        processor.config.dictionary_path = mock_dictionary_path
        dwi_data = np.random.rand(32, 32, 16, 60)
        bvals = np.ones(50)  # Wrong number of b-values
        bvecs = np.random.randn(3, 60)
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs)
        assert result is False
    
    def test_validate_inputs_mask_shape_mismatch(self, processor, sample_dwi_data, mock_dictionary_path):
        """Test input validation with wrong mask shape"""
        processor.config.dictionary_path = mock_dictionary_path
        dwi_data, bvals, bvecs, _, peaks, numfasc = sample_dwi_data
        wrong_mask = np.ones((16, 16, 8), dtype=bool)  # Wrong shape
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs, wrong_mask, peaks, numfasc)
        assert result is False
    
    def test_validate_inputs_peaks_shape_mismatch(self, processor, sample_dwi_data, mock_dictionary_path):
        """Test input validation with wrong peaks shape"""
        processor.config.dictionary_path = mock_dictionary_path
        dwi_data, bvals, bvecs, mask, _, numfasc = sample_dwi_data
        wrong_peaks = np.random.randn(16, 16, 8, 3, 2)  # Wrong shape
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs, mask, wrong_peaks, numfasc)
        assert result is False
    
    def test_validate_inputs_invalid_peaks_type(self, processor, sample_dwi_data, mock_dictionary_path):
        """Test input validation with invalid peaks type"""
        processor.config.dictionary_path = mock_dictionary_path
        processor.config.peaks_type = "INVALID"
        dwi_data, bvals, bvecs, mask, peaks, numfasc = sample_dwi_data
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs, mask, peaks, numfasc)
        assert result is False
    
    @patch('elikopy.processing.fingerprinting.mf')
    def test_load_dictionary(self, mock_mf, processor, mock_dictionary_path):
        """Test dictionary loading"""
        # Mock the MFModel
        mock_model = Mock()
        mock_mf.MFModel.return_value = mock_model
        
        dictionary = processor.load_dictionary(mock_dictionary_path)
        
        assert isinstance(dictionary, FingerprintingDictionary)
        assert dictionary.model == mock_model
        assert dictionary.path == mock_dictionary_path
        mock_mf.MFModel.assert_called_once_with(str(mock_dictionary_path))
    
    def test_load_dictionary_import_error(self, processor, mock_dictionary_path):
        """Test dictionary loading with import error"""
        with patch('elikopy.processing.fingerprinting.mf', side_effect=ImportError()):
            with pytest.raises(ImportError, match="microstructure-fingerprinting library is required"):
                processor.load_dictionary(mock_dictionary_path)
    
    @patch('elikopy.processing.fingerprinting.mf')
    def test_fit_fingerprinting(self, mock_mf, processor, sample_dwi_data, mock_dictionary_path):
        """Test fingerprinting model fitting"""
        dwi_data, bvals, bvecs, mask, peaks, numfasc = sample_dwi_data
        
        # Mock the MFModel and fit result
        mock_model = Mock()
        mock_fit = Mock()
        mock_fit.frac_f0 = np.random.rand(*dwi_data.shape[:3])
        mock_fit.fvf_tot = np.random.rand(*dwi_data.shape[:3])
        mock_fit.MSE = np.random.rand(*dwi_data.shape[:3])
        mock_fit.R2 = np.random.rand(*dwi_data.shape[:3])
        
        mock_model.fit.return_value = mock_fit
        mock_mf.MFModel.return_value = mock_model
        
        processor.config.dictionary_path = mock_dictionary_path
        
        result = processor.fit_fingerprinting(dwi_data, bvals, bvecs, mask, peaks, numfasc)
        
        assert isinstance(result, FingerprintingResult)
        assert result.frac_f0.shape == dwi_data.shape[:3]
        assert result.fvf_tot.shape == dwi_data.shape[:3]
        assert result.mse.shape == dwi_data.shape[:3]
        assert result.r2.shape == dwi_data.shape[:3]
        
        # Check that fit was called with correct parameters
        mock_model.fit.assert_called_once()
        call_args = mock_model.fit.call_args
        assert np.array_equal(call_args[0][0], dwi_data)  # dwi_data
        assert np.array_equal(call_args[0][1], mask)      # mask
        assert np.array_equal(call_args[0][2], numfasc)   # numfasc
    
    def test_compute_metrics(self, processor):
        """Test metrics computation"""
        shape = (32, 32, 16)
        
        model_params = {
            "frac_f0": np.random.rand(*shape),
            "fvf_tot": np.random.rand(*shape),
            "mse": np.random.rand(*shape),
            "r2": np.random.rand(*shape)
        }
        
        metrics = processor.compute_metrics(model_params)
        
        assert "frac_f0" in metrics
        assert "fvf_tot" in metrics
        assert "fvf_f0" in metrics  # Derived metric
        assert "fvf_f1" in metrics  # Derived metric
        
        # Check derived metrics calculation
        expected_fvf_f0 = model_params["fvf_tot"] * model_params["frac_f0"]
        expected_fvf_f1 = model_params["fvf_tot"] * (1 - model_params["frac_f0"])
        
        np.testing.assert_array_equal(metrics["fvf_f0"], expected_fvf_f0)
        np.testing.assert_array_equal(metrics["fvf_f1"], expected_fvf_f1)
    
    def test_save_outputs(self, processor):
        """Test saving fingerprinting outputs"""
        shape = (32, 32, 16)
        
        fingerprinting_result = FingerprintingResult(
            frac_f0=np.random.rand(*shape),
            fvf_tot=np.random.rand(*shape),
            mse=np.random.rand(*shape),
            r2=np.random.rand(*shape),
            mask=np.ones(shape, dtype=bool)
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            affine = np.eye(4)
            
            output_files = processor._save_outputs(
                fingerprinting_result, output_dir, "sub-01", "ses-01", affine
            )
            
            # Check that files were created
            assert len(output_files) > 0
            
            # Check fingerprinting directory was created
            fingerprinting_dir = output_dir / "fingerprinting"
            assert fingerprinting_dir.exists()
            
            # Check that NIfTI files were created
            nifti_files = [f for f in output_files if f.suffix == '.gz']
            json_files = [f for f in output_files if f.suffix == '.json']
            
            assert len(nifti_files) > 0
            assert len(json_files) > 0
            
            # Check specific files
            expected_files = [
                "sub-01_ses-01_model-MF_parameter-frac_f0.nii.gz",
                "sub-01_ses-01_model-MF_parameter-fvf_tot.nii.gz",
                "sub-01_ses-01_model-MF_mask.nii.gz"
            ]
            
            for expected_file in expected_files:
                expected_path = fingerprinting_dir / expected_file
                assert expected_path in output_files
                assert expected_path.exists()
            
            # Check JSON metadata
            json_path = fingerprinting_dir / "sub-01_ses-01_model-MF_parameter-frac_f0.json"
            assert json_path.exists()
            
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                assert metadata["Model"] == "Microstructure Fingerprinting"
                assert metadata["Parameter"] == "frac_f0"
    
    @patch('elikopy.processing.fingerprinting.mf')
    def test_process_success(self, mock_mf, processor, sample_dwi_data, mock_dictionary_path):
        """Test successful processing"""
        dwi_data, bvals, bvecs, mask, peaks, numfasc = sample_dwi_data
        
        # Mock the MFModel and fit result
        mock_model = Mock()
        mock_fit = Mock()
        mock_fit.frac_f0 = np.random.rand(*dwi_data.shape[:3])
        mock_fit.fvf_tot = np.random.rand(*dwi_data.shape[:3])
        mock_fit.MSE = np.random.rand(*dwi_data.shape[:3])
        mock_fit.R2 = np.random.rand(*dwi_data.shape[:3])
        
        mock_model.fit.return_value = mock_fit
        mock_mf.MFModel.return_value = mock_model
        
        processor.config.dictionary_path = mock_dictionary_path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            affine = np.eye(4)
            
            result = processor.process(
                dwi_data, bvals, bvecs, mask, peaks, numfasc, affine,
                output_dir, "sub-01", "ses-01"
            )
            
            assert result.status == ProcessingStatus.COMPLETED
            assert len(result.output_files) > 0
            assert "processing_method" in result.metadata
            assert result.metadata["processing_method"] == "Microstructure Fingerprinting"
    
    def test_process_validation_failure(self, processor):
        """Test processing with validation failure"""
        # Invalid data (3D instead of 4D)
        dwi_data = np.random.rand(32, 32, 16)
        bvals = np.ones(60)
        bvecs = np.random.randn(3, 60)
        
        result = processor.process(dwi_data, bvals, bvecs)
        
        assert result.status == ProcessingStatus.FAILED
        assert "Input validation failed" in result.error_message
        assert len(result.output_files) == 0
    
    def test_process_missing_peaks(self, processor, sample_dwi_data, mock_dictionary_path):
        """Test processing with missing peaks"""
        processor.config.dictionary_path = mock_dictionary_path
        dwi_data, bvals, bvecs, mask, _, _ = sample_dwi_data
        
        result = processor.process(dwi_data, bvals, bvecs, mask)
        
        assert result.status == ProcessingStatus.FAILED
        assert "Peaks and numfasc are required" in result.error_message
    
    def test_validate_config_valid(self, processor, mock_dictionary_path):
        """Test configuration validation with valid parameters"""
        config = {
            "dictionary_path": str(mock_dictionary_path),
            "peaks_type": "MSMT-CSD",
            "core_count": 4,
            "verbose": 3,
            "color_order": "rgb"
        }
        
        result = processor.validate_config(config)
        assert result is True
    
    def test_validate_config_invalid_dictionary_path(self, processor):
        """Test configuration validation with invalid dictionary path"""
        config = {
            "dictionary_path": "/nonexistent/path"
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_peaks_type(self, processor):
        """Test configuration validation with invalid peaks type"""
        config = {
            "peaks_type": "INVALID_TYPE"
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_core_count(self, processor):
        """Test configuration validation with invalid core count"""
        config = {
            "core_count": 0
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_verbose(self, processor):
        """Test configuration validation with invalid verbose level"""
        config = {
            "verbose": -1
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_color_order(self, processor):
        """Test configuration validation with invalid color order"""
        config = {
            "color_order": "invalid"
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_fit_model_interface_compliance(self, processor, sample_dwi_data, mock_dictionary_path):
        """Test fit_model method for interface compliance"""
        processor.config.dictionary_path = mock_dictionary_path
        dwi_data, bvals, bvecs, mask, _, _ = sample_dwi_data
        
        with patch.object(processor, 'fit_fingerprinting') as mock_fit:
            mock_result = FingerprintingResult(
                frac_f0=np.random.rand(*dwi_data.shape[:3]),
                fvf_tot=np.random.rand(*dwi_data.shape[:3]),
                mse=np.random.rand(*dwi_data.shape[:3]),
                r2=np.random.rand(*dwi_data.shape[:3])
            )
            mock_fit.return_value = mock_result
            
            result = processor.fit_model(dwi_data, bvals, bvecs, mask)
            
            assert isinstance(result, dict)
            assert "frac_f0" in result
            assert "fvf_tot" in result
            assert "mse" in result
            assert "r2" in result


class TestFingerprintingIntegration:
    """Integration tests for fingerprinting processing"""
    
    def test_fingerprinting_processor_interface_compliance(self):
        """Test that MicrostructureFingerprintingProcessor implements required interfaces"""
        processor = MicrostructureFingerprintingProcessor()
        
        # Check that it has required methods from base classes
        assert hasattr(processor, 'validate_inputs')
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'fit_model')  # From ModelProcessor
        assert hasattr(processor, 'compute_metrics')  # From ModelProcessor
        assert hasattr(processor, 'validate_config')  # From ConfigurableComponent
        assert hasattr(processor, 'configure')  # From ConfigurableComponent
        assert hasattr(processor, 'get_default_config')  # From ConfigurableComponent
    
    def test_fingerprinting_config_serialization(self):
        """Test fingerprinting configuration serialization"""
        config = FingerprintingConfig(
            dictionary_path=Path("/path/to/dict"),
            peaks_type="CSD",
            use_parallel_processing=False,
            output_metrics=["frac_f0", "fvf_tot"]
        )
        
        # Test that config can be converted to dict (for JSON serialization)
        config_dict = {
            "dictionary_path": str(config.dictionary_path),
            "peaks_type": config.peaks_type,
            "use_parallel_processing": config.use_parallel_processing,
            "output_metrics": config.output_metrics
        }
        
        assert config_dict["dictionary_path"] == "/path/to/dict"
        assert config_dict["peaks_type"] == "CSD"
        assert config_dict["use_parallel_processing"] is False
        assert "frac_f0" in config_dict["output_metrics"]
    
    def test_configure_method(self):
        """Test processor configuration method"""
        processor = MicrostructureFingerprintingProcessor()
        
        config = {
            "peaks_type": "CSD",
            "use_parallel_processing": False,
            "core_count": 8,
            "verbose": 1
        }
        
        processor.configure(config)
        
        assert processor.config.peaks_type == "CSD"
        assert processor.config.use_parallel_processing is False
        assert processor.config.core_count == 8
        assert processor.config.verbose == 1
    
    def test_get_default_config_method(self):
        """Test get_default_config method"""
        processor = MicrostructureFingerprintingProcessor()
        
        default_config = processor.get_default_config()
        
        assert isinstance(default_config, dict)
        assert "dictionary_path" in default_config
        assert "peaks_type" in default_config
        assert "use_parallel_processing" in default_config
        assert "output_metrics" in default_config
        assert default_config["peaks_type"] == "MSMT-CSD"
        assert default_config["use_parallel_processing"] is True