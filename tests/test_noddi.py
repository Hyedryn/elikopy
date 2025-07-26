"""
Unit tests for NODDI processing module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import nibabel as nib

from elikopy.processing.noddi import NODDIProcessor, NODDIConfig, NODDIResult
from elikopy.core.base import ProcessingStatus


class TestNODDIConfig:
    """Test NODDI configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = NODDIConfig()
        
        assert config.lambda_iso_diff == 3.0e-9
        assert config.lambda_par_diff == 1.7e-9
        assert config.use_amico is False
        assert config.use_parallel_processing is True
        assert config.number_of_processors == 4
        assert config.solver == "brute2fine"
        assert config.maxiter == 300
        assert "mu" in config.output_metrics
        assert "odi" in config.output_metrics
        assert "icvf" in config.output_metrics
        assert config.save_quality_control is True
        assert config.auto_mask is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = NODDIConfig(
            lambda_iso_diff=2.5e-9,
            lambda_par_diff=1.5e-9,
            use_amico=True,
            number_of_processors=8,
            output_metrics=["odi", "icvf"]
        )
        
        assert config.lambda_iso_diff == 2.5e-9
        assert config.lambda_par_diff == 1.5e-9
        assert config.use_amico is True
        assert config.number_of_processors == 8
        assert config.output_metrics == ["odi", "icvf"]


class TestNODDIResult:
    """Test NODDI result data structure"""
    
    def test_noddi_result_creation(self):
        """Test NODDIResult creation"""
        shape = (64, 64, 32)
        
        result = NODDIResult(
            mu=np.random.rand(*shape, 3),
            odi=np.random.rand(*shape),
            fiso=np.random.rand(*shape),
            fbundle=np.random.rand(*shape),
            fintra=np.random.rand(*shape),
            icvf=np.random.rand(*shape),
            fextra=np.random.rand(*shape),
            mse=np.random.rand(*shape),
            R2=np.random.rand(*shape),
            mask=np.ones(shape, dtype=bool)
        )
        
        assert result.mu.shape == (*shape, 3)
        assert result.odi.shape == shape
        assert result.icvf.shape == shape
        assert result.mask.shape == shape


class TestNODDIProcessor:
    """Test NODDI processor"""
    
    @pytest.fixture
    def processor(self):
        """Create NODDI processor for testing"""
        return NODDIProcessor()
    
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
        
        return dwi_data, bvals, bvecs, mask
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert isinstance(processor.config, NODDIConfig)
        assert processor._noddi_model is None
    
    def test_processor_initialization_with_config(self):
        """Test processor initialization with custom config"""
        config = NODDIConfig(lambda_iso_diff=2.5e-9, use_amico=True)
        processor = NODDIProcessor(config)
        
        assert processor.config.lambda_iso_diff == 2.5e-9
        assert processor.config.use_amico is True
    
    def test_validate_inputs_valid(self, processor, sample_dwi_data):
        """Test input validation with valid data"""
        dwi_data, bvals, bvecs, mask = sample_dwi_data
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs, mask)
        assert result is True
    
    def test_validate_inputs_invalid_dwi_shape(self, processor):
        """Test input validation with invalid DWI shape"""
        dwi_data = np.random.rand(32, 32, 16)  # 3D instead of 4D
        bvals = np.ones(60)
        bvecs = np.random.randn(3, 60)
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs)
        assert result is False
    
    def test_validate_inputs_mismatched_volumes(self, processor):
        """Test input validation with mismatched volumes"""
        dwi_data = np.random.rand(32, 32, 16, 60)
        bvals = np.ones(50)  # Wrong number of b-values
        bvecs = np.random.randn(3, 60)
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs)
        assert result is False
    
    def test_validate_inputs_insufficient_bvalues(self, processor):
        """Test input validation with insufficient high b-values"""
        dwi_data = np.random.rand(32, 32, 16, 30)
        bvals = np.concatenate([np.zeros(6), np.ones(24) * 500])  # No high b-values
        bvecs = np.random.randn(3, 30)
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs)
        assert result is False
    
    def test_validate_inputs_mask_shape_mismatch(self, processor, sample_dwi_data):
        """Test input validation with wrong mask shape"""
        dwi_data, bvals, bvecs, _ = sample_dwi_data
        wrong_mask = np.ones((16, 16, 8), dtype=bool)  # Wrong shape
        
        result = processor.validate_inputs(dwi_data, bvals, bvecs, wrong_mask)
        assert result is False
    
    @patch('elikopy.processing.noddi.NODDIProcessor._create_noddi_model')
    def test_fit_noddi_model_creation(self, mock_create_model, processor, sample_dwi_data):
        """Test NODDI model creation during fitting"""
        dwi_data, bvals, bvecs, mask = sample_dwi_data
        
        # Mock the model and its methods
        mock_model = Mock()
        mock_fit = Mock()
        mock_fit.fitted_parameters = {
            "SD1WatsonDistributed_1_SD1Watson_1_mu": np.random.rand(*dwi_data.shape[:3], 3),
            "SD1WatsonDistributed_1_SD1Watson_1_odi": np.random.rand(*dwi_data.shape[:3]),
            "partial_volume_0": np.random.rand(*dwi_data.shape[:3]),
            "partial_volume_1": np.random.rand(*dwi_data.shape[:3]),
            "SD1WatsonDistributed_1_partial_volume_0": np.random.rand(*dwi_data.shape[:3])
        }
        mock_fit.mean_squared_error.return_value = np.random.rand(*dwi_data.shape[:3])
        mock_fit.R2_coefficient_of_determination.return_value = np.random.rand(*dwi_data.shape[:3])
        
        mock_model.fit.return_value = mock_fit
        mock_create_model.return_value = mock_model
        
        with patch('elikopy.processing.noddi.gradient_table'), \
             patch('elikopy.processing.noddi.gtab_dipy2dmipy'):
            
            result = processor.fit_noddi(dwi_data, bvals, bvecs, mask)
            
            assert isinstance(result, NODDIResult)
            assert result.odi.shape == dwi_data.shape[:3]
            assert result.icvf.shape == dwi_data.shape[:3]
            mock_create_model.assert_called_once()
    
    def test_compute_metrics(self, processor):
        """Test metrics computation"""
        shape = (32, 32, 16)
        
        noddi_result = NODDIResult(
            mu=np.random.rand(*shape, 3),
            odi=np.random.rand(*shape),
            fiso=np.random.rand(*shape),
            fbundle=np.random.rand(*shape),
            fintra=np.random.rand(*shape),
            icvf=np.random.rand(*shape),
            fextra=np.random.rand(*shape),
            mse=np.random.rand(*shape),
            R2=np.random.rand(*shape),
            mask=np.ones(shape, dtype=bool)
        )
        
        metrics = processor.compute_metrics(noddi_result)
        
        assert "odi" in metrics
        assert "icvf" in metrics
        assert "ndi" in metrics  # Should be same as icvf
        assert "total_tissue_fraction" in metrics
        
        # Check that NDI equals ICVF
        np.testing.assert_array_equal(metrics["ndi"], metrics["icvf"])
    
    def test_compute_metrics_with_mask(self, processor):
        """Test metrics computation with mask applied"""
        shape = (32, 32, 16)
        mask = np.ones(shape, dtype=bool)
        mask[0, 0, 0] = False  # Mask out one voxel
        
        noddi_result = NODDIResult(
            mu=np.random.rand(*shape, 3),
            odi=np.random.rand(*shape),
            fiso=np.random.rand(*shape),
            fbundle=np.random.rand(*shape),
            fintra=np.random.rand(*shape),
            icvf=np.random.rand(*shape),
            fextra=np.random.rand(*shape),
            mse=np.random.rand(*shape),
            R2=np.random.rand(*shape),
            mask=mask
        )
        
        metrics = processor.compute_metrics(noddi_result)
        
        # Check that masked voxels are zero
        assert metrics["odi"][0, 0, 0] == 0
        assert metrics["icvf"][0, 0, 0] == 0
    
    def test_save_outputs(self, processor):
        """Test saving NODDI outputs"""
        shape = (32, 32, 16)
        
        noddi_result = NODDIResult(
            mu=np.random.rand(*shape, 3),
            odi=np.random.rand(*shape),
            fiso=np.random.rand(*shape),
            fbundle=np.random.rand(*shape),
            fintra=np.random.rand(*shape),
            icvf=np.random.rand(*shape),
            fextra=np.random.rand(*shape),
            mse=np.random.rand(*shape),
            R2=np.random.rand(*shape),
            mask=np.ones(shape, dtype=bool)
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            affine = np.eye(4)
            
            output_files = processor._save_outputs(
                noddi_result, output_dir, "sub-01", "ses-01", affine
            )
            
            # Check that files were created
            assert len(output_files) > 0
            
            # Check NODDI directory was created
            noddi_dir = output_dir / "noddi"
            assert noddi_dir.exists()
            
            # Check that NIfTI files were created
            nifti_files = [f for f in output_files if f.suffix == '.gz']
            json_files = [f for f in output_files if f.suffix == '.json']
            
            assert len(nifti_files) > 0
            assert len(json_files) > 0
            
            # Check specific files
            expected_files = [
                "sub-01_ses-01_model-NODDI_parameter-odi.nii.gz",
                "sub-01_ses-01_model-NODDI_parameter-icvf.nii.gz",
                "sub-01_ses-01_model-NODDI_mask.nii.gz"
            ]
            
            for expected_file in expected_files:
                expected_path = noddi_dir / expected_file
                assert expected_path in output_files
                assert expected_path.exists()
            
            # Check JSON metadata
            json_path = noddi_dir / "sub-01_ses-01_model-NODDI_parameter-odi.json"
            assert json_path.exists()
            
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                assert metadata["Model"] == "NODDI"
                assert metadata["Parameter"] == "odi"
    
    @patch('elikopy.processing.noddi.NODDIProcessor.fit_noddi')
    def test_process_success(self, mock_fit_noddi, processor, sample_dwi_data):
        """Test successful processing"""
        dwi_data, bvals, bvecs, mask = sample_dwi_data
        
        # Mock the fit result
        mock_result = NODDIResult(
            mu=np.random.rand(*dwi_data.shape[:3], 3),
            odi=np.random.rand(*dwi_data.shape[:3]),
            fiso=np.random.rand(*dwi_data.shape[:3]),
            fbundle=np.random.rand(*dwi_data.shape[:3]),
            fintra=np.random.rand(*dwi_data.shape[:3]),
            icvf=np.random.rand(*dwi_data.shape[:3]),
            fextra=np.random.rand(*dwi_data.shape[:3]),
            mse=np.random.rand(*dwi_data.shape[:3]),
            R2=np.random.rand(*dwi_data.shape[:3]),
            mask=mask
        )
        mock_fit_noddi.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            affine = np.eye(4)
            
            result = processor.process(
                dwi_data, bvals, bvecs, mask, affine, 
                output_dir, "sub-01", "ses-01"
            )
            
            assert result.status == ProcessingStatus.COMPLETED
            assert len(result.output_files) > 0
            assert "processing_method" in result.metadata
            assert result.metadata["processing_method"] == "NODDI"
    
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
    
    @patch('elikopy.processing.noddi.NODDIProcessor.fit_noddi')
    def test_process_with_auto_mask(self, mock_fit_noddi, processor, sample_dwi_data):
        """Test processing with automatic masking"""
        dwi_data, bvals, bvecs, _ = sample_dwi_data
        
        # Mock the fit result
        mock_result = NODDIResult(
            mu=np.random.rand(*dwi_data.shape[:3], 3),
            odi=np.random.rand(*dwi_data.shape[:3]),
            fiso=np.random.rand(*dwi_data.shape[:3]),
            fbundle=np.random.rand(*dwi_data.shape[:3]),
            fintra=np.random.rand(*dwi_data.shape[:3]),
            icvf=np.random.rand(*dwi_data.shape[:3]),
            fextra=np.random.rand(*dwi_data.shape[:3]),
            mse=np.random.rand(*dwi_data.shape[:3]),
            R2=np.random.rand(*dwi_data.shape[:3]),
            mask=np.ones(dwi_data.shape[:3], dtype=bool)
        )
        mock_fit_noddi.return_value = mock_result
        
        with patch('dipy.segment.mask.median_otsu') as mock_median_otsu:
            mock_median_otsu.return_value = (dwi_data, np.ones(dwi_data.shape[:3], dtype=bool))
            
            result = processor.process(dwi_data, bvals, bvecs, mask=None)
            
            assert result.status == ProcessingStatus.COMPLETED
            mock_median_otsu.assert_called_once()
    
    def test_validate_config_valid(self, processor):
        """Test configuration validation with valid parameters"""
        config = {
            "lambda_iso_diff": 3.0e-9,
            "lambda_par_diff": 1.7e-9,
            "solver": "brute2fine",
            "maxiter": 300,
            "number_of_processors": 4
        }
        
        result = processor.validate_config(config)
        assert result is True
    
    def test_validate_config_invalid_lambda(self, processor):
        """Test configuration validation with invalid lambda values"""
        config = {
            "lambda_iso_diff": 1e-5,  # Too high
            "lambda_par_diff": 1.7e-9
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_solver(self, processor):
        """Test configuration validation with invalid solver"""
        config = {
            "solver": "invalid_solver"
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_maxiter(self, processor):
        """Test configuration validation with invalid maxiter"""
        config = {
            "maxiter": -1
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_processors(self, processor):
        """Test configuration validation with invalid number of processors"""
        config = {
            "number_of_processors": 0
        }
        
        result = processor.validate_config(config)
        assert result is False
    
    @patch('elikopy.processing.noddi.load_nifti')
    @patch('elikopy.processing.noddi.read_bvals_bvecs')
    @patch('elikopy.processing.noddi.NODDIProcessor.fit_noddi')
    def test_process_noddi_solo(self, mock_fit_noddi, mock_read_bvals_bvecs, 
                               mock_load_nifti, processor):
        """Test process_noddi_solo method"""
        # Setup mocks
        shape = (32, 32, 16)
        dwi_data = np.random.rand(*shape, 60)
        affine = np.eye(4)
        bvals = np.concatenate([np.zeros(6), np.ones(54) * 1000])
        bvecs = np.random.randn(3, 60)
        mask = np.ones(shape, dtype=bool)
        
        mock_load_nifti.side_effect = [(dwi_data, affine), (mask, affine)]
        mock_read_bvals_bvecs.return_value = (bvals, bvecs)
        
        mock_result = NODDIResult(
            mu=np.random.rand(*shape, 3),
            odi=np.random.rand(*shape),
            fiso=np.random.rand(*shape),
            fbundle=np.random.rand(*shape),
            fintra=np.random.rand(*shape),
            icvf=np.random.rand(*shape),
            fextra=np.random.rand(*shape),
            mse=np.random.rand(*shape),
            R2=np.random.rand(*shape),
            mask=mask
        )
        mock_fit_noddi.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dwi_path = temp_path / "dwi.nii.gz"
            bval_path = temp_path / "dwi.bval"
            bvec_path = temp_path / "dwi.bvec"
            mask_path = temp_path / "mask.nii.gz"
            output_dir = temp_path / "output"
            
            # Create dummy files
            dwi_path.touch()
            bval_path.touch()
            bvec_path.touch()
            mask_path.touch()
            
            with patch('elikopy.processing.noddi.save_nifti'):
                result = processor.process_noddi_solo(
                    dwi_path, bval_path, bvec_path, mask_path,
                    output_dir, "sub-01"
                )
                
                assert result.status == ProcessingStatus.COMPLETED
                assert result.metadata["subject_id"] == "sub-01"
                assert result.metadata["processing_method"] == "NODDI"


class TestNODDIIntegration:
    """Integration tests for NODDI processing"""
    
    def test_noddi_processor_interface_compliance(self):
        """Test that NODDIProcessor implements required interfaces"""
        processor = NODDIProcessor()
        
        # Check that it has required methods from base classes
        assert hasattr(processor, 'validate_inputs')
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'fit_model')  # From ModelProcessor
        assert hasattr(processor, 'compute_metrics')  # From ModelProcessor
        assert hasattr(processor, 'validate_config')  # From ConfigurableComponent
    
    def test_noddi_config_serialization(self):
        """Test NODDI configuration serialization"""
        config = NODDIConfig(
            lambda_iso_diff=2.5e-9,
            use_amico=True,
            output_metrics=["odi", "icvf"]
        )
        
        # Test that config can be converted to dict (for JSON serialization)
        config_dict = {
            "lambda_iso_diff": config.lambda_iso_diff,
            "lambda_par_diff": config.lambda_par_diff,
            "use_amico": config.use_amico,
            "output_metrics": config.output_metrics
        }
        
        assert config_dict["lambda_iso_diff"] == 2.5e-9
        assert config_dict["use_amico"] is True
        assert "odi" in config_dict["output_metrics"]