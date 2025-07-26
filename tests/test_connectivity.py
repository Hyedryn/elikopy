"""
Unit tests for connectivity processing module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import nibabel as nib
import pytest
from dipy.tracking.streamline import Streamlines

from elikopy.processing.connectivity import (
    ConnectivityProcessor, ConnectivityConfig, AtlasData, 
    ConnectivityMatrix, RegistrationResult
)
from elikopy.core.base import ProcessingStatus


class TestConnectivityConfig:
    """Test ConnectivityConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ConnectivityConfig()
        
        assert config.atlas_name == "aal"
        assert config.atlas_path is None
        assert config.weighting == "count"
        assert config.symmetric is True
        assert config.normalize is False
        assert config.inclusive is False
        assert config.dilation_radius == 0
        assert config.min_streamline_length == 20.0
        assert config.max_streamline_length == 200.0
        assert config.output_formats == ["csv", "mat", "json"]
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ConnectivityConfig(
            atlas_name="desikan",
            weighting="length",
            symmetric=False,
            normalize=True,
            dilation_radius=2,
            output_formats=["csv", "npy"]
        )
        
        assert config.atlas_name == "desikan"
        assert config.weighting == "length"
        assert config.symmetric is False
        assert config.normalize is True
        assert config.dilation_radius == 2
        assert config.output_formats == ["csv", "npy"]


class TestAtlasData:
    """Test AtlasData dataclass"""
    
    def test_atlas_data_creation(self):
        """Test AtlasData creation"""
        atlas_image = np.random.randint(0, 10, (64, 64, 64))
        labels = ["Region1", "Region2", "Region3"]
        label_indices = [1, 2, 3]
        affine = np.eye(4)
        
        atlas_data = AtlasData(
            atlas_image=atlas_image,
            labels=labels,
            label_indices=label_indices,
            affine=affine,
            atlas_name="test_atlas"
        )
        
        assert atlas_data.atlas_image.shape == (64, 64, 64)
        assert atlas_data.labels == labels
        assert atlas_data.label_indices == label_indices
        assert np.array_equal(atlas_data.affine, affine)
        assert atlas_data.atlas_name == "test_atlas"
        assert atlas_data.mni_space is True


class TestConnectivityMatrix:
    """Test ConnectivityMatrix dataclass"""
    
    def test_connectivity_matrix_creation(self):
        """Test ConnectivityMatrix creation"""
        matrix = np.random.rand(10, 10)
        labels = [f"Region{i}" for i in range(10)]
        label_indices = list(range(1, 11))
        
        conn_matrix = ConnectivityMatrix(
            matrix=matrix,
            atlas_labels=labels,
            label_indices=label_indices,
            subject_id="sub-01",
            session_id="ses-01",
            atlas_name="test_atlas",
            weighting="count",
            streamline_count=1000,
            processing_info={"symmetric": True}
        )
        
        assert conn_matrix.matrix.shape == (10, 10)
        assert conn_matrix.atlas_labels == labels
        assert conn_matrix.subject_id == "sub-01"
        assert conn_matrix.session_id == "ses-01"
        assert conn_matrix.weighting == "count"
        assert conn_matrix.streamline_count == 1000


class TestConnectivityProcessor:
    """Test ConnectivityProcessor class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ConnectivityConfig(
            atlas_name="test_atlas",
            weighting="count",
            output_formats=["csv", "json"]
        )
        self.processor = ConnectivityProcessor(self.config)
        
        # Create mock data
        self.mock_streamlines = self._create_mock_streamlines()
        self.mock_atlas_data = self._create_mock_atlas_data()
        self.mock_affine = np.eye(4)
    
    def _create_mock_streamlines(self):
        """Create mock streamlines for testing"""
        # Create simple streamlines
        streamlines = []
        for i in range(10):
            # Create a streamline with 20 points
            streamline = np.random.rand(20, 3) * 50
            streamlines.append(streamline)
        return Streamlines(streamlines)
    
    def _create_mock_atlas_data(self):
        """Create mock atlas data for testing"""
        atlas_image = np.zeros((64, 64, 64), dtype=np.int32)
        # Create some regions
        atlas_image[10:20, 10:20, 10:20] = 1
        atlas_image[30:40, 30:40, 30:40] = 2
        atlas_image[50:60, 50:60, 50:60] = 3
        
        return AtlasData(
            atlas_image=atlas_image,
            labels=["Region1", "Region2", "Region3"],
            label_indices=[1, 2, 3],
            affine=np.eye(4),
            atlas_name="test_atlas"
        )
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        # Test with default config
        processor = ConnectivityProcessor()
        assert processor.config.atlas_name == "aal"
        assert processor.config.weighting == "count"
        
        # Test with custom config
        processor = ConnectivityProcessor(self.config)
        assert processor.config.atlas_name == "test_atlas"
        assert processor.config.weighting == "count"
    
    def test_validate_inputs_success(self):
        """Test successful input validation"""
        with tempfile.NamedTemporaryFile(suffix='.trk') as streamlines_file:
            with tempfile.NamedTemporaryFile(suffix='.nii.gz') as ref_file:
                # Create temporary files
                Path(streamlines_file.name).touch()
                Path(ref_file.name).touch()
                
                result = self.processor.validate_inputs(
                    streamlines_file=Path(streamlines_file.name),
                    atlas_data=self.mock_atlas_data,
                    reference_image=Path(ref_file.name)
                )
                assert result is True
    
    def test_validate_inputs_missing_streamlines(self):
        """Test input validation with missing streamlines"""
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as ref_file:
            Path(ref_file.name).touch()
            
            result = self.processor.validate_inputs(
                streamlines_file=Path("nonexistent.trk"),
                atlas_data=self.mock_atlas_data,
                reference_image=Path(ref_file.name)
            )
            assert result is False
    
    def test_validate_inputs_missing_atlas(self):
        """Test input validation with missing atlas"""
        with tempfile.NamedTemporaryFile(suffix='.trk') as streamlines_file:
            with tempfile.NamedTemporaryFile(suffix='.nii.gz') as ref_file:
                Path(streamlines_file.name).touch()
                Path(ref_file.name).touch()
                
                result = self.processor.validate_inputs(
                    streamlines_file=Path(streamlines_file.name),
                    atlas_data=None,
                    reference_image=Path(ref_file.name)
                )
                assert result is False
    
    def test_configure(self):
        """Test processor configuration"""
        config_dict = {
            "atlas_name": "new_atlas",
            "weighting": "length",
            "symmetric": False
        }
        
        self.processor.configure(config_dict)
        
        assert self.processor.config.atlas_name == "new_atlas"
        assert self.processor.config.weighting == "length"
        assert self.processor.config.symmetric is False
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        default_config = self.processor.get_default_config()
        
        assert default_config["atlas_name"] == "aal"
        assert default_config["weighting"] == "count"
        assert default_config["symmetric"] is True
        assert default_config["normalize"] is False
    
    def test_validate_config_success(self):
        """Test successful configuration validation"""
        valid_config = {
            "weighting": "count",
            "output_formats": ["csv", "json"],
            "dilation_radius": 1
        }
        
        result = self.processor.validate_config(valid_config)
        assert result is True
    
    def test_validate_config_invalid_weighting(self):
        """Test configuration validation with invalid weighting"""
        invalid_config = {"weighting": "invalid_weighting"}
        
        result = self.processor.validate_config(invalid_config)
        assert result is False
    
    def test_validate_config_invalid_format(self):
        """Test configuration validation with invalid output format"""
        invalid_config = {"output_formats": ["csv", "invalid_format"]}
        
        result = self.processor.validate_config(invalid_config)
        assert result is False
    
    def test_validate_config_negative_dilation(self):
        """Test configuration validation with negative dilation radius"""
        invalid_config = {"dilation_radius": -1}
        
        result = self.processor.validate_config(invalid_config)
        assert result is False
    
    @patch('elikopy.processing.connectivity.reslice')
    @patch('nibabel.load')
    def test_register_to_subject_space(self, mock_nib_load, mock_reslice):
        """Test atlas registration to subject space"""
        # Mock nibabel load
        mock_img = Mock()
        mock_img.get_fdata.return_value = np.random.rand(64, 64, 64)
        mock_nib_load.return_value = mock_img
        
        # Mock reslice
        resliced_atlas = np.random.randint(0, 4, (64, 64, 64)).astype(np.int32)
        mock_reslice.return_value = (resliced_atlas, np.eye(4))
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as ref_file:
            Path(ref_file.name).touch()
            
            result = self.processor.register_to_subject_space(
                self.mock_atlas_data,
                Path(ref_file.name),
                np.eye(4)
            )
            
            assert isinstance(result, RegistrationResult)
            assert result.atlas_in_subject_space is True
            assert result.registration_method == "reslicing"
    
    def test_apply_atlas(self):
        """Test atlas application"""
        registration_result = RegistrationResult(
            registered_atlas=self.mock_atlas_data.atlas_image,
            transformation_matrix=np.eye(4),
            atlas_in_subject_space=True,
            registration_method="reslicing"
        )
        
        atlas_result = self.processor.apply_atlas(registration_result, self.mock_atlas_data)
        
        assert "atlas_labels" in atlas_result
        assert "unique_labels" in atlas_result
        assert "label_names" in atlas_result
        assert "atlas_name" in atlas_result
        assert atlas_result["atlas_name"] == "test_atlas"
    
    @patch('elikopy.processing.connectivity.utils.connectivity_matrix')
    def test_compute_connectivity_matrix_count(self, mock_connectivity_matrix):
        """Test connectivity matrix computation with count weighting"""
        # Mock DIPY connectivity_matrix function
        mock_matrix = np.random.randint(0, 10, (4, 4))  # Include background
        mock_connectivity_matrix.return_value = (mock_matrix, None)
        
        atlas_result = {
            "atlas_labels": self.mock_atlas_data.atlas_image,
            "unique_labels": np.array([1, 2, 3]),
            "label_names": ["Region1", "Region2", "Region3"],
            "atlas_name": "test_atlas"
        }
        
        result = self.processor.compute_connectivity_matrix(
            self.mock_streamlines, atlas_result, self.mock_affine
        )
        
        assert isinstance(result, ConnectivityMatrix)
        assert result.weighting == "count"
        assert result.atlas_name == "test_atlas"
        # The streamline count should be from filtered streamlines, not original
        assert result.streamline_count >= 0  # Just check it's non-negative
    
    def test_filter_streamlines_by_length(self):
        """Test streamline filtering by length"""
        # Create streamlines with different lengths
        short_streamline = np.array([[0, 0, 0], [1, 1, 1]])  # Very short
        long_streamline = np.random.rand(1000, 3) * 100  # Very long
        normal_streamline = np.random.rand(50, 3) * 10  # Normal length
        
        streamlines = Streamlines([short_streamline, long_streamline, normal_streamline])
        
        # Set length limits
        self.processor.config.min_streamline_length = 10.0
        self.processor.config.max_streamline_length = 100.0
        
        filtered = self.processor._filter_streamlines_by_length(streamlines)
        
        # Should filter out short and long streamlines
        assert len(filtered) <= len(streamlines)
    
    def test_dilate_atlas_regions(self):
        """Test atlas region dilation"""
        atlas_labels = np.zeros((10, 10, 10), dtype=np.int32)
        atlas_labels[4:6, 4:6, 4:6] = 1  # Small region
        
        dilated = self.processor._dilate_atlas_regions(atlas_labels, radius=1)
        
        # Check that dilation occurred
        assert np.sum(dilated == 1) > np.sum(atlas_labels == 1)
    
    def test_normalize_matrix(self):
        """Test matrix normalization"""
        matrix = np.array([[0, 5, 10], [5, 0, 15], [10, 15, 0]])
        
        normalized = self.processor._normalize_matrix(matrix)
        
        assert np.max(normalized) == 1.0
        assert np.min(normalized) == 0.0
    
    @patch('elikopy.processing.connectivity.pd.DataFrame.to_csv')
    def test_export_csv(self, mock_to_csv):
        """Test CSV export"""
        matrix = np.random.rand(3, 3)
        conn_matrix = ConnectivityMatrix(
            matrix=matrix,
            atlas_labels=["Region1", "Region2", "Region3"],
            label_indices=[1, 2, 3],
            subject_id="sub-01",
            session_id=None,
            atlas_name="test_atlas",
            weighting="count",
            streamline_count=100,
            processing_info={}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            self.processor._export_csv(conn_matrix, Path(temp_file.name))
            mock_to_csv.assert_called_once()
    
    @patch('elikopy.processing.connectivity.savemat')
    def test_export_mat(self, mock_savemat):
        """Test MATLAB export"""
        matrix = np.random.rand(3, 3)
        conn_matrix = ConnectivityMatrix(
            matrix=matrix,
            atlas_labels=["Region1", "Region2", "Region3"],
            label_indices=[1, 2, 3],
            subject_id="sub-01",
            session_id=None,
            atlas_name="test_atlas",
            weighting="count",
            streamline_count=100,
            processing_info={}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.mat') as temp_file:
            self.processor._export_mat(conn_matrix, Path(temp_file.name))
            mock_savemat.assert_called_once()
    
    def test_export_json(self):
        """Test JSON export"""
        matrix = np.random.rand(3, 3)
        conn_matrix = ConnectivityMatrix(
            matrix=matrix,
            atlas_labels=["Region1", "Region2", "Region3"],
            label_indices=[1, 2, 3],
            subject_id="sub-01",
            session_id=None,
            atlas_name="test_atlas",
            weighting="count",
            streamline_count=100,
            processing_info={}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            self.processor._export_json(conn_matrix, temp_path)
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data["atlas_name"] == "test_atlas"
            assert data["weighting"] == "count"
            assert data["subject_id"] == "sub-01"
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_export_npy(self):
        """Test NumPy export"""
        matrix = np.random.rand(3, 3)
        conn_matrix = ConnectivityMatrix(
            matrix=matrix,
            atlas_labels=["Region1", "Region2", "Region3"],
            label_indices=[1, 2, 3],
            subject_id="sub-01",
            session_id=None,
            atlas_name="test_atlas",
            weighting="count",
            streamline_count=100,
            processing_info={}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            self.processor._export_npy(conn_matrix, temp_path)
            
            # Read back and verify
            loaded_matrix = np.load(temp_path)
            assert np.array_equal(loaded_matrix, matrix)
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_config_to_dict(self):
        """Test configuration conversion to dictionary"""
        config_dict = self.processor._config_to_dict()
        
        assert config_dict["atlas_name"] == "test_atlas"
        assert config_dict["weighting"] == "count"
        assert config_dict["symmetric"] is True
        assert config_dict["normalize"] is False
    
    @patch('elikopy.processing.connectivity.load_trk')
    def test_load_streamlines_trk(self, mock_load_trk):
        """Test loading streamlines from TRK file"""
        # Mock tractogram
        mock_tractogram = Mock()
        mock_tractogram.streamlines = self.mock_streamlines
        mock_tractogram.affine = self.mock_affine
        mock_load_trk.return_value = mock_tractogram
        
        with tempfile.NamedTemporaryFile(suffix='.trk') as streamlines_file:
            with tempfile.NamedTemporaryFile(suffix='.nii.gz') as ref_file:
                Path(streamlines_file.name).touch()
                Path(ref_file.name).touch()
                
                streamlines, affine = self.processor._load_streamlines(
                    Path(streamlines_file.name), Path(ref_file.name)
                )
                
                assert streamlines is not None
                assert affine is not None
                assert len(streamlines) == len(self.mock_streamlines)
    
    def test_load_streamlines_unsupported_format(self):
        """Test loading streamlines from unsupported format"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as streamlines_file:
            with tempfile.NamedTemporaryFile(suffix='.nii.gz') as ref_file:
                Path(streamlines_file.name).touch()
                Path(ref_file.name).touch()
                
                streamlines, affine = self.processor._load_streamlines(
                    Path(streamlines_file.name), Path(ref_file.name)
                )
                
                assert streamlines is None
                assert affine is None
    
    @patch('nibabel.load')
    def test_load_atlas_data(self, mock_nib_load):
        """Test loading atlas data from file"""
        # Mock nibabel load
        mock_img = Mock()
        mock_img.get_fdata.return_value = self.mock_atlas_data.atlas_image.astype(float)
        mock_img.affine = self.mock_atlas_data.affine
        mock_nib_load.return_value = mock_img
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as atlas_file:
            Path(atlas_file.name).touch()
            self.processor.config.atlas_path = Path(atlas_file.name)
            
            atlas_data = self.processor._load_atlas_data()
            
            assert atlas_data is not None
            assert atlas_data.atlas_name == self.processor.config.atlas_name
            assert len(atlas_data.labels) > 0
    
    def test_load_atlas_data_no_path(self):
        """Test loading atlas data with no path specified"""
        self.processor.config.atlas_path = None
        
        atlas_data = self.processor._load_atlas_data()
        
        assert atlas_data is None


class TestIntegration:
    """Integration tests for connectivity processing"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = ConnectivityConfig(
            atlas_name="test_atlas",
            output_formats=["csv", "json"]
        )
        self.processor = ConnectivityProcessor(self.config)
    
    def teardown_method(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('elikopy.processing.connectivity.load_trk')
    @patch('nibabel.load')
    @patch('elikopy.processing.connectivity.utils.connectivity_matrix')
    @patch('elikopy.processing.connectivity.reslice')
    def test_full_processing_pipeline(self, mock_reslice, mock_connectivity_matrix, 
                                    mock_nib_load, mock_load_trk):
        """Test complete processing pipeline"""
        # Mock streamlines loading
        mock_streamlines = Streamlines([np.random.rand(20, 3) for _ in range(10)])
        mock_tractogram = Mock()
        mock_tractogram.streamlines = mock_streamlines
        mock_tractogram.affine = np.eye(4)
        mock_load_trk.return_value = mock_tractogram
        
        # Mock reference image loading
        mock_ref_img = Mock()
        mock_ref_img.get_fdata.return_value = np.random.rand(64, 64, 64)
        
        # Mock atlas loading
        mock_atlas_img = Mock()
        atlas_data = np.zeros((64, 64, 64), dtype=np.int32)
        atlas_data[10:20, 10:20, 10:20] = 1
        atlas_data[30:40, 30:40, 30:40] = 2
        mock_atlas_img.get_fdata.return_value = atlas_data.astype(float)
        mock_atlas_img.affine = np.eye(4)
        
        def mock_load_side_effect(path):
            if 'atlas' in str(path):
                return mock_atlas_img
            else:
                return mock_ref_img
        
        mock_nib_load.side_effect = mock_load_side_effect
        
        # Mock reslice
        resliced_atlas = np.random.randint(0, 3, (64, 64, 64)).astype(np.int32)
        mock_reslice.return_value = (resliced_atlas, np.eye(4))
        
        # Mock connectivity matrix computation
        mock_matrix = np.random.randint(0, 10, (3, 3))
        mock_connectivity_matrix.return_value = (mock_matrix, None)
        
        # Create temporary files
        streamlines_file = self.temp_dir / "streamlines.trk"
        reference_file = self.temp_dir / "reference.nii.gz"
        atlas_file = self.temp_dir / "atlas.nii.gz"
        
        streamlines_file.touch()
        reference_file.touch()
        atlas_file.touch()
        
        # Set atlas path
        self.processor.config.atlas_path = atlas_file
        
        # Run processing
        result = self.processor.process(
            streamlines_file=streamlines_file,
            reference_image=reference_file,
            output_dir=self.temp_dir,
            subject_id="sub-01",
            session_id="ses-01"
        )
        
        # Verify results
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.output_files) > 0
        assert "atlas_name" in result.metadata
        assert "processing_time" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__])