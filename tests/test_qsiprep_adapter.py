"""
Unit tests for QsiPrepAdapter class
"""

import json
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from elikopy.processing.qsiprep_adapter import QsiPrepAdapter, DWIData, AnatomicalData
from elikopy.data.bids_handler import BIDSHandler, DWIFile, AnatomicalFile
from elikopy.core.base import ValidationResult, ValidationError, ValidationWarning
from elikopy.infrastructure.exceptions import DataValidationError


@pytest.fixture
def mock_qsiprep_dir(tmp_path):
    """Create a mock qsiprep directory structure"""
    qsiprep_dir = tmp_path / "qsiprep"
    qsiprep_dir.mkdir()
    
    # Create dataset_description.json
    dataset_desc = {
        "Name": "QSIPrep outputs",
        "BIDSVersion": "1.4.0",
        "GeneratedBy": [
            {
                "Name": "qsiprep",
                "Version": "0.16.1",
                "CodeURL": "https://github.com/PennLINC/qsiprep"
            }
        ]
    }
    
    with open(qsiprep_dir / "dataset_description.json", "w") as f:
        json.dump(dataset_desc, f)
    
    # Create subject directory structure
    sub_dir = qsiprep_dir / "sub-01"
    sub_dir.mkdir()
    
    ses_dir = sub_dir / "ses-01"
    ses_dir.mkdir()
    
    # Create DWI directory
    dwi_dir = ses_dir / "dwi"
    dwi_dir.mkdir()
    
    # Create anatomical directory
    anat_dir = ses_dir / "anat"
    anat_dir.mkdir()
    
    return qsiprep_dir

@pytest.fixture
def mock_dwi_files(mock_qsiprep_dir):
    """Create mock DWI files"""
    dwi_dir = mock_qsiprep_dir / "sub-01" / "ses-01" / "dwi"
    
    # Create DWI file
    dwi_path = dwi_dir / "sub-01_ses-01_space-ACPC_desc-preproc_dwi.nii.gz"
    
    # Create mock NIfTI data (4D: 64x64x40x30 volumes)
    dwi_data = np.random.rand(64, 64, 40, 30).astype(np.float32)
    dwi_img = nib.Nifti1Image(dwi_data, np.eye(4))
    nib.save(dwi_img, dwi_path)
    
    # Create bval file
    bval_path = dwi_dir / "sub-01_ses-01_space-ACPC_desc-preproc_dwi.bval"
    bvals = np.array([0, 0, 0] + [1000] * 27)  # 3 b=0, 27 b=1000
    np.savetxt(bval_path, bvals, fmt='%d')
    
    # Create bvec file
    bvec_path = dwi_dir / "sub-01_ses-01_space-ACPC_desc-preproc_dwi.bvec"
    bvecs = np.random.rand(3, 30)
    bvecs[:, :3] = 0  # b=0 directions should be zero
    np.savetxt(bvec_path, bvecs, fmt='%.6f')
    
    # Create JSON sidecar
    json_path = dwi_dir / "sub-01_ses-01_space-ACPC_desc-preproc_dwi.json"
    metadata = {
        "RepetitionTime": 2.5,
        "EchoTime": 0.089,
        "FlipAngle": 90,
        "ProcessingSteps": [
            "Susceptibility distortion correction",
            "Eddy current correction",
            "Motion correction"
        ]
    }
    
    with open(json_path, "w") as f:
        json.dump(metadata, f)
    
    return {
        "dwi_path": dwi_path,
        "bval_path": bval_path,
        "bvec_path": bvec_path,
        "json_path": json_path,
        "metadata": metadata
    }

@pytest.fixture
def mock_anat_files(mock_qsiprep_dir):
    """Create mock anatomical files"""
    anat_dir = mock_qsiprep_dir / "sub-01" / "ses-01" / "anat"
    
    # Create T1w file
    t1w_path = anat_dir / "sub-01_ses-01_space-ACPC_desc-preproc_T1w.nii.gz"
    
    # Create mock T1w data (3D: 256x256x256)
    t1w_data = np.random.rand(256, 256, 256).astype(np.float32)
    t1w_img = nib.Nifti1Image(t1w_data, np.eye(4))
    nib.save(t1w_img, t1w_path)
    
    # Create JSON sidecar
    json_path = anat_dir / "sub-01_ses-01_space-ACPC_desc-preproc_T1w.json"
    metadata = {
        "RepetitionTime": 2.3,
        "EchoTime": 0.00456,
        "FlipAngle": 9,
        "ProcessingSteps": [
            "Skull stripping",
            "Bias field correction",
            "Spatial normalization"
        ]
    }
    
    with open(json_path, "w") as f:
        json.dump(metadata, f)
    
    return {
        "t1w_path": t1w_path,
        "json_path": json_path,
        "metadata": metadata
    }

@pytest.fixture
def mock_bids_handler():
    """Create a mock BIDS handler"""
    mock_handler = Mock(spec=BIDSHandler)
    
    # Mock validation result
    mock_validation = Mock()
    mock_validation.is_valid = True
    mock_validation.errors = []
    mock_validation.warnings = []
    mock_handler.validate_qsiprep_structure.return_value = mock_validation
    
    return mock_handler


class TestQsiPrepAdapter:
    """Test cases for QsiPrepAdapter class"""
    
    def test_init_success(self, mock_qsiprep_dir, mock_bids_handler):
        """Test successful initialization of QsiPrepAdapter"""
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        assert adapter.qsiprep_dir == mock_qsiprep_dir
        assert adapter.bids_handler == mock_bids_handler
    
    def test_init_invalid_structure(self, tmp_path):
        """Test initialization with invalid qsiprep structure"""
        invalid_dir = tmp_path / "invalid_qsiprep"
        invalid_dir.mkdir()
        
        with pytest.raises(DataValidationError, match="Failed to initialize BIDS handler"):
            QsiPrepAdapter(invalid_dir)
    
    def test_validate_inputs_success(self, mock_qsiprep_dir, mock_bids_handler):
        """Test successful input validation"""
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        assert adapter.validate_inputs() is True
        # Called once during init and once during validate_inputs
        assert mock_bids_handler.validate_qsiprep_structure.call_count >= 1
    
    def test_validate_inputs_failure(self, mock_qsiprep_dir, mock_bids_handler):
        """Test input validation failure"""
        # Mock validation failure
        mock_validation = Mock()
        mock_validation.is_valid = False
        mock_bids_handler.validate_qsiprep_structure.return_value = mock_validation
        
        # This should raise an error during initialization
        with pytest.raises(DataValidationError, match="Invalid qsiprep directory structure"):
            QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
    
    def test_load_preprocessed_data_success(self, mock_qsiprep_dir, mock_bids_handler, mock_dwi_files):
        """Test successful loading of preprocessed DWI data"""
        # Mock DWI file
        dwi_file = DWIFile(
            path=mock_dwi_files["dwi_path"],
            bval_path=mock_dwi_files["bval_path"],
            bvec_path=mock_dwi_files["bvec_path"],
            json_path=mock_dwi_files["json_path"],
            acquisition_params=mock_dwi_files["metadata"]
        )
        
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file]
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        # Mock brain mask loading
        with patch.object(adapter, '_load_brain_mask', return_value=None):
            dwi_data = adapter.load_preprocessed_data("sub-01", "ses-01")
        
        # Verify DWIData container
        assert isinstance(dwi_data, DWIData)
        assert dwi_data.shape == (64, 64, 40, 30)
        assert dwi_data.n_volumes == 30
        assert dwi_data.n_bvals == 2  # b=0 and b=1000
        assert len(dwi_data.bvals) == 30
        assert dwi_data.bvecs.shape == (3, 30)
        assert dwi_data.metadata == mock_dwi_files["metadata"]
        
        # Verify BIDS handler was called correctly
        mock_bids_handler.get_preprocessed_dwi_files.assert_called_once_with(
            subject="sub-01", session="ses-01", run=None, task=None
        )
    
    def test_load_preprocessed_data_no_files(self, mock_qsiprep_dir, mock_bids_handler):
        """Test loading preprocessed data when no files found"""
        mock_bids_handler.get_preprocessed_dwi_files.return_value = []
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        with pytest.raises(DataValidationError, match="No preprocessed DWI files found"):
            adapter.load_preprocessed_data("sub-01", "ses-01")
    
    def test_load_preprocessed_data_multiple_files(self, mock_qsiprep_dir, mock_bids_handler, mock_dwi_files):
        """Test loading preprocessed data with multiple files (uses first one)"""
        # Create two DWI files
        dwi_file1 = DWIFile(
            path=mock_dwi_files["dwi_path"],
            bval_path=mock_dwi_files["bval_path"],
            bvec_path=mock_dwi_files["bvec_path"],
            json_path=mock_dwi_files["json_path"]
        )
        
        dwi_file2 = DWIFile(
            path=mock_dwi_files["dwi_path"],
            bval_path=mock_dwi_files["bval_path"],
            bvec_path=mock_dwi_files["bvec_path"],
            json_path=mock_dwi_files["json_path"]
        )
        
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file1, dwi_file2]
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        with patch.object(adapter, '_load_brain_mask', return_value=None):
            dwi_data = adapter.load_preprocessed_data("sub-01")
        
        assert isinstance(dwi_data, DWIData)
    
    def test_load_anatomical_data_success(self, mock_qsiprep_dir, mock_bids_handler, mock_anat_files):
        """Test successful loading of anatomical data"""
        # Mock anatomical file
        anat_file = AnatomicalFile(
            path=mock_anat_files["t1w_path"],
            json_path=mock_anat_files["json_path"],
            acquisition_params=mock_anat_files["metadata"]
        )
        
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = [anat_file]
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        # Mock mask and segmentation loading
        with patch.object(adapter, '_load_anatomical_brain_mask', return_value=None), \
             patch.object(adapter, '_load_tissue_segmentation', return_value=None):
            anat_data = adapter.load_anatomical_data("sub-01", "ses-01")
        
        # Verify AnatomicalData container
        assert isinstance(anat_data, AnatomicalData)
        assert anat_data.shape == (256, 256, 256)
        assert anat_data.metadata == mock_anat_files["metadata"]
        
        # Verify BIDS handler was called correctly
        mock_bids_handler.get_preprocessed_anatomical_files.assert_called_once_with(
            subject="sub-01", session="ses-01"
        )
    
    def test_load_anatomical_data_no_files(self, mock_qsiprep_dir, mock_bids_handler):
        """Test loading anatomical data when no files found"""
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = []
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        with pytest.raises(DataValidationError, match="No preprocessed anatomical files found"):
            adapter.load_anatomical_data("sub-01", "ses-01")
    
    def test_get_preprocessing_metadata(self, mock_qsiprep_dir, mock_bids_handler, mock_dwi_files, mock_anat_files):
        """Test getting preprocessing metadata"""
        # Mock dataset description
        dataset_desc_path = mock_qsiprep_dir / "dataset_description.json"
        
        # Mock DWI and anatomical files
        dwi_file = DWIFile(
            path=mock_dwi_files["dwi_path"],
            bval_path=mock_dwi_files["bval_path"],
            bvec_path=mock_dwi_files["bvec_path"],
            json_path=mock_dwi_files["json_path"]
        )
        
        anat_file = AnatomicalFile(
            path=mock_anat_files["t1w_path"],
            json_path=mock_anat_files["json_path"]
        )
        
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file]
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = [anat_file]
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        metadata = adapter.get_preprocessing_metadata("sub-01", "ses-01")
        
        # Verify metadata structure
        assert metadata["subject"] == "sub-01"
        assert metadata["session"] == "ses-01"
        assert metadata["qsiprep_version"] == "0.16.1"
        assert "preprocessing_steps" in metadata
        assert "acquisition_parameters" in metadata
        assert "anatomical_parameters" in metadata
    
    def test_validate_qsiprep_outputs_success(self, mock_qsiprep_dir, mock_bids_handler, mock_dwi_files, mock_anat_files):
        """Test successful validation of qsiprep outputs"""
        # Mock DWI and anatomical files
        dwi_file = DWIFile(
            path=mock_dwi_files["dwi_path"],
            bval_path=mock_dwi_files["bval_path"],
            bvec_path=mock_dwi_files["bvec_path"],
            json_path=mock_dwi_files["json_path"]
        )
        
        anat_file = AnatomicalFile(
            path=mock_anat_files["t1w_path"],
            json_path=mock_anat_files["json_path"]
        )
        
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file]
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = [anat_file]
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        # Mock brain mask check
        with patch.object(adapter, '_check_brain_mask_exists', return_value=True), \
             patch.object(adapter, 'get_preprocessing_metadata', return_value={
                 'qsiprep_version': '0.16.1',
                 'preprocessing_steps': ['step1', 'step2']
             }):
            
            validation_result = adapter.validate_qsiprep_outputs("sub-01", "ses-01")
        
        # Verify validation result
        assert isinstance(validation_result, ValidationResult)
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
    
    def test_validate_qsiprep_outputs_no_dwi_files(self, mock_qsiprep_dir, mock_bids_handler):
        """Test validation when no DWI files found"""
        mock_bids_handler.get_preprocessed_dwi_files.return_value = []
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = []
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        with patch.object(adapter, '_check_brain_mask_exists', return_value=False), \
             patch.object(adapter, 'get_preprocessing_metadata', return_value={}):
            
            validation_result = adapter.validate_qsiprep_outputs("sub-01", "ses-01")
        
        # Verify validation result
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        assert any("No preprocessed DWI files found" in error.message for error in validation_result.errors)
    
    def test_validate_qsiprep_outputs_missing_files(self, mock_qsiprep_dir, mock_bids_handler, tmp_path):
        """Test validation with missing files"""
        # Create DWI file with missing bval/bvec
        missing_bval = tmp_path / "missing.bval"
        missing_bvec = tmp_path / "missing.bvec"
        existing_dwi = tmp_path / "existing.nii.gz"
        
        # Create only the DWI file
        dwi_data = np.random.rand(64, 64, 40, 30).astype(np.float32)
        dwi_img = nib.Nifti1Image(dwi_data, np.eye(4))
        nib.save(dwi_img, existing_dwi)
        
        dwi_file = DWIFile(
            path=existing_dwi,
            bval_path=missing_bval,
            bvec_path=missing_bvec,
            json_path=None
        )
        
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file]
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = []
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        with patch.object(adapter, '_check_brain_mask_exists', return_value=False), \
             patch.object(adapter, 'get_preprocessing_metadata', return_value={}):
            
            validation_result = adapter.validate_qsiprep_outputs("sub-01", "ses-01")
        
        # Verify validation result
        assert validation_result.is_valid is False
        assert len(validation_result.errors) >= 2  # Missing bval and bvec
        assert any("bval file not found" in error.message for error in validation_result.errors)
        assert any("bvec file not found" in error.message for error in validation_result.errors)
    
    def test_validate_qsiprep_outputs_dimension_mismatch(self, mock_qsiprep_dir, mock_bids_handler, tmp_path):
        """Test validation with dimension mismatch between DWI and bvals/bvecs"""
        # Create files with mismatched dimensions
        dwi_path = tmp_path / "dwi.nii.gz"
        bval_path = tmp_path / "dwi.bval"
        bvec_path = tmp_path / "dwi.bvec"
        
        # Create DWI with 30 volumes
        dwi_data = np.random.rand(64, 64, 40, 30).astype(np.float32)
        dwi_img = nib.Nifti1Image(dwi_data, np.eye(4))
        nib.save(dwi_img, dwi_path)
        
        # Create bvals/bvecs with wrong number of values (20 instead of 30)
        bvals = np.array([0, 0, 0] + [1000] * 17)  # 20 values
        bvecs = np.random.rand(3, 20)  # 20 directions
        
        np.savetxt(bval_path, bvals, fmt='%d')
        np.savetxt(bvec_path, bvecs, fmt='%.6f')
        
        dwi_file = DWIFile(
            path=dwi_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            json_path=None
        )
        
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file]
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = []
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        with patch.object(adapter, '_check_brain_mask_exists', return_value=False), \
             patch.object(adapter, 'get_preprocessing_metadata', return_value={}):
            
            validation_result = adapter.validate_qsiprep_outputs("sub-01", "ses-01")
        
        # Verify validation result
        assert validation_result.is_valid is False
        assert any("Mismatch between DWI volumes" in error.message for error in validation_result.errors)
    
    def test_get_available_subjects(self, mock_qsiprep_dir, mock_bids_handler):
        """Test getting available subjects"""
        mock_bids_handler.get_subjects_info.return_value = {
            "sub-01": ["ses-01", "ses-02"],
            "sub-02": ["ses-01"]
        }
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        subjects = adapter.get_available_subjects()
        
        assert subjects == ["sub-01", "sub-02"]
        mock_bids_handler.get_subjects_info.assert_called_once()
    
    def test_get_available_sessions(self, mock_qsiprep_dir, mock_bids_handler):
        """Test getting available sessions for a subject"""
        mock_bids_handler.get_subjects_info.return_value = {
            "sub-01": ["ses-01", "ses-02"],
            "sub-02": ["ses-01"]
        }
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        sessions = adapter.get_available_sessions("sub-01")
        
        assert sessions == ["ses-01", "ses-02"]
        
        # Test with sub- prefix
        sessions = adapter.get_available_sessions("01")
        assert sessions == ["ses-01", "ses-02"]
    
    def test_load_brain_mask(self, mock_qsiprep_dir, mock_bids_handler):
        """Test loading brain mask"""
        # Mock pybids layout
        mock_layout = Mock()
        mock_file = Mock()
        mock_file.path = "/path/to/mask.nii.gz"
        mock_layout.get.return_value = [mock_file]
        
        mock_bids_handler._qsiprep_layout = mock_layout
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        # Mock nibabel load
        mock_mask_data = np.ones((64, 64, 40))
        mock_img = Mock()
        mock_img.get_fdata.return_value = mock_mask_data
        
        with patch('nibabel.load', return_value=mock_img):
            mask = adapter._load_brain_mask("sub-01", "ses-01")
        
        assert mask is not None
        assert mask.shape == (64, 64, 40)
        
        # Verify pybids query
        mock_layout.get.assert_called_once_with(
            subject="01",
            datatype="dwi",
            suffix="mask",
            extension=".nii.gz",
            session="01"
        )
    
    def test_load_brain_mask_not_found(self, mock_qsiprep_dir, mock_bids_handler):
        """Test loading brain mask when not found"""
        # Mock pybids layout with no results
        mock_layout = Mock()
        mock_layout.get.return_value = []
        
        mock_bids_handler._qsiprep_layout = mock_layout
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        mask = adapter._load_brain_mask("sub-01", "ses-01")
        
        assert mask is None
    
    def test_dwi_data_properties(self):
        """Test DWIData container properties"""
        dwi_image = np.random.rand(64, 64, 40, 30)
        bvals = np.array([0, 0, 0] + [1000] * 27)
        bvecs = np.random.rand(3, 30)
        
        dwi_data = DWIData(
            dwi_image=dwi_image,
            bvals=bvals,
            bvecs=bvecs,
            affine=np.eye(4),
            header=None,
            metadata={}
        )
        
        assert dwi_data.shape == (64, 64, 40, 30)
        assert dwi_data.n_volumes == 30
        assert dwi_data.n_bvals == 2  # b=0 and b=1000
    
    def test_anatomical_data_properties(self):
        """Test AnatomicalData container properties"""
        t1w_image = np.random.rand(256, 256, 256)
        
        anat_data = AnatomicalData(
            t1w_image=t1w_image,
            affine=np.eye(4),
            header=None,
            metadata={}
        )
        
        assert anat_data.shape == (256, 256, 256)


class TestQsiPrepAdapterIntegration:
    """Integration tests for QsiPrepAdapter"""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow with real file structure"""
        # Create BIDS root directory with dataset_description.json
        bids_root = tmp_path
        dataset_desc = {
            "Name": "Test BIDS dataset",
            "BIDSVersion": "1.4.0"
        }
        
        with open(bids_root / "dataset_description.json", "w") as f:
            json.dump(dataset_desc, f)
        
        # Create realistic qsiprep directory structure
        qsiprep_dir = tmp_path / "qsiprep"
        qsiprep_dir.mkdir()
        
        # Create qsiprep dataset_description.json
        qsiprep_dataset_desc = {
            "Name": "QSIPrep outputs",
            "BIDSVersion": "1.4.0",
            "GeneratedBy": [{"Name": "qsiprep", "Version": "0.16.1"}]
        }
        
        with open(qsiprep_dir / "dataset_description.json", "w") as f:
            json.dump(qsiprep_dataset_desc, f)
        
        # Create subject structure
        sub_dir = qsiprep_dir / "sub-01" / "ses-01"
        dwi_dir = sub_dir / "dwi"
        anat_dir = sub_dir / "anat"
        
        dwi_dir.mkdir(parents=True)
        anat_dir.mkdir(parents=True)
        
        # Create DWI files
        dwi_data = np.random.rand(32, 32, 20, 10).astype(np.float32)
        dwi_img = nib.Nifti1Image(dwi_data, np.eye(4))
        dwi_path = dwi_dir / "sub-01_ses-01_space-ACPC_desc-preproc_dwi.nii.gz"
        nib.save(dwi_img, dwi_path)
        
        bvals = np.array([0, 0] + [1000] * 8)
        bvecs = np.random.rand(3, 10)
        bvecs[:, :2] = 0
        
        np.savetxt(dwi_dir / "sub-01_ses-01_space-ACPC_desc-preproc_dwi.bval", bvals, fmt='%d')
        np.savetxt(dwi_dir / "sub-01_ses-01_space-ACPC_desc-preproc_dwi.bvec", bvecs, fmt='%.6f')
        
        # Create T1w files
        t1w_data = np.random.rand(128, 128, 128).astype(np.float32)
        t1w_img = nib.Nifti1Image(t1w_data, np.eye(4))
        t1w_path = anat_dir / "sub-01_ses-01_space-ACPC_desc-preproc_T1w.nii.gz"
        nib.save(t1w_img, t1w_path)
        
        # Test with real BIDS handler (if pybids is available)
        try:
            from bids.layout import BIDSLayout
            
            # This test requires pybids to be installed
            adapter = QsiPrepAdapter(qsiprep_dir)
            
            # Test basic functionality
            subjects = adapter.get_available_subjects()
            assert "sub-01" in subjects
            
            sessions = adapter.get_available_sessions("sub-01")
            assert "ses-01" in sessions
            
            # Test validation
            validation_result = adapter.validate_qsiprep_outputs("sub-01", "ses-01")
            assert isinstance(validation_result, ValidationResult)
            
        except ImportError:
            # Skip if pybids not available
            pytest.skip("pybids not available for integration test")


class TestQsiPrepAdapterQualityControl:
    """Test cases for QsiPrepAdapter quality control functionality"""
    
    @pytest.fixture
    def adapter_with_data(self, mock_qsiprep_dir, mock_bids_handler, mock_dwi_files, mock_anat_files):
        """Create adapter with mock data for quality control tests"""
        # Mock DWI and anatomical files
        dwi_file = DWIFile(
            path=mock_dwi_files["dwi_path"],
            bval_path=mock_dwi_files["bval_path"],
            bvec_path=mock_dwi_files["bvec_path"],
            json_path=mock_dwi_files["json_path"]
        )
        
        anat_file = AnatomicalFile(
            path=mock_anat_files["t1w_path"],
            json_path=mock_anat_files["json_path"]
        )
        
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file]
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = [anat_file]
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        return adapter
    
    def test_check_data_completeness_success(self, adapter_with_data):
        """Test successful data completeness check"""
        with patch.object(adapter_with_data, '_check_brain_mask_exists', return_value=True):
            result = adapter_with_data._check_data_completeness("sub-01", "ses-01")
        
        assert result['status'] == 'pass'
        assert result['score'] > 0.8
        assert result['details']['dwi_files_count'] == 1
        assert result['details']['anatomical_files_count'] == 1
        assert result['details']['brain_mask_available'] is True
    
    def test_check_data_completeness_missing_files(self, mock_qsiprep_dir, mock_bids_handler):
        """Test data completeness check with missing files"""
        # Mock no files found
        mock_bids_handler.get_preprocessed_dwi_files.return_value = []
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = []
        
        adapter = QsiPrepAdapter(mock_qsiprep_dir, mock_bids_handler)
        
        with patch.object(adapter, '_check_brain_mask_exists', return_value=False):
            result = adapter._check_data_completeness("sub-01", "ses-01")
        
        assert result['status'] == 'fail'
        assert result['score'] == 0.0
        assert len(result['errors']) > 0
        assert any("No DWI files found" in error for error in result['errors'])
    
    def test_check_dwi_quality_success(self, adapter_with_data):
        """Test successful DWI quality check"""
        # Mock load_preprocessed_data
        mock_dwi_data = DWIData(
            dwi_image=np.random.rand(64, 64, 40, 30).astype(np.float32) * 1000,
            bvals=np.array([0, 0, 0] + [1000] * 27),
            bvecs=np.random.rand(3, 30),
            affine=np.eye(4),
            header=None,
            metadata={}
        )
        # Set b=0 directions to zero
        mock_dwi_data.bvecs[:, :3] = 0
        
        with patch.object(adapter_with_data, 'load_preprocessed_data', return_value=mock_dwi_data):
            result = adapter_with_data._check_dwi_quality("sub-01", "ses-01")
        
        assert result['status'] in ['pass', 'warning']
        assert result['score'] > 0.5
        assert result['details']['n_volumes'] == 30
        assert result['details']['n_bvals'] == 2
        assert result['details']['b0_count'] == 3
    
    def test_check_dwi_quality_poor_data(self, adapter_with_data):
        """Test DWI quality check with poor quality data"""
        # Mock poor quality DWI data
        mock_dwi_data = DWIData(
            dwi_image=np.random.rand(64, 64, 40, 5).astype(np.float32) * 1000,  # Only 5 volumes
            bvals=np.array([1000] * 5),  # No b=0 images
            bvecs=np.random.rand(3, 5),
            affine=np.eye(4),
            header=None,
            metadata={}
        )
        
        with patch.object(adapter_with_data, 'load_preprocessed_data', return_value=mock_dwi_data):
            result = adapter_with_data._check_dwi_quality("sub-01", "ses-01")
        
        assert result['status'] == 'fail'
        assert result['score'] < 0.8
        assert result['details']['b0_count'] == 0
        assert any("No b=0 images found" in error for error in result['errors'])
    
    def test_check_anatomical_quality_success(self, adapter_with_data):
        """Test successful anatomical quality check"""
        # Mock load_anatomical_data
        mock_anat_data = AnatomicalData(
            t1w_image=np.random.rand(256, 256, 256).astype(np.float32) * 1000,
            affine=np.eye(4),
            header=None,
            metadata={},
            brain_mask=np.ones((256, 256, 256)),
            tissue_segmentation=np.random.randint(0, 4, (256, 256, 256))
        )
        
        with patch.object(adapter_with_data, 'load_anatomical_data', return_value=mock_anat_data):
            result = adapter_with_data._check_anatomical_quality("sub-01", "ses-01")
        
        assert result['status'] == 'pass'
        assert result['score'] > 0.8
        assert result['details']['t1w_shape'] == (256, 256, 256)
        assert result['details']['brain_mask_available'] is True
        assert result['details']['tissue_segmentation_available'] is True
    
    def test_check_anatomical_quality_low_resolution(self, adapter_with_data):
        """Test anatomical quality check with low resolution data"""
        # Mock low resolution anatomical data
        mock_anat_data = AnatomicalData(
            t1w_image=np.random.rand(64, 64, 64).astype(np.float32) * 1000,  # Low resolution
            affine=np.eye(4),
            header=None,
            metadata={},
            brain_mask=None,  # No brain mask
            tissue_segmentation=None  # No tissue segmentation
        )
        
        with patch.object(adapter_with_data, 'load_anatomical_data', return_value=mock_anat_data):
            result = adapter_with_data._check_anatomical_quality("sub-01", "ses-01")
        
        assert result['status'] == 'warning'
        assert result['score'] < 1.0
        assert result['details']['brain_mask_available'] is False
        assert result['details']['tissue_segmentation_available'] is False
        assert len(result['warnings']) > 0
    
    def test_check_preprocessing_metadata_quality_complete(self, adapter_with_data):
        """Test metadata quality check with complete metadata"""
        mock_metadata = {
            'qsiprep_version': '0.16.1',
            'preprocessing_steps': ['step1', 'step2', 'step3'],
            'acquisition_parameters': {
                'RepetitionTime': 2.5,
                'EchoTime': 0.089,
                'FlipAngle': 90
            },
            'software_versions': {'qsiprep': '0.16.1'}
        }
        
        with patch.object(adapter_with_data, 'get_preprocessing_metadata', return_value=mock_metadata):
            result = adapter_with_data._check_preprocessing_metadata_quality("sub-01", "ses-01")
        
        assert result['status'] == 'pass'
        assert result['score'] == 1.0
        assert result['details']['qsiprep_version'] == '0.16.1'
        assert result['details']['n_preprocessing_steps'] == 3
        assert result['details']['has_acquisition_params'] is True
    
    def test_check_preprocessing_metadata_quality_incomplete(self, adapter_with_data):
        """Test metadata quality check with incomplete metadata"""
        mock_metadata = {
            'qsiprep_version': None,
            'preprocessing_steps': [],
            'acquisition_parameters': {},
            'software_versions': {}
        }
        
        with patch.object(adapter_with_data, 'get_preprocessing_metadata', return_value=mock_metadata):
            result = adapter_with_data._check_preprocessing_metadata_quality("sub-01", "ses-01")
        
        assert result['status'] == 'warning'
        assert result['score'] < 1.0
        assert len(result['warnings']) > 0
    
    def test_check_motion_parameters_no_files(self, adapter_with_data):
        """Test motion parameter check when no files found"""
        with patch.object(adapter_with_data, '_find_motion_files', return_value=[]):
            result = adapter_with_data._check_motion_parameters("sub-01", "ses-01")
        
        assert result['status'] == 'warning'
        assert result['score'] == 0.8
        assert result['details']['motion_files_found'] == 0
        assert any("No motion parameter files found" in warning for warning in result['warnings'])
    
    def test_check_motion_parameters_with_files(self, adapter_with_data, tmp_path):
        """Test motion parameter check with motion files"""
        # Create mock motion file
        motion_file = tmp_path / "motion.txt"
        
        # Create motion parameters (6 columns: 3 translations, 3 rotations)
        # Low motion case
        motion_params = np.random.randn(100, 6) * 0.1  # Small motion
        np.savetxt(motion_file, motion_params)
        
        with patch.object(adapter_with_data, '_find_motion_files', return_value=[motion_file]):
            result = adapter_with_data._check_motion_parameters("sub-01", "ses-01")
        
        assert result['status'] in ['pass', 'warning']
        assert result['score'] >= 0.8
        assert result['details']['motion_files_found'] == 1
        assert 'motion_stats' in result['details']
    
    def test_check_motion_parameters_high_motion(self, adapter_with_data, tmp_path):
        """Test motion parameter check with high motion"""
        # Create mock motion file with high motion
        motion_file = tmp_path / "motion.txt"
        
        # Create motion parameters with high motion
        motion_params = np.random.randn(100, 6)
        motion_params[:, :3] *= 3.0  # High translation (3mm)
        motion_params[:, 3:] *= 0.03  # High rotation (~2 degrees)
        np.savetxt(motion_file, motion_params)
        
        with patch.object(adapter_with_data, '_find_motion_files', return_value=[motion_file]):
            result = adapter_with_data._check_motion_parameters("sub-01", "ses-01")
        
        assert result['status'] == 'warning'
        assert result['score'] < 0.8
        assert len(result['warnings']) > 0
        assert any("High translation motion detected" in warning for warning in result['warnings'])
    
    def test_check_preprocessing_quality_comprehensive(self, adapter_with_data):
        """Test comprehensive preprocessing quality check"""
        # Mock all individual check methods
        with patch.object(adapter_with_data, '_check_data_completeness') as mock_completeness, \
             patch.object(adapter_with_data, '_check_dwi_quality') as mock_dwi, \
             patch.object(adapter_with_data, '_check_anatomical_quality') as mock_anat, \
             patch.object(adapter_with_data, '_check_preprocessing_metadata_quality') as mock_metadata, \
             patch.object(adapter_with_data, '_check_motion_parameters') as mock_motion:
            
            # Mock return values
            mock_completeness.return_value = {'status': 'pass', 'score': 1.0, 'warnings': [], 'errors': []}
            mock_dwi.return_value = {'status': 'pass', 'score': 0.9, 'warnings': [], 'errors': []}
            mock_anat.return_value = {'status': 'warning', 'score': 0.8, 'warnings': ['test warning'], 'errors': []}
            mock_metadata.return_value = {'status': 'pass', 'score': 1.0, 'warnings': [], 'errors': []}
            mock_motion.return_value = {'status': 'pass', 'score': 0.9, 'warnings': [], 'errors': []}
            
            result = adapter_with_data.check_preprocessing_quality("sub-01", "ses-01")
        
        assert result['subject'] == "sub-01"
        assert result['session'] == "ses-01"
        assert 'overall_quality' in result
        assert 'quality_score' in result
        assert 'checks' in result
        assert 'recommendations' in result
        assert len(result['checks']) == 5
    
    def test_calculate_quality_score(self, adapter_with_data):
        """Test quality score calculation"""
        checks = {
            'data_completeness': {'score': 1.0},
            'dwi_quality': {'score': 0.8},
            'anatomical_quality': {'score': 0.9},
            'metadata_quality': {'score': 0.7},
            'motion_quality': {'score': 0.9}
        }
        
        score = adapter_with_data._calculate_quality_score(checks)
        
        # Expected: 0.3*1.0 + 0.3*0.8 + 0.2*0.9 + 0.1*0.7 + 0.1*0.9 = 0.88
        assert 0.85 <= score <= 0.9
    
    def test_determine_overall_quality(self, adapter_with_data):
        """Test overall quality determination"""
        assert adapter_with_data._determine_overall_quality(0.95) == 'excellent'
        assert adapter_with_data._determine_overall_quality(0.85) == 'good'
        assert adapter_with_data._determine_overall_quality(0.75) == 'acceptable'
        assert adapter_with_data._determine_overall_quality(0.6) == 'poor'
        assert adapter_with_data._determine_overall_quality(0.3) == 'failed'
    
    def test_generate_quality_recommendations(self, adapter_with_data):
        """Test quality recommendation generation"""
        checks = {
            'data_completeness': {
                'status': 'fail',
                'details': {'brain_mask_available': False}
            },
            'dwi_quality': {
                'details': {
                    'b0_count': 1,
                    'n_volumes': 20,
                    'high_bval_count': 0
                }
            },
            'motion_quality': {
                'details': {
                    'motion_stats': {'max_translation_rms': 2.5}
                }
            }
        }
        
        recommendations = adapter_with_data._generate_quality_recommendations(checks)
        
        assert len(recommendations) > 0
        assert any("required DWI files" in rec for rec in recommendations)
        assert any("brain masks" in rec for rec in recommendations)
        assert any("b=0 images" in rec for rec in recommendations)
        assert any("motion" in rec for rec in recommendations)
    
    def test_generate_quality_report(self, adapter_with_data, tmp_path):
        """Test quality report generation"""
        output_path = tmp_path / "quality_report.json"
        
        # Mock check_preprocessing_quality
        mock_qc_results = {
            'subject': 'sub-01',
            'session': 'ses-01',
            'overall_quality': 'good',
            'quality_score': 0.85,
            'checks': {},
            'recommendations': ['test recommendation'],
            'warnings': ['test warning'],
            'errors': []
        }
        
        with patch.object(adapter_with_data, 'check_preprocessing_quality', return_value=mock_qc_results):
            report = adapter_with_data.generate_quality_report("sub-01", "ses-01", output_path)
        
        # Check report structure
        assert 'report_info' in report
        assert 'quality_assessment' in report
        assert 'summary' in report
        
        assert report['report_info']['subject'] == 'sub-01'
        assert report['summary']['overall_quality'] == 'good'
        assert report['summary']['quality_score'] == 0.85
        
        # Check file was saved
        assert output_path.exists()
        
        # Load and verify saved report
        with open(output_path, 'r') as f:
            saved_report = json.load(f)
        
        assert saved_report['summary']['overall_quality'] == 'good'