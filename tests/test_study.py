"""
Unit tests for ElikopyStudy class
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from elikopy.core.study import ElikopyStudy, Subject
from elikopy.core.config import ElikopyConfig
from elikopy.core.base import DWIFile
from elikopy.data.bids_handler import BIDSHandler, AnatomicalFile, BIDSValidationResult


class TestSubject:
    """Test cases for Subject class"""
    
    def test_subject_initialization(self):
        """Test Subject initialization"""
        subject = Subject("sub-01", ["ses-01", "ses-02"])
        
        assert subject.id == "sub-01"
        assert subject.sessions == ["ses-01", "ses-02"]
        assert subject.metadata == {}
        assert subject.dwi_files == []
        assert subject.anatomical_files == []
    
    def test_subject_initialization_no_sessions(self):
        """Test Subject initialization without sessions"""
        subject = Subject("sub-01")
        
        assert subject.id == "sub-01"
        assert subject.sessions == [""]
        assert subject.metadata == {}
    
    def test_subject_add_dwi_file(self):
        """Test adding DWI file to subject"""
        subject = Subject("sub-01")
        dwi_file = DWIFile(
            path=Path("/test/dwi.nii.gz"),
            bval_path=Path("/test/dwi.bval"),
            bvec_path=Path("/test/dwi.bvec"),
            json_path=Path("/test/dwi.json")
        )
        
        subject.add_dwi_file(dwi_file)
        
        assert len(subject.dwi_files) == 1
        assert subject.dwi_files[0] == dwi_file
    
    def test_subject_add_anatomical_file(self):
        """Test adding anatomical file to subject"""
        subject = Subject("sub-01")
        anat_file = AnatomicalFile(
            path=Path("/test/T1w.nii.gz"),
            json_path=Path("/test/T1w.json")
        )
        
        subject.add_anatomical_file(anat_file)
        
        assert len(subject.anatomical_files) == 1
        assert subject.anatomical_files[0] == anat_file
    
    def test_subject_get_dwi_files(self):
        """Test getting DWI files from subject"""
        subject = Subject("sub-01")
        dwi_file = DWIFile(
            path=Path("/test/dwi.nii.gz"),
            bval_path=Path("/test/dwi.bval"),
            bvec_path=Path("/test/dwi.bvec"),
            json_path=Path("/test/dwi.json")
        )
        subject.add_dwi_file(dwi_file)
        
        files = subject.get_dwi_files()
        assert len(files) == 1
        assert files[0] == dwi_file
    
    def test_subject_get_anatomical_files(self):
        """Test getting anatomical files from subject"""
        subject = Subject("sub-01")
        anat_file = AnatomicalFile(
            path=Path("/test/T1w.nii.gz"),
            json_path=Path("/test/T1w.json")
        )
        subject.add_anatomical_file(anat_file)
        
        files = subject.get_anatomical_files()
        assert len(files) == 1
        assert files[0] == anat_file
    
    def test_subject_repr(self):
        """Test Subject string representation"""
        subject = Subject("sub-01", ["ses-01"])
        dwi_file = DWIFile(
            path=Path("/test/dwi.nii.gz"),
            bval_path=Path("/test/dwi.bval"),
            bvec_path=Path("/test/dwi.bvec"),
            json_path=Path("/test/dwi.json")
        )
        subject.add_dwi_file(dwi_file)
        
        repr_str = repr(subject)
        assert "sub-01" in repr_str
        assert "ses-01" in repr_str
        assert "dwi_files=1" in repr_str
        assert "anat_files=0" in repr_str


class TestElikopyStudy:
    """Test cases for ElikopyStudy class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.study_path = Path(self.temp_dir) / "study"
        self.bids_root = Path(self.temp_dir) / "bids"
        self.qsiprep_dir = Path(self.temp_dir) / "qsiprep"
        
        # Create directories
        self.bids_root.mkdir(parents=True)
        self.qsiprep_dir.mkdir(parents=True)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_study_initialization(self):
        """Test ElikopyStudy initialization"""
        config = ElikopyConfig()
        study = ElikopyStudy(
            study_path=self.study_path,
            bids_root=self.bids_root,
            derivatives_name="test_elikopy",
            config=config
        )
        
        assert study.study_path == self.study_path
        assert study.bids_root == self.bids_root
        assert study.derivatives_name == "test_elikopy"
        assert study.config == config
        assert study.subjects == {}
        assert study.bids_handler is None
        assert study.qsiprep_dir is None
        assert study.derivatives_dir is None
        assert study.study_path.exists()
    
    def test_study_initialization_minimal(self):
        """Test ElikopyStudy initialization with minimal parameters"""
        study = ElikopyStudy(study_path=self.study_path)
        
        assert study.study_path == self.study_path
        assert study.bids_root is None
        assert study.derivatives_name == "elikopy"
        assert isinstance(study.config, ElikopyConfig)
        assert study.study_path.exists()
    
    @patch('elikopy.core.study.BIDSHandler')
    def test_setup_from_qsiprep_success(self, mock_bids_handler_class):
        """Test successful setup from qsiprep"""
        # Mock BIDSHandler
        mock_bids_handler = Mock()
        mock_bids_handler.validate_qsiprep_structure.return_value = BIDSValidationResult(
            is_valid=True, errors=[], warnings=[]
        )
        mock_bids_handler.create_derivatives_structure.return_value = Path("/derivatives/elikopy")
        mock_bids_handler.get_subjects_info.return_value = {
            "sub-01": ["ses-01", "ses-02"],
            "sub-02": [""]
        }
        mock_bids_handler.get_preprocessed_dwi_files.return_value = []
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = []
        mock_bids_handler_class.return_value = mock_bids_handler
        
        # Create dataset_description.json in bids_root
        (self.bids_root / "dataset_description.json").write_text('{"Name": "Test"}')
        
        study = ElikopyStudy(study_path=self.study_path, bids_root=self.bids_root)
        study.setup_from_qsiprep(self.qsiprep_dir)
        
        assert study.qsiprep_dir == self.qsiprep_dir
        assert study.bids_handler is not None
        assert len(study.subjects) == 2
        assert "sub-01" in study.subjects
        assert "sub-02" in study.subjects
        
        # Check subjects
        subject_01 = study.subjects["sub-01"]
        assert subject_01.id == "sub-01"
        assert subject_01.sessions == ["ses-01", "ses-02"]
        
        subject_02 = study.subjects["sub-02"]
        assert subject_02.id == "sub-02"
        assert subject_02.sessions == [""]
    
    def test_setup_from_qsiprep_directory_not_found(self):
        """Test setup from qsiprep with non-existent directory"""
        study = ElikopyStudy(study_path=self.study_path)
        
        with pytest.raises(FileNotFoundError, match="QSIPrep directory not found"):
            study.setup_from_qsiprep(Path("/nonexistent/path"))
    
    @patch('elikopy.core.study.BIDSHandler')
    def test_setup_from_qsiprep_invalid_structure(self, mock_bids_handler_class):
        """Test setup from qsiprep with invalid structure"""
        # Mock BIDSHandler with validation failure
        mock_bids_handler = Mock()
        mock_bids_handler.validate_qsiprep_structure.return_value = BIDSValidationResult(
            is_valid=False, errors=["Invalid structure"], warnings=[]
        )
        mock_bids_handler_class.return_value = mock_bids_handler
        
        study = ElikopyStudy(study_path=self.study_path, bids_root=self.bids_root)
        
        with pytest.raises(ValueError, match="Invalid QSIPrep structure"):
            study.setup_from_qsiprep(self.qsiprep_dir)
    
    @patch('elikopy.core.study.BIDSHandler')
    def test_setup_from_qsiprep_bids_handler_failure(self, mock_bids_handler_class):
        """Test setup from qsiprep with BIDSHandler initialization failure"""
        # Mock BIDSHandler to raise exception
        mock_bids_handler_class.side_effect = Exception("BIDS handler failed")
        
        study = ElikopyStudy(study_path=self.study_path, bids_root=self.bids_root)
        
        with pytest.raises(ValueError, match="Failed to initialize BIDS handler"):
            study.setup_from_qsiprep(self.qsiprep_dir)
    
    @patch('elikopy.core.study.BIDSHandler')
    def test_setup_from_qsiprep_with_warnings(self, mock_bids_handler_class):
        """Test setup from qsiprep with validation warnings"""
        # Mock BIDSHandler with warnings
        mock_bids_handler = Mock()
        mock_bids_handler.validate_qsiprep_structure.return_value = BIDSValidationResult(
            is_valid=True, errors=[], warnings=["Some warning"]
        )
        mock_bids_handler.create_derivatives_structure.return_value = Path("/derivatives/elikopy")
        mock_bids_handler.get_subjects_info.return_value = {}
        mock_bids_handler_class.return_value = mock_bids_handler
        
        study = ElikopyStudy(study_path=self.study_path, bids_root=self.bids_root)
        study.setup_from_qsiprep(self.qsiprep_dir)
        
        # Should succeed despite warnings
        assert study.bids_handler is not None
    
    @patch('elikopy.core.study.BIDSHandler')
    def test_discover_subjects_with_data(self, mock_bids_handler_class):
        """Test subject discovery with DWI and anatomical data"""
        # Mock DWI and anatomical files
        dwi_file = DWIFile(
            path=Path("/qsiprep/sub-01/dwi/sub-01_dwi.nii.gz"),
            bval_path=Path("/qsiprep/sub-01/dwi/sub-01_dwi.bval"),
            bvec_path=Path("/qsiprep/sub-01/dwi/sub-01_dwi.bvec"),
            json_path=Path("/qsiprep/sub-01/dwi/sub-01_dwi.json")
        )
        anat_file = AnatomicalFile(
            path=Path("/qsiprep/sub-01/anat/sub-01_T1w.nii.gz"),
            json_path=Path("/qsiprep/sub-01/anat/sub-01_T1w.json")
        )
        
        # Mock BIDSHandler
        mock_bids_handler = Mock()
        mock_bids_handler.validate_qsiprep_structure.return_value = BIDSValidationResult(
            is_valid=True, errors=[], warnings=[]
        )
        mock_bids_handler.create_derivatives_structure.return_value = Path("/derivatives/elikopy")
        mock_bids_handler.get_subjects_info.return_value = {"sub-01": [""]}
        mock_bids_handler.get_preprocessed_dwi_files.return_value = [dwi_file]
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = [anat_file]
        mock_bids_handler_class.return_value = mock_bids_handler
        
        study = ElikopyStudy(study_path=self.study_path, bids_root=self.bids_root)
        study.setup_from_qsiprep(self.qsiprep_dir)
        
        assert len(study.subjects) == 1
        subject = study.subjects["sub-01"]
        assert len(subject.dwi_files) == 1
        assert len(subject.anatomical_files) == 1
        assert subject.dwi_files[0] == dwi_file
        assert subject.anatomical_files[0] == anat_file
        
        # Check metadata
        assert "qsiprep_dir" in subject.metadata
        assert "bids_root" in subject.metadata
        assert "total_dwi_files" in subject.metadata
        assert "total_anatomical_files" in subject.metadata
        assert subject.metadata["total_dwi_files"] == 1
        assert subject.metadata["total_anatomical_files"] == 1
    
    def test_get_subjects(self):
        """Test getting list of subjects"""
        study = ElikopyStudy(study_path=self.study_path)
        study.subjects = {
            "sub-01": Subject("sub-01"),
            "sub-02": Subject("sub-02")
        }
        
        subjects = study.get_subjects()
        assert len(subjects) == 2
        assert all(isinstance(s, Subject) for s in subjects)
        subject_ids = [s.id for s in subjects]
        assert "sub-01" in subject_ids
        assert "sub-02" in subject_ids
    
    def test_get_subject(self):
        """Test getting specific subject"""
        study = ElikopyStudy(study_path=self.study_path)
        subject_01 = Subject("sub-01")
        study.subjects = {"sub-01": subject_01}
        
        # Test with full ID
        result = study.get_subject("sub-01")
        assert result == subject_01
        
        # Test with short ID
        result = study.get_subject("01")
        assert result == subject_01
        
        # Test non-existent subject
        result = study.get_subject("sub-99")
        assert result is None
    
    def test_get_subject_ids(self):
        """Test getting subject IDs"""
        study = ElikopyStudy(study_path=self.study_path)
        study.subjects = {
            "sub-01": Subject("sub-01"),
            "sub-02": Subject("sub-02")
        }
        
        subject_ids = study.get_subject_ids()
        assert len(subject_ids) == 2
        assert "sub-01" in subject_ids
        assert "sub-02" in subject_ids
    
    def test_has_subject(self):
        """Test checking if subject exists"""
        study = ElikopyStudy(study_path=self.study_path)
        study.subjects = {"sub-01": Subject("sub-01")}
        
        assert study.has_subject("sub-01") is True
        assert study.has_subject("01") is True
        assert study.has_subject("sub-99") is False
    
    def test_get_study_summary(self):
        """Test getting study summary"""
        study = ElikopyStudy(study_path=self.study_path, bids_root=self.bids_root)
        study.qsiprep_dir = self.qsiprep_dir
        study.derivatives_dir = Path("/derivatives/elikopy")
        
        # Add subjects with files
        subject_01 = Subject("sub-01", ["ses-01"])
        dwi_file = DWIFile(
            path=Path("/test/dwi.nii.gz"),
            bval_path=Path("/test/dwi.bval"),
            bvec_path=Path("/test/dwi.bvec"),
            json_path=Path("/test/dwi.json")
        )
        anat_file = AnatomicalFile(
            path=Path("/test/T1w.nii.gz"),
            json_path=Path("/test/T1w.json")
        )
        subject_01.add_dwi_file(dwi_file)
        subject_01.add_anatomical_file(anat_file)
        
        subject_02 = Subject("sub-02", [""])
        
        study.subjects = {"sub-01": subject_01, "sub-02": subject_02}
        
        summary = study.get_study_summary()
        
        assert summary["study_path"] == str(self.study_path)
        assert summary["bids_root"] == str(self.bids_root)
        assert summary["qsiprep_dir"] == str(self.qsiprep_dir)
        assert summary["derivatives_name"] == "elikopy"
        assert summary["derivatives_dir"] == str(Path("/derivatives/elikopy"))
        assert summary["total_subjects"] == 2
        assert summary["total_dwi_files"] == 1
        assert summary["total_anatomical_files"] == 1
        assert summary["sessions_per_subject"]["sub-01"] == 1
        assert summary["sessions_per_subject"]["sub-02"] == 1
        assert "config_summary" in summary
    
    def test_create_processor(self):
        """Test creating processor"""
        study = ElikopyStudy(study_path=self.study_path)
        
        # Mock the import inside the method
        with patch('elikopy.core.processor.ElikopyProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            
            processor = study.create_processor("dti", param1="value1")
            
            mock_processor_class.assert_called_once_with(
                study=study,
                processing_type="dti",
                param1="value1"
            )
            assert processor == mock_processor
    
    def test_configure(self):
        """Test configuring study"""
        study = ElikopyStudy(study_path=self.study_path)
        config_dict = {"study_name": "test_study"}
        
        study.configure(config_dict)
        
        # Should call config.update
        assert study.config.study_name == "test_study"
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        study = ElikopyStudy(study_path=self.study_path)
        
        default_config = study.get_default_config()
        
        assert isinstance(default_config, dict)
        assert "study_name" in default_config
    
    def test_validate_config(self):
        """Test validating configuration"""
        study = ElikopyStudy(study_path=self.study_path)
        config_dict = {"study_name": "test_study"}
        
        is_valid = study.validate_config(config_dict)
        
        assert isinstance(is_valid, bool)
    
    def test_discover_subjects_no_bids_handler(self):
        """Test subject discovery without BIDS handler"""
        study = ElikopyStudy(study_path=self.study_path)
        
        with pytest.raises(RuntimeError, match="BIDS handler not initialized"):
            study._discover_subjects()
    
    @patch('elikopy.core.study.BIDSHandler')
    def test_discover_subjects_with_data_loading_error(self, mock_bids_handler_class):
        """Test subject discovery with data loading errors"""
        # Mock BIDSHandler that raises exception for data loading
        mock_bids_handler = Mock()
        mock_bids_handler.validate_qsiprep_structure.return_value = BIDSValidationResult(
            is_valid=True, errors=[], warnings=[]
        )
        mock_bids_handler.create_derivatives_structure.return_value = Path("/derivatives/elikopy")
        mock_bids_handler.get_subjects_info.return_value = {"sub-01": [""]}
        mock_bids_handler.get_preprocessed_dwi_files.side_effect = Exception("Data loading failed")
        mock_bids_handler.get_preprocessed_anatomical_files.return_value = []
        mock_bids_handler_class.return_value = mock_bids_handler
        
        study = ElikopyStudy(study_path=self.study_path, bids_root=self.bids_root)
        
        # Should not raise exception, but log warning
        study.setup_from_qsiprep(self.qsiprep_dir)
        
        # Subject should still be created
        assert len(study.subjects) == 1
        assert "sub-01" in study.subjects
    
    @patch('elikopy.core.study.BIDSHandler')
    def test_setup_from_qsiprep_infer_bids_root(self, mock_bids_handler_class):
        """Test setup from qsiprep with BIDS root inference"""
        # Create directory structure for BIDS root inference
        derivatives_dir = self.qsiprep_dir.parent
        inferred_bids_root = derivatives_dir.parent
        (inferred_bids_root / "dataset_description.json").write_text('{"Name": "Test"}')
        
        # Mock BIDSHandler
        mock_bids_handler = Mock()
        mock_bids_handler.validate_qsiprep_structure.return_value = BIDSValidationResult(
            is_valid=True, errors=[], warnings=[]
        )
        mock_bids_handler.create_derivatives_structure.return_value = Path("/derivatives/elikopy")
        mock_bids_handler.get_subjects_info.return_value = {}
        mock_bids_handler_class.return_value = mock_bids_handler
        
        study = ElikopyStudy(study_path=self.study_path)  # No bids_root provided
        study.setup_from_qsiprep(self.qsiprep_dir)
        
        assert study.bids_root == inferred_bids_root
        
        # Check that BIDSHandler was called with inferred BIDS root
        mock_bids_handler_class.assert_called_once_with(
            bids_root=inferred_bids_root,
            qsiprep_dir=self.qsiprep_dir
        )


if __name__ == "__main__":
    pytest.main([__file__])