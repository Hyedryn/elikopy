"""
Unit tests for BIDSHandler class - pybids-based derivatives parsing
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from elikopy.data.bids_handler import BIDSHandler, BIDSValidationResult, AnatomicalFile
from elikopy.core.base import DWIFile, ValidationResult, ValidationError, ValidationWarning


class TestBIDSHandler:
    """Test suite for BIDSHandler class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.bids_root = Path(self.temp_dir) / "bids_root"
        self.qsiprep_dir = Path(self.temp_dir) / "qsiprep"
        
        # Create basic BIDS structure
        self.bids_root.mkdir(parents=True)
        self.qsiprep_dir.mkdir(parents=True)
        
        # Create dataset_description.json files
        self._create_dataset_description(self.bids_root)
        self._create_dataset_description(self.qsiprep_dir, is_derivative=True)
        
        # Create mock subject structure
        self._create_mock_subject_structure()
    
    def _create_dataset_description(self, path: Path, is_derivative: bool = False):
        """Create a dataset_description.json file"""
        desc = {
            "Name": "Test Dataset",
            "BIDSVersion": "1.4.0"
        }
        
        if is_derivative:
            desc.update({
                "PipelineDescription": {
                    "Name": "qsiprep",
                    "Version": "0.16.0"
                }
            })
        
        with open(path / "dataset_description.json", "w") as f:
            json.dump(desc, f)
    
    def _create_mock_subject_structure(self):
        """Create mock subject structure for testing"""
        # Subject 01, session 01
        sub01_ses01_dwi = self.qsiprep_dir / "sub-01" / "ses-01" / "dwi"
        sub01_ses01_dwi.mkdir(parents=True)
        
        # Create DWI files
        dwi_files = [
            "sub-01_ses-01_desc-preproc_dwi.nii.gz",
            "sub-01_ses-01_desc-preproc_dwi.bval",
            "sub-01_ses-01_desc-preproc_dwi.bvec",
            "sub-01_ses-01_desc-preproc_dwi.json"
        ]
        
        for dwi_file in dwi_files:
            (sub01_ses01_dwi / dwi_file).touch()
        
        # Create JSON content
        json_content = {
            "RepetitionTime": 2.0,
            "EchoTime": 0.03,
            "FlipAngle": 90
        }
        with open(sub01_ses01_dwi / "sub-01_ses-01_desc-preproc_dwi.json", "w") as f:
            json.dump(json_content, f)
        
        # Subject 01, session 01, anatomical
        sub01_ses01_anat = self.qsiprep_dir / "sub-01" / "ses-01" / "anat"
        sub01_ses01_anat.mkdir(parents=True)
        
        anat_files = [
            "sub-01_ses-01_desc-preproc_T1w.nii.gz",
            "sub-01_ses-01_desc-preproc_T1w.json"
        ]
        
        for anat_file in anat_files:
            (sub01_ses01_anat / anat_file).touch()
        
        # Subject 02, no sessions
        sub02_dwi = self.qsiprep_dir / "sub-02" / "dwi"
        sub02_dwi.mkdir(parents=True)
        
        dwi_files_sub02 = [
            "sub-02_desc-preproc_dwi.nii.gz",
            "sub-02_desc-preproc_dwi.bval",
            "sub-02_desc-preproc_dwi.bvec",
            "sub-02_desc-preproc_dwi.json"
        ]
        
        for dwi_file in dwi_files_sub02:
            (sub02_dwi / dwi_file).touch()
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_init_with_pybids(self, mock_bids_layout):
        """Test BIDSHandler initialization with pybids"""
        # Mock BIDSLayout
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        # Initialize handler
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        
        # Check initialization
        assert handler.bids_root == self.bids_root
        assert handler.qsiprep_dir == self.qsiprep_dir
        assert handler._layout == mock_layout
        assert handler._qsiprep_layout == mock_layout
        
        # Check BIDSLayout was called (validation + 2 layouts)
        assert mock_bids_layout.call_count >= 2
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', False)
    def test_init_without_pybids(self):
        """Test BIDSHandler initialization without pybids"""
        with pytest.raises(ImportError, match="pybids is required"):
            BIDSHandler(self.bids_root, self.qsiprep_dir)
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_validate_bids_structure_valid(self, mock_bids_layout):
        """Test BIDS structure validation with valid structure"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        result = handler.validate_bids_structure(self.bids_root)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_validate_bids_structure_invalid(self, mock_bids_layout):
        """Test BIDS structure validation with invalid structure"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        # Create a separate invalid BIDS directory
        invalid_bids = Path(self.temp_dir) / "invalid_bids"
        invalid_bids.mkdir()
        
        # Initialize handler with valid directories first
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        
        # Then test validation on invalid directory
        result = handler.validate_bids_structure(invalid_bids)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("dataset_description.json not found" in error.message for error in result.errors)
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_validate_qsiprep_structure_valid(self, mock_bids_layout):
        """Test qsiprep structure validation with valid structure"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        result = handler.validate_qsiprep_structure()
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_validate_qsiprep_structure_no_subjects(self, mock_bids_layout):
        """Test qsiprep structure validation with no subjects"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        # Create empty qsiprep directory
        empty_qsiprep = Path(self.temp_dir) / "empty_qsiprep"
        empty_qsiprep.mkdir()
        self._create_dataset_description(empty_qsiprep, is_derivative=True)
        
        handler = BIDSHandler(self.bids_root, empty_qsiprep)
        result = handler.validate_qsiprep_structure()
        
        assert not result.is_valid
        assert any("No subject directories found" in error for error in result.errors)
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_get_subjects_info(self, mock_bids_layout):
        """Test getting subjects information"""
        # Mock BIDSLayout
        mock_layout = Mock()
        mock_layout.get_subjects.return_value = ['01', '02']
        mock_layout.get_sessions.side_effect = lambda subject: ['01'] if subject == '01' else []
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        subjects_info = handler.get_subjects_info()
        
        expected = {
            'sub-01': ['ses-01'],
            'sub-02': ['']
        }
        
        assert subjects_info == expected
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_get_preprocessed_dwi_files(self, mock_bids_layout):
        """Test getting preprocessed DWI files"""
        # Mock BIDSLayout and file objects
        mock_layout = Mock()
        
        # Mock DWI file
        mock_dwi_file = Mock()
        mock_dwi_file.path = str(self.qsiprep_dir / "sub-01" / "ses-01" / "dwi" / "sub-01_ses-01_desc-preproc_dwi.nii.gz")
        mock_dwi_file.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 'suffix': 'dwi'
        }
        
        # Mock bval file
        mock_bval_file = Mock()
        mock_bval_file.path = str(self.qsiprep_dir / "sub-01" / "ses-01" / "dwi" / "sub-01_ses-01_desc-preproc_dwi.bval")
        mock_bval_file.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 'suffix': 'dwi'
        }
        
        # Mock bvec file
        mock_bvec_file = Mock()
        mock_bvec_file.path = str(self.qsiprep_dir / "sub-01" / "ses-01" / "dwi" / "sub-01_ses-01_desc-preproc_dwi.bvec")
        mock_bvec_file.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 'suffix': 'dwi'
        }
        
        # Mock JSON file
        mock_json_file = Mock()
        mock_json_file.path = str(self.qsiprep_dir / "sub-01" / "ses-01" / "dwi" / "sub-01_ses-01_desc-preproc_dwi.json")
        mock_json_file.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 'suffix': 'dwi'
        }
        
        # Configure mock layout responses
        def mock_get(**kwargs):
            if kwargs.get('extension') == '.nii.gz':
                return [mock_dwi_file]
            elif kwargs.get('extension') == '.bval':
                return [mock_bval_file]
            elif kwargs.get('extension') == '.bvec':
                return [mock_bvec_file]
            elif kwargs.get('extension') == '.json':
                return [mock_json_file]
            return []
        
        mock_layout.get.side_effect = mock_get
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        dwi_files = handler.get_preprocessed_dwi_files('01', session='01')
        
        assert len(dwi_files) == 1
        assert isinstance(dwi_files[0], DWIFile)
        assert dwi_files[0].path.name == "sub-01_ses-01_desc-preproc_dwi.nii.gz"
        assert dwi_files[0].bval_path.name == "sub-01_ses-01_desc-preproc_dwi.bval"
        assert dwi_files[0].bvec_path.name == "sub-01_ses-01_desc-preproc_dwi.bvec"
        assert dwi_files[0].json_path.name == "sub-01_ses-01_desc-preproc_dwi.json"
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_get_preprocessed_anatomical_files(self, mock_bids_layout):
        """Test getting preprocessed anatomical files"""
        # Mock BIDSLayout and file objects
        mock_layout = Mock()
        
        # Mock T1w file
        mock_t1w_file = Mock()
        mock_t1w_file.path = str(self.qsiprep_dir / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_desc-preproc_T1w.nii.gz")
        mock_t1w_file.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 'suffix': 'T1w'
        }
        
        # Mock JSON file
        mock_json_file = Mock()
        mock_json_file.path = str(self.qsiprep_dir / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_desc-preproc_T1w.json")
        mock_json_file.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 'suffix': 'T1w'
        }
        
        # Configure mock layout responses
        def mock_get(**kwargs):
            if kwargs.get('extension') == '.nii.gz':
                return [mock_t1w_file]
            elif kwargs.get('extension') == '.json':
                return [mock_json_file]
            return []
        
        mock_layout.get.side_effect = mock_get
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        anat_files = handler.get_preprocessed_anatomical_files('01', session='01')
        
        assert len(anat_files) == 1
        assert isinstance(anat_files[0], AnatomicalFile)
        assert anat_files[0].path.name == "sub-01_ses-01_desc-preproc_T1w.nii.gz"
        assert anat_files[0].json_path.name == "sub-01_ses-01_desc-preproc_T1w.json"
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_create_bids_derivatives(self, mock_bids_layout):
        """Test creating BIDS derivatives structure"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        
        output_dir = Path(self.temp_dir) / "output"
        derivatives_dir = handler.create_bids_derivatives(output_dir, "elikopy")
        
        # Check directory was created
        assert derivatives_dir.exists()
        assert derivatives_dir == output_dir / "derivatives" / "elikopy"
        
        # Check dataset_description.json was created
        dataset_desc_path = derivatives_dir / "dataset_description.json"
        assert dataset_desc_path.exists()
        
        with open(dataset_desc_path, 'r') as f:
            dataset_desc = json.load(f)
        
        assert dataset_desc["Name"] == "elikopy outputs"
        assert dataset_desc["BIDSVersion"] == "1.4.0"
        assert dataset_desc["PipelineDescription"]["Name"] == "elikopy"
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_get_bids_layout(self, mock_bids_layout):
        """Test getting BIDS layout"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        layout = handler.get_bids_layout()
        
        assert layout == mock_layout
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_get_qsiprep_layout(self, mock_bids_layout):
        """Test getting qsiprep layout"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        layout = handler.get_qsiprep_layout()
        
        assert layout == mock_layout
    
    @patch('elikopy.data.bids_handler.PYBIDS_AVAILABLE', True)
    @patch('elikopy.data.bids_handler.BIDSLayout')
    def test_files_match_entities(self, mock_bids_layout):
        """Test entity matching between files"""
        mock_layout = Mock()
        mock_bids_layout.return_value = mock_layout
        
        handler = BIDSHandler(self.bids_root, self.qsiprep_dir)
        
        # Mock files with matching entities
        file1 = Mock()
        file1.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 
            'suffix': 'dwi', 'extension': '.nii.gz'
        }
        
        file2 = Mock()
        file2.get_entities.return_value = {
            'subject': '01', 'session': '01', 'desc': 'preproc', 
            'suffix': 'dwi', 'extension': '.bval'
        }
        
        # Should match (same entities except suffix and extension)
        assert handler._files_match_entities(file1, file2)
        
        # Mock files with different entities
        file3 = Mock()
        file3.get_entities.return_value = {
            'subject': '02', 'session': '01', 'desc': 'preproc', 
            'suffix': 'dwi', 'extension': '.bval'
        }
        
        # Should not match (different subject)
        assert not handler._files_match_entities(file1, file3)


if __name__ == "__main__":
    pytest.main([__file__])