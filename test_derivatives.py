"""
Unit tests for DerivativesManager class - BIDS derivatives structure creation
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from elikopy.data.derivatives import DerivativesManager
from elikopy.core.base import ValidationResult, ValidationError, ValidationWarning


class TestDerivativesManager:
    """Test suite for DerivativesManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.bids_root = Path(self.temp_dir) / "bids_root"
        
        # Create basic BIDS structure
        self.bids_root.mkdir(parents=True)
        
        # Create dataset_description.json
        self._create_dataset_description(self.bids_root)
    
    def _create_dataset_description(self, path: Path):
        """Create a dataset_description.json file"""
        desc = {
            "Name": "Test Dataset",
            "BIDSVersion": "1.4.0"
        }
        
        with open(path / "dataset_description.json", "w") as f:
            json.dump(desc, f)
    
    def test_init_valid_bids(self):
        """Test DerivativesManager initialization with valid BIDS structure"""
        manager = DerivativesManager(self.bids_root, "elikopy", "0.5.0")
        
        assert manager.bids_root == self.bids_root
        assert manager.pipeline_name == "elikopy"
        assert manager.pipeline_version == "0.5.0"
        assert manager.derivatives_dir == self.bids_root / "derivatives" / "elikopy"
        
        # Check derivatives structure was created
        assert manager.derivatives_dir.exists()
        assert (manager.derivatives_dir / "dataset_description.json").exists()
        assert (manager.derivatives_dir / "README.md").exists()
        assert (manager.derivatives_dir / "logs").exists()
    
    def test_init_invalid_bids(self):
        """Test DerivativesManager initialization with invalid BIDS structure"""
        # Remove dataset_description.json
        (self.bids_root / "dataset_description.json").unlink()
        
        with pytest.raises(ValueError, match="Invalid BIDS structure"):
            DerivativesManager(self.bids_root, "elikopy")
    
    def test_validate_bids_structure_valid(self):
        """Test BIDS structure validation with valid structure"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        result = manager.validate_bids_structure(self.bids_root)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_bids_structure_invalid(self):
        """Test BIDS structure validation with invalid structure"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        
        # Test with non-existent directory
        invalid_dir = Path(self.temp_dir) / "nonexistent"
        result = manager.validate_bids_structure(invalid_dir)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("not found" in error.message for error in result.errors)
    
    def test_create_bids_derivatives(self):
        """Test creating BIDS derivatives structure"""
        manager = DerivativesManager(self.bids_root, "test_pipeline", "1.0.0")
        
        output_dir = Path(self.temp_dir) / "output"
        derivatives_dir = manager.create_bids_derivatives(output_dir, "test_pipeline")
        
        # Check directory structure
        assert derivatives_dir.exists()
        assert derivatives_dir == output_dir / "derivatives" / "test_pipeline"
        assert (derivatives_dir / "logs").exists()
        
        # Check dataset_description.json
        dataset_desc_path = derivatives_dir / "dataset_description.json"
        assert dataset_desc_path.exists()
        
        with open(dataset_desc_path, 'r') as f:
            dataset_desc = json.load(f)
        
        assert dataset_desc["Name"] == "test_pipeline outputs"
        assert dataset_desc["BIDSVersion"] == "1.4.0"
        assert dataset_desc["DatasetType"] == "derivative"
        assert dataset_desc["PipelineDescription"]["Name"] == "test_pipeline"
        assert dataset_desc["PipelineDescription"]["Version"] == "1.0.0"
        
        # Check README.md
        readme_path = derivatives_dir / "README.md"
        assert readme_path.exists()
        
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        assert "test_pipeline Derivatives" in readme_content
        assert "v1.0.0" in readme_content
    
    def test_get_subject_dir(self):
        """Test getting subject directory"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        
        # Test without session
        subject_dir = manager.get_subject_dir("01")
        expected_dir = manager.derivatives_dir / "sub-01"
        assert subject_dir == expected_dir
        assert subject_dir.exists()
        
        # Test with session
        session_dir = manager.get_subject_dir("01", "01")
        expected_session_dir = manager.derivatives_dir / "sub-01" / "ses-01"
        assert session_dir == expected_session_dir
        assert session_dir.exists()
        
        # Test with prefixes already present
        subject_dir2 = manager.get_subject_dir("sub-02", "ses-02")
        expected_dir2 = manager.derivatives_dir / "sub-02" / "ses-02"
        assert subject_dir2 == expected_dir2
        assert subject_dir2.exists()
    
    def test_get_modality_dir(self):
        """Test getting modality directory"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        
        # Test without session
        modality_dir = manager.get_modality_dir("01", "dti")
        expected_dir = manager.derivatives_dir / "sub-01" / "dti"
        assert modality_dir == expected_dir
        assert modality_dir.exists()
        
        # Test with session
        modality_dir2 = manager.get_modality_dir("01", "noddi", "01")
        expected_dir2 = manager.derivatives_dir / "sub-01" / "ses-01" / "noddi"
        assert modality_dir2 == expected_dir2
        assert modality_dir2.exists()
    
    def test_get_output_path(self):
        """Test getting output file path with BIDS conventions"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        
        # Test basic path
        output_path = manager.get_output_path("01", "dti", "FA")
        expected_path = manager.derivatives_dir / "sub-01" / "dti" / "sub-01_FA.nii.gz"
        assert output_path == expected_path
        assert output_path.parent.exists()
        
        # Test with session
        output_path2 = manager.get_output_path("01", "dti", "MD", session_id="01")
        expected_path2 = manager.derivatives_dir / "sub-01" / "ses-01" / "dti" / "sub-01_ses-01_MD.nii.gz"
        assert output_path2 == expected_path2
        
        # Test with entities
        output_path3 = manager.get_output_path(
            "01", "dti", "FA", 
            session_id="01",
            model="DTI",
            parameter="FA",
            desc="preproc"
        )
        expected_path3 = manager.derivatives_dir / "sub-01" / "ses-01" / "dti" / "sub-01_ses-01_desc-preproc_model-DTI_parameter-FA_FA.nii.gz"
        assert output_path3 == expected_path3
        
        # Test with different extension
        output_path4 = manager.get_output_path("01", "connectivity", "connectivity", extension=".csv")
        expected_path4 = manager.derivatives_dir / "sub-01" / "connectivity" / "sub-01_connectivity.csv"
        assert output_path4 == expected_path4
    
    def test_create_provenance(self):
        """Test creating provenance sidecar files"""
        manager = DerivativesManager(self.bids_root, "elikopy", "0.5.0")
        
        # Create test output file path
        output_file = manager.get_output_path("01", "dti", "FA")
        
        # Create test input files
        input_files = [
            self.bids_root / "sub-01" / "dwi" / "sub-01_dwi.nii.gz",
            self.bids_root / "sub-01" / "dwi" / "sub-01_dwi.bval",
            self.bids_root / "sub-01" / "dwi" / "sub-01_dwi.bvec"
        ]
        
        # Create input files for testing
        for input_file in input_files:
            input_file.parent.mkdir(parents=True, exist_ok=True)
            input_file.touch()
        
        parameters = {
            "algorithm": "WLS",
            "mask_threshold": 0.1
        }
        
        processing_info = {
            "processing_time": 120.5,
            "memory_usage": "2.1 GB"
        }
        
        # Create provenance
        json_file = manager.create_provenance(output_file, input_files, parameters, processing_info)
        
        # Check JSON file was created
        expected_json = output_file.with_suffix(".json")
        assert json_file == expected_json
        assert json_file.exists()
        
        # Check JSON content
        with open(json_file, 'r') as f:
            provenance = json.load(f)
        
        assert len(provenance["Sources"]) == 3
        assert provenance["Parameters"] == parameters
        assert provenance["ProcessedBy"]["Name"] == "elikopy"
        assert provenance["ProcessedBy"]["Version"] == "0.5.0"
        assert "ProcessingDate" in provenance
        assert "Environment" in provenance
        assert provenance["processing_time"] == 120.5
        assert provenance["memory_usage"] == "2.1 GB"
    
    def test_update_dataset_description(self):
        """Test updating dataset_description.json"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        
        updates = {
            "License": "MIT",
            "DatasetDOI": "10.1234/test.doi"
        }
        
        manager.update_dataset_description(updates)
        
        # Check updates were applied
        dataset_desc_path = manager.derivatives_dir / "dataset_description.json"
        with open(dataset_desc_path, 'r') as f:
            dataset_desc = json.load(f)
        
        assert dataset_desc["License"] == "MIT"
        assert dataset_desc["DatasetDOI"] == "10.1234/test.doi"
    
    def test_create_pipeline_description(self):
        """Test creating pipeline description"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        
        manager.create_pipeline_description(
            "custom_pipeline", 
            "2.0.0", 
            "Custom diffusion processing pipeline",
            "https://github.com/example/custom_pipeline"
        )
        
        # Check pipeline description was updated
        dataset_desc_path = manager.derivatives_dir / "dataset_description.json"
        with open(dataset_desc_path, 'r') as f:
            dataset_desc = json.load(f)
        
        pipeline_desc = dataset_desc["PipelineDescription"]
        assert pipeline_desc["Name"] == "custom_pipeline"
        assert pipeline_desc["Version"] == "2.0.0"
        assert pipeline_desc["Description"] == "Custom diffusion processing pipeline"
        assert pipeline_desc["CodeURL"] == "https://github.com/example/custom_pipeline"
    
    def test_add_source_dataset(self):
        """Test adding source dataset information"""
        manager = DerivativesManager(self.bids_root, "elikopy")
        
        manager.add_source_dataset(
            "HCP", 
            "https://www.humanconnectome.org/",
            "1.0"
        )
        
        # Check source dataset was added
        dataset_desc_path = manager.derivatives_dir / "dataset_description.json"
        with open(dataset_desc_path, 'r') as f:
            dataset_desc = json.load(f)
        
        source_datasets = dataset_desc["SourceDatasets"]
        assert len(source_datasets) == 1
        assert source_datasets[0]["Name"] == "HCP"
        assert source_datasets[0]["URL"] == "https://www.humanconnectome.org/"
        assert source_datasets[0]["Version"] == "1.0"
        
        # Test adding another source dataset
        manager.add_source_dataset("ADNI", "https://adni.loni.usc.edu/")
        
        with open(dataset_desc_path, 'r') as f:
            dataset_desc = json.load(f)
        
        source_datasets = dataset_desc["SourceDatasets"]
        assert len(source_datasets) == 2
        assert source_datasets[1]["Name"] == "ADNI"
        assert source_datasets[1]["URL"] == "https://adni.loni.usc.edu/"
    
    def test_create_processing_log(self):
        """Test creating processing log"""
        manager = DerivativesManager(self.bids_root, "elikopy", "0.5.0")
        
        log_data = {
            "subject": "sub-01",
            "processing_steps": ["DTI", "NODDI"],
            "status": "completed",
            "errors": []
        }
        
        log_file = manager.create_processing_log(log_data)
        
        # Check log file was created
        assert log_file.exists()
        assert log_file.parent == manager.derivatives_dir / "logs"
        assert log_file.name.startswith("processing_log_")
        assert log_file.suffix == ".json"
        
        # Check log content
        with open(log_file, 'r') as f:
            log_content = json.load(f)
        
        assert log_content["subject"] == "sub-01"
        assert log_content["processing_steps"] == ["DTI", "NODDI"]
        assert log_content["status"] == "completed"
        assert "timestamp" in log_content
        assert log_content["pipeline"]["name"] == "elikopy"
        assert log_content["pipeline"]["version"] == "0.5.0"
    
    def test_get_derivatives_structure(self):
        """Test getting derivatives structure overview"""
        manager = DerivativesManager(self.bids_root, "elikopy", "0.5.0")
        
        # Create some test structure
        manager.get_modality_dir("01", "dti")
        manager.get_modality_dir("01", "noddi", "01")
        manager.get_modality_dir("02", "dti")
        
        # Create some test files
        test_file1 = manager.get_output_path("01", "dti", "FA")
        test_file2 = manager.get_output_path("01", "noddi", "ICVF", session_id="01")
        test_file3 = manager.get_output_path("02", "dti", "MD")
        
        test_file1.touch()
        test_file2.touch()
        test_file3.touch()
        
        structure = manager.get_derivatives_structure()
        
        assert structure["pipeline_name"] == "elikopy"
        assert structure["pipeline_version"] == "0.5.0"
        assert len(structure["subjects"]) == 2
        assert "dti" in structure["modalities"]
        assert "noddi" in structure["modalities"]
        assert structure["total_files"] == 3
        
        # Check subject structure
        subject_01 = next(s for s in structure["subjects"] if s["subject_id"] == "sub-01")
        assert len(subject_01["sessions"]) == 1
        assert subject_01["sessions"][0]["session_id"] == "ses-01"
        assert "noddi" in subject_01["sessions"][0]["modalities"]
        assert "dti" in subject_01["modalities"]
        
        subject_02 = next(s for s in structure["subjects"] if s["subject_id"] == "sub-02")
        assert len(subject_02["sessions"]) == 0
        assert "dti" in subject_02["modalities"]


if __name__ == "__main__":
    pytest.main([__file__])