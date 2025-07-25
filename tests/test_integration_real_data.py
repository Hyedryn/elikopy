"""
Integration tests using real BIDS example dataset with qsiprep derivatives
"""

import json
import pytest
from pathlib import Path

from elikopy.data.bids_handler import BIDSHandler, BIDSValidationResult
from elikopy.data.derivatives import DerivativesManager
from elikopy.core.base import DWIFile


class TestRealDataIntegration:
    """Integration tests using the real BIDS example dataset"""
    
    def setup_method(self):
        """Set up test fixtures with real data"""
        self.bids_example_dir = Path("bids_example")
        self.qsiprep_dir = self.bids_example_dir / "derivatives" / "qsiprep"
        
        # Skip tests if example data is not available
        if not self.bids_example_dir.exists():
            pytest.skip("BIDS example dataset not found")
        
        if not self.qsiprep_dir.exists():
            pytest.skip("QSIPrep derivatives not found in example dataset")
    
    def test_bids_handler_with_real_qsiprep_data(self):
        """Test BIDSHandler with real qsiprep derivatives"""
        # Initialize handler with real data
        handler = BIDSHandler(self.bids_example_dir, self.qsiprep_dir)
        
        # Test BIDS structure validation
        bids_validation = handler.validate_bids_structure(self.bids_example_dir)
        assert bids_validation.is_valid, f"BIDS validation failed: {bids_validation.errors}"
        
        # Test qsiprep structure validation
        qsiprep_validation = handler.validate_qsiprep_structure()
        assert qsiprep_validation.is_valid, f"QSIPrep validation failed: {qsiprep_validation.errors}"
        
        # Test getting subjects info
        subjects_info = handler.get_subjects_info()
        print(f"Found subjects: {subjects_info}")
        
        # We expect at least sub-01 with ses-01
        assert "sub-01" in subjects_info
        assert "ses-01" in subjects_info["sub-01"]
    
    def test_get_preprocessed_dwi_files_real_data(self):
        """Test getting preprocessed DWI files from real qsiprep data"""
        handler = BIDSHandler(self.bids_example_dir, self.qsiprep_dir)
        
        # Get DWI files for sub-01, ses-01
        dwi_files = handler.get_preprocessed_dwi_files("sub-01", session="ses-01")
        
        print(f"Found {len(dwi_files)} DWI files")
        for dwi_file in dwi_files:
            print(f"  DWI: {dwi_file.path}")
            print(f"  BVAL: {dwi_file.bval_path}")
            print(f"  BVEC: {dwi_file.bvec_path}")
            print(f"  JSON: {dwi_file.json_path}")
        
        # Validate we found DWI files
        assert len(dwi_files) > 0, "No DWI files found"
        
        # Validate each DWI file
        for dwi_file in dwi_files:
            assert isinstance(dwi_file, DWIFile)
            assert dwi_file.path.exists(), f"DWI file not found: {dwi_file.path}"
            assert dwi_file.bval_path.exists(), f"BVAL file not found: {dwi_file.bval_path}"
            assert dwi_file.bvec_path.exists(), f"BVEC file not found: {dwi_file.bvec_path}"
            
            # Check file naming follows BIDS conventions
            assert "sub-01" in dwi_file.path.name
            assert "ses-01" in dwi_file.path.name
            assert "desc-preproc" in dwi_file.path.name
            assert dwi_file.path.name.endswith("_dwi.nii.gz")
            
            # Check JSON sidecar if present
            if dwi_file.json_path and dwi_file.json_path.exists():
                with open(dwi_file.json_path, 'r') as f:
                    json_data = json.load(f)
                assert isinstance(json_data, dict)
                print(f"  JSON metadata keys: {list(json_data.keys())}")  
  
    def test_get_preprocessed_anatomical_files_real_data(self):
        """Test getting preprocessed anatomical files from real qsiprep data"""
        handler = BIDSHandler(self.bids_example_dir, self.qsiprep_dir)
        
        # Get anatomical files for sub-01
        anat_files = handler.get_preprocessed_anatomical_files("sub-01")
        
        print(f"Found {len(anat_files)} anatomical files")
        for anat_file in anat_files:
            print(f"  T1w: {anat_file.path}")
            print(f"  JSON: {anat_file.json_path}")
        
        # Validate we found anatomical files
        assert len(anat_files) > 0, "No anatomical files found"
        
        # Validate each anatomical file
        for anat_file in anat_files:
            assert anat_file.path.exists(), f"Anatomical file not found: {anat_file.path}"
            
            # Check file naming follows BIDS conventions
            assert "sub-01" in anat_file.path.name
            assert "desc-preproc" in anat_file.path.name
            assert anat_file.path.name.endswith("_T1w.nii.gz")
            
            # Check JSON sidecar if present
            if anat_file.json_path and anat_file.json_path.exists():
                with open(anat_file.json_path, 'r') as f:
                    json_data = json.load(f)
                assert isinstance(json_data, dict)
                print(f"  JSON metadata keys: {list(json_data.keys())}")
    
    def test_derivatives_manager_with_real_data(self):
        """Test DerivativesManager with real BIDS data"""
        # Create derivatives manager
        manager = DerivativesManager(self.bids_example_dir, "elikopy", "0.5.0")
        
        # Test creating subject directories
        subject_dir = manager.get_subject_dir("sub-01", "ses-01")
        assert subject_dir.exists()
        assert subject_dir == manager.derivatives_dir / "sub-01" / "ses-01"
        
        # Test creating modality directories
        dti_dir = manager.get_modality_dir("sub-01", "dti", "ses-01")
        assert dti_dir.exists()
        assert dti_dir == manager.derivatives_dir / "sub-01" / "ses-01" / "dti"
        
        # Test getting output paths with BIDS naming
        fa_path = manager.get_output_path(
            "sub-01", "dti", "FA", 
            session_id="ses-01",
            model="DTI",
            parameter="FA"
        )
        
        expected_name = "sub-01_ses-01_model-DTI_parameter-FA_FA.nii.gz"
        assert fa_path.name == expected_name
        assert fa_path.parent == dti_dir
        
        # Test creating provenance with real input files
        handler = BIDSHandler(self.bids_example_dir, self.qsiprep_dir)
        dwi_files = handler.get_preprocessed_dwi_files("sub-01", session="ses-01")
        
        if dwi_files:
            input_files = [
                dwi_files[0].path,
                dwi_files[0].bval_path,
                dwi_files[0].bvec_path
            ]
            
            parameters = {
                "algorithm": "WLS",
                "mask_threshold": 0.1
            }
            
            # Create test output file
            fa_path.parent.mkdir(parents=True, exist_ok=True)
            fa_path.touch()
            
            # Create provenance
            json_file = manager.create_provenance(fa_path, input_files, parameters)
            
            assert json_file.exists()
            assert json_file == fa_path.with_suffix(".json")
            
            # Validate provenance content
            with open(json_file, 'r') as f:
                provenance = json.load(f)
            
            assert len(provenance["Sources"]) == 3
            assert provenance["Parameters"] == parameters
            assert provenance["ProcessedBy"]["Name"] == "elikopy"
            
            print(f"Created provenance file: {json_file}")
            print(f"Sources: {provenance['Sources']}")
    
    def test_bids_compliance_validation(self):
        """Test BIDS compliance validation with real data"""
        handler = BIDSHandler(self.bids_example_dir, self.qsiprep_dir)
        
        # Test dataset description validation
        dataset_desc_path = self.qsiprep_dir / "dataset_description.json"
        assert dataset_desc_path.exists()
        
        with open(dataset_desc_path, 'r') as f:
            dataset_desc = json.load(f)
        
        # Validate required BIDS fields
        required_fields = ["Name", "BIDSVersion", "PipelineDescription"]
        for field in required_fields:
            assert field in dataset_desc, f"Missing required field: {field}"
        
        # Validate pipeline description
        pipeline_desc = dataset_desc["PipelineDescription"]
        assert "Name" in pipeline_desc
        assert "Version" in pipeline_desc
        assert pipeline_desc["Name"] == "qsiprep"
        
        print(f"QSIPrep version: {pipeline_desc['Version']}")
        print(f"Dataset name: {dataset_desc['Name']}")
        
        # Test file naming compliance
        dwi_files = handler.get_preprocessed_dwi_files("sub-01", session="ses-01")
        
        for dwi_file in dwi_files:
            filename = dwi_file.path.name
            
            # Check BIDS entity order and format
            assert filename.startswith("sub-01_")
            assert "ses-01_" in filename
            assert "desc-preproc_" in filename
            assert filename.endswith("_dwi.nii.gz")
            
            # Validate associated files follow same pattern
            bval_name = dwi_file.bval_path.name
            bvec_name = dwi_file.bvec_path.name
            
            assert bval_name.replace(".bval", ".nii.gz") == filename
            assert bvec_name.replace(".bvec", ".nii.gz") == filename
            
            if dwi_file.json_path:
                json_name = dwi_file.json_path.name
                assert json_name.replace(".json", ".nii.gz") == filename
        
        print("BIDS compliance validation passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])