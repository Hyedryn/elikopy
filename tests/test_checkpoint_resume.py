"""
Test checkpoint and resume functionality for ElikopyProcessor

This module tests the comprehensive checkpoint and resume capabilities
including state persistence, progress tracking, and error recovery.
"""

import json
import pickle
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from elikopy.core.processor import ElikopyProcessor, ProcessingPipeline, ProcessingStep, PipelineStage
from elikopy.core.study import ElikopyStudy, Subject
from elikopy.core.base import ProcessingStatus, ProcessingResult, ValidationResult
from elikopy.data.bids_handler import DWIFile
from elikopy.infrastructure.scheduler import JobScheduler


class TestCheckpointResume:
    """Test checkpoint and resume functionality"""
    
    @pytest.fixture
    def temp_study_dir(self):
        """Create temporary study directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_study(self, temp_study_dir):
        """Create mock study"""
        study = Mock(spec=ElikopyStudy)
        study.study_path = temp_study_dir
        study.bids_root = temp_study_dir / "bids"
        study.qsiprep_dir = temp_study_dir / "qsiprep"
        study.derivatives_name = "elikopy"
        study.derivatives_dir = temp_study_dir / "derivatives" / "elikopy"
        study.bids_handler = Mock()
        
        # Create mock subjects
        subjects = []
        for i in range(3):
            subject = Mock(spec=Subject)
            subject.id = f"sub-{i+1:02d}"
            
            # Create mock DWI file with required attributes
            dwi_file = Mock(spec=DWIFile)
            dwi_file.path = temp_study_dir / f"sub-{i+1:02d}_dwi.nii.gz"
            dwi_file.bval_path = temp_study_dir / f"sub-{i+1:02d}_dwi.bval"
            dwi_file.bvec_path = temp_study_dir / f"sub-{i+1:02d}_dwi.bvec"
            dwi_file.json_path = temp_study_dir / f"sub-{i+1:02d}_dwi.json"
            
            subject.dwi_files = [dwi_file]
            subject.anatomical_files = []
            subjects.append(subject)
        
        study.get_subjects.return_value = subjects
        study.get_subject.side_effect = lambda sid: next(
            (s for s in subjects if s.id == sid), None
        )
        study.get_study_summary.return_value = {"study_name": "test_study"}
        
        return study
    
    @pytest.fixture
    def processor(self, mock_study):
        """Create processor for testing"""
        return ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=mock_study.study_path / "processing_results"
        )
    
    def test_checkpoint_creation(self, processor):
        """Test checkpoint file creation"""
        # Initialize processing state
        processor.pipeline.status = ProcessingStatus.IN_PROGRESS
        processor.pipeline.start_time = datetime.now()
        processor.pipeline.total_subjects = 3
        processor.pipeline.completed_subjects = 1
        processor.processing_state = {
            'subjects': ['sub-01', 'sub-02', 'sub-03'],
            'completed_subjects': ['sub-01'],
            'failed_subjects': [],
            'current_subject_index': 1
        }
        
        # Save checkpoint
        checkpoint_path = processor._save_checkpoint("test_checkpoint")
        
        # Verify checkpoint file exists
        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == '.pkl'
        
        # Verify metadata file exists
        metadata_path = checkpoint_path.with_suffix('.json')
        assert metadata_path.exists()
        
        # Verify checkpoint content
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        assert 'processor_config' in checkpoint_data
        assert 'study_config' in checkpoint_data
        assert 'pipeline_state' in checkpoint_data
        assert 'processing_state' in checkpoint_data
        assert 'checkpoint_metadata' in checkpoint_data
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['processing_type'] == 'dti'
        assert metadata['total_subjects'] == 3
        assert metadata['completed_subjects'] == 1
        assert metadata['progress_percent'] == 33.33333333333333
    
    def test_checkpoint_resume(self, processor):
        """Test resuming from checkpoint"""
        # Create initial processing state
        processor.pipeline.status = ProcessingStatus.IN_PROGRESS
        processor.pipeline.start_time = datetime.now()
        processor.pipeline.total_subjects = 3
        processor.pipeline.completed_subjects = 1
        processor.processing_state = {
            'subjects': ['sub-01', 'sub-02', 'sub-03'],
            'completed_subjects': ['sub-01'],
            'failed_subjects': [],
            'current_subject_index': 1,
            'parallel': False,
            'save_checkpoints': True,
            'checkpoint_interval': 1
        }
        
        # Save checkpoint
        checkpoint_path = processor._save_checkpoint("resume_test")
        
        # Create new processor instance
        new_processor = ElikopyProcessor(
            study=processor.study,
            processing_type="dti",
            output_dir=processor.output_dir
        )
        
        # Mock the run method to avoid actual processing
        with patch.object(new_processor, 'run') as mock_run:
            mock_run.return_value = {'sub-02': [], 'sub-03': []}
            
            # Resume from checkpoint
            results = new_processor.resume(checkpoint_path)
            
            # Verify resume was called with correct parameters
            mock_run.assert_called_once_with(
                subjects=['sub-02', 'sub-03'],
                parallel=False,
                save_checkpoints=True,
                checkpoint_interval=1
            )
    
    def test_checkpoint_status(self, processor):
        """Test getting checkpoint status"""
        # Create checkpoint
        processor.pipeline.status = ProcessingStatus.IN_PROGRESS
        processor.pipeline.start_time = datetime.now()
        processor.pipeline.total_subjects = 5
        processor.pipeline.completed_subjects = 2
        processor.processing_state = {
            'subjects': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'],
            'completed_subjects': ['sub-01', 'sub-02'],
            'failed_subjects': [],
            'current_subject_index': 2
        }
        
        checkpoint_path = processor._save_checkpoint("status_test")
        
        # Get status from metadata file
        status = processor.get_checkpoint_status(checkpoint_path)
        
        assert status['processing_type'] == 'dti'
        assert status['total_subjects'] == 5
        assert status['completed_subjects'] == 2
        assert status['progress_percent'] == 40.0
        assert status['status'] == ProcessingStatus.IN_PROGRESS.value
    
    def test_list_checkpoints(self, processor):
        """Test listing available checkpoints"""
        # Create multiple checkpoints
        checkpoint_paths = []
        for i in range(3):
            processor.pipeline.completed_subjects = i
            checkpoint_path = processor._save_checkpoint(f"list_test_{i}")
            checkpoint_paths.append(checkpoint_path)
        
        # List checkpoints
        checkpoints = processor.list_checkpoints(processor.checkpoint_dir)
        
        assert len(checkpoints) == 3
        
        # Verify they are sorted by timestamp (newest first)
        timestamps = [cp['timestamp'] for cp in checkpoints]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_delete_checkpoint(self, processor):
        """Test deleting checkpoint files"""
        # Create checkpoint
        checkpoint_path = processor._save_checkpoint("delete_test")
        metadata_path = checkpoint_path.with_suffix('.json')
        
        # Verify files exist
        assert checkpoint_path.exists()
        assert metadata_path.exists()
        
        # Delete checkpoint
        success = processor.delete_checkpoint(checkpoint_path)
        
        assert success
        assert not checkpoint_path.exists()
        assert not metadata_path.exists()
    
    def test_cleanup_old_checkpoints(self, processor):
        """Test cleaning up old checkpoints"""
        # Create multiple checkpoints with different timestamps
        checkpoint_paths = []
        for i in range(7):
            # Mock different timestamps
            with patch('elikopy.core.processor.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime.now() - timedelta(hours=i)
                mock_datetime.fromisoformat = datetime.fromisoformat
                checkpoint_path = processor._save_checkpoint(f"cleanup_test_{i}")
                checkpoint_paths.append(checkpoint_path)
        
        # Cleanup, keeping only 3 most recent
        deleted_count = processor.cleanup_old_checkpoints(keep_count=3)
        
        assert deleted_count == 4  # Should delete 4 old checkpoints
        
        # Verify only 3 checkpoints remain
        remaining_checkpoints = processor.list_checkpoints(processor.checkpoint_dir)
        assert len(remaining_checkpoints) == 3
    
    def test_processing_rate_calculation(self, processor):
        """Test processing rate calculation"""
        # Test with no processing started
        rate = processor._calculate_processing_rate()
        assert rate is None
        
        # Test with processing in progress
        processor.pipeline.start_time = datetime.now() - timedelta(hours=2)
        processor.pipeline.completed_subjects = 4
        
        rate = processor._calculate_processing_rate()
        assert rate is not None
        assert abs(rate - 2.0) < 0.01  # 4 subjects in 2 hours ≈ 2 subjects/hour
    
    def test_remaining_time_estimation(self, processor):
        """Test remaining time estimation"""
        # Test with no processing started
        remaining = processor._estimate_remaining_time()
        assert remaining is None
        
        # Test with processing in progress
        processor.pipeline.start_time = datetime.now() - timedelta(hours=1)
        processor.pipeline.total_subjects = 10
        processor.pipeline.completed_subjects = 2
        
        remaining = processor._estimate_remaining_time()
        assert remaining is not None
        # 2 subjects in 1 hour = 0.5 hours per subject
        # 8 remaining subjects = 4 hours = 14400 seconds
        assert abs(remaining - 14400.0) < 1.0  # Allow small floating point differences
    
    def test_progress_report(self, processor):
        """Test progress report generation"""
        # Set up processing state
        processor.pipeline.status = ProcessingStatus.IN_PROGRESS
        processor.pipeline.start_time = datetime.now() - timedelta(minutes=30)
        processor.pipeline.total_subjects = 5
        processor.pipeline.completed_subjects = 2
        processor.processing_state = {'test': 'data'}
        
        report = processor.get_progress_report()
        
        assert 'pipeline' in report
        assert 'steps' in report
        assert 'processing_state' in report
        
        pipeline_info = report['pipeline']
        assert pipeline_info['status'] == ProcessingStatus.IN_PROGRESS.value
        assert pipeline_info['progress_percent'] == 40.0
        assert pipeline_info['total_subjects'] == 5
        assert pipeline_info['completed_subjects'] == 2
        assert pipeline_info['current_duration'] is not None
    
    def test_resume_with_no_remaining_subjects(self, processor):
        """Test resume when all subjects are already processed"""
        # Create checkpoint with all subjects completed
        processor.processing_state = {
            'subjects': ['sub-01', 'sub-02'],
            'completed_subjects': ['sub-01', 'sub-02'],
            'failed_subjects': [],
            'current_subject_index': 2
        }
        
        checkpoint_path = processor._save_checkpoint("complete_test")
        
        # Create new processor and resume
        new_processor = ElikopyProcessor(
            study=processor.study,
            processing_type="dti",
            output_dir=processor.output_dir
        )
        
        results = new_processor.resume(checkpoint_path)
        
        # Should return empty results since no subjects remain
        assert results == {}
    
    def test_resume_validation_errors(self, processor):
        """Test resume with validation errors"""
        # Test with non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            processor.resume("non_existent_checkpoint.pkl")
        
        # Test with invalid checkpoint format
        invalid_checkpoint = processor.checkpoint_dir / "invalid.pkl"
        with open(invalid_checkpoint, 'wb') as f:
            pickle.dump({'invalid': 'data'}, f)
        
        with pytest.raises(RuntimeError, match="Failed to resume from checkpoint"):
            processor.resume(invalid_checkpoint)
        
        # Test with mismatched processing type
        processor.processing_state = {'subjects': ['sub-01']}
        checkpoint_path = processor._save_checkpoint("mismatch_test")
        
        # Create processor with different type
        different_processor = ElikopyProcessor(
            study=processor.study,
            processing_type="noddi",  # Different type
            output_dir=processor.output_dir
        )
        
        with pytest.raises(RuntimeError, match="Failed to resume from checkpoint"):
            different_processor.resume(checkpoint_path)
    
    @patch('elikopy.core.processor.ElikopyProcessor._process_subject')
    @patch('elikopy.core.processor.ElikopyProcessor.validate_inputs')
    def test_checkpoint_during_processing(self, mock_validate, mock_process, processor):
        """Test checkpoint creation during actual processing"""
        # Mock validation to pass
        mock_validate.return_value = ValidationResult(True, [], [], [])
        
        # Mock successful processing
        mock_result = Mock(spec=ProcessingResult)
        mock_result.status = ProcessingStatus.COMPLETED
        mock_result.output_files = []
        mock_result.metadata = {}
        mock_process.return_value = [mock_result]
        
        # Run processing with checkpoint interval of 1
        results = processor.run(
            subjects=['sub-01', 'sub-02'],
            parallel=False,
            save_checkpoints=True,
            checkpoint_interval=1
        )
        
        # Verify checkpoints were created
        checkpoints = processor.list_checkpoints(processor.checkpoint_dir)
        assert len(checkpoints) >= 1  # At least one checkpoint should be created
        
        # Verify processing completed
        assert len(results) == 2
        assert 'sub-01' in results
        assert 'sub-02' in results


if __name__ == "__main__":
    pytest.main([__file__])