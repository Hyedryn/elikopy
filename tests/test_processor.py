"""
Unit tests for ElikopyProcessor class
"""

import json
import pickle
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from elikopy.core.processor import (
    ElikopyProcessor, ProcessingStep, ProcessingPipeline, PipelineStage
)
from elikopy.core.base import ProcessingStatus, ProcessingResult, ValidationResult, ValidationError
from elikopy.core.study import ElikopyStudy, Subject
from elikopy.infrastructure.scheduler import JobScheduler
from elikopy.data.bids_handler import DWIFile


class TestProcessingStep:
    """Test ProcessingStep class"""
    
    def test_processing_step_creation(self):
        """Test ProcessingStep creation"""
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING,
            config={"param1": "value1"},
            dependencies=["dep1"]
        )
        
        assert step.name == "test_step"
        assert step.processor_class == "TestProcessor"
        assert step.stage == PipelineStage.MODELING
        assert step.config == {"param1": "value1"}
        assert step.dependencies == ["dep1"]
        assert step.status == ProcessingStatus.NOT_STARTED
        assert step.result is None
    
    def test_processing_step_execute_success(self):
        """Test successful step execution"""
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING
        )
        
        # Mock processor instance
        mock_processor = Mock()
        mock_result = ProcessingResult(
            status=ProcessingStatus.COMPLETED,
            output_files=[Path("test_output.nii.gz")],
            metadata={"test": "data"}
        )
        mock_processor.process.return_value = mock_result
        
        # Execute step
        result = step.execute(mock_processor, test_arg="test_value")
        
        assert step.status == ProcessingStatus.COMPLETED
        assert step.result == mock_result
        assert step.start_time is not None
        assert step.end_time is not None
        assert result == mock_result
        
        # Verify processor was called with correct arguments
        mock_processor.process.assert_called_once_with(test_arg="test_value")
    
    def test_processing_step_execute_with_validation(self):
        """Test step execution with input validation"""
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING
        )
        
        # Mock processor with validation
        mock_processor = Mock()
        mock_processor.validate_inputs.return_value = ValidationResult(
            is_valid=True, errors=[], warnings=[], suggestions=[]
        )
        mock_result = ProcessingResult(
            status=ProcessingStatus.COMPLETED,
            output_files=[],
            metadata={}
        )
        mock_processor.process.return_value = mock_result
        
        # Execute step
        result = step.execute(mock_processor)
        
        assert step.status == ProcessingStatus.COMPLETED
        mock_processor.validate_inputs.assert_called_once()
        mock_processor.process.assert_called_once()
    
    def test_processing_step_execute_validation_failure(self):
        """Test step execution with validation failure"""
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING
        )
        
        # Mock processor with failed validation
        mock_processor = Mock()
        mock_processor.validate_inputs.return_value = ValidationResult(
            is_valid=False, 
            errors=[ValidationError("Validation failed")], 
            warnings=[], 
            suggestions=[]
        )
        
        # Execute step should raise ValueError
        with pytest.raises(ValueError, match="Input validation failed"):
            step.execute(mock_processor)
        
        assert step.status == ProcessingStatus.FAILED
        assert step.result is not None
        assert step.result.status == ProcessingStatus.FAILED
    
    def test_processing_step_execute_exception(self):
        """Test step execution with exception"""
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING
        )
        
        # Mock processor that raises exception
        mock_processor = Mock()
        mock_processor.process.side_effect = RuntimeError("Processing failed")
        
        # Execute step should raise exception
        with pytest.raises(RuntimeError, match="Processing failed"):
            step.execute(mock_processor)
        
        assert step.status == ProcessingStatus.FAILED
        assert step.result is not None
        assert step.result.status == ProcessingStatus.FAILED
        assert "Processing failed" in step.result.error_message
    
    def test_processing_step_get_duration(self):
        """Test step duration calculation"""
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING
        )
        
        # No duration before execution
        assert step.get_duration() is None
        
        # Set times manually for testing
        step.start_time = datetime(2023, 1, 1, 10, 0, 0)
        step.end_time = datetime(2023, 1, 1, 10, 0, 30)
        
        assert step.get_duration() == 30.0
    
    def test_processing_step_to_dict(self):
        """Test step serialization to dictionary"""
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING,
            config={"param": "value"},
            dependencies=["dep1"]
        )
        
        step_dict = step.to_dict()
        
        assert step_dict["name"] == "test_step"
        assert step_dict["processor_class"] == "TestProcessor"
        assert step_dict["stage"] == "modeling"
        assert step_dict["config"] == {"param": "value"}
        assert step_dict["dependencies"] == ["dep1"]
        assert step_dict["status"] == "not_started"


class TestProcessingPipeline:
    """Test ProcessingPipeline class"""
    
    def test_pipeline_creation(self):
        """Test pipeline creation"""
        pipeline = ProcessingPipeline(name="test_pipeline")
        
        assert pipeline.name == "test_pipeline"
        assert pipeline.steps == []
        assert pipeline.status == ProcessingStatus.NOT_STARTED
        assert pipeline.total_subjects == 0
        assert pipeline.completed_subjects == 0
        assert pipeline.failed_subjects == 0
    
    def test_pipeline_progress(self):
        """Test pipeline progress calculation"""
        pipeline = ProcessingPipeline(name="test_pipeline")
        
        # No subjects
        assert pipeline.get_progress() == 0.0
        
        # With subjects
        pipeline.total_subjects = 10
        pipeline.completed_subjects = 3
        assert pipeline.get_progress() == 30.0
        
        # All completed
        pipeline.completed_subjects = 10
        assert pipeline.get_progress() == 100.0
    
    def test_pipeline_duration(self):
        """Test pipeline duration calculation"""
        pipeline = ProcessingPipeline(name="test_pipeline")
        
        # No duration before execution
        assert pipeline.get_duration() is None
        
        # Set times manually
        pipeline.start_time = datetime(2023, 1, 1, 10, 0, 0)
        pipeline.end_time = datetime(2023, 1, 1, 10, 5, 0)
        
        assert pipeline.get_duration() == 300.0
    
    def test_pipeline_to_dict(self):
        """Test pipeline serialization"""
        pipeline = ProcessingPipeline(name="test_pipeline")
        pipeline.total_subjects = 5
        pipeline.completed_subjects = 2
        
        step = ProcessingStep(
            name="test_step",
            processor_class="TestProcessor",
            stage=PipelineStage.MODELING
        )
        pipeline.steps = [step]
        
        pipeline_dict = pipeline.to_dict()
        
        assert pipeline_dict["name"] == "test_pipeline"
        assert pipeline_dict["status"] == "not_started"
        assert pipeline_dict["total_subjects"] == 5
        assert pipeline_dict["completed_subjects"] == 2
        assert pipeline_dict["progress"] == 40.0
        assert len(pipeline_dict["steps"]) == 1


class TestElikopyProcessor:
    """Test ElikopyProcessor class"""
    
    @pytest.fixture
    def mock_study(self):
        """Create mock study"""
        study = Mock(spec=ElikopyStudy)
        study.study_path = Path("/test/study")
        study.bids_handler = Mock()
        study.derivatives_dir = Path("/test/derivatives")
        study.derivatives_name = "elikopy"
        study.bids_root = Path("/test/bids")
        study.qsiprep_dir = Path("/test/qsiprep")
        
        # Mock subjects
        subject1 = Mock(spec=Subject)
        subject1.id = "sub-01"
        subject1.dwi_files = [Mock(spec=DWIFile)]
        subject1.anatomical_files = []
        
        subject2 = Mock(spec=Subject)
        subject2.id = "sub-02"
        subject2.dwi_files = [Mock(spec=DWIFile)]
        subject2.anatomical_files = []
        
        study.get_subjects.return_value = [subject1, subject2]
        study.get_subject.side_effect = lambda sid: subject1 if sid == "sub-01" else subject2 if sid == "sub-02" else None
        study.get_study_summary.return_value = {"study": "test"}
        
        return study
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_processor_initialization(self, mock_study, temp_output_dir):
        """Test processor initialization"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        assert processor.study == mock_study
        assert processor.processing_type == "dti"
        assert processor.output_dir == temp_output_dir
        assert processor.scheduler is None
        assert processor.pipeline.name == "dti_pipeline"
        assert len(processor.pipeline.steps) == 1
        assert processor.pipeline.steps[0].name == "dti_fitting"
    
    def test_processor_pipeline_creation_dti(self, mock_study, temp_output_dir):
        """Test DTI pipeline creation"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        assert len(processor.pipeline.steps) == 1
        step = processor.pipeline.steps[0]
        assert step.name == "dti_fitting"
        assert step.processor_class == "DTIProcessor"
        assert step.stage == PipelineStage.MODELING
        assert step.dependencies == []
    
    def test_processor_pipeline_creation_connectivity(self, mock_study, temp_output_dir):
        """Test connectivity pipeline creation"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="connectivity",
            output_dir=temp_output_dir
        )
        
        assert len(processor.pipeline.steps) == 3
        
        # Check step names and dependencies
        step_names = [step.name for step in processor.pipeline.steps]
        assert "csd_fitting" in step_names
        assert "tracking" in step_names
        assert "connectivity" in step_names
        
        # Check dependencies
        tracking_step = next(s for s in processor.pipeline.steps if s.name == "tracking")
        assert "csd_fitting" in tracking_step.dependencies
        
        connectivity_step = next(s for s in processor.pipeline.steps if s.name == "connectivity")
        assert "tracking" in connectivity_step.dependencies
    
    def test_processor_unknown_type(self, mock_study, temp_output_dir):
        """Test processor with unknown processing type"""
        with pytest.raises(ValueError, match="Unknown processing type"):
            ElikopyProcessor(
                study=mock_study,
                processing_type="unknown_type",
                output_dir=temp_output_dir
            )
    
    @patch('importlib.import_module')
    def test_get_processor_instance(self, mock_import, mock_study, temp_output_dir):
        """Test processor instance creation"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        # Mock the import
        mock_module = Mock()
        mock_processor_class = Mock()
        mock_module.DTIProcessor = mock_processor_class
        mock_import.return_value = mock_module
        
        # Get processor instance
        instance = processor._get_processor_instance("DTIProcessor")
        
        mock_import.assert_called_once_with("elikopy.processing.dti")
        mock_processor_class.assert_called_once()
    
    def test_get_processor_instance_unknown(self, mock_study, temp_output_dir):
        """Test processor instance creation with unknown class"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        with pytest.raises(ValueError, match="Unknown processor class"):
            processor._get_processor_instance("UnknownProcessor")
    
    @patch('importlib.import_module')
    def test_get_processor_instance_import_error(self, mock_import, mock_study, temp_output_dir):
        """Test processor instance creation with import error"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        mock_import.side_effect = ImportError("Module not found")
        
        with pytest.raises(ImportError, match="Failed to import processor"):
            processor._get_processor_instance("DTIProcessor")
    
    @patch('elikopy.core.processor.DataValidator')
    def test_validate_inputs_success(self, mock_validator_class, mock_study, temp_output_dir):
        """Test successful input validation"""
        # Mock validator instance
        mock_validator = Mock()
        mock_validator.validate_dwi_data.return_value = ValidationResult(
            is_valid=True, errors=[], warnings=[], suggestions=[]
        )
        mock_validator_class.return_value = mock_validator
        
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        result = processor.validate_inputs(["sub-01"])
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_inputs_no_subjects(self, mock_study, temp_output_dir):
        """Test input validation with no subjects"""
        mock_study.get_subjects.return_value = []
        
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        result = processor.validate_inputs()
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "No subjects found" in result.errors[0].message
    
    @patch('elikopy.core.processor.DataValidator')
    def test_validate_inputs_missing_study_components(self, mock_validator_class, mock_study, temp_output_dir):
        """Test input validation with missing study components"""
        mock_study.bids_handler = None
        mock_study.derivatives_dir = None
        
        # Mock validator instance
        mock_validator = Mock()
        mock_validator.validate_dwi_data.return_value = ValidationResult(
            is_valid=True, errors=[], warnings=[], suggestions=[]
        )
        mock_validator_class.return_value = mock_validator
        
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        result = processor.validate_inputs(["sub-01"])
        
        assert not result.is_valid
        assert len(result.errors) >= 2
        error_messages = [error.message for error in result.errors]
        assert any("BIDS handler not initialized" in msg for msg in error_messages)
        assert any("derivatives directory not set" in msg for msg in error_messages)
    
    def test_validate_pipeline_dependencies_circular(self, mock_study, temp_output_dir):
        """Test validation of circular dependencies"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        # Create circular dependency
        step1 = ProcessingStep("step1", "Processor1", PipelineStage.MODELING, dependencies=["step2"])
        step2 = ProcessingStep("step2", "Processor2", PipelineStage.MODELING, dependencies=["step1"])
        processor.pipeline.steps = [step1, step2]
        
        result = processor._validate_pipeline_dependencies()
        
        assert not result.is_valid
        assert any("circular dependencies" in error.message.lower() for error in result.errors)
    
    def test_get_execution_order(self, mock_study, temp_output_dir):
        """Test execution order calculation"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="connectivity",
            output_dir=temp_output_dir
        )
        
        execution_order = processor._get_execution_order()
        
        # CSD should come first, then tracking, then connectivity
        assert execution_order.index("csd_fitting") < execution_order.index("tracking")
        assert execution_order.index("tracking") < execution_order.index("connectivity")
    
    def test_get_execution_order_circular_dependency(self, mock_study, temp_output_dir):
        """Test execution order with circular dependency"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        # Create circular dependency
        step1 = ProcessingStep("step1", "Processor1", PipelineStage.MODELING, dependencies=["step2"])
        step2 = ProcessingStep("step2", "Processor2", PipelineStage.MODELING, dependencies=["step1"])
        processor.pipeline.steps = [step1, step2]
        
        with pytest.raises(ValueError, match="Circular dependencies detected"):
            processor._get_execution_order()
    
    @patch('elikopy.core.processor.ElikopyProcessor._get_processor_instance')
    def test_process_subject_success(self, mock_get_processor, mock_study, temp_output_dir):
        """Test successful subject processing"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        # Mock processor instance
        mock_processor_instance = Mock()
        mock_result = ProcessingResult(
            status=ProcessingStatus.COMPLETED,
            output_files=[Path("test.nii.gz")],
            metadata={"test": "data"}
        )
        mock_processor_instance.process.return_value = mock_result
        mock_get_processor.return_value = mock_processor_instance
        
        results = processor._process_subject("sub-01")
        
        assert len(results) == 1
        assert results[0].status == ProcessingStatus.COMPLETED
        assert len(results[0].output_files) == 1
    
    def test_process_subject_not_found(self, mock_study, temp_output_dir):
        """Test processing non-existent subject"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        results = processor._process_subject("sub-nonexistent")
        
        assert len(results) == 1
        assert results[0].status == ProcessingStatus.FAILED
        assert "Subject not found" in results[0].error_message
    
    @patch('elikopy.core.processor.ElikopyProcessor._get_processor_instance')
    def test_process_subject_step_failure(self, mock_get_processor, mock_study, temp_output_dir):
        """Test subject processing with step failure"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        # Mock processor instance that raises exception
        mock_processor_instance = Mock()
        mock_processor_instance.process.side_effect = RuntimeError("Processing failed")
        mock_get_processor.return_value = mock_processor_instance
        
        results = processor._process_subject("sub-01")
        
        assert len(results) == 1
        assert results[0].status == ProcessingStatus.FAILED
        assert "Processing failed" in results[0].error_message
    
    def test_save_checkpoint(self, mock_study, temp_output_dir):
        """Test checkpoint saving"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        checkpoint_path = processor._save_checkpoint("test_checkpoint")
        
        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pkl"
        
        # Check metadata file
        metadata_file = checkpoint_path.with_suffix('.json')
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata["checkpoint_name"] == "test_checkpoint"
        assert metadata["processing_type"] == "dti"
    
    def test_save_processing_summary(self, mock_study, temp_output_dir):
        """Test processing summary saving"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        # Mock results
        results = {
            "sub-01": [ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=[Path("test.nii.gz")],
                metadata={}
            )],
            "sub-02": [ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message="Failed"
            )]
        }
        
        processor._save_processing_summary(results)
        
        summary_file = temp_output_dir / "processing_summary.json"
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        assert summary["subject_statistics"]["total_subjects"] == 2
        assert summary["subject_statistics"]["successful_subjects"] == 1  # One subject completed successfully
        assert summary["subject_statistics"]["failed_subjects"] == 1
    
    def test_cleanup(self, mock_study, temp_output_dir):
        """Test processor cleanup"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        # Create some temporary files
        temp_file = temp_output_dir / "test.tmp"
        temp_file.touch()
        
        scripts_dir = temp_output_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "test_script.py").touch()
        
        processor.cleanup_outputs(keep_checkpoints=False)
        
        assert not temp_file.exists()
        assert not scripts_dir.exists()
    
    def test_str_and_repr(self, mock_study, temp_output_dir):
        """Test string representations"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        str_repr = str(processor)
        assert "ElikopyProcessor" in str_repr
        assert "dti" in str_repr
        assert "not_started" in str_repr
        
        repr_str = repr(processor)
        assert "ElikopyProcessor" in repr_str
        assert "processing_type='dti'" in repr_str
        assert "dti_fitting" in repr_str
    
    @patch('elikopy.core.processor.DataValidator')
    @patch('elikopy.core.processor.ElikopyProcessor._process_subject')
    def test_run_sequential_success(self, mock_process_subject, mock_validator_class, mock_study, temp_output_dir):
        """Test successful sequential processing"""
        # Mock validator
        mock_validator = Mock()
        mock_validator.validate_dwi_data.return_value = ValidationResult(
            is_valid=True, errors=[], warnings=[], suggestions=[]
        )
        mock_validator_class.return_value = mock_validator
        
        # Mock successful processing results
        mock_result = ProcessingResult(
            status=ProcessingStatus.COMPLETED,
            output_files=[Path("test.nii.gz")],
            metadata={"test": "data"}
        )
        mock_process_subject.return_value = [mock_result]
        
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        results = processor.run(subjects=["sub-01"], parallel=False)
        
        assert "sub-01" in results
        assert len(results["sub-01"]) == 1
        assert results["sub-01"][0].status == ProcessingStatus.COMPLETED
        assert processor.pipeline.status == ProcessingStatus.COMPLETED
        assert processor.pipeline.completed_subjects == 1
        assert processor.pipeline.failed_subjects == 0
    
    @patch('elikopy.core.processor.DataValidator')
    def test_run_validation_failure(self, mock_validator_class, mock_study, temp_output_dir):
        """Test processing with validation failure"""
        # Mock validator with validation failure
        mock_validator = Mock()
        mock_validator.validate_dwi_data.return_value = ValidationResult(
            is_valid=False, 
            errors=[ValidationError("Validation failed")], 
            warnings=[], 
            suggestions=[]
        )
        mock_validator_class.return_value = mock_validator
        
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        with pytest.raises(ValueError, match="Input validation failed"):
            processor.run(subjects=["sub-01"], parallel=False)
    
    @patch('elikopy.core.processor.DataValidator')
    @patch('elikopy.core.processor.ElikopyProcessor._process_subject')
    def test_run_with_subject_failure(self, mock_process_subject, mock_validator_class, mock_study, temp_output_dir):
        """Test processing with subject failure"""
        # Mock validator
        mock_validator = Mock()
        mock_validator.validate_dwi_data.return_value = ValidationResult(
            is_valid=True, errors=[], warnings=[], suggestions=[]
        )
        mock_validator_class.return_value = mock_validator
        
        # Mock processing failure
        mock_process_subject.side_effect = RuntimeError("Processing failed")
        
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        results = processor.run(subjects=["sub-01"], parallel=False)
        
        assert "sub-01" in results
        assert len(results["sub-01"]) == 1
        assert results["sub-01"][0].status == ProcessingStatus.FAILED
        assert "Processing failed" in results["sub-01"][0].error_message
        assert processor.pipeline.failed_subjects == 1
    
    def test_resume_checkpoint_not_found(self, mock_study, temp_output_dir):
        """Test resume with non-existent checkpoint"""
        processor = ElikopyProcessor(
            study=mock_study,
            processing_type="dti",
            output_dir=temp_output_dir
        )
        
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            processor.resume("/nonexistent/checkpoint.pkl")


if __name__ == "__main__":
    pytest.main([__file__])