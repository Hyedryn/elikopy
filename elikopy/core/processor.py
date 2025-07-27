"""
ElikopyProcessor class - Processing orchestration

This module provides comprehensive processing orchestration with pipeline management,
input validation, parallel processing, and job submission capabilities.
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from elikopy.core.base import ConfigurableComponent, ProcessingResult, ProcessingStatus, ValidationResult, ValidationError, ValidationWarning
from elikopy.core.study import ElikopyStudy, Subject
from elikopy.infrastructure.scheduler import JobScheduler, ProcessingJob, JobID, JobStatus
from elikopy.infrastructure.logging import get_logger
from elikopy.data.validator import DataValidator


class PipelineStage(Enum):
    """Pipeline processing stages"""
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    MODELING = "modeling"
    POSTPROCESSING = "postprocessing"
    OUTPUT = "output"


@dataclass
class ProcessingStep:
    """Class representing a processing step in the pipeline"""
    
    name: str
    processor_class: str
    stage: PipelineStage
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    result: Optional[ProcessingResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    job_id: Optional[JobID] = None
    
    def execute(self, processor_instance: Any, **kwargs) -> ProcessingResult:
        """Execute the processing step
        
        Args:
            processor_instance: Instantiated processor object
            **kwargs: Additional arguments for processor
            
        Returns:
            ProcessingResult object
        """
        self.status = ProcessingStatus.IN_PROGRESS
        self.start_time = datetime.now()
        
        try:
            # Validate inputs before processing
            if hasattr(processor_instance, 'validate_inputs'):
                validation_result = processor_instance.validate_inputs(**kwargs)
                if not validation_result.is_valid:
                    raise ValueError(f"Input validation failed: {validation_result.errors}")
            
            # Execute processing
            self.result = processor_instance.process(**{**self.config, **kwargs})
            self.status = self.result.status
            self.end_time = datetime.now()
            
            return self.result
            
        except Exception as e:
            self.status = ProcessingStatus.FAILED
            self.end_time = datetime.now()
            self.result = ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={
                    'error': str(e),
                    'step_name': self.name,
                    'stage': self.stage.value,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': self.end_time.isoformat() if self.end_time else None
                },
                error_message=str(e)
            )
            raise
    
    def get_duration(self) -> Optional[float]:
        """Get step execution duration in seconds
        
        Returns:
            Duration in seconds or None if not completed
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {
            'name': self.name,
            'processor_class': self.processor_class,
            'stage': self.stage.value,
            'config': self.config,
            'dependencies': self.dependencies,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.get_duration(),
            'job_id': str(self.job_id) if self.job_id else None
        }


@dataclass
class ProcessingPipeline:
    """Class representing a complete processing pipeline"""
    
    name: str
    steps: List[ProcessingStep] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_subjects: int = 0
    completed_subjects: int = 0
    failed_subjects: int = 0
    
    def get_progress(self) -> float:
        """Get pipeline progress as percentage
        
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if self.total_subjects == 0:
            return 0.0
        return (self.completed_subjects / self.total_subjects) * 100.0
    
    def get_duration(self) -> Optional[float]:
        """Get pipeline execution duration in seconds
        
        Returns:
            Duration in seconds or None if not completed
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {
            'name': self.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.get_duration(),
            'total_subjects': self.total_subjects,
            'completed_subjects': self.completed_subjects,
            'failed_subjects': self.failed_subjects,
            'progress': self.get_progress(),
            'steps': [step.to_dict() for step in self.steps]
        }


class ElikopyProcessor(ConfigurableComponent):
    """Enhanced class for processing orchestration with pipeline management"""
    
    def __init__(
        self, 
        study: ElikopyStudy,
        processing_type: str,
        scheduler: Optional[JobScheduler] = None,
        output_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize processor
        
        Args:
            study: ElikopyStudy object
            processing_type: Type of processing to perform
            scheduler: JobScheduler object (optional)
            output_dir: Output directory for results (optional)
            **kwargs: Additional configuration
        """
        self.study = study
        self.processing_type = processing_type
        self.scheduler = scheduler
        self.output_dir = output_dir or (study.study_path / "processing_results")
        self.config = kwargs
        self.logger = get_logger(__name__).get_logger()
        
        # Initialize data validator
        self.validator = DataValidator()
        
        # Create processing pipeline
        self.pipeline = self._create_pipeline()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing state
        self.processing_state: Dict[str, Any] = {}
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized ElikopyProcessor for {processing_type} processing")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Pipeline has {len(self.pipeline.steps)} steps")
        
    def _create_pipeline(self) -> ProcessingPipeline:
        """Create processing pipeline based on type
        
        Returns:
            ProcessingPipeline object
        """
        pipeline = ProcessingPipeline(name=f"{self.processing_type}_pipeline")
        
        # Define processing steps based on type
        if self.processing_type == "dti":
            pipeline.steps = [
                ProcessingStep(
                    name="dti_fitting",
                    processor_class="DTIProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("dti", {}),
                    dependencies=[]
                )
            ]
            
        elif self.processing_type == "noddi":
            pipeline.steps = [
                ProcessingStep(
                    name="noddi_fitting",
                    processor_class="NODDIProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("noddi", {}),
                    dependencies=[]
                )
            ]
            
        elif self.processing_type == "csd":
            pipeline.steps = [
                ProcessingStep(
                    name="csd_fitting",
                    processor_class="CSDProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("csd", {}),
                    dependencies=[]
                )
            ]
            
        elif self.processing_type == "msmt_csd":
            pipeline.steps = [
                ProcessingStep(
                    name="msmt_csd_fitting",
                    processor_class="CSDProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("msmt_csd", {}),
                    dependencies=[]
                )
            ]
            
        elif self.processing_type == "fingerprinting":
            pipeline.steps = [
                ProcessingStep(
                    name="fingerprinting_fitting",
                    processor_class="MicrostructureFingerprintingProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("fingerprinting", {}),
                    dependencies=[]
                )
            ]
            
        elif self.processing_type == "tracking":
            pipeline.steps = [
                ProcessingStep(
                    name="csd_fitting",
                    processor_class="CSDProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("csd", {}),
                    dependencies=[]
                ),
                ProcessingStep(
                    name="tracking",
                    processor_class="TrackingProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("tracking", {}),
                    dependencies=["csd_fitting"]
                )
            ]
            
        elif self.processing_type == "connectivity":
            pipeline.steps = [
                ProcessingStep(
                    name="csd_fitting",
                    processor_class="CSDProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("csd", {}),
                    dependencies=[]
                ),
                ProcessingStep(
                    name="tracking",
                    processor_class="TrackingProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("tracking", {}),
                    dependencies=["csd_fitting"]
                ),
                ProcessingStep(
                    name="connectivity",
                    processor_class="ConnectivityProcessor",
                    stage=PipelineStage.POSTPROCESSING,
                    config=self.config.get("connectivity", {}),
                    dependencies=["tracking"]
                )
            ]
            
        elif self.processing_type == "full_pipeline":
            # Complete processing pipeline
            pipeline.steps = [
                ProcessingStep(
                    name="dti_fitting",
                    processor_class="DTIProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("dti", {}),
                    dependencies=[]
                ),
                ProcessingStep(
                    name="csd_fitting",
                    processor_class="CSDProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("csd", {}),
                    dependencies=[]
                ),
                ProcessingStep(
                    name="noddi_fitting",
                    processor_class="NODDIProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("noddi", {}),
                    dependencies=[]
                ),
                ProcessingStep(
                    name="tracking",
                    processor_class="TrackingProcessor",
                    stage=PipelineStage.MODELING,
                    config=self.config.get("tracking", {}),
                    dependencies=["csd_fitting"]
                ),
                ProcessingStep(
                    name="connectivity",
                    processor_class="ConnectivityProcessor",
                    stage=PipelineStage.POSTPROCESSING,
                    config=self.config.get("connectivity", {}),
                    dependencies=["tracking"]
                )
            ]
            
        else:
            raise ValueError(f"Unknown processing type: {self.processing_type}")
        
        return pipeline
    
    def _get_processor_instance(self, processor_class: str) -> Any:
        """Get processor instance by class name
        
        Args:
            processor_class: Processor class name
            
        Returns:
            Processor instance
        """
        # Import processors here to avoid circular imports
        processor_map = {
            "DTIProcessor": ("elikopy.processing.dti", "DTIProcessor"),
            "NODDIProcessor": ("elikopy.processing.noddi", "NODDIProcessor"),
            "CSDProcessor": ("elikopy.processing.csd", "CSDProcessor"),
            "TrackingProcessor": ("elikopy.processing.tracking", "TrackingProcessor"),
            "ConnectivityProcessor": ("elikopy.processing.connectivity", "ConnectivityProcessor"),
            "MicrostructureFingerprintingProcessor": ("elikopy.processing.fingerprinting", "MicrostructureFingerprintingProcessor")
        }
        
        if processor_class not in processor_map:
            raise ValueError(f"Unknown processor class: {processor_class}")
        
        module_name, class_name = processor_map[processor_class]
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            processor_cls = getattr(module, class_name)
            return processor_cls()
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import processor {processor_class}: {e}")
    
    def validate_inputs(self, subjects: Optional[List[str]] = None) -> ValidationResult:
        """Comprehensive input validation before processing
        
        Args:
            subjects: List of subject IDs to validate (None for all)
            
        Returns:
            ValidationResult object
        """
        self.logger.info("Starting comprehensive input validation...")
        
        errors = []
        warnings = []
        suggestions = []
        
        # Get subjects to validate
        subject_list = subjects or [s.id for s in self.study.get_subjects()]
        
        if not subject_list:
            errors.append(ValidationError("No subjects found for processing"))
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Validate study setup
        if not self.study.bids_handler:
            errors.append(ValidationError("Study BIDS handler not initialized"))
        
        if not self.study.derivatives_dir:
            errors.append(ValidationError("Study derivatives directory not set"))
        
        # Validate each subject
        for subject_id in subject_list:
            subject = self.study.get_subject(subject_id)
            if not subject:
                errors.append(ValidationError(f"Subject not found: {subject_id}"))
                continue
            
            # Validate subject data
            subject_validation = self._validate_subject_data(subject)
            errors.extend(subject_validation.errors)
            warnings.extend(subject_validation.warnings)
            suggestions.extend(subject_validation.suggestions)
        
        # Validate processing configuration
        config_validation = self._validate_processing_config()
        errors.extend(config_validation.errors)
        warnings.extend(config_validation.warnings)
        suggestions.extend(config_validation.suggestions)
        
        # Validate pipeline dependencies
        dependency_validation = self._validate_pipeline_dependencies()
        errors.extend(dependency_validation.errors)
        warnings.extend(dependency_validation.warnings)
        
        # Validate scheduler if parallel processing is requested
        if self.scheduler:
            scheduler_validation = self._validate_scheduler()
            errors.extend(scheduler_validation.errors)
            warnings.extend(scheduler_validation.warnings)
        
        is_valid = len(errors) == 0
        
        self.logger.info(f"Input validation completed: {len(errors)} errors, {len(warnings)} warnings")
        
        return ValidationResult(is_valid, errors, warnings, suggestions)
    
    def _validate_subject_data(self, subject: Subject) -> ValidationResult:
        """Validate data for a single subject
        
        Args:
            subject: Subject object
            
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check if subject has DWI files
        if not subject.dwi_files:
            errors.append(ValidationError(f"No DWI files found for subject {subject.id}"))
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Validate each DWI file
        for dwi_file in subject.dwi_files:
            # Use data validator
            dwi_validation = self.validator.validate_dwi_data(dwi_file.path, dwi_file.bval_path, dwi_file.bvec_path, dwi_file.json_path)
            errors.extend(dwi_validation.errors)
            warnings.extend(dwi_validation.warnings)
            suggestions.extend(dwi_validation.suggestions)
        
        # Check for anatomical files if needed
        processing_needs_anat = self.processing_type in ["connectivity", "full_pipeline"]
        if processing_needs_anat and not subject.anatomical_files:
            warnings.append(ValidationWarning(
                f"No anatomical files found for subject {subject.id}. "
                "This may limit connectivity analysis capabilities."
            ))
            suggestions.append(
                f"Consider adding anatomical data for subject {subject.id} "
                "to improve registration and connectivity analysis"
            )
        
        return ValidationResult(len(errors) == 0, errors, warnings, suggestions)
    
    def _validate_processing_config(self) -> ValidationResult:
        """Validate processing configuration
        
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Validate configuration for each step
        for step in self.pipeline.steps:
            try:
                processor_instance = self._get_processor_instance(step.processor_class)
                
                # Check if processor has validate_config method
                if hasattr(processor_instance, 'validate_config'):
                    config_validation = processor_instance.validate_config(step.config)
                    if hasattr(config_validation, 'errors'):
                        errors.extend(config_validation.errors)
                    if hasattr(config_validation, 'warnings'):
                        warnings.extend(config_validation.warnings)
                    elif not config_validation:  # Boolean return
                        errors.append(ValidationError(f"Invalid configuration for step {step.name}"))
                        
            except Exception as e:
                errors.append(ValidationError(f"Failed to validate config for step {step.name}: {e}"))
        
        return ValidationResult(len(errors) == 0, errors, warnings, suggestions)
    
    def _validate_pipeline_dependencies(self) -> ValidationResult:
        """Validate pipeline step dependencies
        
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        
        step_names = {step.name for step in self.pipeline.steps}
        
        # Check that all dependencies exist
        for step in self.pipeline.steps:
            for dependency in step.dependencies:
                if dependency not in step_names:
                    errors.append(ValidationError(
                        f"Step {step.name} depends on {dependency}, but {dependency} is not in pipeline"
                    ))
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append(ValidationError("Pipeline has circular dependencies"))
        
        return ValidationResult(len(errors) == 0, errors, warnings, [])
    
    def _has_circular_dependencies(self) -> bool:
        """Check if pipeline has circular dependencies
        
        Returns:
            True if circular dependencies exist
        """
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_name: str) -> bool:
            visited.add(step_name)
            rec_stack.add(step_name)
            
            # Find step by name
            step = next((s for s in self.pipeline.steps if s.name == step_name), None)
            if not step:
                return False
            
            # Check all dependencies
            for dependency in step.dependencies:
                if dependency not in visited:
                    if has_cycle(dependency):
                        return True
                elif dependency in rec_stack:
                    return True
            
            rec_stack.remove(step_name)
            return False
        
        # Check each step
        for step in self.pipeline.steps:
            if step.name not in visited:
                if has_cycle(step.name):
                    return True
        
        return False
    
    def _validate_scheduler(self) -> ValidationResult:
        """Validate scheduler configuration
        
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        
        if not self.scheduler:
            return ValidationResult(True, errors, warnings, [])
        
        # Check scheduler type
        if self.scheduler.scheduler_type not in ["slurm", "local"]:
            errors.append(ValidationError(f"Unsupported scheduler type: {self.scheduler.scheduler_type}"))
        
        # Check SLURM availability if needed
        if self.scheduler.scheduler_type == "slurm":
            try:
                import subprocess
                subprocess.run(["sinfo", "--version"], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, 
                             check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                errors.append(ValidationError("SLURM is not available on this system"))
        
        return ValidationResult(len(errors) == 0, errors, warnings, [])
    
    def run(
        self, 
        subjects: Optional[List[str]] = None,
        parallel: bool = True,
        save_checkpoints: bool = True,
        checkpoint_interval: int = 5
    ) -> Dict[str, List[ProcessingResult]]:
        """Execute processing pipeline with comprehensive orchestration
        
        Args:
            subjects: List of subject IDs to process (None for all)
            parallel: Whether to run in parallel using scheduler
            save_checkpoints: Whether to save processing checkpoints
            checkpoint_interval: Save checkpoint every N subjects
            
        Returns:
            Dictionary of processing results by subject
        """
        # Get subjects to process
        subject_list = subjects or [s.id for s in self.study.get_subjects()]
        
        self.logger.info(f"Starting {self.processing_type} processing for {len(subject_list)} subjects")
        self.logger.info(f"Parallel processing: {parallel and self.scheduler is not None}")
        
        # Comprehensive input validation
        validation_result = self.validate_inputs(subject_list)
        if not validation_result.is_valid:
            error_msg = f"Input validation failed with {len(validation_result.errors)} errors"
            self.logger.error(error_msg)
            for error in validation_result.errors:
                self.logger.error(f"  - {error.message}")
            raise ValueError(error_msg)
        
        # Log warnings
        for warning in validation_result.warnings:
            self.logger.warning(f"  - {warning.message}")
        
        # Initialize pipeline
        self.pipeline.status = ProcessingStatus.IN_PROGRESS
        self.pipeline.start_time = datetime.now()
        self.pipeline.total_subjects = len(subject_list)
        self.pipeline.completed_subjects = 0
        self.pipeline.failed_subjects = 0
        
        # Initialize processing state
        self.processing_state = {
            'pipeline': self.pipeline.to_dict(),
            'subjects': subject_list,
            'completed_subjects': [],
            'failed_subjects': [],
            'current_subject_index': 0,
            'start_time': self.pipeline.start_time.isoformat(),
            'parallel': parallel,
            'save_checkpoints': save_checkpoints,
            'checkpoint_interval': checkpoint_interval
        }
        
        results: Dict[str, List[ProcessingResult]] = {}
        
        try:
            if parallel and self.scheduler:
                # Parallel processing with job scheduler
                results = self._run_parallel_processing(subject_list, save_checkpoints, checkpoint_interval)
            else:
                # Sequential processing
                results = self._run_sequential_processing(subject_list, save_checkpoints, checkpoint_interval)
            
            # Update pipeline status
            self.pipeline.status = ProcessingStatus.COMPLETED
            self.pipeline.end_time = datetime.now()
            
            self.logger.info(f"Processing completed successfully")
            self.logger.info(f"Total duration: {self.pipeline.get_duration():.2f} seconds")
            self.logger.info(f"Completed subjects: {self.pipeline.completed_subjects}")
            self.logger.info(f"Failed subjects: {self.pipeline.failed_subjects}")
            
        except Exception as e:
            self.pipeline.status = ProcessingStatus.FAILED
            self.pipeline.end_time = datetime.now()
            self.logger.error(f"Processing failed: {e}")
            
            # Save final checkpoint on failure
            if save_checkpoints:
                self._save_checkpoint("processing_failed")
            
            raise
        
        # Save final processing state
        self._save_processing_summary(results)
        
        return results
    
    def _run_sequential_processing(
        self, 
        subject_list: List[str], 
        save_checkpoints: bool, 
        checkpoint_interval: int
    ) -> Dict[str, List[ProcessingResult]]:
        """Run sequential processing
        
        Args:
            subject_list: List of subject IDs
            save_checkpoints: Whether to save checkpoints
            checkpoint_interval: Checkpoint interval
            
        Returns:
            Dictionary of processing results by subject
        """
        results: Dict[str, List[ProcessingResult]] = {}
        
        for i, subject_id in enumerate(subject_list):
            self.logger.info(f"Processing subject {subject_id} ({i+1}/{len(subject_list)})")
            
            try:
                # Process subject
                subject_results = self._process_subject(subject_id)
                results[subject_id] = subject_results
                
                # Update state
                self.pipeline.completed_subjects += 1
                self.processing_state['completed_subjects'].append(subject_id)
                self.processing_state['current_subject_index'] = i + 1
                
                # Check if all steps completed successfully
                all_successful = all(
                    result.status == ProcessingStatus.COMPLETED 
                    for result in subject_results
                )
                
                if not all_successful:
                    self.pipeline.failed_subjects += 1
                    self.processing_state['failed_subjects'].append(subject_id)
                    self.logger.warning(f"Some steps failed for subject {subject_id}")
                
                # Save checkpoint if needed
                if save_checkpoints and (i + 1) % checkpoint_interval == 0:
                    checkpoint_name = f"checkpoint_subject_{i+1}"
                    self._save_checkpoint(checkpoint_name)
                    self.logger.info(f"Saved checkpoint: {checkpoint_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process subject {subject_id}: {e}")
                self.pipeline.failed_subjects += 1
                self.processing_state['failed_subjects'].append(subject_id)
                
                # Create failed result
                results[subject_id] = [ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={'subject_id': subject_id, 'error': str(e)},
                    error_message=str(e)
                )]
        
        return results
    
    def _run_parallel_processing(
        self, 
        subject_list: List[str], 
        save_checkpoints: bool, 
        checkpoint_interval: int
    ) -> Dict[str, List[ProcessingResult]]:
        """Run parallel processing using job scheduler
        
        Args:
            subject_list: List of subject IDs
            save_checkpoints: Whether to save checkpoints
            checkpoint_interval: Checkpoint interval
            
        Returns:
            Dictionary of processing results by subject
        """
        if not self.scheduler:
            raise RuntimeError("Scheduler not available for parallel processing")
        
        self.logger.info("Submitting jobs for parallel processing...")
        
        # Submit jobs for all subjects
        job_ids = []
        subject_job_map = {}
        
        for subject_id in subject_list:
            try:
                job_id = self._submit_subject_job(subject_id)
                job_ids.append(job_id)
                subject_job_map[str(job_id)] = subject_id
                self.logger.debug(f"Submitted job {job_id} for subject {subject_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to submit job for subject {subject_id}: {e}")
                self.pipeline.failed_subjects += 1
                self.processing_state['failed_subjects'].append(subject_id)
        
        self.logger.info(f"Submitted {len(job_ids)} jobs, monitoring progress...")
        
        # Monitor jobs and collect results
        results = self._monitor_parallel_jobs(
            job_ids, subject_job_map, save_checkpoints, checkpoint_interval
        )
        
        return results
    
    def _process_subject(self, subject_id: str) -> List[ProcessingResult]:
        """Process a single subject through the pipeline
        
        Args:
            subject_id: Subject ID
            
        Returns:
            List of processing results for each step
        """
        results = []
        subject = self.study.get_subject(subject_id)
        
        if not subject:
            error_msg = f"Subject not found: {subject_id}"
            self.logger.error(error_msg)
            return [ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={'subject_id': subject_id},
                error_message=error_msg
            )]
        
        self.logger.debug(f"Processing subject {subject_id} with {len(self.pipeline.steps)} steps")
        
        # Create subject-specific output directory
        subject_output_dir = self.output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute steps in dependency order
        step_results = {}
        execution_order = self._get_execution_order()
        
        for step_name in execution_order:
            step = next(s for s in self.pipeline.steps if s.name == step_name)
            
            self.logger.debug(f"Executing step {step.name} for subject {subject_id}")
            
            try:
                # Get processor instance
                processor_instance = self._get_processor_instance(step.processor_class)
                
                # Prepare step arguments
                step_kwargs = {
                    'subject': subject,
                    'subject_id': subject_id,
                    'output_dir': subject_output_dir,
                    'study': self.study,
                    'previous_results': step_results
                }
                
                # Execute step
                result = step.execute(processor_instance, **step_kwargs)
                results.append(result)
                step_results[step.name] = result
                
                self.logger.debug(f"Step {step.name} completed with status {result.status.value}")
                
                # Stop processing if step failed and it's critical
                if result.status == ProcessingStatus.FAILED:
                    self.logger.warning(f"Step {step.name} failed for subject {subject_id}")
                    # Continue with remaining steps but mark them as skipped
                    remaining_steps = execution_order[execution_order.index(step_name) + 1:]
                    for remaining_step_name in remaining_steps:
                        remaining_step = next(s for s in self.pipeline.steps if s.name == remaining_step_name)
                        skipped_result = ProcessingResult(
                            status=ProcessingStatus.FAILED,
                            output_files=[],
                            metadata={
                                'subject_id': subject_id,
                                'step_name': remaining_step.name,
                                'skipped_due_to_failure': step.name
                            },
                            error_message=f"Skipped due to failure in step {step.name}"
                        )
                        results.append(skipped_result)
                    break
                    
            except Exception as e:
                error_msg = f"Failed to execute step {step.name} for subject {subject_id}: {e}"
                self.logger.error(error_msg)
                
                failed_result = ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={
                        'subject_id': subject_id,
                        'step_name': step.name,
                        'error': str(e)
                    },
                    error_message=error_msg
                )
                results.append(failed_result)
                step_results[step.name] = failed_result
                
                # Stop processing on critical failure
                break
        
        return results
    
    def _get_execution_order(self) -> List[str]:
        """Get step execution order based on dependencies
        
        Returns:
            List of step names in execution order
        """
        # Topological sort of steps based on dependencies
        in_degree = {step.name: 0 for step in self.pipeline.steps}
        graph = {step.name: [] for step in self.pipeline.steps}
        
        # Build dependency graph
        for step in self.pipeline.steps:
            for dependency in step.dependencies:
                graph[dependency].append(step.name)
                in_degree[step.name] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [step_name for step_name, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(execution_order) != len(self.pipeline.steps):
            raise ValueError("Circular dependencies detected in pipeline")
        
        return execution_order
    
    def _submit_subject_job(self, subject_id: str) -> JobID:
        """Submit job for processing a single subject
        
        Args:
            subject_id: Subject ID
            
        Returns:
            JobID object
        """
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        # Create processing script for subject
        script_path = self._create_processing_script(subject_id)
        
        # Create job configuration
        job_config = ProcessingJob(
            name=f"elikopy_{self.processing_type}_{subject_id}",
            script_path=script_path,
            script_args=[subject_id, str(self.output_dir)],
            working_dir=self.study.study_path,
            output_file=f"elikopy_{self.processing_type}_{subject_id}.out",
            error_file=f"elikopy_{self.processing_type}_{subject_id}.err",
            cpus_per_task=self.config.get('cpus_per_task', 4),
            mem_per_cpu=self.config.get('mem_per_cpu', 4),
            time_limit=self.config.get('time_limit', "24:00:00"),
            partition=self.config.get('partition'),
            account=self.config.get('account'),
            use_gpu=self.config.get('use_gpu', False),
            gpu_count=self.config.get('gpu_count', 0),
            environment_vars={
                'ELIKOPY_PROCESSING_TYPE': self.processing_type,
                'ELIKOPY_STUDY_PATH': str(self.study.study_path),
                'ELIKOPY_OUTPUT_DIR': str(self.output_dir)
            }
        )
        
        # Submit job
        job_id = self.scheduler.submit_job(job_config)
        self.logger.debug(f"Submitted job {job_id} for subject {subject_id}")
        
        return job_id
    
    def _create_processing_script(self, subject_id: str) -> Path:
        """Create processing script for a subject
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Path to processing script
        """
        script_dir = self.output_dir / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = script_dir / f"process_{subject_id}.py"
        
        # Create script content
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated processing script for subject {subject_id}
Processing type: {self.processing_type}
"""

import sys
import json
import pickle
from pathlib import Path

# Add elikopy to path
sys.path.insert(0, "{self.study.study_path}")

from elikopy.core.study import ElikopyStudy
from elikopy.core.processor import ElikopyProcessor

def main():
    if len(sys.argv) < 3:
        print("Usage: python process_subject.py <subject_id> <output_dir>")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    output_dir = Path(sys.argv[2])
    
    try:
        # Load study
        study = ElikopyStudy.load_from_checkpoint("{self.study.study_path}")
        
        # Create processor
        processor = ElikopyProcessor(
            study=study,
            processing_type="{self.processing_type}",
            output_dir=output_dir,
            **{json.dumps(self.config)}
        )
        
        # Process subject
        results = processor._process_subject(subject_id)
        
        # Save results
        result_file = output_dir / subject_id / "processing_results.pkl"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file, "wb") as f:
            pickle.dump(results, f)
        
        # Save summary
        summary_file = output_dir / subject_id / "processing_summary.json"
        summary = {{
            "subject_id": subject_id,
            "processing_type": "{self.processing_type}",
            "num_steps": len(results),
            "successful_steps": sum(1 for r in results if r.status.value == "completed"),
            "failed_steps": sum(1 for r in results if r.status.value == "failed"),
            "output_files": [str(f) for result in results for f in result.output_files]
        }}
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Processing completed for subject {{subject_id}}")
        
    except Exception as e:
        print(f"Processing failed for subject {{subject_id}}: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # Write script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        return script_path
    
    def _monitor_parallel_jobs(
        self, 
        job_ids: List[JobID], 
        subject_job_map: Dict[str, str],
        save_checkpoints: bool,
        checkpoint_interval: int
    ) -> Dict[str, List[ProcessingResult]]:
        """Monitor parallel jobs and collect results
        
        Args:
            job_ids: List of job IDs
            subject_job_map: Mapping from job ID to subject ID
            save_checkpoints: Whether to save checkpoints
            checkpoint_interval: Checkpoint interval
            
        Returns:
            Dictionary of processing results by subject
        """
        results: Dict[str, List[ProcessingResult]] = {}
        completed_count = 0
        
        # Monitor jobs until completion
        final_statuses = self.scheduler.wait_for_jobs(job_ids, poll_interval=30)
        
        # Collect results for each job
        for job_id_str, job_status in final_statuses.items():
            subject_id = subject_job_map.get(job_id_str)
            if not subject_id:
                self.logger.warning(f"No subject found for job {job_id_str}")
                continue
            
            if job_status.status == JobStatus.COMPLETED:
                # Load results from file
                try:
                    results[subject_id] = self._load_subject_results(subject_id)
                    self.pipeline.completed_subjects += 1
                    self.processing_state['completed_subjects'].append(subject_id)
                    completed_count += 1
                    
                    self.logger.info(f"Completed processing for subject {subject_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load results for subject {subject_id}: {e}")
                    results[subject_id] = [ProcessingResult(
                        status=ProcessingStatus.FAILED,
                        output_files=[],
                        metadata={'subject_id': subject_id, 'error': str(e)},
                        error_message=f"Failed to load results: {e}"
                    )]
                    self.pipeline.failed_subjects += 1
                    self.processing_state['failed_subjects'].append(subject_id)
            else:
                # Job failed
                self.logger.error(f"Job failed for subject {subject_id}: {job_status.status}")
                results[subject_id] = [ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={'subject_id': subject_id, 'job_status': job_status.status},
                    error_message=f"Job failed with status: {job_status.status}"
                )]
                self.pipeline.failed_subjects += 1
                self.processing_state['failed_subjects'].append(subject_id)
            
            # Save checkpoint if needed
            if save_checkpoints and completed_count % checkpoint_interval == 0:
                checkpoint_name = f"parallel_checkpoint_{completed_count}"
                self._save_checkpoint(checkpoint_name)
                self.logger.info(f"Saved checkpoint: {checkpoint_name}")
        
        return results
    
    def _load_subject_results(self, subject_id: str) -> List[ProcessingResult]:
        """Load processing results for a subject
        
        Args:
            subject_id: Subject ID
            
        Returns:
            List of processing results
        """
        result_file = self.output_dir / subject_id / "processing_results.pkl"
        
        if not result_file.exists():
            raise FileNotFoundError(f"Results file not found: {result_file}")
        
        with open(result_file, "rb") as f:
            results = pickle.load(f)
        
        return results
    
    def _save_checkpoint(self, checkpoint_name: str) -> Path:
        """Save processing checkpoint with comprehensive state persistence
        
        Args:
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.pkl"
        
        # Prepare comprehensive checkpoint data
        checkpoint_data = {
            'processor_config': {
                'processing_type': self.processing_type,
                'config': self.config,
                'output_dir': str(self.output_dir)
            },
            'study_config': {
                'study_path': str(self.study.study_path),
                'bids_root': str(self.study.bids_root) if self.study.bids_root else None,
                'qsiprep_dir': str(self.study.qsiprep_dir) if self.study.qsiprep_dir else None,
                'derivatives_name': self.study.derivatives_name
            },
            'pipeline_state': self.pipeline.to_dict(),
            'processing_state': self.processing_state.copy(),
            'step_states': {
                step.name: {
                    'status': step.status.value,
                    'start_time': step.start_time.isoformat() if step.start_time else None,
                    'end_time': step.end_time.isoformat() if step.end_time else None,
                    'duration': step.get_duration(),
                    'job_id': str(step.job_id) if step.job_id else None,
                    'result_metadata': step.result.metadata if step.result else None
                }
                for step in self.pipeline.steps
            },
            'checkpoint_metadata': {
                'checkpoint_name': checkpoint_name,
                'timestamp': timestamp,
                'created_by': 'ElikopyProcessor',
                'version': '2.0',
                'elikopy_version': getattr(self, '_version', '1.0'),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            }
        }
        
        # Save checkpoint with atomic write
        try:
            # Write to temporary file first
            temp_file = checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Atomic move to final location
            temp_file.replace(checkpoint_file)
            
            # Save JSON metadata for easier inspection and status reporting
            metadata_file = checkpoint_file.with_suffix('.json')
            metadata = {
                'checkpoint_file': str(checkpoint_file),
                'checkpoint_name': checkpoint_name,
                'timestamp': timestamp,
                'processing_type': self.processing_type,
                'total_subjects': self.pipeline.total_subjects,
                'completed_subjects': self.pipeline.completed_subjects,
                'failed_subjects': self.pipeline.failed_subjects,
                'progress_percent': self.pipeline.get_progress(),
                'status': self.pipeline.status.value,
                'duration_seconds': self.pipeline.get_duration(),
                'estimated_remaining_time': self._estimate_remaining_time(),
                'subjects_per_hour': self._calculate_processing_rate(),
                'step_progress': {
                    step.name: {
                        'status': step.status.value,
                        'duration': step.get_duration()
                    }
                    for step in self.pipeline.steps
                }
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved checkpoint: {checkpoint_file}")
            self.logger.debug(f"Checkpoint metadata: {metadata_file}")
            return checkpoint_file
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_name}: {e}")
            # Clean up temporary file if it exists
            temp_file = checkpoint_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise
    
    def _save_processing_summary(self, results: Dict[str, List[ProcessingResult]]) -> None:
        """Save processing summary
        
        Args:
            results: Processing results by subject
        """
        summary_file = self.output_dir / "processing_summary.json"
        
        # Calculate summary statistics
        total_subjects = len(results)
        successful_subjects = sum(
            1 for subject_results in results.values()
            if all(r.status == ProcessingStatus.COMPLETED for r in subject_results)
        )
        failed_subjects = total_subjects - successful_subjects
        
        # Collect all output files
        all_output_files = []
        for subject_results in results.values():
            for result in subject_results:
                all_output_files.extend([str(f) for f in result.output_files])
        
        # Create summary
        summary = {
            'processing_summary': {
                'processing_type': self.processing_type,
                'pipeline_name': self.pipeline.name,
                'start_time': self.pipeline.start_time.isoformat() if self.pipeline.start_time else None,
                'end_time': self.pipeline.end_time.isoformat() if self.pipeline.end_time else None,
                'duration_seconds': self.pipeline.get_duration(),
                'status': self.pipeline.status.value
            },
            'subject_statistics': {
                'total_subjects': total_subjects,
                'successful_subjects': successful_subjects,
                'failed_subjects': failed_subjects,
                'success_rate': (successful_subjects / total_subjects * 100) if total_subjects > 0 else 0
            },
            'pipeline_steps': [step.to_dict() for step in self.pipeline.steps],
            'output_files': {
                'total_files': len(all_output_files),
                'files_by_subject': {
                    subject_id: [str(f) for result in subject_results for f in result.output_files]
                    for subject_id, subject_results in results.items()
                }
            },
            'configuration': self.config,
            'study_info': self.study.get_study_summary()
        }
        
        # Save summary
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Saved processing summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processing summary: {e}")
    
    def resume(self, checkpoint_path: Union[str, Path]) -> Dict[str, List[ProcessingResult]]:
        """Resume processing from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary of processing results by subject
        """
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        self.logger.info(f"Resuming processing from checkpoint: {checkpoint_file}")
        
        try:
            # Load checkpoint data
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint format
            required_keys = ['processor_config', 'study_config', 'pipeline_state', 'processing_state']
            for key in required_keys:
                if key not in checkpoint_data:
                    raise ValueError(f"Invalid checkpoint format: missing key '{key}'")
            
            # Restore processor configuration
            processor_config = checkpoint_data['processor_config']
            if processor_config['processing_type'] != self.processing_type:
                raise ValueError(
                    f"Checkpoint processing type ({processor_config['processing_type']}) "
                    f"does not match current processor ({self.processing_type})"
                )
            
            # Update configuration with checkpoint data
            self.config.update(processor_config['config'])
            
            # Restore pipeline state
            pipeline_state = checkpoint_data['pipeline_state']
            self.pipeline.status = ProcessingStatus(pipeline_state['status'])
            self.pipeline.total_subjects = pipeline_state['total_subjects']
            self.pipeline.completed_subjects = pipeline_state['completed_subjects']
            self.pipeline.failed_subjects = pipeline_state['failed_subjects']
            
            if pipeline_state['start_time']:
                self.pipeline.start_time = datetime.fromisoformat(pipeline_state['start_time'])
            
            # Restore processing state
            self.processing_state = checkpoint_data['processing_state'].copy()
            
            # Determine remaining subjects to process
            all_subjects = self.processing_state['subjects']
            completed_subjects = set(self.processing_state.get('completed_subjects', []))
            failed_subjects = set(self.processing_state.get('failed_subjects', []))
            processed_subjects = completed_subjects | failed_subjects
            
            remaining_subjects = [s for s in all_subjects if s not in processed_subjects]
            
            self.logger.info(f"Checkpoint loaded successfully")
            self.logger.info(f"Total subjects: {len(all_subjects)}")
            self.logger.info(f"Completed subjects: {len(completed_subjects)}")
            self.logger.info(f"Failed subjects: {len(failed_subjects)}")
            self.logger.info(f"Remaining subjects: {len(remaining_subjects)}")
            
            if not remaining_subjects:
                self.logger.info("No remaining subjects to process")
                return {}
            
            # Resume processing with remaining subjects
            parallel = self.processing_state.get('parallel', False)
            save_checkpoints = self.processing_state.get('save_checkpoints', True)
            checkpoint_interval = self.processing_state.get('checkpoint_interval', 5)
            
            return self.run(
                subjects=remaining_subjects,
                parallel=parallel,
                save_checkpoints=save_checkpoints,
                checkpoint_interval=checkpoint_interval
            )
            
        except Exception as e:
            error_msg = f"Failed to resume from checkpoint: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @classmethod
    def list_checkpoints(cls, checkpoint_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List available checkpoints
        
        Args:
            checkpoint_dir: Directory to search for checkpoints (optional)
            
        Returns:
            List of checkpoint metadata
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / "processing_results" / "checkpoints"
        
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        
        # Search for checkpoint metadata files
        for metadata_file in checkpoint_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if corresponding pickle file exists
                checkpoint_file = Path(metadata["checkpoint_file"])
                if checkpoint_file.exists():
                    checkpoints.append(metadata)
                    
            except Exception:
                # Skip invalid metadata files
                continue
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return checkpoints
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get current processing progress report
        
        Returns:
            Dictionary with progress information
        """
        return {
            'pipeline': {
                'name': self.pipeline.name,
                'status': self.pipeline.status.value,
                'progress_percent': self.pipeline.get_progress(),
                'total_subjects': self.pipeline.total_subjects,
                'completed_subjects': self.pipeline.completed_subjects,
                'failed_subjects': self.pipeline.failed_subjects,
                'start_time': self.pipeline.start_time.isoformat() if self.pipeline.start_time else None,
                'current_duration': (
                    (datetime.now() - self.pipeline.start_time).total_seconds()
                    if self.pipeline.start_time else None
                )
            },
            'steps': [
                {
                    'name': step.name,
                    'stage': step.stage.value,
                    'status': step.status.value,
                    'duration': step.get_duration()
                }
                for step in self.pipeline.steps
            ],
            'processing_state': self.processing_state.copy() if hasattr(self, 'processing_state') else {}
        }
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the processor
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Updating processor configuration")
        
        # Update main configuration
        self.config.update(config)
        
        # Update step configurations
        for step in self.pipeline.steps:
            if step.name in config:
                step.config.update(config[step.name])
                self.logger.debug(f"Updated configuration for step {step.name}")
        
        # Recreate pipeline if processing type changed
        if 'processing_type' in config and config['processing_type'] != self.processing_type:
            self.processing_type = config['processing_type']
            self.pipeline = self._create_pipeline()
            self.logger.info(f"Recreated pipeline for processing type: {self.processing_type}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all processing steps
        
        Returns:
            Default configuration dictionary
        """
        defaults = {
            'processing_type': self.processing_type,
            'parallel': True,
            'save_checkpoints': True,
            'checkpoint_interval': 5,
            'cpus_per_task': 4,
            'mem_per_cpu': 4,
            'time_limit': "24:00:00"
        }
        
        # Get defaults for each step
        for step in self.pipeline.steps:
            try:
                processor_instance = self._get_processor_instance(step.processor_class)
                if hasattr(processor_instance, 'get_default_config'):
                    defaults[step.name] = processor_instance.get_default_config()
                else:
                    defaults[step.name] = {}
            except Exception as e:
                self.logger.warning(f"Could not get default config for step {step.name}: {e}")
                defaults[step.name] = {}
        
        return defaults
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Validate processing type
        valid_types = [
            "dti", "noddi", "csd", "msmt_csd", "fingerprinting", 
            "tracking", "connectivity", "full_pipeline"
        ]
        processing_type = config.get('processing_type', self.processing_type)
        if processing_type not in valid_types:
            errors.append(ValidationError(
                f"Invalid processing type: {processing_type}. Valid types: {valid_types}"
            ))
        
        # Validate scheduler configuration
        if 'parallel' in config and config['parallel']:
            if not self.scheduler:
                warnings.append(ValidationWarning(
                    "Parallel processing requested but no scheduler available"
                ))
                suggestions.append("Initialize scheduler for parallel processing")
        
        # Validate resource requirements
        if 'cpus_per_task' in config:
            cpus = config['cpus_per_task']
            if not isinstance(cpus, int) or cpus <= 0:
                errors.append(ValidationError("cpus_per_task must be a positive integer"))
        
        if 'mem_per_cpu' in config:
            mem = config['mem_per_cpu']
            if not isinstance(mem, (int, float)) or mem <= 0:
                errors.append(ValidationError("mem_per_cpu must be a positive number"))
        
        # Validate checkpoint settings
        if 'checkpoint_interval' in config:
            interval = config['checkpoint_interval']
            if not isinstance(interval, int) or interval <= 0:
                errors.append(ValidationError("checkpoint_interval must be a positive integer"))
        
        # Validate each step's configuration
        for step in self.pipeline.steps:
            if step.name in config:
                try:
                    processor_instance = self._get_processor_instance(step.processor_class)
                    if hasattr(processor_instance, 'validate_config'):
                        step_validation = processor_instance.validate_config(config[step.name])
                        if hasattr(step_validation, 'errors'):
                            errors.extend([
                                ValidationError(f"Step {step.name}: {error.message}")
                                for error in step_validation.errors
                            ])
                        elif not step_validation:  # Boolean return
                            errors.append(ValidationError(f"Invalid configuration for step {step.name}"))
                except Exception as e:
                    errors.append(ValidationError(f"Failed to validate config for step {step.name}: {e}"))
        
        return ValidationResult(len(errors) == 0, errors, warnings, suggestions)
    
    def cleanup_outputs(self, keep_checkpoints: bool = True, keep_logs: bool = True) -> None:
        """Clean up processing outputs
        
        Args:
            keep_checkpoints: Whether to keep checkpoint files
            keep_logs: Whether to keep log files
        """
        self.logger.info("Cleaning up processing outputs...")
        
        if self.output_dir.exists():
            # Clean up temporary files
            for temp_file in self.output_dir.rglob("*.tmp"):
                try:
                    temp_file.unlink()
                    self.logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
            
            # Clean up processing scripts
            scripts_dir = self.output_dir / "scripts"
            if scripts_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(scripts_dir)
                    self.logger.debug("Removed processing scripts directory")
                except Exception as e:
                    self.logger.warning(f"Failed to remove scripts directory: {e}")
            
            # Clean up checkpoints if requested
            if not keep_checkpoints and self.checkpoint_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(self.checkpoint_dir)
                    self.logger.debug("Removed checkpoints directory")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoints directory: {e}")
        
        self.logger.info("Cleanup completed")
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"ElikopyProcessor(type={self.processing_type}, "
            f"steps={len(self.pipeline.steps)}, "
            f"status={self.pipeline.status.value})"
        )
    
    def _estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining processing time in seconds
        
        Returns:
            Estimated remaining time in seconds or None if cannot estimate
        """
        if not self.pipeline.start_time or self.pipeline.completed_subjects == 0:
            return None
        
        # Calculate elapsed time
        elapsed_time = (datetime.now() - self.pipeline.start_time).total_seconds()
        
        # Calculate average time per subject
        avg_time_per_subject = elapsed_time / self.pipeline.completed_subjects
        
        # Calculate remaining subjects
        remaining_subjects = self.pipeline.total_subjects - self.pipeline.completed_subjects
        
        # Estimate remaining time
        estimated_remaining = avg_time_per_subject * remaining_subjects
        
        return estimated_remaining
    
    def _calculate_processing_rate(self) -> Optional[float]:
        """Calculate processing rate in subjects per hour
        
        Returns:
            Processing rate in subjects per hour or None if cannot calculate
        """
        if not self.pipeline.start_time or self.pipeline.completed_subjects == 0:
            return None
        
        # Calculate elapsed time in hours
        elapsed_time_hours = (datetime.now() - self.pipeline.start_time).total_seconds() / 3600
        
        # Calculate subjects per hour
        subjects_per_hour = self.pipeline.completed_subjects / elapsed_time_hours
        
        return subjects_per_hour
    
    def get_checkpoint_status(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Get status information from a checkpoint file
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint status information
        """
        checkpoint_file = Path(checkpoint_path)
        
        # Try to load JSON metadata first (faster)
        metadata_file = checkpoint_file.with_suffix('.json')
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Fall back to loading pickle file
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Extract status information
            pipeline_state = checkpoint_data.get('pipeline_state', {})
            processing_state = checkpoint_data.get('processing_state', {})
            
            return {
                'checkpoint_file': str(checkpoint_file),
                'checkpoint_name': checkpoint_data.get('checkpoint_metadata', {}).get('checkpoint_name', 'unknown'),
                'timestamp': checkpoint_data.get('checkpoint_metadata', {}).get('timestamp', 'unknown'),
                'processing_type': checkpoint_data.get('processor_config', {}).get('processing_type', 'unknown'),
                'total_subjects': pipeline_state.get('total_subjects', 0),
                'completed_subjects': pipeline_state.get('completed_subjects', 0),
                'failed_subjects': pipeline_state.get('failed_subjects', 0),
                'progress_percent': (
                    (pipeline_state.get('completed_subjects', 0) / pipeline_state.get('total_subjects', 1)) * 100
                    if pipeline_state.get('total_subjects', 0) > 0 else 0
                ),
                'status': pipeline_state.get('status', 'unknown')
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to read checkpoint status: {e}")
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Delete a checkpoint file and its metadata
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successfully deleted, False otherwise
        """
        checkpoint_file = Path(checkpoint_path)
        metadata_file = checkpoint_file.with_suffix('.json')
        
        success = True
        
        # Delete checkpoint file
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                self.logger.info(f"Deleted checkpoint file: {checkpoint_file}")
            except Exception as e:
                self.logger.error(f"Failed to delete checkpoint file {checkpoint_file}: {e}")
                success = False
        
        # Delete metadata file
        if metadata_file.exists():
            try:
                metadata_file.unlink()
                self.logger.debug(f"Deleted checkpoint metadata: {metadata_file}")
            except Exception as e:
                self.logger.warning(f"Failed to delete checkpoint metadata {metadata_file}: {e}")
                # Don't mark as failure since main file was deleted
        
        return success
    
    def cleanup_old_checkpoints(self, keep_count: int = 5) -> int:
        """Clean up old checkpoint files, keeping only the most recent ones
        
        Args:
            keep_count: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        if not self.checkpoint_dir.exists():
            return 0
        
        # Get all checkpoints sorted by timestamp (newest first)
        checkpoints = self.list_checkpoints(self.checkpoint_dir)
        
        if len(checkpoints) <= keep_count:
            return 0
        
        # Delete old checkpoints
        deleted_count = 0
        for checkpoint in checkpoints[keep_count:]:
            try:
                checkpoint_path = Path(checkpoint['checkpoint_file'])
                if self.delete_checkpoint(checkpoint_path):
                    deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to delete old checkpoint: {e}")
        
        self.logger.info(f"Cleaned up {deleted_count} old checkpoints, kept {keep_count} most recent")
        return deleted_count
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"ElikopyProcessor("
            f"processing_type='{self.processing_type}', "
            f"pipeline_steps={[step.name for step in self.pipeline.steps]}, "
            f"status='{self.pipeline.status.value}', "
            f"scheduler={'available' if self.scheduler else 'none'}, "
            f"output_dir='{self.output_dir}'"
            f")"
        )