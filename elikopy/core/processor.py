"""
ElikopyProcessor class - Processing orchestration
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from elikopy.core.base import ConfigurableComponent, ProcessingResult, ProcessingStatus
from elikopy.core.study import ElikopyStudy
from elikopy.infrastructure.scheduler import JobScheduler


class ProcessingStep:
    """Class representing a processing step"""
    
    def __init__(self, name: str, processor: Any, config: Dict[str, Any]):
        """Initialize a processing step
        
        Args:
            name: Step name
            processor: Processor object
            config: Step configuration
        """
        self.name = name
        self.processor = processor
        self.config = config
        self.status = ProcessingStatus.NOT_STARTED
        self.result: Optional[ProcessingResult] = None
        
    def execute(self, **kwargs) -> ProcessingResult:
        """Execute the processing step
        
        Args:
            **kwargs: Additional arguments for processor
            
        Returns:
            ProcessingResult object
        """
        self.status = ProcessingStatus.IN_PROGRESS
        try:
            self.result = self.processor.process(**{**self.config, **kwargs})
            self.status = self.result.status
            return self.result
        except Exception as e:
            self.status = ProcessingStatus.FAILED
            self.result = ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
            raise


class ElikopyProcessor(ConfigurableComponent):
    """Class for processing orchestration"""
    
    def __init__(
        self, 
        study: ElikopyStudy,
        processing_type: str,
        scheduler: Optional[JobScheduler] = None,
        **kwargs
    ):
        """Initialize processor
        
        Args:
            study: ElikopyStudy object
            processing_type: Type of processing to perform
            scheduler: JobScheduler object (optional)
            **kwargs: Additional configuration
        """
        self.study = study
        self.processing_type = processing_type
        self.scheduler = scheduler
        self.config = kwargs
        self.steps: List[ProcessingStep] = []
        
        # Initialize processor based on type
        self._initialize_processor()
        
    def _initialize_processor(self) -> None:
        """Initialize processor based on type"""
        # Import processors here to avoid circular imports
        from elikopy.processing.dti import DTIProcessor
        from elikopy.processing.noddi import NODDIProcessor
        from elikopy.processing.csd import CSDProcessor
        from elikopy.processing.tracking import TrackingProcessor
        
        # Create processing steps based on type
        if self.processing_type == "dti":
            self.steps.append(ProcessingStep(
                name="dti_fitting",
                processor=DTIProcessor(self.config.get("dti", {})),
                config=self.config.get("dti", {})
            ))
        elif self.processing_type == "noddi":
            self.steps.append(ProcessingStep(
                name="noddi_fitting",
                processor=NODDIProcessor(self.config.get("noddi", {})),
                config=self.config.get("noddi", {})
            ))
        elif self.processing_type == "csd":
            self.steps.append(ProcessingStep(
                name="csd_fitting",
                processor=CSDProcessor(self.config.get("csd", {})),
                config=self.config.get("csd", {})
            ))
        elif self.processing_type == "tracking":
            # Add CSD step if not already done
            self.steps.append(ProcessingStep(
                name="csd_fitting",
                processor=CSDProcessor(self.config.get("csd", {})),
                config=self.config.get("csd", {})
            ))
            # Add tracking step
            self.steps.append(ProcessingStep(
                name="tracking",
                processor=TrackingProcessor(self.config.get("tracking", {})),
                config=self.config.get("tracking", {})
            ))
        else:
            raise ValueError(f"Unknown processing type: {self.processing_type}")
    
    def validate_inputs(self) -> bool:
        """Validate inputs before processing
        
        Returns:
            True if inputs are valid
        """
        # Validate each step
        for step in self.steps:
            if not step.processor.validate_inputs():
                return False
        return True
    
    def run(
        self, 
        subjects: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, List[ProcessingResult]]:
        """Execute processing pipeline
        
        Args:
            subjects: List of subject IDs to process (None for all)
            parallel: Whether to run in parallel
            
        Returns:
            Dictionary of processing results by subject
        """
        # Get subjects to process
        subject_list = subjects or [s.id for s in self.study.get_subjects()]
        
        # Validate inputs
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        results: Dict[str, List[ProcessingResult]] = {}
        
        # Process each subject
        if parallel and self.scheduler:
            # Submit jobs for parallel processing
            job_ids = self._submit_parallel_jobs(subject_list)
            # Wait for jobs to complete
            self.scheduler.wait_for_jobs(job_ids)
            # Collect results
            for subject_id in subject_list:
                results[subject_id] = self._collect_results(subject_id)
        else:
            # Process sequentially
            for subject_id in subject_list:
                results[subject_id] = self._process_subject(subject_id)
        
        return results
    
    def _process_subject(self, subject_id: str) -> List[ProcessingResult]:
        """Process a single subject
        
        Args:
            subject_id: Subject ID
            
        Returns:
            List of processing results
        """
        results = []
        
        # Execute each step
        for step in self.steps:
            result = step.execute(subject_id=subject_id)
            results.append(result)
            
            # Stop if step failed
            if result.status == ProcessingStatus.FAILED:
                break
        
        return results
    
    def _submit_parallel_jobs(self, subject_list: List[str]) -> List[str]:
        """Submit jobs for parallel processing
        
        Args:
            subject_list: List of subject IDs
            
        Returns:
            List of job IDs
        """
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        job_ids = []
        
        # Submit job for each subject
        for subject_id in subject_list:
            job_id = self.scheduler.submit_job({
                "type": self.processing_type,
                "subject_id": subject_id,
                "config": self.config
            })
            job_ids.append(job_id)
        
        return job_ids
    
    def _collect_results(self, subject_id: str) -> List[ProcessingResult]:
        """Collect results for a subject
        
        Args:
            subject_id: Subject ID
            
        Returns:
            List of processing results
        """
        results = []
        
        # Collect results for each step
        for step in self.steps:
            # Check if result file exists
            result_file = self.study.study_path / "results" / subject_id / f"{step.name}_result.json"
            if result_file.exists():
                # Load result from file
                with open(result_file, "r") as f:
                    result_data = f.read()
                
                # Parse result
                import json
                result_dict = json.loads(result_data)
                
                # Create ProcessingResult object
                result = ProcessingResult(
                    status=ProcessingStatus(result_dict["status"]),
                    output_files=[Path(p) for p in result_dict["output_files"]],
                    metadata=result_dict["metadata"],
                    error_message=result_dict.get("error_message")
                )
                
                results.append(result)
            else:
                # Result not found
                results.append(ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message=f"Result file not found: {result_file}"
                ))
        
        return results
    
    def resume(self, checkpoint_path: Union[str, Path]) -> Dict[str, List[ProcessingResult]]:
        """Resume processing from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary of processing results by subject
        """
        # Load checkpoint
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
        
        # Parse checkpoint
        import json
        with open(checkpoint, "r") as f:
            checkpoint_data = json.load(f)
        
        # Get subjects to process
        subject_list = checkpoint_data.get("remaining_subjects", [])
        
        # Resume processing
        return self.run(subjects=subject_list)
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the processor
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        
        # Update step configurations
        for step in self.steps:
            if step.name in config:
                step.config.update(config[step.name])
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        defaults = {}
        
        # Get defaults for each step
        for step in self.steps:
            defaults[step.name] = step.processor.get_default_config()
        
        return defaults
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        # Validate each step's configuration
        for step in self.steps:
            if step.name in config:
                if not step.processor.validate_config(config[step.name]):
                    return False
        
        return True