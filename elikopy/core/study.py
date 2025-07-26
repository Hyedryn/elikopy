"""
ElikopyStudy class - Main entry point for study management
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from elikopy.core.base import ConfigurableComponent, DWIFile
from elikopy.core.config import ElikopyConfig
from elikopy.data.bids_handler import BIDSHandler, AnatomicalFile
from elikopy.infrastructure.logging import get_logger


class Subject:
    """Class representing a subject in the study"""
    
    def __init__(self, subject_id: str, sessions: Optional[List[str]] = None):
        """Initialize a subject
        
        Args:
            subject_id: Subject identifier
            sessions: List of session identifiers
        """
        self.id = subject_id
        self.sessions = sessions or [""]
        self.metadata: Dict[str, Any] = {}
        self.dwi_files: List[DWIFile] = []
        self.anatomical_files: List[AnatomicalFile] = []
        
    def add_dwi_file(self, dwi_file: DWIFile) -> None:
        """Add a DWI file to this subject
        
        Args:
            dwi_file: DWI file to add
        """
        self.dwi_files.append(dwi_file)
    
    def add_anatomical_file(self, anat_file: AnatomicalFile) -> None:
        """Add an anatomical file to this subject
        
        Args:
            anat_file: Anatomical file to add
        """
        self.anatomical_files.append(anat_file)
    
    def get_dwi_files(self, session: Optional[str] = None) -> List[DWIFile]:
        """Get DWI files for this subject, optionally filtered by session
        
        Args:
            session: Session to filter by (optional)
            
        Returns:
            List of DWI files
        """
        if session is None:
            return self.dwi_files
        
        # Filter by session - this would need to be implemented based on file metadata
        # For now, return all files
        return self.dwi_files
    
    def get_anatomical_files(self, session: Optional[str] = None) -> List[AnatomicalFile]:
        """Get anatomical files for this subject, optionally filtered by session
        
        Args:
            session: Session to filter by (optional)
            
        Returns:
            List of anatomical files
        """
        if session is None:
            return self.anatomical_files
        
        # Filter by session - this would need to be implemented based on file metadata
        # For now, return all files
        return self.anatomical_files
        
    def __repr__(self) -> str:
        """String representation"""
        return f"Subject(id={self.id}, sessions={self.sessions}, " \
               f"dwi_files={len(self.dwi_files)}, anat_files={len(self.anatomical_files)})"


class ElikopyStudy(ConfigurableComponent):
    """Main class for managing an ElikoPy study with qsiprep derivatives support"""
    
    def __init__(
        self, 
        study_path: Union[str, Path],
        bids_root: Optional[Union[str, Path]] = None,
        derivatives_name: str = "elikopy",
        config: Optional[ElikopyConfig] = None
    ):
        """Initialize an ElikoPy study
        
        Args:
            study_path: Path to the study directory
            bids_root: Path to the BIDS root directory (optional)
            derivatives_name: Name of the derivatives directory
            config: Study configuration
        """
        self.study_path = Path(study_path)
        self.bids_root = Path(bids_root) if bids_root else None
        self.derivatives_name = derivatives_name
        self.config = config or ElikopyConfig()
        self.subjects: Dict[str, Subject] = {}
        self.bids_handler: Optional[BIDSHandler] = None
        self.qsiprep_dir: Optional[Path] = None
        self.derivatives_dir: Optional[Path] = None
        self.logger = get_logger(__name__).get_logger()
        
        # Create study directory if it doesn't exist
        self.study_path.mkdir(exist_ok=True, parents=True)
        
        # Set up logging for this study
        self.logger.info(f"Initialized ElikopyStudy at {self.study_path}")
        if self.bids_root:
            self.logger.info(f"BIDS root set to {self.bids_root}")
        
    def setup_from_qsiprep(self, qsiprep_dir: Union[str, Path]) -> None:
        """Setup study from qsiprep derivatives
        
        Args:
            qsiprep_dir: Path to qsiprep derivatives directory
            
        Raises:
            FileNotFoundError: If qsiprep directory doesn't exist
            ValueError: If qsiprep structure is invalid
        """
        qsiprep_path = Path(qsiprep_dir)
        if not qsiprep_path.exists():
            raise FileNotFoundError(f"QSIPrep directory not found: {qsiprep_path}")
        
        self.qsiprep_dir = qsiprep_path
        self.logger.info(f"Setting up study from QSIPrep directory: {qsiprep_path}")
        
        # Determine BIDS root if not provided
        if not self.bids_root:
            # Assume qsiprep is in derivatives/qsiprep under BIDS root
            potential_bids_root = qsiprep_path.parent.parent
            if (potential_bids_root / "dataset_description.json").exists():
                self.bids_root = potential_bids_root
                self.logger.info(f"Inferred BIDS root: {self.bids_root}")
            else:
                # Use qsiprep parent as BIDS root
                self.bids_root = qsiprep_path.parent
                self.logger.warning(f"Could not find BIDS root, using: {self.bids_root}")
        
        # Initialize BIDS handler
        try:
            self.bids_handler = BIDSHandler(
                bids_root=self.bids_root,
                qsiprep_dir=qsiprep_path
            )
            self.logger.info("BIDS handler initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize BIDS handler: {e}")
            raise ValueError(f"Failed to initialize BIDS handler: {e}") from e
        
        # Validate qsiprep structure
        validation_result = self.bids_handler.validate_qsiprep_structure()
        if not validation_result.is_valid:
            error_msg = f"Invalid QSIPrep structure: {validation_result.errors}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(f"QSIPrep validation warning: {warning}")
        
        # Create derivatives directory structure
        self.derivatives_dir = self.bids_handler.create_derivatives_structure(self.derivatives_name)
        self.logger.info(f"Created derivatives directory: {self.derivatives_dir}")
        
        # Discover subjects and their data
        self._discover_subjects()
        self.logger.info(f"Discovered {len(self.subjects)} subjects")
        
    def _discover_subjects(self) -> None:
        """Discover subjects from BIDS dataset and load their data"""
        if not self.bids_handler:
            raise RuntimeError("BIDS handler not initialized. Call setup_from_qsiprep first.")
        
        self.logger.info("Discovering subjects from BIDS dataset...")
        
        # Get subjects from BIDS layout
        subjects_info = self.bids_handler.get_subjects_info()
        
        # Create Subject objects and load their data
        for subject_id, sessions in subjects_info.items():
            self.logger.debug(f"Processing subject {subject_id} with sessions {sessions}")
            
            subject = Subject(subject_id, sessions)
            
            # Load DWI and anatomical files for each session
            for session in sessions:
                session_id = session if session else None
                
                try:
                    # Get DWI files
                    dwi_files = self.bids_handler.get_preprocessed_dwi_files(
                        subject=subject_id, 
                        session=session_id
                    )
                    for dwi_file in dwi_files:
                        subject.add_dwi_file(dwi_file)
                    
                    # Get anatomical files
                    anat_files = self.bids_handler.get_preprocessed_anatomical_files(
                        subject=subject_id,
                        session=session_id
                    )
                    for anat_file in anat_files:
                        subject.add_anatomical_file(anat_file)
                    
                    self.logger.debug(f"Subject {subject_id}, session {session}: "
                                    f"{len(dwi_files)} DWI files, {len(anat_files)} anatomical files")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load data for subject {subject_id}, "
                                      f"session {session}: {e}")
            
            # Store subject metadata
            subject.metadata.update({
                'qsiprep_dir': str(self.qsiprep_dir),
                'bids_root': str(self.bids_root),
                'discovery_timestamp': str(Path().cwd()),  # Could use actual timestamp
                'total_dwi_files': len(subject.dwi_files),
                'total_anatomical_files': len(subject.anatomical_files)
            })
            
            self.subjects[subject_id] = subject
            
        self.logger.info(f"Successfully discovered {len(self.subjects)} subjects")
    
    def get_subjects(self) -> List[Subject]:
        """Get list of available subjects
        
        Returns:
            List of Subject objects
        """
        return list(self.subjects.values())
    
    def get_subject(self, subject_id: str) -> Optional[Subject]:
        """Get a specific subject
        
        Args:
            subject_id: Subject identifier (with or without 'sub-' prefix)
            
        Returns:
            Subject object or None if not found
        """
        # Normalize subject ID
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'
        
        return self.subjects.get(subject_id)
    
    def get_subject_ids(self) -> List[str]:
        """Get list of subject IDs
        
        Returns:
            List of subject ID strings
        """
        return list(self.subjects.keys())
    
    def has_subject(self, subject_id: str) -> bool:
        """Check if a subject exists in the study
        
        Args:
            subject_id: Subject identifier (with or without 'sub-' prefix)
            
        Returns:
            True if subject exists, False otherwise
        """
        return self.get_subject(subject_id) is not None
    
    def get_study_summary(self) -> Dict[str, Any]:
        """Get a summary of the study
        
        Returns:
            Dictionary with study summary information
        """
        total_dwi_files = sum(len(subject.dwi_files) for subject in self.subjects.values())
        total_anat_files = sum(len(subject.anatomical_files) for subject in self.subjects.values())
        
        sessions_per_subject = {}
        for subject_id, subject in self.subjects.items():
            sessions_per_subject[subject_id] = len(subject.sessions)
        
        return {
            'study_path': str(self.study_path),
            'bids_root': str(self.bids_root) if self.bids_root else None,
            'qsiprep_dir': str(self.qsiprep_dir) if self.qsiprep_dir else None,
            'derivatives_name': self.derivatives_name,
            'derivatives_dir': str(self.derivatives_dir) if self.derivatives_dir else None,
            'total_subjects': len(self.subjects),
            'total_dwi_files': total_dwi_files,
            'total_anatomical_files': total_anat_files,
            'sessions_per_subject': sessions_per_subject,
            'config_summary': self.config.get_summary() if self.config else None
        }
    
    def create_processor(self, processing_type: str, **kwargs) -> "ElikopyProcessor":
        """Create processor for specific analysis type
        
        Args:
            processing_type: Type of processing to perform
            **kwargs: Additional arguments for processor
            
        Returns:
            ElikopyProcessor object
        """
        # Import here to avoid circular imports
        from elikopy.core.processor import ElikopyProcessor
        
        return ElikopyProcessor(
            study=self,
            processing_type=processing_type,
            **kwargs
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the study
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return self.config.get_defaults()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        # Create a temporary config to validate
        temp_config = ElikopyConfig()
        temp_config.update(config)
        return temp_config.is_valid()