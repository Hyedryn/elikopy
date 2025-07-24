"""
ElikopyStudy class - Main entry point for study management
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from elikopy.core.base import ConfigurableComponent
from elikopy.core.config import ElikopyConfig
from elikopy.data.bids_handler import BIDSHandler


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
        
    def __repr__(self) -> str:
        """String representation"""
        return f"Subject(id={self.id}, sessions={self.sessions})"


class ElikopyStudy(ConfigurableComponent):
    """Main class for managing an ElikoPy study"""
    
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
        self.bids_handler = None
        
        # Create study directory if it doesn't exist
        self.study_path.mkdir(exist_ok=True, parents=True)
        
    def setup_from_qsiprep(self, qsiprep_dir: Union[str, Path]) -> None:
        """Setup study from qsiprep derivatives
        
        Args:
            qsiprep_dir: Path to qsiprep derivatives directory
        """
        qsiprep_path = Path(qsiprep_dir)
        if not qsiprep_path.exists():
            raise FileNotFoundError(f"QSIPrep directory not found: {qsiprep_path}")
        
        # Initialize BIDS handler
        self.bids_handler = BIDSHandler(
            bids_root=self.bids_root or qsiprep_path.parent.parent,
            qsiprep_dir=qsiprep_path
        )
        
        # Discover subjects
        self._discover_subjects()
        
    def _discover_subjects(self) -> None:
        """Discover subjects from BIDS dataset"""
        if not self.bids_handler:
            raise RuntimeError("BIDS handler not initialized. Call setup_from_qsiprep first.")
        
        # Get subjects from BIDS layout
        subjects_info = self.bids_handler.get_subjects_info()
        
        # Create Subject objects
        for subject_id, sessions in subjects_info.items():
            self.subjects[subject_id] = Subject(subject_id, sessions)
    
    def get_subjects(self) -> List[Subject]:
        """Get list of available subjects
        
        Returns:
            List of Subject objects
        """
        return list(self.subjects.values())
    
    def get_subject(self, subject_id: str) -> Optional[Subject]:
        """Get a specific subject
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Subject object or None if not found
        """
        return self.subjects.get(subject_id)
    
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
        return self.config.validate(config)