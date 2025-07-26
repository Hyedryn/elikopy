"""
QSIPrep adapter for ElikoPy
==========================

This module provides an adapter for loading and working with qsiprep preprocessed data.
It handles extraction of preprocessed DWI and anatomical data, metadata extraction,
and validation of qsiprep outputs.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
import nibabel as nib

from elikopy.core.base import ProcessingComponent, ProcessingResult, ProcessingStatus, ValidationResult, ValidationError, ValidationWarning, DWIFile
from elikopy.data.bids_handler import BIDSHandler, AnatomicalFile
from elikopy.infrastructure.exceptions import ElikopyError, DataValidationError


logger = logging.getLogger(__name__)


@dataclass
class DWIData:
    """Container for DWI data and associated information"""
    dwi_image: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    affine: np.ndarray
    header: Any
    metadata: Dict[str, Any]
    mask: Optional[np.ndarray] = None
    
    @property
    def shape(self) -> tuple:
        """Get shape of DWI data"""
        return self.dwi_image.shape
    
    @property
    def n_volumes(self) -> int:
        """Get number of DWI volumes"""
        return self.dwi_image.shape[-1] if len(self.dwi_image.shape) == 4 else 1
    
    @property
    def n_bvals(self) -> int:
        """Get number of b-values"""
        return len(np.unique(self.bvals))


@dataclass
class AnatomicalData:
    """Container for anatomical data and associated information"""
    t1w_image: np.ndarray
    affine: np.ndarray
    header: Any
    metadata: Dict[str, Any]
    brain_mask: Optional[np.ndarray] = None
    tissue_segmentation: Optional[np.ndarray] = None
    
    @property
    def shape(self) -> tuple:
        """Get shape of anatomical data"""
        return self.t1w_image.shape


class QsiPrepAdapter(ProcessingComponent):
    """
    Adapter for loading and working with qsiprep preprocessed data.
    
    This class provides methods to:
    - Load preprocessed DWI and anatomical data from qsiprep outputs
    - Extract and validate metadata
    - Validate qsiprep output completeness and quality
    """
    
    def __init__(self, qsiprep_dir: Path, bids_handler: Optional[BIDSHandler] = None):
        """
        Initialize qsiprep adapter.
        
        Parameters
        ----------
        qsiprep_dir : Path
            Path to qsiprep derivatives directory
        bids_handler : BIDSHandler, optional
            BIDS handler instance, if None a new one will be created
            
        Raises
        ------
        DataValidationError
            If qsiprep directory structure is invalid
        """
        self.qsiprep_dir = Path(qsiprep_dir)
        
        # Initialize BIDS handler
        if bids_handler is None:
            try:
                self.bids_handler = BIDSHandler(bids_root=self.qsiprep_dir.parent, 
                                              qsiprep_dir=self.qsiprep_dir)
            except Exception as e:
                raise DataValidationError(f"Failed to initialize BIDS handler: {e}")
        else:
            self.bids_handler = bids_handler
        
        # Validate qsiprep structure
        validation_result = self.validate_inputs()
        if not validation_result:
            raise DataValidationError("Invalid qsiprep directory structure")
        
        logger.info(f"Initialized QsiPrepAdapter for directory: {self.qsiprep_dir}")
    
    def validate_inputs(self) -> bool:
        """
        Validate qsiprep directory structure.
        
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        try:
            validation_result = self.bids_handler.validate_qsiprep_structure()
            return validation_result.is_valid
        except Exception as e:
            logger.error(f"Failed to validate qsiprep structure: {e}")
            return False
    
    def process(self, **kwargs) -> ProcessingResult:
        """
        Process qsiprep outputs (placeholder for base class compatibility).
        
        Parameters
        ----------
        **kwargs
            Additional arguments
            
        Returns
        -------
        ProcessingResult
            Processing result
        """
        return ProcessingResult(
            status=ProcessingStatus.COMPLETED,
            output_files=[],
            metadata={"adapter": "qsiprep", "directory": str(self.qsiprep_dir)}
        )
    
    def load_preprocessed_data(self, subject: str, session: Optional[str] = None, 
                             run: Optional[str] = None, task: Optional[str] = None,
                             **filters) -> DWIData:
        """
        Load preprocessed DWI data from qsiprep outputs.
        
        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str, optional
            Session ID (with or without 'ses-' prefix)
        run : str, optional
            Run ID (with or without 'run-' prefix)
        task : str, optional
            Task ID (with or without 'task-' prefix)
        **filters
            Additional BIDS entity filters
            
        Returns
        -------
        DWIData
            Container with preprocessed DWI data and metadata
            
        Raises
        ------
        DataValidationError
            If no DWI files found or data loading fails
        """
        logger.info(f"Loading preprocessed DWI data for subject: {subject}")
        
        try:
            # Get DWI files using BIDS handler
            dwi_files = self.bids_handler.get_preprocessed_dwi_files(
                subject=subject, session=session, run=run, task=task, **filters
            )
            
            if not dwi_files:
                raise DataValidationError(f"No preprocessed DWI files found for subject {subject}")
            
            # Use the first DWI file if multiple found
            if len(dwi_files) > 1:
                logger.warning(f"Multiple DWI files found for subject {subject}, using first one")
            
            dwi_file = dwi_files[0]
            
            # Load DWI image
            dwi_img = nib.load(dwi_file.path)
            dwi_data = dwi_img.get_fdata()
            
            # Load bvals and bvecs
            bvals = np.loadtxt(dwi_file.bval_path)
            bvecs = np.loadtxt(dwi_file.bvec_path)
            
            # Ensure bvecs is 3xN
            if bvecs.shape[0] != 3:
                bvecs = bvecs.T
            
            # Load metadata from JSON sidecar
            metadata = {}
            if dwi_file.json_path and dwi_file.json_path.exists():
                try:
                    with open(dwi_file.json_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {dwi_file.json_path}: {e}")
            
            # Try to load brain mask if available
            mask = self._load_brain_mask(subject, session, run, task, **filters)
            
            # Create DWIData container
            dwi_data_container = DWIData(
                dwi_image=dwi_data,
                bvals=bvals,
                bvecs=bvecs,
                affine=dwi_img.affine,
                header=dwi_img.header,
                metadata=metadata,
                mask=mask
            )
            
            logger.info(f"Successfully loaded DWI data: shape={dwi_data_container.shape}, "
                       f"n_volumes={dwi_data_container.n_volumes}, "
                       f"n_bvals={dwi_data_container.n_bvals}")
            
            return dwi_data_container
            
        except Exception as e:
            raise DataValidationError(f"Failed to load preprocessed DWI data for subject {subject}: {e}")
    
    def load_anatomical_data(self, subject: str, session: Optional[str] = None,
                           **filters) -> AnatomicalData:
        """
        Load preprocessed anatomical data from qsiprep outputs.
        
        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str, optional
            Session ID (with or without 'ses-' prefix)
        **filters
            Additional BIDS entity filters
            
        Returns
        -------
        AnatomicalData
            Container with preprocessed anatomical data and metadata
            
        Raises
        ------
        DataValidationError
            If no anatomical files found or data loading fails
        """
        logger.info(f"Loading preprocessed anatomical data for subject: {subject}")
        
        try:
            # Get anatomical files using BIDS handler
            anat_files = self.bids_handler.get_preprocessed_anatomical_files(
                subject=subject, session=session, **filters
            )
            
            if not anat_files:
                raise DataValidationError(f"No preprocessed anatomical files found for subject {subject}")
            
            # Use the first anatomical file if multiple found
            if len(anat_files) > 1:
                logger.warning(f"Multiple anatomical files found for subject {subject}, using first one")
            
            anat_file = anat_files[0]
            
            # Load T1w image
            t1w_img = nib.load(anat_file.path)
            t1w_data = t1w_img.get_fdata()
            
            # Load metadata from JSON sidecar
            metadata = {}
            if anat_file.json_path and anat_file.json_path.exists():
                try:
                    with open(anat_file.json_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {anat_file.json_path}: {e}")
            
            # Try to load brain mask and tissue segmentation if available
            brain_mask = self._load_anatomical_brain_mask(subject, session, **filters)
            tissue_seg = self._load_tissue_segmentation(subject, session, **filters)
            
            # Create AnatomicalData container
            anat_data_container = AnatomicalData(
                t1w_image=t1w_data,
                affine=t1w_img.affine,
                header=t1w_img.header,
                metadata=metadata,
                brain_mask=brain_mask,
                tissue_segmentation=tissue_seg
            )
            
            logger.info(f"Successfully loaded anatomical data: shape={anat_data_container.shape}")
            
            return anat_data_container
            
        except Exception as e:
            raise DataValidationError(f"Failed to load preprocessed anatomical data for subject {subject}: {e}")
    
    def get_preprocessing_metadata(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """
        Get preprocessing metadata from qsiprep outputs.
        
        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str, optional
            Session ID (with or without 'ses-' prefix)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing preprocessing metadata
        """
        logger.info(f"Getting preprocessing metadata for subject: {subject}")
        
        metadata = {
            'subject': subject,
            'session': session,
            'qsiprep_version': None,
            'preprocessing_steps': [],
            'software_versions': {},
            'acquisition_parameters': {}
        }
        
        try:
            # Get dataset description
            dataset_desc_path = self.qsiprep_dir / "dataset_description.json"
            if dataset_desc_path.exists():
                with open(dataset_desc_path, 'r') as f:
                    dataset_desc = json.load(f)
                    metadata['qsiprep_version'] = dataset_desc.get('GeneratedBy', [{}])[0].get('Version')
                    metadata['software_versions'] = dataset_desc.get('GeneratedBy', [{}])[0]
            
            # Get DWI metadata
            try:
                dwi_files = self.bids_handler.get_preprocessed_dwi_files(subject=subject, session=session)
                if dwi_files and dwi_files[0].json_path:
                    with open(dwi_files[0].json_path, 'r') as f:
                        dwi_metadata = json.load(f)
                        metadata['acquisition_parameters'] = dwi_metadata
                        
                        # Extract preprocessing steps from metadata
                        if 'ProcessingSteps' in dwi_metadata:
                            metadata['preprocessing_steps'] = dwi_metadata['ProcessingSteps']
            except Exception as e:
                logger.warning(f"Failed to load DWI metadata: {e}")
            
            # Get anatomical metadata
            try:
                anat_files = self.bids_handler.get_preprocessed_anatomical_files(subject=subject, session=session)
                if anat_files and anat_files[0].json_path:
                    with open(anat_files[0].json_path, 'r') as f:
                        anat_metadata = json.load(f)
                        metadata['anatomical_parameters'] = anat_metadata
            except Exception as e:
                logger.warning(f"Failed to load anatomical metadata: {e}")
            
        except Exception as e:
            logger.error(f"Failed to get preprocessing metadata for subject {subject}: {e}")
        
        return metadata
    
    def validate_qsiprep_outputs(self, subject: str, session: Optional[str] = None) -> ValidationResult:
        """
        Validate qsiprep outputs for completeness and quality.
        
        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str, optional
            Session ID (with or without 'ses-' prefix)
            
        Returns
        -------
        ValidationResult
            Validation result with errors, warnings, and suggestions
        """
        logger.info(f"Validating qsiprep outputs for subject: {subject}")
        
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Check for DWI files
            dwi_files = self.bids_handler.get_preprocessed_dwi_files(subject=subject, session=session)
            if not dwi_files:
                errors.append(ValidationError(f"No preprocessed DWI files found for subject {subject}"))
            else:
                # Validate each DWI file
                for dwi_file in dwi_files:
                    # Check file existence
                    if not dwi_file.path.exists():
                        errors.append(ValidationError(f"DWI file not found: {dwi_file.path}"))
                    if not dwi_file.bval_path.exists():
                        errors.append(ValidationError(f"bval file not found: {dwi_file.bval_path}"))
                    if not dwi_file.bvec_path.exists():
                        errors.append(ValidationError(f"bvec file not found: {dwi_file.bvec_path}"))
                    
                    # Check JSON sidecar
                    if not dwi_file.json_path or not dwi_file.json_path.exists():
                        warnings.append(ValidationWarning(f"JSON sidecar not found: {dwi_file.json_path}"))
                    
                    # Validate data integrity
                    try:
                        # Load and check DWI data
                        dwi_img = nib.load(dwi_file.path)
                        dwi_shape = dwi_img.shape
                        
                        # Load bvals/bvecs
                        bvals = np.loadtxt(dwi_file.bval_path)
                        bvecs = np.loadtxt(dwi_file.bvec_path)
                        
                        # Check dimensions consistency
                        if len(dwi_shape) == 4:
                            n_volumes = dwi_shape[3]
                        else:
                            n_volumes = 1
                        
                        if len(bvals) != n_volumes:
                            errors.append(ValidationError(f"Mismatch between DWI volumes ({n_volumes}) and bvals ({len(bvals)})"))
                        
                        if bvecs.shape[1] != n_volumes and bvecs.shape[0] != n_volumes:
                            errors.append(ValidationError(f"Mismatch between DWI volumes ({n_volumes}) and bvecs shape ({bvecs.shape})"))
                        
                        # Check for reasonable b-values
                        unique_bvals = np.unique(bvals)
                        if len(unique_bvals) < 2:
                            warnings.append(ValidationWarning("Only one unique b-value found, DTI fitting may be limited"))
                        
                        if np.max(bvals) > 10000:
                            warnings.append(ValidationWarning(f"Very high b-values detected (max: {np.max(bvals)}), check units"))
                        
                        # Check for b=0 images
                        b0_count = np.sum(bvals < 100)
                        if b0_count == 0:
                            errors.append(ValidationError("No b=0 images found"))
                        elif b0_count < 3:
                            warnings.append(ValidationWarning(f"Only {b0_count} b=0 images found, consider having more for better preprocessing"))
                        
                    except Exception as e:
                        errors.append(ValidationError(f"Failed to validate DWI data integrity: {e}"))
            
            # Check for anatomical files
            anat_files = self.bids_handler.get_preprocessed_anatomical_files(subject=subject, session=session)
            if not anat_files:
                warnings.append(ValidationWarning(f"No preprocessed anatomical files found for subject {subject}"))
                suggestions.append("Anatomical data is recommended for registration and tissue segmentation")
            else:
                # Validate anatomical files
                for anat_file in anat_files:
                    if not anat_file.path.exists():
                        errors.append(ValidationError(f"Anatomical file not found: {anat_file.path}"))
                    
                    # Check JSON sidecar
                    if not anat_file.json_path or not anat_file.json_path.exists():
                        warnings.append(ValidationWarning(f"Anatomical JSON sidecar not found: {anat_file.json_path}"))
            
            # Check for brain masks
            mask_found = self._check_brain_mask_exists(subject, session)
            if not mask_found:
                warnings.append(ValidationWarning("No brain mask found, processing may be less accurate"))
                suggestions.append("Consider generating brain masks for improved processing")
            
            # Check preprocessing metadata
            try:
                metadata = self.get_preprocessing_metadata(subject, session)
                if not metadata.get('qsiprep_version'):
                    warnings.append(ValidationWarning("QSIPrep version information not found"))
                
                if not metadata.get('preprocessing_steps'):
                    warnings.append(ValidationWarning("Preprocessing steps information not found"))
                    
            except Exception as e:
                warnings.append(ValidationWarning(f"Failed to validate preprocessing metadata: {e}"))
            
        except Exception as e:
            errors.append(ValidationError(f"Validation failed: {e}"))
        
        is_valid = len(errors) == 0
        
        logger.info(f"Validation completed for subject {subject}: "
                   f"valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _load_brain_mask(self, subject: str, session: Optional[str] = None,
                        run: Optional[str] = None, task: Optional[str] = None,
                        **filters) -> Optional[np.ndarray]:
        """Load brain mask for DWI data if available"""
        try:
            # Try to find brain mask using pybids
            if self.bids_handler._qsiprep_layout:
                subject_id = subject.replace("sub-", "") if subject.startswith("sub-") else subject
                
                query_params = {
                    'subject': subject_id,
                    'datatype': 'dwi',
                    'suffix': 'mask',
                    'extension': '.nii.gz'
                }
                
                if session:
                    session_id = session.replace("ses-", "") if session.startswith("ses-") else session
                    query_params['session'] = session_id
                
                if run:
                    run_id = run.replace("run-", "") if run.startswith("run-") else run
                    query_params['run'] = run_id
                
                if task:
                    task_id = task.replace("task-", "") if task.startswith("task-") else task
                    query_params['task'] = task_id
                
                mask_files = self.bids_handler._qsiprep_layout.get(**query_params)
                
                if mask_files:
                    mask_img = nib.load(mask_files[0].path)
                    return mask_img.get_fdata()
                    
        except Exception as e:
            logger.debug(f"Failed to load brain mask: {e}")
        
        return None
    
    def _load_anatomical_brain_mask(self, subject: str, session: Optional[str] = None,
                                  **filters) -> Optional[np.ndarray]:
        """Load brain mask for anatomical data if available"""
        try:
            # Try to find anatomical brain mask using pybids
            if self.bids_handler._qsiprep_layout:
                subject_id = subject.replace("sub-", "") if subject.startswith("sub-") else subject
                
                query_params = {
                    'subject': subject_id,
                    'datatype': 'anat',
                    'suffix': 'mask',
                    'extension': '.nii.gz'
                }
                
                if session:
                    session_id = session.replace("ses-", "") if session.startswith("ses-") else session
                    query_params['session'] = session_id
                
                mask_files = self.bids_handler._qsiprep_layout.get(**query_params)
                
                if mask_files:
                    mask_img = nib.load(mask_files[0].path)
                    return mask_img.get_fdata()
                    
        except Exception as e:
            logger.debug(f"Failed to load anatomical brain mask: {e}")
        
        return None
    
    def _load_tissue_segmentation(self, subject: str, session: Optional[str] = None,
                                **filters) -> Optional[np.ndarray]:
        """Load tissue segmentation if available"""
        try:
            # Try to find tissue segmentation using pybids
            if self.bids_handler._qsiprep_layout:
                subject_id = subject.replace("sub-", "") if subject.startswith("sub-") else subject
                
                query_params = {
                    'subject': subject_id,
                    'datatype': 'anat',
                    'suffix': 'dseg',  # Discrete segmentation
                    'extension': '.nii.gz'
                }
                
                if session:
                    session_id = session.replace("ses-", "") if session.startswith("ses-") else session
                    query_params['session'] = session_id
                
                seg_files = self.bids_handler._qsiprep_layout.get(**query_params)
                
                if seg_files:
                    seg_img = nib.load(seg_files[0].path)
                    return seg_img.get_fdata()
                    
        except Exception as e:
            logger.debug(f"Failed to load tissue segmentation: {e}")
        
        return None
    
    def _check_brain_mask_exists(self, subject: str, session: Optional[str] = None) -> bool:
        """Check if brain mask exists for the subject"""
        try:
            dwi_mask = self._load_brain_mask(subject, session)
            anat_mask = self._load_anatomical_brain_mask(subject, session)
            return dwi_mask is not None or anat_mask is not None
        except Exception:
            return False
    
    def get_available_subjects(self) -> List[str]:
        """
        Get list of available subjects in qsiprep outputs.
        
        Returns
        -------
        List[str]
            List of subject IDs
        """
        try:
            subjects_info = self.bids_handler.get_subjects_info()
            return list(subjects_info.keys())
        except Exception as e:
            logger.error(f"Failed to get available subjects: {e}")
            return []
    
    def get_available_sessions(self, subject: str) -> List[str]:
        """
        Get list of available sessions for a subject.
        
        Parameters
        ----------
        subject : str
            Subject ID
            
        Returns
        -------
        List[str]
            List of session IDs
        """
        try:
            subjects_info = self.bids_handler.get_subjects_info()
            subject_key = subject if subject.startswith('sub-') else f'sub-{subject}'
            return subjects_info.get(subject_key, [])
        except Exception as e:
            logger.error(f"Failed to get available sessions for subject {subject}: {e}")
            return []
    
    def check_preprocessing_quality(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive quality control checks on qsiprep outputs.
        
        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str, optional
            Session ID (with or without 'ses-' prefix)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing quality control metrics and assessments
        """
        logger.info(f"Performing quality control for subject: {subject}")
        
        qc_results = {
            'subject': subject,
            'session': session,
            'overall_quality': 'unknown',
            'quality_score': 0.0,
            'checks': {},
            'recommendations': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check data completeness
            completeness_check = self._check_data_completeness(subject, session)
            qc_results['checks']['data_completeness'] = completeness_check
            
            # Check DWI data quality
            dwi_quality_check = self._check_dwi_quality(subject, session)
            qc_results['checks']['dwi_quality'] = dwi_quality_check
            
            # Check anatomical data quality
            anat_quality_check = self._check_anatomical_quality(subject, session)
            qc_results['checks']['anatomical_quality'] = anat_quality_check
            
            # Check preprocessing metadata
            metadata_check = self._check_preprocessing_metadata_quality(subject, session)
            qc_results['checks']['metadata_quality'] = metadata_check
            
            # Check motion parameters if available
            motion_check = self._check_motion_parameters(subject, session)
            qc_results['checks']['motion_quality'] = motion_check
            
            # Calculate overall quality score
            qc_results['quality_score'] = self._calculate_quality_score(qc_results['checks'])
            qc_results['overall_quality'] = self._determine_overall_quality(qc_results['quality_score'])
            
            # Generate recommendations
            qc_results['recommendations'] = self._generate_quality_recommendations(qc_results['checks'])
            
            # Collect warnings and errors
            for check_name, check_result in qc_results['checks'].items():
                if 'warnings' in check_result:
                    qc_results['warnings'].extend(check_result['warnings'])
                if 'errors' in check_result:
                    qc_results['errors'].extend(check_result['errors'])
            
        except Exception as e:
            logger.error(f"Quality control failed for subject {subject}: {e}")
            qc_results['errors'].append(f"Quality control failed: {e}")
            qc_results['overall_quality'] = 'failed'
        
        return qc_results
    
    def _check_data_completeness(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """Check completeness of required files"""
        check_result = {
            'status': 'pass',
            'score': 1.0,
            'details': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check DWI files
            dwi_files = self.bids_handler.get_preprocessed_dwi_files(subject=subject, session=session)
            check_result['details']['dwi_files_count'] = len(dwi_files)
            
            if not dwi_files:
                check_result['status'] = 'fail'
                check_result['score'] = 0.0
                check_result['errors'].append("No DWI files found")
            else:
                # Check for required associated files
                for dwi_file in dwi_files:
                    if not dwi_file.bval_path.exists():
                        check_result['errors'].append(f"Missing bval file: {dwi_file.bval_path}")
                        check_result['score'] -= 0.2
                    if not dwi_file.bvec_path.exists():
                        check_result['errors'].append(f"Missing bvec file: {dwi_file.bvec_path}")
                        check_result['score'] -= 0.2
                    if not dwi_file.json_path or not dwi_file.json_path.exists():
                        check_result['warnings'].append(f"Missing JSON sidecar: {dwi_file.json_path}")
                        check_result['score'] -= 0.1
            
            # Check anatomical files
            anat_files = self.bids_handler.get_preprocessed_anatomical_files(subject=subject, session=session)
            check_result['details']['anatomical_files_count'] = len(anat_files)
            
            if not anat_files:
                check_result['warnings'].append("No anatomical files found")
                check_result['score'] -= 0.1
            
            # Check for brain masks
            mask_exists = self._check_brain_mask_exists(subject, session)
            check_result['details']['brain_mask_available'] = mask_exists
            
            if not mask_exists:
                check_result['warnings'].append("No brain mask found")
                check_result['score'] -= 0.1
            
            # Ensure score doesn't go below 0
            check_result['score'] = max(0.0, check_result['score'])
            
            if check_result['errors']:
                check_result['status'] = 'fail'
            elif check_result['warnings']:
                check_result['status'] = 'warning'
                
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['score'] = 0.0
            check_result['errors'].append(f"Data completeness check failed: {e}")
        
        return check_result
    
    def _check_dwi_quality(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """Check DWI data quality metrics"""
        check_result = {
            'status': 'pass',
            'score': 1.0,
            'details': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            dwi_data = self.load_preprocessed_data(subject, session)
            
            # Check data dimensions
            check_result['details']['dwi_shape'] = dwi_data.shape
            check_result['details']['n_volumes'] = dwi_data.n_volumes
            check_result['details']['n_bvals'] = dwi_data.n_bvals
            
            # Check for reasonable number of volumes
            if dwi_data.n_volumes < 10:
                check_result['warnings'].append(f"Low number of DWI volumes: {dwi_data.n_volumes}")
                check_result['score'] -= 0.1
            elif dwi_data.n_volumes < 30:
                check_result['warnings'].append(f"Moderate number of DWI volumes: {dwi_data.n_volumes}")
                check_result['score'] -= 0.05
            
            # Check b-value distribution
            unique_bvals = np.unique(dwi_data.bvals)
            check_result['details']['unique_bvals'] = unique_bvals.tolist()
            check_result['details']['bval_counts'] = {int(bval): int(np.sum(dwi_data.bvals == bval)) for bval in unique_bvals}
            
            # Check for b=0 images
            b0_count = np.sum(dwi_data.bvals < 100)
            check_result['details']['b0_count'] = int(b0_count)
            
            if b0_count == 0:
                check_result['errors'].append("No b=0 images found")
                check_result['score'] -= 0.3
            elif b0_count < 3:
                check_result['warnings'].append(f"Low number of b=0 images: {b0_count}")
                check_result['score'] -= 0.1
            
            # Check for high b-value images
            high_bval_count = np.sum(dwi_data.bvals > 500)
            check_result['details']['high_bval_count'] = int(high_bval_count)
            
            if high_bval_count == 0:
                check_result['warnings'].append("No high b-value images found")
                check_result['score'] -= 0.2
            
            # Check for data range and potential artifacts
            dwi_mean = np.mean(dwi_data.dwi_image)
            dwi_std = np.std(dwi_data.dwi_image)
            dwi_min = np.min(dwi_data.dwi_image)
            dwi_max = np.max(dwi_data.dwi_image)
            
            check_result['details']['signal_stats'] = {
                'mean': float(dwi_mean),
                'std': float(dwi_std),
                'min': float(dwi_min),
                'max': float(dwi_max)
            }
            
            # Check for negative values (shouldn't exist in magnitude data)
            if dwi_min < 0:
                check_result['warnings'].append(f"Negative values found in DWI data (min: {dwi_min})")
                check_result['score'] -= 0.1
            
            # Check for very high signal values (potential artifacts)
            if dwi_max > 10 * dwi_mean:
                check_result['warnings'].append(f"Very high signal values detected (max: {dwi_max}, mean: {dwi_mean})")
                check_result['score'] -= 0.1
            
            # Check gradient directions
            if dwi_data.bvecs.shape[0] != 3:
                check_result['errors'].append(f"Invalid bvec dimensions: {dwi_data.bvecs.shape}")
                check_result['score'] -= 0.2
            
            # Check for zero gradients in non-b0 images
            non_b0_indices = dwi_data.bvals > 100
            if np.any(non_b0_indices):
                non_b0_bvecs = dwi_data.bvecs[:, non_b0_indices]
                bvec_norms = np.linalg.norm(non_b0_bvecs, axis=0)
                zero_gradient_count = np.sum(bvec_norms < 0.1)
                
                if zero_gradient_count > 0:
                    check_result['warnings'].append(f"Zero gradients found in {zero_gradient_count} non-b0 images")
                    check_result['score'] -= 0.1
            
            # Ensure score doesn't go below 0
            check_result['score'] = max(0.0, check_result['score'])
            
            if check_result['errors']:
                check_result['status'] = 'fail'
            elif check_result['warnings']:
                check_result['status'] = 'warning'
                
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['score'] = 0.0
            check_result['errors'].append(f"DWI quality check failed: {e}")
        
        return check_result
    
    def _check_anatomical_quality(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """Check anatomical data quality metrics"""
        check_result = {
            'status': 'pass',
            'score': 1.0,
            'details': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            anat_data = self.load_anatomical_data(subject, session)
            
            # Check data dimensions
            check_result['details']['t1w_shape'] = anat_data.shape
            
            # Check for reasonable dimensions
            if any(dim < 100 for dim in anat_data.shape):
                check_result['warnings'].append(f"Low resolution anatomical data: {anat_data.shape}")
                check_result['score'] -= 0.1
            
            # Check signal statistics
            t1w_mean = np.mean(anat_data.t1w_image)
            t1w_std = np.std(anat_data.t1w_image)
            t1w_min = np.min(anat_data.t1w_image)
            t1w_max = np.max(anat_data.t1w_image)
            
            check_result['details']['signal_stats'] = {
                'mean': float(t1w_mean),
                'std': float(t1w_std),
                'min': float(t1w_min),
                'max': float(t1w_max)
            }
            
            # Check for negative values
            if t1w_min < 0:
                check_result['warnings'].append(f"Negative values in T1w data (min: {t1w_min})")
                check_result['score'] -= 0.1
            
            # Check for brain mask availability
            check_result['details']['brain_mask_available'] = anat_data.brain_mask is not None
            check_result['details']['tissue_segmentation_available'] = anat_data.tissue_segmentation is not None
            
            if anat_data.brain_mask is None:
                check_result['warnings'].append("No brain mask available for anatomical data")
                check_result['score'] -= 0.1
            
            if anat_data.tissue_segmentation is None:
                check_result['warnings'].append("No tissue segmentation available")
                check_result['score'] -= 0.1
            
            # Ensure score doesn't go below 0
            check_result['score'] = max(0.0, check_result['score'])
            
            if check_result['errors']:
                check_result['status'] = 'fail'
            elif check_result['warnings']:
                check_result['status'] = 'warning'
                
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['score'] = 0.0
            check_result['errors'].append(f"Anatomical quality check failed: {e}")
        
        return check_result
    
    def _check_preprocessing_metadata_quality(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """Check preprocessing metadata quality and completeness"""
        check_result = {
            'status': 'pass',
            'score': 1.0,
            'details': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            metadata = self.get_preprocessing_metadata(subject, session)
            
            # Check for QSIPrep version
            qsiprep_version = metadata.get('qsiprep_version')
            check_result['details']['qsiprep_version'] = qsiprep_version
            
            if not qsiprep_version:
                check_result['warnings'].append("QSIPrep version not found in metadata")
                check_result['score'] -= 0.1
            
            # Check for preprocessing steps
            preprocessing_steps = metadata.get('preprocessing_steps', [])
            check_result['details']['preprocessing_steps'] = preprocessing_steps
            check_result['details']['n_preprocessing_steps'] = len(preprocessing_steps)
            
            if not preprocessing_steps:
                check_result['warnings'].append("No preprocessing steps information found")
                check_result['score'] -= 0.1
            
            # Check for acquisition parameters
            acquisition_params = metadata.get('acquisition_parameters', {})
            check_result['details']['has_acquisition_params'] = bool(acquisition_params)
            
            if not acquisition_params:
                check_result['warnings'].append("No acquisition parameters found")
                check_result['score'] -= 0.1
            else:
                # Check for important acquisition parameters
                important_params = ['RepetitionTime', 'EchoTime', 'FlipAngle']
                missing_params = [param for param in important_params if param not in acquisition_params]
                
                if missing_params:
                    check_result['warnings'].append(f"Missing acquisition parameters: {missing_params}")
                    check_result['score'] -= 0.05 * len(missing_params)
            
            # Check for software versions
            software_versions = metadata.get('software_versions', {})
            check_result['details']['has_software_versions'] = bool(software_versions)
            
            if not software_versions:
                check_result['warnings'].append("No software version information found")
                check_result['score'] -= 0.05
            
            # Ensure score doesn't go below 0
            check_result['score'] = max(0.0, check_result['score'])
            
            if check_result['errors']:
                check_result['status'] = 'fail'
            elif check_result['warnings']:
                check_result['status'] = 'warning'
                
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['score'] = 0.0
            check_result['errors'].append(f"Metadata quality check failed: {e}")
        
        return check_result
    
    def _check_motion_parameters(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """Check motion parameters if available"""
        check_result = {
            'status': 'pass',
            'score': 1.0,
            'details': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Try to find motion parameters files
            motion_files = self._find_motion_files(subject, session)
            check_result['details']['motion_files_found'] = len(motion_files)
            
            if not motion_files:
                check_result['warnings'].append("No motion parameter files found")
                check_result['score'] = 0.8  # Not critical, but useful
                check_result['status'] = 'warning'
                return check_result
            
            # Analyze motion parameters if found
            for motion_file in motion_files:
                try:
                    motion_params = np.loadtxt(motion_file)
                    
                    # Calculate motion statistics
                    if motion_params.ndim == 2 and motion_params.shape[1] >= 6:
                        # Assume first 3 columns are translations, next 3 are rotations
                        translations = motion_params[:, :3]
                        rotations = motion_params[:, 3:6]
                        
                        # Calculate RMS motion
                        trans_rms = np.sqrt(np.mean(translations**2, axis=1))
                        rot_rms = np.sqrt(np.mean(rotations**2, axis=1))
                        
                        mean_trans_rms = np.mean(trans_rms)
                        mean_rot_rms = np.mean(rot_rms)
                        max_trans_rms = np.max(trans_rms)
                        max_rot_rms = np.max(rot_rms)
                        
                        check_result['details']['motion_stats'] = {
                            'mean_translation_rms': float(mean_trans_rms),
                            'mean_rotation_rms': float(mean_rot_rms),
                            'max_translation_rms': float(max_trans_rms),
                            'max_rotation_rms': float(max_rot_rms)
                        }
                        
                        # Check for excessive motion
                        if max_trans_rms > 2.0:  # 2mm threshold
                            check_result['warnings'].append(f"High translation motion detected: {max_trans_rms:.2f}mm")
                            check_result['score'] -= 0.2
                        elif max_trans_rms > 1.0:
                            check_result['warnings'].append(f"Moderate translation motion detected: {max_trans_rms:.2f}mm")
                            check_result['score'] -= 0.1
                        
                        if max_rot_rms > 0.02:  # ~1 degree threshold
                            check_result['warnings'].append(f"High rotational motion detected: {max_rot_rms:.4f}rad")
                            check_result['score'] -= 0.2
                        elif max_rot_rms > 0.01:
                            check_result['warnings'].append(f"Moderate rotational motion detected: {max_rot_rms:.4f}rad")
                            check_result['score'] -= 0.1
                    
                except Exception as e:
                    check_result['warnings'].append(f"Failed to analyze motion file {motion_file}: {e}")
                    check_result['score'] -= 0.1
            
            # Ensure score doesn't go below 0
            check_result['score'] = max(0.0, check_result['score'])
            
            if check_result['warnings']:
                check_result['status'] = 'warning'
                
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['score'] = 0.0
            check_result['errors'].append(f"Motion parameter check failed: {e}")
        
        return check_result
    
    def _find_motion_files(self, subject: str, session: Optional[str] = None) -> List[Path]:
        """Find motion parameter files for a subject"""
        motion_files = []
        
        try:
            if self.bids_handler._qsiprep_layout:
                subject_id = subject.replace("sub-", "") if subject.startswith("sub-") else subject
                
                # Common motion file patterns in qsiprep
                motion_suffixes = ['motion', 'confounds', 'regressors']
                
                for suffix in motion_suffixes:
                    query_params = {
                        'subject': subject_id,
                        'suffix': suffix,
                        'extension': ['.tsv', '.txt']
                    }
                    
                    if session:
                        session_id = session.replace("ses-", "") if session.startswith("ses-") else session
                        query_params['session'] = session_id
                    
                    files = self.bids_handler._qsiprep_layout.get(**query_params)
                    motion_files.extend([Path(f.path) for f in files])
                    
        except Exception as e:
            logger.debug(f"Failed to find motion files: {e}")
        
        return motion_files
    
    def _calculate_quality_score(self, checks: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall quality score from individual checks"""
        if not checks:
            return 0.0
        
        # Weight different checks
        weights = {
            'data_completeness': 0.3,
            'dwi_quality': 0.3,
            'anatomical_quality': 0.2,
            'metadata_quality': 0.1,
            'motion_quality': 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for check_name, check_result in checks.items():
            if check_name in weights and 'score' in check_result:
                weight = weights[check_name]
                score = check_result['score']
                total_score += weight * score
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _determine_overall_quality(self, quality_score: float) -> str:
        """Determine overall quality rating from score"""
        if quality_score >= 0.9:
            return 'excellent'
        elif quality_score >= 0.8:
            return 'good'
        elif quality_score >= 0.7:
            return 'acceptable'
        elif quality_score >= 0.5:
            return 'poor'
        else:
            return 'failed'
    
    def _generate_quality_recommendations(self, checks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on quality checks"""
        recommendations = []
        
        # Data completeness recommendations
        completeness_check = checks.get('data_completeness', {})
        if completeness_check.get('status') == 'fail':
            recommendations.append("Ensure all required DWI files (nii.gz, bval, bvec) are present")
        if not completeness_check.get('details', {}).get('brain_mask_available', True):
            recommendations.append("Consider generating brain masks for improved processing accuracy")
        
        # DWI quality recommendations
        dwi_check = checks.get('dwi_quality', {})
        dwi_details = dwi_check.get('details', {})
        
        if dwi_details.get('b0_count', 0) < 3:
            recommendations.append("Consider acquiring more b=0 images for better preprocessing")
        if dwi_details.get('n_volumes', 0) < 30:
            recommendations.append("Consider acquiring more DWI volumes for robust tensor fitting")
        if dwi_details.get('high_bval_count', 0) == 0:
            recommendations.append("Consider acquiring high b-value images for advanced modeling")
        
        # Motion recommendations
        motion_check = checks.get('motion_quality', {})
        motion_stats = motion_check.get('details', {}).get('motion_stats', {})
        
        if motion_stats.get('max_translation_rms', 0) > 1.0:
            recommendations.append("High motion detected - consider excluding volumes or using motion correction")
        
        # Anatomical recommendations
        anat_check = checks.get('anatomical_quality', {})
        anat_details = anat_check.get('details', {})
        
        if not anat_details.get('tissue_segmentation_available', True):
            recommendations.append("Consider generating tissue segmentation for advanced processing")
        
        # Metadata recommendations
        metadata_check = checks.get('metadata_quality', {})
        if not metadata_check.get('details', {}).get('has_acquisition_params', True):
            recommendations.append("Ensure acquisition parameters are properly documented")
        
        return recommendations
    
    def generate_quality_report(self, subject: str, session: Optional[str] = None, 
                              output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive quality control report.
        
        Parameters
        ----------
        subject : str
            Subject ID
        session : str, optional
            Session ID
        output_path : Path, optional
            Path to save the report (JSON format)
            
        Returns
        -------
        Dict[str, Any]
            Complete quality control report
        """
        logger.info(f"Generating quality control report for subject: {subject}")
        
        # Perform quality checks
        qc_results = self.check_preprocessing_quality(subject, session)
        
        # Add additional report metadata
        report = {
            'report_info': {
                'generated_by': 'elikopy QsiPrepAdapter',
                'generation_time': str(np.datetime64('now')),
                'qsiprep_directory': str(self.qsiprep_dir),
                'subject': subject,
                'session': session
            },
            'quality_assessment': qc_results,
            'summary': {
                'overall_quality': qc_results['overall_quality'],
                'quality_score': qc_results['quality_score'],
                'total_errors': len(qc_results['errors']),
                'total_warnings': len(qc_results['warnings']),
                'total_recommendations': len(qc_results['recommendations'])
            }
        }
        
        # Save report if output path provided
        if output_path:
            try:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                logger.info(f"Quality control report saved to: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save quality control report: {e}")
        
        return report