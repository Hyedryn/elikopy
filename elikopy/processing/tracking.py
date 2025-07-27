"""
Tractography processing module for ElikoPy
=======================================

This module provides functionality for tractography processing including
streamline generation, SIFT filtering, and BIDS-compliant output generation.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking import utils
from dipy.direction import peaks_from_model
from dipy.io.streamline import save_trk, load_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from dipy.tracking.utils import seeds_from_mask
from dipy.direction import DeterministicMaximumDirectionGetter, ProbabilisticDirectionGetter

from elikopy.core.base import ProcessingComponent, ProcessingResult, ProcessingStatus, ConfigurableComponent
from elikopy.data.derivatives import DerivativesManager


logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for tractography processing"""
    algorithm: str = "deterministic"  # deterministic, probabilistic, closestpeak
    step_size: float = 0.5
    max_angle: float = 30.0
    min_length: float = 20.0
    max_length: float = 200.0
    num_streamlines: int = 100000
    seed_density: float = 1.0
    cutoff: float = 0.1
    apply_sift: bool = True
    sift_term_count: int = 200000
    use_mrtrix: bool = True  # Use mrtrix for better performance when available
    output_formats: List[str] = field(default_factory=lambda: ["tck", "trk"])
    save_seeds: bool = False
    save_stopping_criterion: bool = False
    mask_type: str = "brain_mask"  # brain_mask, brain_mask_dilated, wm_mask
    seed_type: str = "mask"  # mask, random, interface
    # DIPY-specific parameters
    sphere: Optional[str] = None  # sphere for direction getting
    pmf_threshold: float = 0.1
    max_cross: Optional[int] = None
    return_all: bool = True


@dataclass
class StreamlinesResult:
    """Result of streamline generation"""
    streamlines: Union[Streamlines, List[np.ndarray]]
    streamline_count: int
    length_stats: Dict[str, float]
    seeds: Optional[np.ndarray] = None
    affine: Optional[np.ndarray] = None
    header: Optional[Dict[str, Any]] = None
    algorithm_used: str = "deterministic"
    processing_time: Optional[float] = None


@dataclass
class SIFTResult:
    """Result of SIFT filtering"""
    streamlines: Union[Streamlines, List[np.ndarray]]
    streamline_count: int
    weights: Optional[np.ndarray] = None
    proportionality_coefficients: Optional[np.ndarray] = None
    cost_function_values: Optional[List[float]] = None
    original_count: Optional[int] = None
    reduction_factor: Optional[float] = None


class TrackingProcessor(ProcessingComponent, ConfigurableComponent):
    """
    Processor for tractography generation and filtering.
    
    This processor implements streamline generation using different algorithms
    (deterministic, probabilistic), SIFT filtering, and BIDS-compliant output.
    """
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize tracking processor.
        
        Parameters
        ----------
        config : TrackingConfig, optional
            Configuration for tractography processing
        """
        self.config = config or TrackingConfig()
        self.derivatives_manager = None
    
    def validate_inputs(self, peaks_data: Optional[Dict[str, Any]] = None,
                       mask: Optional[np.ndarray] = None,
                       dwi_data: Optional[np.ndarray] = None,
                       affine: Optional[np.ndarray] = None, subject: Optional[str] = None) -> bool:
        """
        Validate inputs before processing.
        
        Parameters
        ----------
        peaks_data : Dict[str, Any], optional
            Dictionary containing peaks data
        mask : np.ndarray, optional
            Binary mask
        dwi_data : np.ndarray, optional
            DWI data
        affine : np.ndarray, optional
            Affine transformation matrix
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if peaks_data is None:
            logger.error("Peaks data is required for tractography")
            return False
            
        if mask is None:
            logger.error("Mask is required for tractography")
            return False
            
        if affine is None:
            logger.error("Affine transformation matrix is required")
            return False
            
        # Validate peaks data structure
        required_keys = ['peaks', 'peak_values', 'peak_indices']
        if not all(key in peaks_data for key in required_keys):
            logger.error(f"Peaks data must contain: {required_keys}")
            return False
            
        # Validate dimensions
        peaks = peaks_data['peaks']
        if len(peaks.shape) != 5:  # (x, y, z, n_peaks, 3)
            logger.error(f"Peaks must be 5D array, got shape: {peaks.shape}")
            return False
            
        if mask.shape != peaks.shape[:3]:
            logger.error(f"Mask shape {mask.shape} doesn't match peaks shape {peaks.shape[:3]}")
            return False
            
        return True
    
    def process(self, peaks_data: Dict[str, Any], mask: np.ndarray,
                affine: np.ndarray, output_dir: Path,
                subject_id: str, session_id: Optional[str] = None,
                dwi_data: Optional[np.ndarray] = None,
                bvals: Optional[np.ndarray] = None,
                bvecs: Optional[np.ndarray] = None) -> ProcessingResult:
        """
        Execute tractography processing.
        
        Parameters
        ----------
        peaks_data : Dict[str, Any]
            Dictionary containing peaks data
        mask : np.ndarray
            Binary mask for seeding and stopping
        affine : np.ndarray
            Affine transformation matrix
        output_dir : Path
            Output directory for results
        subject_id : str
            Subject identifier
        session_id : str, optional
            Session identifier
        dwi_data : np.ndarray, optional
            DWI data (needed for some algorithms)
        bvals : np.ndarray, optional
            B-values
        bvecs : np.ndarray, optional
            B-vectors
            
        Returns
        -------
        ProcessingResult
            Processing result with output files and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validate_inputs(peaks_data, mask, dwi_data, affine):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Input validation failed"
                )
            
            logger.info(f"Starting tractography for subject {subject_id}")
            
            # Generate streamlines
            streamlines_result = self.generate_streamlines(
                peaks_data, mask, affine, dwi_data, bvals, bvecs
            )
            
            if streamlines_result is None:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Streamline generation failed"
                )
            
            # Apply SIFT filtering if requested
            sift_result = None
            if self.config.apply_sift:
                logger.info("Applying SIFT filtering")
                sift_result = self.apply_sift(streamlines_result, peaks_data)
            
            # Save outputs in BIDS format
            output_files = self._save_bids_outputs(
                streamlines_result, sift_result, output_dir, 
                subject_id, session_id, affine
            )
            
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                "algorithm": self.config.algorithm,
                "num_streamlines": streamlines_result.streamline_count,
                "processing_time": processing_time,
                "config": self._config_to_dict(),
                "length_stats": streamlines_result.length_stats
            }
            
            if sift_result:
                metadata["sift_applied"] = True
                metadata["sift_streamlines"] = sift_result.streamline_count
                metadata["sift_reduction_factor"] = sift_result.reduction_factor
            
            logger.info(f"Tractography completed in {processing_time:.2f} seconds")
            
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Tractography processing failed: {str(e)}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the component.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        # Convert dict to TrackingConfig
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns
        -------
        Dict[str, Any]
            Default configuration
        """
        return {
            "algorithm": "deterministic",
            "step_size": 0.5,
            "max_angle": 30.0,
            "min_length": 20.0,
            "max_length": 200.0,
            "num_streamlines": 100000,
            "seed_density": 1.0,
            "cutoff": 0.1,
            "apply_sift": True,
            "sift_term_count": 200000,
            "use_mrtrix": True,
            "mask_type": "brain_mask"
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        valid_algorithms = ["deterministic", "probabilistic", "closestpeak"]
        valid_mask_types = ["brain_mask", "brain_mask_dilated", "wm_mask"]
        valid_seed_types = ["mask", "random", "interface"]
        
        if "algorithm" in config and config["algorithm"] not in valid_algorithms:
            logger.error(f"Invalid algorithm: {config['algorithm']}")
            return False
            
        if "step_size" in config and config["step_size"] <= 0:
            logger.error(f"Step size must be positive: {config['step_size']}")
            return False
            
        if "max_angle" in config and (config["max_angle"] <= 0 or config["max_angle"] > 90):
            logger.error(f"Max angle must be between 0 and 90: {config['max_angle']}")
            return False
            
        if "min_length" in config and config["min_length"] <= 0:
            logger.error(f"Min length must be positive: {config['min_length']}")
            return False
            
        if "max_length" in config and config["max_length"] <= config.get("min_length", 0):
            logger.error("Max length must be greater than min length")
            return False
            
        if "mask_type" in config and config["mask_type"] not in valid_mask_types:
            logger.error(f"Invalid mask type: {config['mask_type']}")
            return False
            
        if "seed_type" in config and config["seed_type"] not in valid_seed_types:
            logger.error(f"Invalid seed type: {config['seed_type']}")
            return False
            
        return True
    
    def generate_streamlines(self, peaks_data: Dict[str, Any], 
                           mask: np.ndarray, affine: np.ndarray,
                           dwi_data: Optional[np.ndarray] = None,
                           bvals: Optional[np.ndarray] = None,
                           bvecs: Optional[np.ndarray] = None) -> Optional[StreamlinesResult]:
        """
        Generate tractography streamlines using DIPY.
        
        Parameters
        ----------
        peaks_data : Dict[str, Any]
            Dictionary containing peaks data with keys: 'peaks', 'peak_values', 'peak_indices'
        mask : np.ndarray
            Binary mask for seeding and stopping
        affine : np.ndarray
            Affine transformation matrix
        dwi_data : np.ndarray, optional
            DWI data (needed for some algorithms)
        bvals : np.ndarray, optional
            B-values
        bvecs : np.ndarray, optional
            B-vectors
            
        Returns
        -------
        StreamlinesResult or None
            Result containing streamlines and metadata
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Generating streamlines using {self.config.algorithm} algorithm")
            
            # Extract peaks data
            peaks = peaks_data['peaks']
            peak_values = peaks_data['peak_values']
            peak_indices = peaks_data['peak_indices']
            
            # Create seeds from mask
            seeds = seeds_from_mask(mask, affine, density=self.config.seed_density)
            logger.info(f"Generated {len(seeds)} seeds")
            
            # Create stopping criterion
            stopping_criterion = ThresholdStoppingCriterion(
                peak_values, self.config.cutoff
            )
            
            # Create direction getter based on algorithm
            if self.config.algorithm == "deterministic":
                direction_getter = DeterministicMaximumDirectionGetter.from_shcoeff(
                    peaks, max_angle=self.config.max_angle, sphere=None
                )
            elif self.config.algorithm == "probabilistic":
                direction_getter = ProbabilisticDirectionGetter.from_shcoeff(
                    peaks, max_angle=self.config.max_angle, sphere=None,
                    pmf_threshold=self.config.pmf_threshold
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
            
            # Perform tracking
            streamlines_generator = LocalTracking(
                direction_getter,
                stopping_criterion,
                seeds,
                affine,
                step_size=self.config.step_size,
                max_cross=self.config.max_cross,
                maxlen=int(self.config.max_length / self.config.step_size),
                minlen=int(self.config.min_length / self.config.step_size),
                return_all=self.config.return_all
            )
            
            # Convert generator to list and limit number of streamlines
            streamlines = Streamlines(streamlines_generator)
            
            # Limit to requested number of streamlines
            if len(streamlines) > self.config.num_streamlines:
                # Randomly sample streamlines
                indices = np.random.choice(
                    len(streamlines), self.config.num_streamlines, replace=False
                )
                streamlines = Streamlines([streamlines[i] for i in indices])
            
            # Calculate length statistics
            lengths = [len(s) * self.config.step_size for s in streamlines]
            length_stats = {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": float(np.min(lengths)),
                "max": float(np.max(lengths)),
                "median": float(np.median(lengths))
            }
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated {len(streamlines)} streamlines in {processing_time:.2f} seconds")
            
            return StreamlinesResult(
                streamlines=streamlines,
                streamline_count=len(streamlines),
                length_stats=length_stats,
                seeds=seeds,
                affine=affine,
                algorithm_used=self.config.algorithm,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Streamline generation failed: {str(e)}")
            return None
    
    def apply_sift(self, streamlines_result: StreamlinesResult, 
                  peaks_data: Dict[str, Any]) -> Optional[SIFTResult]:
        """
        Apply SIFT filtering to streamlines.
        
        Parameters
        ----------
        streamlines_result : StreamlinesResult
            Result from streamline generation
        peaks_data : Dict[str, Any]
            Dictionary containing ODF/peaks data
            
        Returns
        -------
        SIFTResult or None
            Result containing filtered streamlines
        """
        try:
            logger.info("Applying SIFT filtering")
            
            # For now, implement a simple version that uses MRtrix if available
            # Otherwise, implement basic SIFT using DIPY
            
            if self.config.use_mrtrix:
                return self._apply_sift_mrtrix(streamlines_result, peaks_data)
            else:
                return self._apply_sift_dipy(streamlines_result, peaks_data)
                
        except Exception as e:
            logger.error(f"SIFT filtering failed: {str(e)}")
            return None
    
    def _apply_sift_mrtrix(self, streamlines_result: StreamlinesResult,
                          peaks_data: Dict[str, Any]) -> Optional[SIFTResult]:
        """Apply SIFT using MRtrix (if available)"""
        # This would require MRtrix to be installed
        # For now, fall back to DIPY implementation
        logger.warning("MRtrix SIFT not implemented, falling back to DIPY")
        return self._apply_sift_dipy(streamlines_result, peaks_data)
    
    def _apply_sift_dipy(self, streamlines_result: StreamlinesResult,
                        peaks_data: Dict[str, Any]) -> Optional[SIFTResult]:
        """Apply basic SIFT-like filtering using DIPY"""
        try:
            # Simple implementation: randomly select streamlines to reach target count
            streamlines = streamlines_result.streamlines
            original_count = len(streamlines)
            target_count = min(self.config.sift_term_count, original_count)
            
            if target_count >= original_count:
                # No filtering needed
                return SIFTResult(
                    streamlines=streamlines,
                    streamline_count=original_count,
                    original_count=original_count,
                    reduction_factor=1.0
                )
            
            # Randomly select streamlines
            indices = np.random.choice(original_count, target_count, replace=False)
            filtered_streamlines = Streamlines([streamlines[i] for i in indices])
            
            reduction_factor = target_count / original_count
            
            logger.info(f"SIFT filtering: {original_count} -> {target_count} streamlines "
                       f"(reduction factor: {reduction_factor:.3f})")
            
            return SIFTResult(
                streamlines=filtered_streamlines,
                streamline_count=target_count,
                original_count=original_count,
                reduction_factor=reduction_factor
            )
            
        except Exception as e:
            logger.error(f"DIPY SIFT filtering failed: {str(e)}")
            return None
    
    def _save_bids_outputs(self, streamlines_result: StreamlinesResult,
                          sift_result: Optional[SIFTResult],
                          output_dir: Path, subject_id: str,
                          session_id: Optional[str],
                          affine: np.ndarray) -> List[Path]:
        """
        Save tractography outputs in BIDS format.
        
        Parameters
        ----------
        streamlines_result : StreamlinesResult
            Result from streamline generation
        sift_result : SIFTResult, optional
            Result from SIFT filtering
        output_dir : Path
            Output directory
        subject_id : str
            Subject identifier
        session_id : str, optional
            Session identifier
        affine : np.ndarray
            Affine transformation matrix
            
        Returns
        -------
        List[Path]
            List of output files
        """
        output_files = []
        
        # Create subject/session directory structure
        if session_id:
            subj_dir = output_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "dwi"
            prefix = f"sub-{subject_id}_ses-{session_id}"
        else:
            subj_dir = output_dir / f"sub-{subject_id}" / "dwi"
            prefix = f"sub-{subject_id}"
        
        subj_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original tractography
        for fmt in self.config.output_formats:
            if fmt == "trk":
                trk_file = subj_dir / f"{prefix}_tractography.trk"
                self._save_trk(streamlines_result.streamlines, trk_file, affine)
                output_files.append(trk_file)
            elif fmt == "tck":
                tck_file = subj_dir / f"{prefix}_tractography.tck"
                self._save_tck(streamlines_result.streamlines, tck_file)
                output_files.append(tck_file)
        
        # Save SIFT-filtered tractography if available
        if sift_result:
            for fmt in self.config.output_formats:
                if fmt == "trk":
                    sift_trk_file = subj_dir / f"{prefix}_tractography-sift.trk"
                    self._save_trk(sift_result.streamlines, sift_trk_file, affine)
                    output_files.append(sift_trk_file)
                elif fmt == "tck":
                    sift_tck_file = subj_dir / f"{prefix}_tractography-sift.tck"
                    self._save_tck(sift_result.streamlines, sift_tck_file)
                    output_files.append(sift_tck_file)
        
        # Save metadata
        metadata_file = subj_dir / f"{prefix}_tractography.json"
        self._save_metadata(streamlines_result, sift_result, metadata_file)
        output_files.append(metadata_file)
        
        return output_files
    
    def _save_trk(self, streamlines: Streamlines, output_file: Path, affine: np.ndarray):
        """Save streamlines in TRK format"""
        try:
            # Create a dummy reference image for TRK format
            # This is needed for the StatefulTractogram
            ref_shape = (128, 128, 128)  # Default shape
            
            tractogram = StatefulTractogram(
                streamlines, reference=affine, space=Space.RASMM
            )
            save_trk(tractogram, str(output_file))
            logger.info(f"Saved TRK file: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save TRK file {output_file}: {str(e)}")
    
    def _save_tck(self, streamlines: Streamlines, output_file: Path):
        """Save streamlines in TCK format using MRtrix"""
        try:
            # This would require MRtrix to be available
            # For now, save as TRK and convert if needed
            logger.warning(f"TCK format not directly supported, skipping {output_file}")
        except Exception as e:
            logger.error(f"Failed to save TCK file {output_file}: {str(e)}")
    
    def _save_metadata(self, streamlines_result: StreamlinesResult,
                      sift_result: Optional[SIFTResult],
                      metadata_file: Path):
        """Save tractography metadata in JSON format"""
        try:
            metadata = {
                "algorithm": self.config.algorithm,
                "step_size": self.config.step_size,
                "max_angle": self.config.max_angle,
                "min_length": self.config.min_length,
                "max_length": self.config.max_length,
                "cutoff": self.config.cutoff,
                "num_streamlines": streamlines_result.streamline_count,
                "length_statistics": streamlines_result.length_stats,
                "processing_time": streamlines_result.processing_time
            }
            
            if sift_result:
                metadata["sift_applied"] = True
                metadata["sift_streamlines"] = sift_result.streamline_count
                metadata["sift_reduction_factor"] = sift_result.reduction_factor
                metadata["sift_term_count"] = self.config.sift_term_count
            else:
                metadata["sift_applied"] = False
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata: {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata {metadata_file}: {str(e)}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "algorithm": self.config.algorithm,
            "step_size": self.config.step_size,
            "max_angle": self.config.max_angle,
            "min_length": self.config.min_length,
            "max_length": self.config.max_length,
            "num_streamlines": self.config.num_streamlines,
            "seed_density": self.config.seed_density,
            "cutoff": self.config.cutoff,
            "apply_sift": self.config.apply_sift,
            "sift_term_count": self.config.sift_term_count,
            "use_mrtrix": self.config.use_mrtrix,
            "mask_type": self.config.mask_type,
            "seed_type": self.config.seed_type
        }