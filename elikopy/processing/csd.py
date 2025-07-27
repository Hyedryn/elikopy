"""
CSD processing module for ElikoPy
===============================

This module provides functionality for CSD and MSMT-CSD processing including
single-shell CSD fitting, response function estimation, peak extraction,
and BIDS-compliant output generation.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel, auto_response_ssst
)
from dipy.reconst.mcsd import (
    MultiShellDeconvModel, multi_shell_fiber_response
)
from dipy.direction import peaks_from_model
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs

from elikopy.core.base import ModelProcessor, ProcessingResult, ProcessingStatus


logger = logging.getLogger(__name__)


@dataclass
class CSDConfig:
    """Configuration for CSD processing"""
    response_algorithm: str = "auto"
    sh_order: int = 8
    relative_peak_threshold: float = 0.5
    min_separation_angle: float = 25
    output_types: List[str] = field(default_factory=lambda: ["odf", "peaks", "peak_indices", "peak_values"])
    auto_mask: bool = True
    mask_median_radius: int = 4
    mask_numpass: int = 1
    fa_threshold: float = 0.7
    roi_radii: int = 10
    roi_center: Optional[Tuple[int, int, int]] = None
    save_response: bool = True
    save_odf: bool = True
    save_peaks: bool = True
    # MSMT-CSD specific options
    msmt_response_estimation: str = "auto"  # Method for MSMT response estimation
    tissue_types: List[str] = field(default_factory=lambda: ["wm", "gm", "csf"])
    save_tissue_fractions: bool = True
    compute_tissue_metrics: bool = True


@dataclass
class CSDResult:
    """Result of CSD model fitting"""
    odf_data: np.ndarray
    peaks: np.ndarray
    peak_values: np.ndarray
    peak_indices: np.ndarray
    response_function: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    fit_quality: Optional[np.ndarray] = None


@dataclass
class MSMTCSDResult:
    """Result of MSMT-CSD model fitting"""
    wm_odf: np.ndarray
    gm_signal: np.ndarray
    csf_signal: np.ndarray
    peaks: np.ndarray
    peak_values: np.ndarray
    peak_indices: np.ndarray
    response_functions: Optional[Dict[str, np.ndarray]] = None
    mask: Optional[np.ndarray] = None


class CSDProcessor(ModelProcessor):
    """
    Processor for CSD and MSMT-CSD model fitting.
    
    This processor implements single-shell Constrained Spherical Deconvolution
    with automatic response function estimation, peak extraction, and 
    BIDS-compliant output generation.
    """
    
    def __init__(self, config: Optional[CSDConfig] = None):
        """Initialize CSD processor."""
        self.config = config or CSDConfig()
        
        # Validate configuration during initialization
        config_dict = {
            "response_algorithm": self.config.response_algorithm,
            "sh_order": self.config.sh_order,
            "relative_peak_threshold": self.config.relative_peak_threshold,
            "min_separation_angle": self.config.min_separation_angle,
            "output_types": self.config.output_types
        }
        
        if not self.validate_config(config_dict):
            raise ValueError("Invalid CSD configuration provided")
        
        self._model: Optional[ConstrainedSphericalDeconvModel] = None
        logger.info("CSD processor initialized with response algorithm: %s", 
                   self.config.response_algorithm)
    
    def validate_inputs(self, dwi_data: np.ndarray, bvals: np.ndarray, 
                       bvecs: np.ndarray, mask: Optional[np.ndarray] = None, subject: Optional[str] = None) -> bool:
        """Validate inputs before processing."""
        try:
            if dwi_data.ndim != 4:
                logger.error("DWI data must be 4D, got %dD", dwi_data.ndim)
                return False
            
            if bvals.ndim != 1:
                logger.error("B-values must be 1D array, got %dD", bvals.ndim)
                return False
            
            if len(bvals) != dwi_data.shape[3]:
                logger.error("Number of b-values does not match DWI volumes")
                return False
            
            if bvecs.ndim != 2:
                logger.error("B-vectors must be 2D array, got %dD", bvecs.ndim)
                return False
            
            if mask is not None and mask.shape != dwi_data.shape[:3]:
                logger.error("Mask shape does not match DWI spatial dimensions")
                return False
            
            # Check for sufficient b-values for CSD
            unique_bvals = np.unique(bvals)
            high_b_shells = unique_bvals[unique_bvals >= 1000]
            if len(high_b_shells) < 1:
                logger.error("CSD requires at least one high b-value shell (>=1000 s/mm²)")
                return False
            
            # Check for sufficient directions
            high_b_indices = np.where(bvals >= 1000)[0]
            if len(high_b_indices) < 30:
                logger.warning("CSD typically requires at least 30 high b-value directions for reliable results")
            
            logger.info("Input validation passed")
            return True
            
        except Exception as e:
            logger.error("Error during input validation: %s", str(e))
            return False
    
    def process(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                mask: Optional[np.ndarray] = None, affine: Optional[np.ndarray] = None,
                output_dir: Optional[Path] = None, subject_id: Optional[str] = None,
                session_id: Optional[str] = None, **kwargs) -> ProcessingResult:
        """Execute CSD processing pipeline."""
        try:
            logger.info("Starting CSD processing")
            
            if not self.validate_inputs(dwi_data, bvals, bvecs, mask):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Input validation failed"
                )
            
            if mask is None and self.config.auto_mask:
                logger.info("Creating automatic mask using median_otsu")
                _, mask = median_otsu(
                    dwi_data, 
                    median_radius=self.config.mask_median_radius,
                    numpass=self.config.mask_numpass
                )
            
            csd_result = self.fit_model(dwi_data, bvals, bvecs, mask)
            
            output_files = []
            metadata = {
                "processing_method": "CSD",
                "response_algorithm": self.config.response_algorithm,
                "sh_order": self.config.sh_order,
                "output_types": self.config.output_types,
                "mask_applied": mask is not None,
                "auto_mask": self.config.auto_mask
            }
            
            if output_dir is not None:
                output_files = self._save_outputs(
                    csd_result, output_dir, 
                    subject_id, session_id, affine
                )
            
            logger.info("CSD processing completed successfully")
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("CSD processing failed: %s", str(e))
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def fit_model(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> CSDResult:
        """Fit Constrained Spherical Deconvolution model to data."""
        gtab = gradient_table(bvals, bvecs=bvecs)
        
        # Estimate response function
        logger.info("Estimating response function using %s algorithm", 
                   self.config.response_algorithm)
        
        if self.config.response_algorithm == "auto":
            response, ratio = auto_response_ssst(
                gtab, dwi_data, 
                roi_radii=self.config.roi_radii,
                fa_thr=self.config.fa_threshold
            )
            logger.info("Auto response estimation completed with ratio: %.3f", ratio)
        else:
            # For other algorithms, we'll use auto as fallback for now
            # In a full implementation, you'd add support for other methods
            logger.warning("Algorithm %s not fully implemented, using auto", 
                          self.config.response_algorithm)
            response, ratio = auto_response_ssst(
                gtab, dwi_data, 
                roi_radii=self.config.roi_radii,
                fa_thr=self.config.fa_threshold
            )
        
        # Create and fit CSD model
        logger.info("Fitting CSD model with SH order %d", self.config.sh_order)
        self._model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=self.config.sh_order)
        
        if mask is not None:
            csd_fit = self._model.fit(dwi_data, mask=mask)
        else:
            csd_fit = self._model.fit(dwi_data)
        
        # Get ODF data
        odf_data = csd_fit.odf(self._model.sphere)
        
        # Extract peaks
        logger.info("Extracting peaks with threshold %.2f and min angle %.1f°", 
                   self.config.relative_peak_threshold, self.config.min_separation_angle)
        
        peaks_result = peaks_from_model(
            model=self._model,
            data=dwi_data,
            sphere=self._model.sphere,
            relative_peak_threshold=self.config.relative_peak_threshold,
            min_separation_angle=self.config.min_separation_angle,
            mask=mask,
            return_odf=False,
            return_sh=False,
            normalize_peaks=True
        )
        
        # Compute fit quality if possible
        fit_quality = None
        if mask is not None:
            try:
                predicted = csd_fit.predict(gtab, S0=1.0)
                residuals = dwi_data - predicted
                fit_quality = np.sum(residuals**2, axis=-1)
                fit_quality[mask == 0] = 0
            except Exception as e:
                logger.warning("Could not compute fit quality: %s", str(e))
        
        return CSDResult(
            odf_data=odf_data,
            peaks=peaks_result.peak_dirs,
            peak_values=peaks_result.peak_values,
            peak_indices=peaks_result.peak_indices,
            response_function=response,
            mask=mask,
            fit_quality=fit_quality
        )
    
    def compute_metrics(self, csd_result: CSDResult) -> Dict[str, np.ndarray]:
        """Compute derived metrics from CSD results."""
        metrics = {}
        
        # Generalized Fractional Anisotropy (GFA) from ODF
        logger.info("Computing GFA from ODF")
        odf = csd_result.odf_data
        
        # Compute GFA: std(odf) / rms(odf)
        odf_mean = np.mean(odf, axis=-1, keepdims=True)
        odf_std = np.std(odf, axis=-1)
        odf_rms = np.sqrt(np.mean(odf**2, axis=-1))
        
        # Avoid division by zero
        gfa = np.zeros_like(odf_std)
        valid_mask = odf_rms > 0
        gfa[valid_mask] = odf_std[valid_mask] / odf_rms[valid_mask]
        
        if csd_result.mask is not None:
            gfa[csd_result.mask == 0] = 0
        
        metrics["gfa"] = np.clip(gfa, 0, 1)
        
        # Number of fiber orientations (peaks)
        logger.info("Computing number of fiber orientations")
        num_peaks = np.sum(csd_result.peak_values > 0, axis=-1)
        if csd_result.mask is not None:
            num_peaks[csd_result.mask == 0] = 0
        
        metrics["num_peaks"] = num_peaks
        
        # Peak amplitudes
        if csd_result.peak_values is not None:
            logger.info("Computing peak amplitude metrics")
            
            # Primary peak amplitude
            primary_peak = csd_result.peak_values[..., 0]
            if csd_result.mask is not None:
                primary_peak[csd_result.mask == 0] = 0
            metrics["primary_peak_amplitude"] = primary_peak
            
            # Secondary peak amplitude (if exists)
            if csd_result.peak_values.shape[-1] > 1:
                secondary_peak = csd_result.peak_values[..., 1]
                if csd_result.mask is not None:
                    secondary_peak[csd_result.mask == 0] = 0
                metrics["secondary_peak_amplitude"] = secondary_peak
        
        return metrics
    
    def compute_msmt_csd_metrics(self, msmt_result: MSMTCSDResult) -> Dict[str, np.ndarray]:
        """Compute derived metrics from MSMT-CSD results."""
        metrics = {}
        
        # Tissue volume fractions are already available
        wm_fraction = np.zeros_like(msmt_result.gm_signal)
        if msmt_result.wm_odf is not None:
            # Estimate WM fraction from ODF amplitude
            wm_fraction = np.mean(msmt_result.wm_odf, axis=-1)
            if msmt_result.mask is not None:
                wm_fraction[msmt_result.mask == 0] = 0
        
        metrics["wm_fraction"] = np.clip(wm_fraction, 0, 1)
        metrics["gm_fraction"] = np.clip(msmt_result.gm_signal, 0, 1)
        metrics["csf_fraction"] = np.clip(msmt_result.csf_signal, 0, 1)
        
        # Total tissue fraction (should sum to ~1 in healthy tissue)
        total_fraction = metrics["wm_fraction"] + metrics["gm_fraction"] + metrics["csf_fraction"]
        metrics["total_fraction"] = total_fraction
        
        # Tissue fraction ratios
        safe_total = np.where(total_fraction > 0, total_fraction, 1)  # Avoid division by zero
        metrics["wm_ratio"] = metrics["wm_fraction"] / safe_total
        metrics["gm_ratio"] = metrics["gm_fraction"] / safe_total
        metrics["csf_ratio"] = metrics["csf_fraction"] / safe_total
        
        # Apply mask to all metrics
        if msmt_result.mask is not None:
            for metric_name in metrics:
                metrics[metric_name][msmt_result.mask == 0] = 0
        
        # Generalized Fractional Anisotropy from WM ODF
        if msmt_result.wm_odf is not None:
            logger.info("Computing GFA from WM ODF")
            odf = msmt_result.wm_odf
            
            # Compute GFA: std(odf) / rms(odf)
            odf_std = np.std(odf, axis=-1)
            odf_rms = np.sqrt(np.mean(odf**2, axis=-1))
            
            # Avoid division by zero
            gfa = np.zeros_like(odf_std)
            valid_mask = odf_rms > 0
            gfa[valid_mask] = odf_std[valid_mask] / odf_rms[valid_mask]
            
            if msmt_result.mask is not None:
                gfa[msmt_result.mask == 0] = 0
            
            metrics["wm_gfa"] = np.clip(gfa, 0, 1)
        
        # Number of fiber orientations (peaks) in WM
        if msmt_result.peak_values is not None:
            logger.info("Computing number of WM fiber orientations")
            num_peaks = np.sum(msmt_result.peak_values > 0, axis=-1)
            if msmt_result.mask is not None:
                num_peaks[msmt_result.mask == 0] = 0
            
            metrics["wm_num_peaks"] = num_peaks
            
            # Primary peak amplitude in WM
            primary_peak = msmt_result.peak_values[..., 0]
            if msmt_result.mask is not None:
                primary_peak[msmt_result.mask == 0] = 0
            metrics["wm_primary_peak_amplitude"] = primary_peak
        
        logger.info("Computed %d MSMT-CSD metrics", len(metrics))
        return metrics
    
    def fit_csd(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                mask: Optional[np.ndarray] = None) -> CSDResult:
        """Fit single-shell CSD model."""
        return self.fit_model(dwi_data, bvals, bvecs, mask)
    
    def process_msmt_csd(self, dwi_data: Optional[np.ndarray] = None, bvals: Optional[np.ndarray] = None, 
                        bvecs: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None, 
                        affine: Optional[np.ndarray] = None, output_dir: Optional[Path] = None, 
                        subject_id: Optional[str] = None, session_id: Optional[str] = None,
                        dwi_path: Optional[Path] = None, bval_path: Optional[Path] = None,
                        bvec_path: Optional[Path] = None, mask_path: Optional[Path] = None,
                        core_count: int = 1, use_mrtrix: bool = True, **kwargs) -> ProcessingResult:
        """Execute MSMT-CSD processing pipeline."""
        try:
            logger.info("Starting MSMT-CSD processing")
            
            # Prefer mrtrix implementation when file paths are provided (following original implementation)
            if use_mrtrix and all(p is not None for p in [dwi_path, bval_path, bvec_path, mask_path, output_dir, subject_id]):
                logger.info("Using mrtrix for MSMT-CSD processing (following original implementation)")
                msmt_result = self.fit_msmt_csd_mrtrix(
                    dwi_path, bval_path, bvec_path, mask_path, 
                    output_dir, subject_id, core_count
                )
                
                # mrtrix implementation saves files directly, so list them
                output_files = list(output_dir.glob(f"{subject_id}_*"))
                
                metadata = {
                    "processing_method": "MSMT-CSD",
                    "implementation": "mrtrix",
                    "response_algorithm": "dhollander",
                    "core_count": core_count,
                    "subject_id": subject_id
                }
                
                logger.info("MSMT-CSD processing completed successfully using mrtrix")
                return ProcessingResult(
                    status=ProcessingStatus.COMPLETED,
                    output_files=output_files,
                    metadata=metadata
                )
            
            # Fallback to DIPY implementation
            logger.info("Using DIPY for MSMT-CSD processing (fallback)")
            
            if dwi_data is None or bvals is None or bvecs is None:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="DWI data, bvals, and bvecs are required for DIPY implementation"
                )
            
            if not self.validate_inputs(dwi_data, bvals, bvecs, mask):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Input validation failed"
                )
            
            # Check for multi-shell data
            unique_bvals = np.unique(bvals)
            high_b_shells = unique_bvals[unique_bvals >= 1000]
            if len(high_b_shells) < 2:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="MSMT-CSD requires multi-shell data with at least 2 high b-value shells"
                )
            
            if mask is None and self.config.auto_mask:
                logger.info("Creating automatic mask using median_otsu")
                _, mask = median_otsu(
                    dwi_data,
                    median_radius=self.config.mask_median_radius,
                    numpass=self.config.mask_numpass
                )
            
            msmt_result = self.fit_msmt_csd(dwi_data, bvals, bvecs, mask)
            
            output_files = []
            metadata = {
                "processing_method": "MSMT-CSD",
                "implementation": "dipy",
                "response_algorithm": self.config.response_algorithm,
                "sh_order": self.config.sh_order,
                "output_types": self.config.output_types,
                "mask_applied": mask is not None,
                "auto_mask": self.config.auto_mask,
                "num_shells": len(high_b_shells),
                "shell_bvals": high_b_shells.tolist()
            }
            
            if output_dir is not None:
                output_files = self._save_msmt_csd_outputs(
                    msmt_result, output_dir,
                    subject_id, session_id, affine
                )
            
            logger.info("MSMT-CSD processing completed successfully using DIPY")
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("MSMT-CSD processing failed: %s", str(e))
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def _run_mrtrix_command(self, command: str, log_file: Optional[Path] = None) -> bool:
        """Run an mrtrix command and handle errors."""
        try:
            logger.info("Running mrtrix command: %s", command)
            
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    process = subprocess.Popen(
                        command, 
                        universal_newlines=True, 
                        shell=True, 
                        stdout=f,
                        stderr=subprocess.STDOUT
                    )
                    output, error = process.communicate()
            else:
                process = subprocess.Popen(
                    command, 
                    universal_newlines=True, 
                    shell=True, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                output, error = process.communicate()
                
                if output:
                    logger.info("Command output: %s", output)
                if error:
                    logger.error("Command error: %s", error)
            
            if process.returncode != 0:
                logger.error("Command failed with return code %d", process.returncode)
                return False
                
            return True
            
        except Exception as e:
            logger.error("Error running mrtrix command: %s", str(e))
            return False

    def fit_msmt_csd_mrtrix(self, dwi_path: Path, bval_path: Path, bvec_path: Path,
                           mask_path: Path, output_dir: Path, 
                           subject_id: str, core_count: int = 1) -> MSMTCSDResult:
        """Fit MSMT-CSD using mrtrix commands (following original implementation)."""
        logger.info("Fitting MSMT-CSD using mrtrix for subject %s", subject_id)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"{subject_id}_MSMT-CSD_logs.txt"
        
        # Define output paths following original naming convention
        wm_response_path = output_dir / f"{subject_id}_dhollander_WM_response.txt"
        gm_response_path = output_dir / f"{subject_id}_dhollander_GM_response.txt"
        csf_response_path = output_dir / f"{subject_id}_dhollander_CSF_response.txt"
        
        wm_odf_path = output_dir / f"{subject_id}_MSMT-CSD_WM_ODF.nii.gz"
        gm_signal_path = output_dir / f"{subject_id}_MSMT-CSD_GM.nii.gz"
        csf_signal_path = output_dir / f"{subject_id}_MSMT-CSD_CSF.nii.gz"
        
        peaks_path = output_dir / f"{subject_id}_MSMT-CSD_peaks.nii.gz"
        peaks_amp_path = output_dir / f"{subject_id}_MSMT-CSD_peaks_amp.nii.gz"
        
        # Step 1: Response function estimation using dhollander algorithm
        dwi2response_cmd = (
            f'dwi2response dhollander -info '
            f'-nthreads {core_count} -fslgrad {bvec_path} {bval_path} '
            f'{dwi_path} '
            f'{wm_response_path} {gm_response_path} {csf_response_path} -force'
        )
        
        if not self._run_mrtrix_command(dwi2response_cmd, log_file):
            raise RuntimeError("dwi2response command failed")
        
        # Step 2: Multi-shell multi-tissue CSD
        dwi2fod_cmd = (
            f'dwi2fod msmt_csd -info '
            f'-nthreads {core_count} -mask {mask_path} '
            f'{dwi_path} -fslgrad {bvec_path} {bval_path} '
            f'{wm_response_path} {wm_odf_path} '
            f'{gm_response_path} {gm_signal_path} '
            f'{csf_response_path} {csf_signal_path} -force'
        )
        
        if not self._run_mrtrix_command(dwi2fod_cmd, log_file):
            raise RuntimeError("dwi2fod command failed")
        
        # Step 3: Extract peaks from WM ODF
        sh2peaks_cmd = (
            f"sh2peaks -force -nthreads {core_count} "
            f"-num 2 {wm_odf_path} {peaks_path}"
        )
        
        if not self._run_mrtrix_command(sh2peaks_cmd, log_file):
            raise RuntimeError("sh2peaks command failed")
        
        # Step 4: Convert peaks to amplitudes
        peaks2amp_cmd = (
            f"peaks2amp -force -nthreads {core_count} "
            f"{peaks_path} {peaks_amp_path}"
        )
        
        if not self._run_mrtrix_command(peaks2amp_cmd, log_file):
            raise RuntimeError("peaks2amp command failed")
        
        # Load results
        wm_odf, affine = load_nifti(str(wm_odf_path))
        gm_signal, _ = load_nifti(str(gm_signal_path))
        csf_signal, _ = load_nifti(str(csf_signal_path))
        peaks, _ = load_nifti(str(peaks_path))
        peak_values, _ = load_nifti(str(peaks_amp_path))
        mask, _ = load_nifti(str(mask_path))
        
        # Load response functions
        response_functions = {}
        for tissue, path in [("wm", wm_response_path), ("gm", gm_response_path), ("csf", csf_response_path)]:
            try:
                response_functions[tissue] = np.loadtxt(path)
            except Exception as e:
                logger.warning("Could not load %s response function: %s", tissue, str(e))
        
        # Create peak indices (mrtrix doesn't provide these directly)
        peak_indices = np.zeros(peak_values.shape, dtype=int)
        
        return MSMTCSDResult(
            wm_odf=wm_odf,
            gm_signal=gm_signal,
            csf_signal=csf_signal,
            peaks=peaks,
            peak_values=peak_values,
            peak_indices=peak_indices,
            response_functions=response_functions,
            mask=mask
        )

    def fit_msmt_csd(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> MSMTCSDResult:
        """Fit Multi-Shell Multi-Tissue CSD model using DIPY (fallback method)."""
        logger.warning("Using DIPY for MSMT-CSD. For better results, use fit_msmt_csd_mrtrix with file paths.")
        
        gtab = gradient_table(bvals, bvecs=bvecs)
        
        # Check for multi-shell data
        unique_bvals = np.unique(bvals)
        high_b_shells = unique_bvals[unique_bvals > 0]  # All non-b0 shells
        high_b_shells = high_b_shells[high_b_shells >= 1000]  # Only high b-value shells
        
        if len(high_b_shells) < 2:
            raise ValueError("MSMT-CSD requires multi-shell data with at least 2 high b-value shells")
        
        logger.info("Fitting MSMT-CSD model with %d shells", len(high_b_shells))
        
        # Create a simple mask for response estimation if none provided
        if mask is None:
            from dipy.segment.mask import median_otsu
            _, mask = median_otsu(dwi_data, median_radius=self.config.mask_median_radius,
                                numpass=self.config.mask_numpass)
        
        # Estimate tissue-specific response functions
        try:
            # Use auto response estimation for WM
            from dipy.reconst.csdeconv import auto_response_ssst
            response_wm, _ = auto_response_ssst(gtab, dwi_data, 
                                              roi_radii=self.config.roi_radii,
                                              fa_thr=self.config.fa_threshold)
            
            # For GM and CSF, we use simplified response functions
            response_gm = np.array([[0.8e-3, 0.2e-3, 0.2e-3]])  # Isotropic GM response
            response_csf = np.array([[3.0e-3, 3.0e-3, 3.0e-3]])  # Isotropic CSF response
            
        except Exception as e:
            logger.warning("Failed to estimate response functions automatically: %s", str(e))
            # Use default response functions
            response_wm = np.array([[1.5e-3, 0.3e-3, 0.3e-3]])
            response_gm = np.array([[0.8e-3, 0.8e-3, 0.8e-3]])
            response_csf = np.array([[3.0e-3, 3.0e-3, 3.0e-3]])
        
        response_functions = [response_wm, response_gm, response_csf]
        
        logger.info("Response functions estimated - WM: %s, GM: %s, CSF: %s", 
                   response_wm.shape, response_gm.shape, response_csf.shape)
        
        # Create and fit MSMT-CSD model
        msmt_model = MultiShellDeconvModel(gtab, response_functions)
        
        if mask is not None:
            msmt_fit = msmt_model.fit(dwi_data, mask=mask)
        else:
            msmt_fit = msmt_model.fit(dwi_data)
        
        # Get tissue-specific signals
        volume_fractions = msmt_fit.volume_fractions
        
        # Extract WM ODF (first tissue type)
        wm_odf = msmt_fit.odf(msmt_model.sphere)
        
        # Extract tissue volume fractions
        wm_signal = volume_fractions[..., 0] if volume_fractions.shape[-1] > 0 else np.zeros(dwi_data.shape[:3])
        gm_signal = volume_fractions[..., 1] if volume_fractions.shape[-1] > 1 else np.zeros(dwi_data.shape[:3])
        csf_signal = volume_fractions[..., 2] if volume_fractions.shape[-1] > 2 else np.zeros(dwi_data.shape[:3])
        
        # Extract peaks from WM ODF
        peaks_result = peaks_from_model(
            model=msmt_model,
            data=dwi_data,
            sphere=msmt_model.sphere,
            relative_peak_threshold=self.config.relative_peak_threshold,
            min_separation_angle=self.config.min_separation_angle,
            mask=mask,
            return_odf=False,
            return_sh=False,
            normalize_peaks=True
        )
        
        response_functions_dict = {
            "wm": response_wm,
            "gm": response_gm,
            "csf": response_csf
        }
        
        return MSMTCSDResult(
            wm_odf=wm_odf,
            gm_signal=gm_signal,
            csf_signal=csf_signal,
            peaks=peaks_result.peak_dirs,
            peak_values=peaks_result.peak_values,
            peak_indices=peaks_result.peak_indices,
            response_functions=response_functions_dict,
            mask=mask
        )
    
    def extract_peaks(self, odf_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract fiber orientation peaks from ODF data."""
        if self._model is None:
            raise ValueError("Model must be fitted before extracting peaks")
        
        # This is a simplified peak extraction - in practice you'd use
        # the full peaks_from_model function with proper sphere sampling
        logger.info("Extracting peaks from ODF data")
        
        # Find local maxima in ODF
        # This is a placeholder - real implementation would use proper peak finding
        peaks = np.zeros(odf_data.shape[:3] + (5, 3))  # Max 5 peaks per voxel
        peak_values = np.zeros(odf_data.shape[:3] + (5,))
        peak_indices = np.zeros(odf_data.shape[:3] + (5,), dtype=int)
        
        return peaks, peak_values, peak_indices
    
    def _save_outputs(self, csd_result: CSDResult, output_dir: Path,
                     subject_id: Optional[str] = None,
                     session_id: Optional[str] = None, 
                     affine: Optional[np.ndarray] = None) -> List[Path]:
        """Save CSD outputs in BIDS-compliant format."""
        output_files = []
        
        csd_dir = output_dir / "csd"
        csd_dir.mkdir(parents=True, exist_ok=True)
        
        if affine is None:
            affine = np.eye(4)
        
        prefix_parts = []
        if subject_id:
            if not subject_id.startswith("sub-"):
                subject_id = f"sub-{subject_id}"
            prefix_parts.append(subject_id)
        
        if session_id:
            if not session_id.startswith("ses-"):
                session_id = f"ses-{session_id}"
            prefix_parts.append(session_id)
        
        prefix = "_".join(prefix_parts) if prefix_parts else "csd"
        
        # Save ODF data
        if self.config.save_odf and "odf" in self.config.output_types:
            odf_path = csd_dir / f"{prefix}_model-CSD_odf.nii.gz"
            odf_img = nib.Nifti1Image(csd_result.odf_data.astype(np.float32), affine)
            nib.save(odf_img, odf_path)
            output_files.append(odf_path)
            
            odf_json = {
                "Description": "Orientation Distribution Function from CSD",
                "Model": "CSD",
                "SHOrder": self.config.sh_order,
                "ResponseAlgorithm": self.config.response_algorithm,
                "Units": "dimensionless"
            }
            json_path = odf_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(odf_json, f, indent=2)
            output_files.append(json_path)
        
        # Save peaks
        if self.config.save_peaks and "peaks" in self.config.output_types:
            peaks_path = csd_dir / f"{prefix}_model-CSD_peaks.nii.gz"
            peaks_img = nib.Nifti1Image(csd_result.peaks.astype(np.float32), affine)
            nib.save(peaks_img, peaks_path)
            output_files.append(peaks_path)
            
            peaks_json = {
                "Description": "Fiber orientation peaks from CSD",
                "Model": "CSD",
                "PeakThreshold": self.config.relative_peak_threshold,
                "MinSeparationAngle": self.config.min_separation_angle,
                "Units": "dimensionless"
            }
            json_path = peaks_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(peaks_json, f, indent=2)
            output_files.append(json_path)
        
        # Save peak values
        if "peak_values" in self.config.output_types:
            peak_values_path = csd_dir / f"{prefix}_model-CSD_peak-values.nii.gz"
            peak_values_img = nib.Nifti1Image(csd_result.peak_values.astype(np.float32), affine)
            nib.save(peak_values_img, peak_values_path)
            output_files.append(peak_values_path)
        
        # Save peak indices
        if "peak_indices" in self.config.output_types:
            peak_indices_path = csd_dir / f"{prefix}_model-CSD_peak-indices.nii.gz"
            peak_indices_img = nib.Nifti1Image(csd_result.peak_indices.astype(np.int16), affine)
            nib.save(peak_indices_img, peak_indices_path)
            output_files.append(peak_indices_path)
        
        # Save response function
        if self.config.save_response and csd_result.response_function is not None:
            response_path = csd_dir / f"{prefix}_model-CSD_response.npy"
            np.save(response_path, csd_result.response_function)
            output_files.append(response_path)
        
        # Save mask
        if csd_result.mask is not None:
            mask_path = csd_dir / f"{prefix}_model-CSD_mask.nii.gz"
            mask_img = nib.Nifti1Image(csd_result.mask.astype(np.uint8), affine)
            nib.save(mask_img, mask_path)
            output_files.append(mask_path)
        
        # Save fit quality
        if csd_result.fit_quality is not None:
            quality_path = csd_dir / f"{prefix}_model-CSD_fit-quality.nii.gz"
            quality_img = nib.Nifti1Image(csd_result.fit_quality.astype(np.float32), affine)
            nib.save(quality_img, quality_path)
            output_files.append(quality_path)
        
        logger.info("Saved %d CSD output files", len(output_files))
        return output_files
    
    def _save_msmt_csd_outputs(self, msmt_result: MSMTCSDResult, output_dir: Path,
                              subject_id: Optional[str] = None,
                              session_id: Optional[str] = None, 
                              affine: Optional[np.ndarray] = None) -> List[Path]:
        """Save MSMT-CSD outputs in BIDS-compliant format."""
        output_files = []
        
        msmt_csd_dir = output_dir / "msmt-csd"
        msmt_csd_dir.mkdir(parents=True, exist_ok=True)
        
        if affine is None:
            affine = np.eye(4)
        
        prefix_parts = []
        if subject_id:
            if not subject_id.startswith("sub-"):
                subject_id = f"sub-{subject_id}"
            prefix_parts.append(subject_id)
        
        if session_id:
            if not session_id.startswith("ses-"):
                session_id = f"ses-{session_id}"
            prefix_parts.append(session_id)
        
        prefix = "_".join(prefix_parts) if prefix_parts else "msmt-csd"
        
        # Save WM ODF
        if msmt_result.wm_odf is not None:
            wm_odf_path = msmt_csd_dir / f"{prefix}_model-MSMTCSD_tissue-WM_odf.nii.gz"
            wm_odf_img = nib.Nifti1Image(msmt_result.wm_odf.astype(np.float32), affine)
            nib.save(wm_odf_img, wm_odf_path)
            output_files.append(wm_odf_path)
            
            wm_odf_json = {
                "Description": "White matter orientation distribution function from MSMT-CSD",
                "Model": "MSMT-CSD",
                "TissueType": "WM",
                "Units": "dimensionless"
            }
            json_path = wm_odf_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(wm_odf_json, f, indent=2)
            output_files.append(json_path)
        
        # Save GM signal
        if msmt_result.gm_signal is not None:
            gm_path = msmt_csd_dir / f"{prefix}_model-MSMTCSD_tissue-GM_signal.nii.gz"
            gm_img = nib.Nifti1Image(msmt_result.gm_signal.astype(np.float32), affine)
            nib.save(gm_img, gm_path)
            output_files.append(gm_path)
            
            gm_json = {
                "Description": "Gray matter signal from MSMT-CSD",
                "Model": "MSMT-CSD",
                "TissueType": "GM",
                "Units": "dimensionless"
            }
            json_path = gm_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(gm_json, f, indent=2)
            output_files.append(json_path)
        
        # Save CSF signal
        if msmt_result.csf_signal is not None:
            csf_path = msmt_csd_dir / f"{prefix}_model-MSMTCSD_tissue-CSF_signal.nii.gz"
            csf_img = nib.Nifti1Image(msmt_result.csf_signal.astype(np.float32), affine)
            nib.save(csf_img, csf_path)
            output_files.append(csf_path)
            
            csf_json = {
                "Description": "CSF signal from MSMT-CSD",
                "Model": "MSMT-CSD",
                "TissueType": "CSF",
                "Units": "dimensionless"
            }
            json_path = csf_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(csf_json, f, indent=2)
            output_files.append(json_path)
        
        # Save peaks
        if msmt_result.peaks is not None:
            peaks_path = msmt_csd_dir / f"{prefix}_model-MSMTCSD_peaks.nii.gz"
            peaks_img = nib.Nifti1Image(msmt_result.peaks.astype(np.float32), affine)
            nib.save(peaks_img, peaks_path)
            output_files.append(peaks_path)
            
            peaks_json = {
                "Description": "Fiber orientation peaks from MSMT-CSD",
                "Model": "MSMT-CSD",
                "Units": "dimensionless"
            }
            json_path = peaks_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(peaks_json, f, indent=2)
            output_files.append(json_path)
        
        # Save peak values
        if msmt_result.peak_values is not None:
            peak_values_path = msmt_csd_dir / f"{prefix}_model-MSMTCSD_peak-values.nii.gz"
            peak_values_img = nib.Nifti1Image(msmt_result.peak_values.astype(np.float32), affine)
            nib.save(peak_values_img, peak_values_path)
            output_files.append(peak_values_path)
        
        # Save response functions
        if msmt_result.response_functions:
            for tissue, response in msmt_result.response_functions.items():
                response_path = msmt_csd_dir / f"{prefix}_model-MSMTCSD_tissue-{tissue.upper()}_response.npy"
                np.save(response_path, response)
                output_files.append(response_path)
        
        # Save mask
        if msmt_result.mask is not None:
            mask_path = msmt_csd_dir / f"{prefix}_model-MSMTCSD_mask.nii.gz"
            mask_img = nib.Nifti1Image(msmt_result.mask.astype(np.uint8), affine)
            nib.save(mask_img, mask_path)
            output_files.append(mask_path)
        
        logger.info("Saved %d MSMT-CSD output files", len(output_files))
        return output_files
    
    def generate_rgb_maps(self, peaks: np.ndarray, peak_values: np.ndarray, 
                         mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Generate RGB color maps from peaks (following original implementation)."""
        try:
            import unravel.utils
            
            rgb_maps = {}
            
            # Primary peak RGB
            if peaks.shape[-2] >= 1:
                primary_peaks = peaks[..., 0, :]
                rgb_primary = unravel.utils.peaks_to_RGB(primary_peaks)
                if mask is not None:
                    rgb_primary[mask == 0] = 0
                rgb_maps["primary_peak_rgb"] = rgb_primary
                
                # Primary peak RGB with amplitude weighting
                if peak_values is not None and peak_values.shape[-1] >= 1:
                    primary_values = peak_values[..., 0]
                    rgb_primary_weighted = unravel.utils.peaks_to_RGB(primary_peaks, primary_values)
                    if mask is not None:
                        rgb_primary_weighted[mask == 0] = 0
                    rgb_maps["primary_peak_rgb_weighted"] = rgb_primary_weighted
            
            # Secondary peak RGB
            if peaks.shape[-2] >= 2:
                secondary_peaks = peaks[..., 1, :]
                rgb_secondary = unravel.utils.peaks_to_RGB(secondary_peaks)
                if mask is not None:
                    rgb_secondary[mask == 0] = 0
                rgb_maps["secondary_peak_rgb"] = rgb_secondary
                
                # Secondary peak RGB with amplitude weighting
                if peak_values is not None and peak_values.shape[-1] >= 2:
                    secondary_values = peak_values[..., 1]
                    rgb_secondary_weighted = unravel.utils.peaks_to_RGB(secondary_peaks, secondary_values)
                    if mask is not None:
                        rgb_secondary_weighted[mask == 0] = 0
                    rgb_maps["secondary_peak_rgb_weighted"] = rgb_secondary_weighted
            
            # Combined peaks RGB
            if peaks.shape[-2] >= 2:
                combined_peaks = np.stack([peaks[..., 0, :], peaks[..., 1, :]], axis=-1)
                rgb_combined = unravel.utils.peaks_to_RGB(combined_peaks)
                if mask is not None:
                    rgb_combined[mask == 0] = 0
                rgb_maps["combined_peaks_rgb"] = rgb_combined
                
                # Combined peaks RGB with amplitude weighting
                if peak_values is not None and peak_values.shape[-1] >= 2:
                    combined_values = np.stack([peak_values[..., 0], peak_values[..., 1]], axis=-1)
                    rgb_combined_weighted = unravel.utils.peaks_to_RGB(combined_peaks, combined_values)
                    if mask is not None:
                        rgb_combined_weighted[mask == 0] = 0
                    rgb_maps["combined_peaks_rgb_weighted"] = rgb_combined_weighted
            
            logger.info("Generated %d RGB maps", len(rgb_maps))
            return rgb_maps
            
        except ImportError:
            logger.warning("unravel.utils not available, skipping RGB map generation")
            return {}
        except Exception as e:
            logger.error("Error generating RGB maps: %s", str(e))
            return {}
    
    def generate_pseudo_tensors(self, peaks: np.ndarray, peak_values: Optional[np.ndarray] = None,
                               pixdim: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Generate pseudo-tensor representations from peaks (following original implementation)."""
        try:
            from elikopy.utils import peak_to_tensor
            
            if pixdim is None:
                pixdim = np.array([1.0, 1.0, 1.0])
            
            pseudo_tensors = {}
            
            # Primary peak tensors
            if peaks.shape[-2] >= 1:
                primary_peaks = peaks[..., 0, :]
                
                # Unnormalized tensor
                tensor_p1 = peak_to_tensor(primary_peaks, norm=None, pixdim=pixdim)
                pseudo_tensors["primary_peak_tensor"] = tensor_p1
                
                # Normalized tensor (if peak values available)
                if peak_values is not None and peak_values.shape[-1] >= 1:
                    primary_values = peak_values[..., 0]
                    tensor_p1_normed = peak_to_tensor(primary_peaks, norm=primary_values, pixdim=pixdim)
                    pseudo_tensors["primary_peak_tensor_normed"] = tensor_p1_normed
            
            # Secondary peak tensors
            if peaks.shape[-2] >= 2:
                secondary_peaks = peaks[..., 1, :]
                
                # Unnormalized tensor
                tensor_p2 = peak_to_tensor(secondary_peaks, norm=None, pixdim=pixdim)
                pseudo_tensors["secondary_peak_tensor"] = tensor_p2
                
                # Normalized tensor (if peak values available)
                if peak_values is not None and peak_values.shape[-1] >= 2:
                    secondary_values = peak_values[..., 1]
                    tensor_p2_normed = peak_to_tensor(secondary_peaks, norm=secondary_values, pixdim=pixdim)
                    pseudo_tensors["secondary_peak_tensor_normed"] = tensor_p2_normed
            
            logger.info("Generated %d pseudo-tensors", len(pseudo_tensors))
            return pseudo_tensors
            
        except ImportError:
            logger.warning("elikopy.utils.peak_to_tensor not available, skipping pseudo-tensor generation")
            return {}
        except Exception as e:
            logger.error("Error generating pseudo-tensors: %s", str(e))
            return {}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate CSD configuration parameters."""
        try:
            # Check response algorithm
            valid_algorithms = ["auto", "manual", "fa", "tournier"]
            if config.get("response_algorithm", "auto") not in valid_algorithms:
                logger.error("Invalid response algorithm: %s", config.get("response_algorithm"))
                return False
            
            # Check SH order
            sh_order = config.get("sh_order", 8)
            if not isinstance(sh_order, int) or sh_order < 2 or sh_order % 2 != 0:
                logger.error("SH order must be a positive even integer, got: %s", sh_order)
                return False
            
            # Check thresholds
            peak_threshold = config.get("relative_peak_threshold", 0.5)
            if not 0 < peak_threshold < 1:
                logger.error("Peak threshold must be between 0 and 1, got: %s", peak_threshold)
                return False
            
            min_angle = config.get("min_separation_angle", 25)
            if not 0 < min_angle < 90:
                logger.error("Min separation angle must be between 0 and 90 degrees, got: %s", min_angle)
                return False
            
            # Check output types
            valid_outputs = ["odf", "peaks", "peak_indices", "peak_values", "response"]
            output_types = config.get("output_types", ["odf", "peaks"])
            if not all(ot in valid_outputs for ot in output_types):
                logger.error("Invalid output types: %s", output_types)
                return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating config: %s", str(e))
            return False
    
    def process_csd_solo(self, dwi_path: Path, bval_path: Path, bvec_path: Path,
                        mask_path: Path, output_dir: Path, subject_id: str,
                        num_peaks: int = 2, peaks_threshold: float = 0.25,
                        csd_bvalue: Optional[int] = None, core_count: int = 1,
                        csd_fa_threshold: float = 0.7, return_odf: bool = False) -> ProcessingResult:
        """
        Process single-shell CSD following the original implementation pattern.
        
        This method closely follows the original odf_csd_solo function.
        """
        try:
            logger.info("Starting CSD processing for subject %s", subject_id)
            
            # Load data
            dwi_data, affine = load_nifti(str(dwi_path))
            bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
            mask, _ = load_nifti(str(mask_path))
            
            # Filter by b-value if specified (following original implementation)
            if csd_bvalue is not None:
                logger.info("Filtering data for b-value: %d", csd_bvalue)
                b0_threshold = np.min(bvals) + 10
                b0_threshold = max(50, b0_threshold)
                
                sel_b = np.logical_or(
                    bvals == 0, 
                    np.logical_and((csd_bvalue - 5) <= bvals, bvals <= (csd_bvalue + 5))
                )
                dwi_data = dwi_data[..., sel_b]
                gtab = gradient_table(bvals[sel_b], bvecs[sel_b], b0_threshold=b0_threshold)
            else:
                b0_threshold = np.min(bvals) + 10
                b0_threshold = max(50, b0_threshold)
                gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
            
            # Estimate response function
            response, ratio = auto_response_ssst(
                gtab, dwi_data, 
                roi_radii=self.config.roi_radii,
                fa_thr=csd_fa_threshold
            )
            logger.info("Response function estimated with ratio: %.3f", ratio)
            
            # Fit CSD model
            csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=self.config.sh_order)
            csd_peaks = peaks_from_model(
                npeaks=num_peaks, 
                model=csd_model, 
                data=dwi_data, 
                sphere=csd_model.sphere,
                relative_peak_threshold=peaks_threshold, 
                min_separation_angle=self.config.min_separation_angle, 
                parallel=False, 
                mask=mask,
                normalize_peaks=True,
                return_odf=return_odf,
                return_sh=True
            )
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = []
            
            # Save main outputs (following original naming convention)
            peaks_path = output_dir / f"{subject_id}_CSD_peaks.nii.gz"
            save_nifti(str(peaks_path), csd_peaks.peak_dirs, affine)
            output_files.append(peaks_path)
            
            values_path = output_dir / f"{subject_id}_CSD_values.nii.gz"
            save_nifti(str(values_path), csd_peaks.peak_values, affine)
            output_files.append(values_path)
            
            if return_odf:
                odf_path = output_dir / f"{subject_id}_CSD_ODF.nii.gz"
                save_nifti(str(odf_path), csd_peaks.odf, affine)
                output_files.append(odf_path)
            
            sh_path = output_dir / f"{subject_id}_CSD_SH_ODF.nii.gz"
            save_nifti(str(sh_path), csd_peaks.shm_coeff, affine)
            output_files.append(sh_path)
            
            # Generate pseudo-tensors (following original implementation)
            img_peaks = nib.load(str(peaks_path))
            img_values = nib.load(str(values_path))
            hdr = img_peaks.header
            pixdim = hdr['pixdim'][1:4]
            
            pseudo_tensors = self.generate_pseudo_tensors(
                csd_peaks.peak_dirs, csd_peaks.peak_values, pixdim
            )
            
            # Save pseudo-tensors with proper header
            tensor_hdr = hdr.copy()
            tensor_hdr['dim'][0] = 5  # 4 scalar, 5 vector
            tensor_hdr['dim'][4] = 1  # 3
            tensor_hdr['dim'][5] = 6  # 1
            tensor_hdr['regular'] = b'r'
            tensor_hdr['intent_code'] = 1005
            
            for tensor_name, tensor_data in pseudo_tensors.items():
                tensor_path = output_dir / f"{subject_id}_CSD_{tensor_name}.nii.gz"
                save_nifti(str(tensor_path), tensor_data, affine, tensor_hdr)
                output_files.append(tensor_path)
            
            # Generate RGB maps (following original implementation)
            rgb_maps = self.generate_rgb_maps(csd_peaks.peak_dirs, csd_peaks.peak_values, mask)
            
            for rgb_name, rgb_data in rgb_maps.items():
                rgb_path = output_dir / f"{subject_id}_CSD_{rgb_name}.nii.gz"
                save_nifti(str(rgb_path), rgb_data, affine)
                output_files.append(rgb_path)
            
            metadata = {
                "processing_method": "CSD",
                "implementation": "dipy",
                "subject_id": subject_id,
                "num_peaks": num_peaks,
                "peaks_threshold": peaks_threshold,
                "csd_bvalue": csd_bvalue,
                "csd_fa_threshold": csd_fa_threshold,
                "return_odf": return_odf,
                "response_ratio": ratio
            }
            
            logger.info("CSD processing completed for subject %s", subject_id)
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("CSD processing failed for subject %s: %s", subject_id, str(e))
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def process_msmt_csd_solo(self, dwi_path: Path, bval_path: Path, bvec_path: Path,
                             mask_path: Path, output_dir: Path, subject_id: str,
                             core_count: int = 1, num_peaks: int = 2, 
                             peaks_threshold: float = 0.25) -> ProcessingResult:
        """
        Process MSMT-CSD following the original implementation pattern using mrtrix.
        
        This method closely follows the original odf_msmtcsd_solo function.
        """
        return self.process_msmt_csd(
            dwi_path=dwi_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            mask_path=mask_path,
            output_dir=output_dir,
            subject_id=subject_id,
            core_count=core_count,
            use_mrtrix=True
        )
    
    def _save_msmt_csd_outputs(self, msmt_result: MSMTCSDResult, output_dir: Path,
                              subject_id: Optional[str] = None,
                              session_id: Optional[str] = None,
                              affine: Optional[np.ndarray] = None) -> List[Path]:
        """Save MSMT-CSD outputs in BIDS-compliant format."""
        output_files = []
        
        msmtcsd_dir = output_dir / "msmtcsd"
        msmtcsd_dir.mkdir(parents=True, exist_ok=True)
        
        if affine is None:
            affine = np.eye(4)
        
        prefix_parts = []
        if subject_id:
            if not subject_id.startswith("sub-"):
                subject_id = f"sub-{subject_id}"
            prefix_parts.append(subject_id)
        
        if session_id:
            if not session_id.startswith("ses-"):
                session_id = f"ses-{session_id}"
            prefix_parts.append(session_id)
        
        prefix = "_".join(prefix_parts) if prefix_parts else "msmtcsd"
        
        # Save WM ODF
        if self.config.save_odf and "odf" in self.config.output_types:
            wm_odf_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_wm-odf.nii.gz"
            wm_odf_img = nib.Nifti1Image(msmt_result.wm_odf.astype(np.float32), affine)
            nib.save(wm_odf_img, wm_odf_path)
            output_files.append(wm_odf_path)
            
            wm_odf_json = {
                "Description": "White matter Orientation Distribution Function from MSMT-CSD",
                "Model": "MSMT-CSD",
                "TissueType": "WhiteMatter",
                "SHOrder": self.config.sh_order,
                "ResponseAlgorithm": self.config.response_algorithm,
                "Units": "dimensionless"
            }
            json_path = wm_odf_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(wm_odf_json, f, indent=2)
            output_files.append(json_path)
        
        # Save GM signal
        gm_signal_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_gm-signal.nii.gz"
        gm_signal_img = nib.Nifti1Image(msmt_result.gm_signal.astype(np.float32), affine)
        nib.save(gm_signal_img, gm_signal_path)
        output_files.append(gm_signal_path)
        
        gm_signal_json = {
            "Description": "Gray matter volume fraction from MSMT-CSD",
            "Model": "MSMT-CSD",
            "TissueType": "GrayMatter",
            "Parameter": "VolumeFraction",
            "Units": "dimensionless"
        }
        json_path = gm_signal_path.with_suffix('').with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(gm_signal_json, f, indent=2)
        output_files.append(json_path)
        
        # Save CSF signal
        csf_signal_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_csf-signal.nii.gz"
        csf_signal_img = nib.Nifti1Image(msmt_result.csf_signal.astype(np.float32), affine)
        nib.save(csf_signal_img, csf_signal_path)
        output_files.append(csf_signal_path)
        
        csf_signal_json = {
            "Description": "Cerebrospinal fluid volume fraction from MSMT-CSD",
            "Model": "MSMT-CSD",
            "TissueType": "CerebrospinalFluid",
            "Parameter": "VolumeFraction",
            "Units": "dimensionless"
        }
        json_path = csf_signal_path.with_suffix('').with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(csf_signal_json, f, indent=2)
        output_files.append(json_path)
        
        # Save peaks from WM ODF
        if self.config.save_peaks and "peaks" in self.config.output_types:
            peaks_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_peaks.nii.gz"
            peaks_img = nib.Nifti1Image(msmt_result.peaks.astype(np.float32), affine)
            nib.save(peaks_img, peaks_path)
            output_files.append(peaks_path)
            
            peaks_json = {
                "Description": "Fiber orientation peaks from MSMT-CSD white matter ODF",
                "Model": "MSMT-CSD",
                "TissueType": "WhiteMatter",
                "PeakThreshold": self.config.relative_peak_threshold,
                "MinSeparationAngle": self.config.min_separation_angle,
                "Units": "dimensionless"
            }
            json_path = peaks_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(peaks_json, f, indent=2)
            output_files.append(json_path)
        
        # Save peak values
        if "peak_values" in self.config.output_types:
            peak_values_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_peak-values.nii.gz"
            peak_values_img = nib.Nifti1Image(msmt_result.peak_values.astype(np.float32), affine)
            nib.save(peak_values_img, peak_values_path)
            output_files.append(peak_values_path)
        
        # Save peak indices
        if "peak_indices" in self.config.output_types:
            peak_indices_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_peak-indices.nii.gz"
            peak_indices_img = nib.Nifti1Image(msmt_result.peak_indices.astype(np.int16), affine)
            nib.save(peak_indices_img, peak_indices_path)
            output_files.append(peak_indices_path)
        
        # Save response functions
        if self.config.save_response and msmt_result.response_functions is not None:
            for tissue_type, response in msmt_result.response_functions.items():
                response_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_{tissue_type}-response.npy"
                np.save(response_path, response)
                output_files.append(response_path)
        
        # Save mask
        if msmt_result.mask is not None:
            mask_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_mask.nii.gz"
            mask_img = nib.Nifti1Image(msmt_result.mask.astype(np.uint8), affine)
            nib.save(mask_img, mask_path)
            output_files.append(mask_path)
        
        # Compute and save MSMT-CSD specific metrics
        msmt_metrics = self.compute_msmt_csd_metrics(msmt_result)
        for metric_name, metric_data in msmt_metrics.items():
            metric_path = msmtcsd_dir / f"{prefix}_model-MSMTCSD_parameter-{metric_name}.nii.gz"
            metric_img = nib.Nifti1Image(metric_data.astype(np.float32), affine)
            nib.save(metric_img, metric_path)
            output_files.append(metric_path)
            
            # Add JSON metadata for metrics
            metric_json = {
                "Description": f"MSMT-CSD-derived {metric_name}",
                "Model": "MSMT-CSD",
                "Parameter": metric_name,
                "Units": "dimensionless"
            }
            json_path = metric_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metric_json, f, indent=2)
            output_files.append(json_path)
        
        logger.info("Saved %d MSMT-CSD output files", len(output_files))
        return output_files
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component."""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info("Updated CSD config: %s = %s", key, value)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "response_algorithm": "auto",
            "sh_order": 8,
            "relative_peak_threshold": 0.5,
            "min_separation_angle": 25,
            "output_types": ["odf", "peaks", "peak_indices", "peak_values"],
            "auto_mask": True,
            "mask_median_radius": 4,
            "mask_numpass": 1,
            "fa_threshold": 0.7,
            "roi_radii": 10,
            "roi_center": None,
            "save_response": True,
            "save_odf": True,
            "save_peaks": True,
            "msmt_response_estimation": "auto",
            "tissue_types": ["wm", "gm", "csf"],
            "save_tissue_fractions": True,
            "compute_tissue_metrics": True
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        try:
            valid_response_algorithms = ["auto", "tournier", "tax", "dhollander"]
            
            if "response_algorithm" in config:
                if config["response_algorithm"] not in valid_response_algorithms:
                    logger.error("Invalid response_algorithm: %s", config["response_algorithm"])
                    return False
            
            if "sh_order" in config:
                sh_order = config["sh_order"]
                if not isinstance(sh_order, int) or sh_order < 2 or sh_order > 12 or sh_order % 2 != 0:
                    logger.error("sh_order must be even integer between 2 and 12")
                    return False
            
            valid_output_types = ["odf", "peaks", "peak_indices", "peak_values"]
            if "output_types" in config:
                if not isinstance(config["output_types"], list):
                    logger.error("output_types must be a list")
                    return False
                
                for output_type in config["output_types"]:
                    if output_type not in valid_output_types:
                        logger.error("Invalid output_type: %s", output_type)
                        return False
            
            numeric_params = {
                "relative_peak_threshold": (0.0, 1.0),
                "min_separation_angle": (0.0, 90.0),
                "fa_threshold": (0.0, 1.0),
                "roi_radii": (1, 50),
                "mask_median_radius": (1, 10),
                "mask_numpass": (1, 5)
            }
            
            for param, (min_val, max_val) in numeric_params.items():
                if param in config:
                    value = config[param]
                    if not isinstance(value, (int, float)):
                        logger.error("%s must be numeric", param)
                        return False
                    if not (min_val <= value <= max_val):
                        logger.error("%s must be between %s and %s", param, min_val, max_val)
                        return False
            
            bool_params = ["auto_mask", "save_response", "save_odf", "save_peaks", 
                          "save_tissue_fractions", "compute_tissue_metrics"]
            for param in bool_params:
                if param in config:
                    if not isinstance(config[param], bool):
                        logger.error("%s must be boolean", param)
                        return False
            
            # Validate MSMT-CSD specific parameters
            valid_msmt_response_methods = ["auto", "manual", "dhollander"]
            if "msmt_response_estimation" in config:
                if config["msmt_response_estimation"] not in valid_msmt_response_methods:
                    logger.error("Invalid msmt_response_estimation: %s", 
                               config["msmt_response_estimation"])
                    return False
            
            valid_tissue_types = ["wm", "gm", "csf"]
            if "tissue_types" in config:
                if not isinstance(config["tissue_types"], list):
                    logger.error("tissue_types must be a list")
                    return False
                
                for tissue_type in config["tissue_types"]:
                    if tissue_type not in valid_tissue_types:
                        logger.error("Invalid tissue_type: %s", tissue_type)
                        return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating configuration: %s", str(e))
            return False