"""
Microstructure fingerprinting processing module for ElikoPy
=========================================================

This module provides functionality for microstructure fingerprinting analysis
using the microstructure-fingerprinting library, following the original implementation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti, save_nifti

from elikopy.core.base import ModelProcessor, ProcessingResult, ProcessingStatus


logger = logging.getLogger(__name__)


@dataclass
class FingerprintingConfig:
    """Configuration for microstructure fingerprinting processing"""
    dictionary_path: Optional[Path] = None
    peaks_type: str = "MSMT-CSD"  # "MSMT-CSD", "CSD", "DIAMOND"
    use_parallel_processing: bool = True
    core_count: int = 4
    csf_mask: bool = True
    ear_mask: bool = False
    verbose: int = 3
    output_metrics: List[str] = field(default_factory=lambda: [
        "frac_f0", "fvf_tot", "MSE", "R2", "peaks", "fractions", "fvf"
    ])
    save_quality_control: bool = True
    save_rgb_maps: bool = True
    save_pseudo_tensors: bool = True
    color_order: str = "rgb"  # "rgb", "brg"


@dataclass
class FingerprintingDictionary:
    """Microstructure fingerprinting dictionary"""
    model: Any  # MFModel instance
    path: Path
    metadata: Dict[str, Any]


@dataclass
class FingerprintingResult:
    """Result of microstructure fingerprinting"""
    frac_f0: np.ndarray  # Fraction of first fascicle
    fvf_tot: np.ndarray  # Total fiber volume fraction
    mse: np.ndarray  # Mean squared error
    r2: np.ndarray  # R² coefficient of determination
    peaks: Optional[Dict[str, np.ndarray]] = None  # Peak directions per fascicle
    fractions: Optional[Dict[str, np.ndarray]] = None  # Fractions per fascicle
    fvf: Optional[Dict[str, np.ndarray]] = None  # FVF per fascicle
    numfasc: Optional[np.ndarray] = None  # Number of fascicles
    mask: Optional[np.ndarray] = None
    fit_object: Optional[Any] = None  # Original MF fit object


class MicrostructureFingerprintingProcessor(ModelProcessor):
    """
    Processor for microstructure fingerprinting analysis.
    
    This processor implements microstructure fingerprinting using the 
    microstructure-fingerprinting library following the original implementation.
    It requires peak directions from CSD, MSMT-CSD, or DIAMOND processing.
    """
    
    def __init__(self, config: Optional[FingerprintingConfig] = None):
        """Initialize microstructure fingerprinting processor."""
        self.config = config or FingerprintingConfig()
        self.dictionary = None
        logger.info("Microstructure fingerprinting processor initialized")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the fingerprinting processor."""
        if 'dictionary_path' in config:
            self.config.dictionary_path = Path(config['dictionary_path'])
        if 'peaks_type' in config:
            self.config.peaks_type = config['peaks_type']
        if 'use_parallel_processing' in config:
            self.config.use_parallel_processing = config['use_parallel_processing']
        if 'core_count' in config:
            self.config.core_count = config['core_count']
        if 'csf_mask' in config:
            self.config.csf_mask = config['csf_mask']
        if 'ear_mask' in config:
            self.config.ear_mask = config['ear_mask']
        if 'verbose' in config:
            self.config.verbose = config['verbose']
        if 'output_metrics' in config:
            self.config.output_metrics = config['output_metrics']
        if 'color_order' in config:
            self.config.color_order = config['color_order']
        
        # Reset dictionary to force reload with new parameters
        self.dictionary = None
        logger.info("Microstructure fingerprinting processor reconfigured")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default fingerprinting configuration."""
        return {
            'dictionary_path': None,
            'peaks_type': 'MSMT-CSD',
            'use_parallel_processing': True,
            'core_count': 4,
            'csf_mask': True,
            'ear_mask': False,
            'verbose': 3,
            'output_metrics': [
                'frac_f0', 'fvf_tot', 'MSE', 'R2', 'peaks', 'fractions', 'fvf'
            ],
            'save_quality_control': True,
            'save_rgb_maps': True,
            'save_pseudo_tensors': True,
            'color_order': 'rgb'
        }
    
    def fit_model(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Fit fingerprinting model to data (interface compliance method)."""
        # This method requires additional peak information, so it's a simplified interface
        # The full implementation is in fit_fingerprinting
        if mask is None:
            mask = np.ones(dwi_data.shape[:3], dtype=bool)
        
        # Create dummy peaks for interface compliance
        peaks = np.random.randn(*dwi_data.shape[:3], 3, 2)  # 2 fascicles
        numfasc = np.ones(dwi_data.shape[:3], dtype=int)
        
        fingerprinting_result = self.fit_fingerprinting(
            dwi_data, bvals, bvecs, mask, peaks, numfasc
        )
        
        # Convert to dictionary format
        model_params = {
            'frac_f0': fingerprinting_result.frac_f0,
            'fvf_tot': fingerprinting_result.fvf_tot,
            'mse': fingerprinting_result.mse,
            'r2': fingerprinting_result.r2
        }
        
        if fingerprinting_result.peaks:
            model_params.update(fingerprinting_result.peaks)
        if fingerprinting_result.fractions:
            model_params.update(fingerprinting_result.fractions)
        
        return model_params
    
    def validate_inputs(self, dwi_data: np.ndarray, bvals: np.ndarray, 
                       bvecs: np.ndarray, mask: Optional[np.ndarray] = None,
                       peaks: Optional[np.ndarray] = None,
                       numfasc: Optional[np.ndarray] = None, subject: Optional[str] = None) -> bool:
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
            
            if peaks is not None and peaks.shape[:3] != dwi_data.shape[:3]:
                logger.error("Peaks shape does not match DWI spatial dimensions")
                return False
            
            if numfasc is not None and numfasc.shape != dwi_data.shape[:3]:
                logger.error("Numfasc shape does not match DWI spatial dimensions")
                return False
            
            # Check dictionary path
            if self.config.dictionary_path is None:
                logger.error("Dictionary path is required for fingerprinting")
                return False
            
            if not self.config.dictionary_path.exists():
                logger.error("Dictionary path does not exist: %s", self.config.dictionary_path)
                return False
            
            # Check peaks type
            valid_peaks_types = ["MSMT-CSD", "CSD", "DIAMOND"]
            if self.config.peaks_type not in valid_peaks_types:
                logger.error("Invalid peaks type: %s", self.config.peaks_type)
                return False
            
            logger.info("Input validation passed")
            return True
            
        except Exception as e:
            logger.error("Error during input validation: %s", str(e))
            return False
    
    def load_dictionary(self, dictionary_path: Path) -> FingerprintingDictionary:
        """Load fingerprinting dictionary."""
        try:
            logger.info("Loading fingerprinting dictionary from %s", dictionary_path)
            
            # Import microstructure fingerprinting library
            try:
                import microstructure_fingerprinting as mf
            except ImportError as e:
                raise ImportError(
                    "microstructure-fingerprinting library is required. "
                    "Please install it with: pip install microstructure-fingerprinting-rensonnetg"
                ) from e
            
            # Load the model
            mf_model = mf.MFModel(str(dictionary_path))
            
            dictionary = FingerprintingDictionary(
                model=mf_model,
                path=dictionary_path,
                metadata={
                    'dictionary_path': str(dictionary_path),
                    'loaded_at': str(np.datetime64('now'))
                }
            )
            
            logger.info("Dictionary loaded successfully")
            return dictionary
            
        except Exception as e:
            logger.error("Failed to load dictionary: %s", str(e))
            raise
    
    def fit_fingerprinting(self, dwi_data: np.ndarray, bvals: np.ndarray, 
                          bvecs: np.ndarray, mask: np.ndarray,
                          peaks: np.ndarray, numfasc: np.ndarray) -> FingerprintingResult:
        """Fit microstructure fingerprinting model to data."""
        logger.info("Fitting microstructure fingerprinting model")
        
        # Load dictionary if not already loaded
        if self.dictionary is None:
            self.dictionary = self.load_dictionary(self.config.dictionary_path)
        
        try:
            # Import microstructure fingerprinting library
            import microstructure_fingerprinting as mf
            
            # Determine parallel processing
            parallel = self.config.use_parallel_processing and self.config.core_count > 1
            
            # Fit the model (following original implementation)
            logger.info("Starting fingerprinting fit with parallel=%s", parallel)
            mf_fit = self.dictionary.model.fit(
                dwi_data, mask, numfasc, peaks=peaks,
                bvals=bvals, bvecs=bvecs,
                csf_mask=self.config.csf_mask,
                ear_mask=self.config.ear_mask,
                verbose=self.config.verbose,
                parallel=parallel
            )
            
            # Extract results (following original implementation)
            frac_f0 = mf_fit.frac_f0
            fvf_tot = mf_fit.fvf_tot
            mse = mf_fit.MSE
            r2 = mf_fit.R2
            
            logger.info("Fingerprinting fitting completed successfully")
            
            return FingerprintingResult(
                frac_f0=frac_f0,
                fvf_tot=fvf_tot,
                mse=mse,
                r2=r2,
                numfasc=numfasc,
                mask=mask,
                fit_object=mf_fit
            )
            
        except Exception as e:
            logger.error("Fingerprinting fitting failed: %s", str(e))
            raise
    
    def compute_metrics(self, model_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute derived metrics from fingerprinting results."""
        metrics = {}
        
        # Basic fingerprinting metrics
        for metric_name in self.config.output_metrics:
            if metric_name in model_params:
                metrics[metric_name] = model_params[metric_name]
        
        # Additional derived metrics
        if 'fvf_tot' in model_params and 'frac_f0' in model_params:
            # Compute fascicle-specific FVF
            fvf_f0 = model_params['fvf_tot'] * model_params['frac_f0']
            fvf_f1 = model_params['fvf_tot'] * (1 - model_params['frac_f0'])
            metrics['fvf_f0'] = fvf_f0
            metrics['fvf_f1'] = fvf_f1
        
        logger.info("Computed %d fingerprinting metrics", len(metrics))
        return metrics
    
    def process(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                mask: Optional[np.ndarray] = None, peaks: Optional[np.ndarray] = None,
                numfasc: Optional[np.ndarray] = None, affine: Optional[np.ndarray] = None,
                output_dir: Optional[Path] = None, subject_id: Optional[str] = None,
                session_id: Optional[str] = None, **kwargs) -> ProcessingResult:
        """Execute fingerprinting processing pipeline."""
        try:
            logger.info("Starting microstructure fingerprinting processing")
            
            if not self.validate_inputs(dwi_data, bvals, bvecs, mask, peaks, numfasc):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Input validation failed"
                )
            
            if mask is None:
                mask = np.ones(dwi_data.shape[:3], dtype=bool)
            
            if peaks is None or numfasc is None:
                logger.error("Peaks and numfasc are required for fingerprinting")
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Peaks and numfasc are required"
                )
            
            fingerprinting_result = self.fit_fingerprinting(
                dwi_data, bvals, bvecs, mask, peaks, numfasc
            )
            
            output_files = []
            metadata = {
                "processing_method": "Microstructure Fingerprinting",
                "dictionary_path": str(self.config.dictionary_path),
                "peaks_type": self.config.peaks_type,
                "use_parallel_processing": self.config.use_parallel_processing,
                "core_count": self.config.core_count,
                "csf_mask": self.config.csf_mask,
                "ear_mask": self.config.ear_mask
            }
            
            if output_dir is not None:
                output_files = self._save_outputs(
                    fingerprinting_result, output_dir,
                    subject_id, session_id, affine
                )
            
            logger.info("Microstructure fingerprinting processing completed successfully")
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("Microstructure fingerprinting processing failed: %s", str(e))
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def _save_outputs(self, fingerprinting_result: FingerprintingResult, output_dir: Path,
                     subject_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     affine: Optional[np.ndarray] = None) -> List[Path]:
        """Save fingerprinting outputs in BIDS-compliant format."""
        output_files = []
        
        fingerprinting_dir = output_dir / "fingerprinting"
        fingerprinting_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        prefix = "_".join(prefix_parts) if prefix_parts else "fingerprinting"
        
        # Save main fingerprinting metrics
        metric_descriptions = {
            "frac_f0": "Fraction of first fascicle from microstructure fingerprinting",
            "fvf_tot": "Total fiber volume fraction from microstructure fingerprinting",
            "mse": "Mean squared error from microstructure fingerprinting fitting",
            "r2": "R² coefficient of determination from microstructure fingerprinting fitting"
        }
        
        for metric_name in ["frac_f0", "fvf_tot", "mse", "r2"]:
            if hasattr(fingerprinting_result, metric_name):
                metric_data = getattr(fingerprinting_result, metric_name)
                
                # Save metric
                metric_path = fingerprinting_dir / f"{prefix}_model-MF_parameter-{metric_name}.nii.gz"
                metric_img = nib.Nifti1Image(metric_data.astype(np.float32), affine)
                nib.save(metric_img, metric_path)
                output_files.append(metric_path)
                
                # Save JSON metadata
                metric_json = {
                    "Description": metric_descriptions.get(metric_name, f"MF {metric_name}"),
                    "Model": "Microstructure Fingerprinting",
                    "Parameter": metric_name,
                    "Units": "dimensionless"
                }
                json_path = metric_path.with_suffix('').with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metric_json, f, indent=2)
                output_files.append(json_path)
        
        # Save individual fascicle outputs if available from fit object
        if fingerprinting_result.fit_object is not None:
            output_files.extend(self._save_fascicle_outputs(
                fingerprinting_result.fit_object, fingerprinting_dir,
                prefix, affine
            ))
        
        # Save mask
        if fingerprinting_result.mask is not None:
            mask_path = fingerprinting_dir / f"{prefix}_model-MF_mask.nii.gz"
            mask_img = nib.Nifti1Image(fingerprinting_result.mask.astype(np.uint8), affine)
            nib.save(mask_img, mask_path)
            output_files.append(mask_path)
        
        logger.info("Saved %d fingerprinting output files", len(output_files))
        return output_files
    
    def _save_fascicle_outputs(self, mf_fit: Any, output_dir: Path,
                              prefix: str, affine: np.ndarray) -> List[Path]:
        """Save individual fascicle outputs following original implementation."""
        output_files = []
        
        try:
            # Save individual fascicle outputs (following original implementation)
            frac = 0
            while hasattr(mf_fit, f'peak_f{frac}') and hasattr(mf_fit, f'frac_f{frac}'):
                # Save peaks
                peaks_data = getattr(mf_fit, f'peak_f{frac}')
                peaks_path = output_dir / f"{prefix}_model-MF_fascicle-{frac}_peaks.nii.gz"
                peaks_img = nib.Nifti1Image(peaks_data.astype(np.float32), affine)
                nib.save(peaks_img, peaks_path)
                output_files.append(peaks_path)
                
                # Save fractions
                frac_data = getattr(mf_fit, f'frac_f{frac}')
                frac_path = output_dir / f"{prefix}_model-MF_fascicle-{frac}_fraction.nii.gz"
                frac_img = nib.Nifti1Image(frac_data.astype(np.float32), affine)
                nib.save(frac_img, frac_path)
                output_files.append(frac_path)
                
                # Save FVF if available
                if hasattr(mf_fit, f'fvf_f{frac}'):
                    fvf_data = getattr(mf_fit, f'fvf_f{frac}')
                    fvf_path = output_dir / f"{prefix}_model-MF_fascicle-{frac}_fvf.nii.gz"
                    fvf_img = nib.Nifti1Image(fvf_data.astype(np.float32), affine)
                    nib.save(fvf_img, fvf_path)
                    output_files.append(fvf_path)
                
                # Generate RGB maps if enabled
                if self.config.save_rgb_maps:
                    rgb_files = self._generate_rgb_maps(
                        peaks_data, frac_data, output_dir, prefix, frac, affine
                    )
                    output_files.extend(rgb_files)
                
                # Generate pseudo tensors if enabled
                if self.config.save_pseudo_tensors:
                    tensor_files = self._generate_pseudo_tensors(
                        peaks_data, frac_data, output_dir, prefix, frac, affine
                    )
                    output_files.extend(tensor_files)
                
                frac += 1
            
        except Exception as e:
            logger.warning("Failed to save fascicle outputs: %s", str(e))
        
        return output_files
    
    def _generate_rgb_maps(self, peaks_data: np.ndarray, frac_data: np.ndarray,
                          output_dir: Path, prefix: str, fascicle_idx: int,
                          affine: np.ndarray) -> List[Path]:
        """Generate RGB maps from peaks (following original implementation)."""
        output_files = []
        
        try:
            import unravel.utils
            
            # Generate RGB map
            rgb_peaks = unravel.utils.peaks_to_RGB(peaks_data, order=self.config.color_order)
            rgb_path = output_dir / f"{prefix}_model-MF_fascicle-{fascicle_idx}_RGB.nii.gz"
            rgb_img = nib.Nifti1Image(rgb_peaks.astype(np.float32), affine)
            nib.save(rgb_img, rgb_path)
            output_files.append(rgb_path)
            
        except ImportError:
            logger.warning("unravel package not available, skipping RGB map generation")
        except Exception as e:
            logger.warning("Failed to generate RGB maps: %s", str(e))
        
        return output_files
    
    def _generate_pseudo_tensors(self, peaks_data: np.ndarray, frac_data: np.ndarray,
                                output_dir: Path, prefix: str, fascicle_idx: int,
                                affine: np.ndarray) -> List[Path]:
        """Generate pseudo tensors from peaks (following original implementation)."""
        output_files = []
        
        try:
            from elikopy.utils import peak_to_tensor
            
            # Generate pseudo tensor
            tensor = peak_to_tensor(peaks_data, norm=None, pixdim=[2, 2, 2])
            tensor_normed = peak_to_tensor(peaks_data, norm=frac_data, pixdim=[2, 2, 2])
            
            # Create proper header for tensor data
            tensor_path = output_dir / f"{prefix}_model-MF_fascicle-{fascicle_idx}_pseudoTensor.nii.gz"
            tensor_normed_path = output_dir / f"{prefix}_model-MF_fascicle-{fascicle_idx}_pseudoTensor_normed.nii.gz"
            
            # Save tensors
            save_nifti(str(tensor_path), tensor.astype(np.float32), affine)
            save_nifti(str(tensor_normed_path), tensor_normed.astype(np.float32), affine)
            
            output_files.extend([tensor_path, tensor_normed_path])
            
        except ImportError:
            logger.warning("Required utilities not available, skipping pseudo tensor generation")
        except Exception as e:
            logger.warning("Failed to generate pseudo tensors: %s", str(e))
        
        return output_files
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate fingerprinting configuration parameters."""
        try:
            # Check dictionary path
            if 'dictionary_path' in config:
                dict_path = Path(config['dictionary_path'])
                if not dict_path.exists():
                    logger.error("Dictionary path does not exist: %s", dict_path)
                    return False
            
            # Check peaks type
            if 'peaks_type' in config:
                valid_peaks_types = ["MSMT-CSD", "CSD", "DIAMOND"]
                if config['peaks_type'] not in valid_peaks_types:
                    logger.error("Invalid peaks type: %s", config['peaks_type'])
                    return False
            
            # Check core count
            if 'core_count' in config:
                core_count = config['core_count']
                if not isinstance(core_count, int) or core_count < 1:
                    logger.error("core_count must be a positive integer, got: %s", core_count)
                    return False
            
            # Check verbose level
            if 'verbose' in config:
                verbose = config['verbose']
                if not isinstance(verbose, int) or verbose < 0:
                    logger.error("verbose must be a non-negative integer, got: %s", verbose)
                    return False
            
            # Check color order
            if 'color_order' in config:
                valid_orders = ["rgb", "brg", "gbr", "grb", "rbg", "bgr"]
                if config['color_order'] not in valid_orders:
                    logger.error("Invalid color order: %s", config['color_order'])
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating config: %s", str(e))
            return False