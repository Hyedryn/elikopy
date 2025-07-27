"""
DTI processing module for ElikoPy
===============================

This module provides functionality for DTI processing including tensor fitting,
metrics computation, and BIDS-compliant output generation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity, \
    axial_diffusivity, radial_diffusivity, geodesic_anisotropy
from dipy.reconst.base import ReconstModel
from dipy.segment.mask import median_otsu

from elikopy.core.base import ProcessingComponent, ProcessingResult, ProcessingStatus, \
    ConfigurableComponent, ModelProcessor, ValidationResult, ValidationError, ValidationWarning


logger = logging.getLogger(__name__)


@dataclass
class DTIConfig:
    """Configuration for DTI processing"""
    fit_method: str = "WLS"
    mask_threshold: float = 0.0
    fa_threshold: float = 0.2
    min_signal: float = 1e-6
    output_metrics: List[str] = field(default_factory=lambda: ["FA", "MD", "AD", "RD", "GA", "RGB"])
    auto_mask: bool = True
    mask_median_radius: int = 4
    mask_numpass: int = 1
    save_tensor: bool = True
    save_eigenvalues: bool = False
    save_eigenvectors: bool = False


@dataclass
class DTIResult:
    """Result of DTI tensor fitting"""
    tensor_data: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    fit_quality: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


@dataclass
class DTIMetrics:
    """DTI scalar metrics"""
    fa: Optional[np.ndarray] = None
    md: Optional[np.ndarray] = None
    ad: Optional[np.ndarray] = None
    rd: Optional[np.ndarray] = None
    ga: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None


class DTIProcessor(ModelProcessor):
    """
    Processor for DTI model fitting and metrics computation.
    """
    
    def __init__(self, config: Optional[DTIConfig] = None):
        """Initialize DTI processor."""
        self.config = config or DTIConfig()
        
        # Validate configuration during initialization
        config_dict = {
            "fit_method": self.config.fit_method,
            "mask_threshold": self.config.mask_threshold,
            "fa_threshold": self.config.fa_threshold,
            "min_signal": self.config.min_signal,
            "output_metrics": self.config.output_metrics
        }
        
        if not self.validate_config(config_dict):
            raise ValueError("Invalid DTI configuration provided")
        
        self._model: Optional[TensorModel] = None
        logger.info(f"DTI processor initialized with fit method: {self.config.fit_method}")
    
    def validate_inputs(self, dwi_data: np.ndarray, bvals: np.ndarray, 
                       bvecs: np.ndarray, mask: Optional[np.ndarray] = None, subject: Optional[str] = None) -> bool:
        """Validate inputs before processing."""
        try:
            if dwi_data.ndim != 4:
                logger.error(f"DWI data must be 4D, got {dwi_data.ndim}D")
                return False
            
            if bvals.ndim != 1:
                logger.error(f"B-values must be 1D array, got {bvals.ndim}D")
                return False
            
            if len(bvals) != dwi_data.shape[3]:
                logger.error(f"Number of b-values does not match DWI volumes")
                return False
            
            if bvecs.ndim != 2:
                logger.error(f"B-vectors must be 2D array, got {bvecs.ndim}D")
                return False
            
            if mask is not None and mask.shape != dwi_data.shape[:3]:
                logger.error(f"Mask shape does not match DWI spatial dimensions")
                return False
            
            unique_bvals = np.unique(bvals)
            if len(unique_bvals) < 2:
                logger.error("DTI requires at least 2 different b-values")
                return False
            
            logger.info("Input validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during input validation: {str(e)}")
            return False
    
    def process(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                mask: Optional[np.ndarray] = None, affine: Optional[np.ndarray] = None,
                output_dir: Optional[Path] = None, subject_id: Optional[str] = None,
                session_id: Optional[str] = None, **kwargs) -> ProcessingResult:
        """Execute DTI processing pipeline."""
        try:
            logger.info("Starting DTI processing")
            
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
            
            dti_result = self.fit_model(dwi_data, bvals, bvecs, mask)
            dti_metrics = self.compute_metrics(dti_result)
            
            output_files = []
            metadata = {
                "processing_method": "DTI",
                "fit_method": self.config.fit_method,
                "output_metrics": self.config.output_metrics,
                "mask_applied": mask is not None,
                "auto_mask": self.config.auto_mask
            }
            
            if output_dir is not None:
                output_files = self._save_outputs(
                    dti_result, dti_metrics, output_dir, 
                    subject_id, session_id, affine
                )
            
            logger.info("DTI processing completed successfully")
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"DTI processing failed: {str(e)}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def fit_model(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> DTIResult:
        """Fit diffusion tensor model to data."""
        gtab = gradient_table(bvals, bvecs=bvecs)
        self._model = TensorModel(gtab, fit_method=self.config.fit_method)
        
        logger.info(f"Fitting tensor model using {self.config.fit_method} method")
        
        if mask is not None:
            tensor_fit = self._model.fit(dwi_data, mask=mask)
        else:
            tensor_fit = self._model.fit(dwi_data)
        
        tensor_data = tensor_fit.quadratic_form
        eigenvalues = tensor_fit.evals
        eigenvectors = tensor_fit.evecs
        
        fit_quality = None
        if hasattr(tensor_fit, 'predict') and mask is not None:
            try:
                predicted = tensor_fit.predict(gtab, S0=1.0)
                residuals = dwi_data - predicted
                fit_quality = np.sum(residuals**2, axis=-1)
                fit_quality[mask == 0] = 0
            except Exception as e:
                logger.warning(f"Could not compute fit quality: {str(e)}")
        
        return DTIResult(
            tensor_data=tensor_data,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            fit_quality=fit_quality,
            mask=mask
        )
    
    def compute_metrics(self, dti_result: DTIResult) -> DTIMetrics:
        """Compute DTI scalar metrics from fitted tensor."""
        metrics = DTIMetrics()
        evals = dti_result.eigenvalues
        evecs = dti_result.eigenvectors
        mask = dti_result.mask
        
        for metric in self.config.output_metrics:
            logger.info(f"Computing {metric} metric")
            
            if metric == "FA":
                fa = fractional_anisotropy(evals)
                if mask is not None:
                    fa[mask == 0] = 0
                fa = np.clip(fa, 0, 1)
                metrics.fa = fa
                
            elif metric == "MD":
                md = mean_diffusivity(evals)
                if mask is not None:
                    md[mask == 0] = 0
                md = np.maximum(md, 0)
                metrics.md = md
                
            elif metric == "AD":
                ad = axial_diffusivity(evals)
                if mask is not None:
                    ad[mask == 0] = 0
                ad = np.maximum(ad, 0)
                metrics.ad = ad
                
            elif metric == "RD":
                rd = radial_diffusivity(evals)
                if mask is not None:
                    rd[mask == 0] = 0
                rd = np.maximum(rd, 0)
                metrics.rd = rd
                
            elif metric == "GA":
                ga = geodesic_anisotropy(evals)
                if mask is not None:
                    ga[mask == 0] = 0
                ga = np.clip(ga, 0, 1)
                metrics.ga = ga
                
            elif metric == "RGB":
                if metrics.fa is None:
                    fa = fractional_anisotropy(evals)
                    if mask is not None:
                        fa[mask == 0] = 0
                    fa = np.clip(fa, 0, 1)
                else:
                    fa = metrics.fa
                
                # Create RGB color map based on principal eigenvector direction
                # RGB = FA * |primary_eigenvector|
                rgb = np.zeros(evecs.shape[:3] + (3,))
                rgb[..., 0] = fa * np.abs(evecs[..., 0, 0])  # Red = FA * |v1_x|
                rgb[..., 1] = fa * np.abs(evecs[..., 1, 0])  # Green = FA * |v1_y|
                rgb[..., 2] = fa * np.abs(evecs[..., 2, 0])  # Blue = FA * |v1_z|
                
                if mask is not None:
                    rgb[mask == 0] = 0
                metrics.rgb = rgb
                
            else:
                logger.warning(f"Unknown metric requested: {metric}")
        
        return metrics
    
    def _save_outputs(self, dti_result: DTIResult, dti_metrics: DTIMetrics,
                     output_dir: Path, subject_id: Optional[str] = None,
                     session_id: Optional[str] = None, 
                     affine: Optional[np.ndarray] = None) -> List[Path]:
        """Save DTI outputs in BIDS-compliant format."""
        output_files = []
        
        dti_dir = output_dir / "dti"
        dti_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        prefix = "_".join(prefix_parts) if prefix_parts else "dti"
        
        if self.config.save_tensor:
            tensor_path = dti_dir / f"{prefix}_model-DTI_tensor.nii.gz"
            tensor_img = nib.Nifti1Image(dti_result.tensor_data.astype(np.float32), affine)
            nib.save(tensor_img, tensor_path)
            output_files.append(tensor_path)
            
            tensor_json = {
                "Description": "Diffusion tensor data",
                "Model": "DTI",
                "FitMethod": self.config.fit_method,
                "Units": "mm²/s"
            }
            json_path = tensor_path.with_suffix('').with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(tensor_json, f, indent=2)
            output_files.append(json_path)
        
        if self.config.save_eigenvalues:
            evals_path = dti_dir / f"{prefix}_model-DTI_eigenvalues.nii.gz"
            evals_img = nib.Nifti1Image(dti_result.eigenvalues.astype(np.float32), affine)
            nib.save(evals_img, evals_path)
            output_files.append(evals_path)
        
        if self.config.save_eigenvectors:
            evecs_path = dti_dir / f"{prefix}_model-DTI_eigenvectors.nii.gz"
            evecs_img = nib.Nifti1Image(dti_result.eigenvectors.astype(np.float32), affine)
            nib.save(evecs_img, evecs_path)
            output_files.append(evecs_path)
        
        metric_info = {
            "FA": {"description": "Fractional Anisotropy", "units": "dimensionless", "range": "[0, 1]"},
            "MD": {"description": "Mean Diffusivity", "units": "mm²/s", "range": "[0, inf)"},
            "AD": {"description": "Axial Diffusivity", "units": "mm²/s", "range": "[0, inf)"},
            "RD": {"description": "Radial Diffusivity", "units": "mm²/s", "range": "[0, inf)"},
            "GA": {"description": "Geodesic Anisotropy", "units": "dimensionless", "range": "[0, 1]"},
            "RGB": {"description": "RGB Color FA", "units": "dimensionless", "range": "[0, 1]"}
        }
        
        for metric in self.config.output_metrics:
            metric_data = getattr(dti_metrics, metric.lower(), None)
            if metric_data is not None:
                metric_path = dti_dir / f"{prefix}_model-DTI_parameter-{metric}.nii.gz"
                metric_img = nib.Nifti1Image(metric_data.astype(np.float32), affine)
                nib.save(metric_img, metric_path)
                output_files.append(metric_path)
                
                if metric in metric_info:
                    info = metric_info[metric]
                    metric_json = {
                        "Description": info["description"],
                        "Model": "DTI",
                        "Parameter": metric,
                        "FitMethod": self.config.fit_method,
                        "Units": info["units"],
                        "ValidRange": info["range"]
                    }
                    json_path = metric_path.with_suffix('').with_suffix('.json')
                    with open(json_path, 'w') as f:
                        json.dump(metric_json, f, indent=2)
                    output_files.append(json_path)
        
        if dti_result.mask is not None:
            mask_path = dti_dir / f"{prefix}_model-DTI_mask.nii.gz"
            mask_img = nib.Nifti1Image(dti_result.mask.astype(np.uint8), affine)
            nib.save(mask_img, mask_path)
            output_files.append(mask_path)
        
        if dti_result.fit_quality is not None:
            quality_path = dti_dir / f"{prefix}_model-DTI_fit-quality.nii.gz"
            quality_img = nib.Nifti1Image(dti_result.fit_quality.astype(np.float32), affine)
            nib.save(quality_img, quality_path)
            output_files.append(quality_path)
        
        logger.info(f"Saved {len(output_files)} DTI output files")
        return output_files
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component."""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated DTI config: {key} = {value}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "fit_method": "WLS",
            "mask_threshold": 0.0,
            "fa_threshold": 0.2,
            "min_signal": 1e-6,
            "output_metrics": ["FA", "MD", "AD", "RD", "GA", "RGB"],
            "auto_mask": True,
            "mask_median_radius": 4,
            "mask_numpass": 1,
            "save_tensor": True,
            "save_eigenvalues": False,
            "save_eigenvectors": False
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        try:
            valid_fit_methods = ["OLS", "WLS", "NLLS", "RESTORE"]
            
            if "fit_method" in config:
                if config["fit_method"] not in valid_fit_methods:
                    logger.error(f"Invalid fit_method: {config['fit_method']}")
                    return False
            
            valid_metrics = ["FA", "MD", "AD", "RD", "GA", "RGB"]
            
            if "output_metrics" in config:
                if not isinstance(config["output_metrics"], list):
                    logger.error("output_metrics must be a list")
                    return False
                
                for metric in config["output_metrics"]:
                    if metric not in valid_metrics:
                        logger.error(f"Invalid metric: {metric}")
                        return False
            
            numeric_params = {
                "mask_threshold": (0.0, float('inf')),
                "fa_threshold": (0.0, 1.0),
                "min_signal": (0.0, float('inf')),
                "mask_median_radius": (1, 10),
                "mask_numpass": (1, 5)
            }
            
            for param, (min_val, max_val) in numeric_params.items():
                if param in config:
                    value = config[param]
                    if not isinstance(value, (int, float)):
                        logger.error(f"{param} must be numeric")
                        return False
                    if not (min_val <= value <= max_val):
                        logger.error(f"{param} must be between {min_val} and {max_val}")
                        return False
            
            bool_params = ["auto_mask", "save_tensor", "save_eigenvalues", "save_eigenvectors"]
            
            for param in bool_params:
                if param in config:
                    if not isinstance(config[param], bool):
                        logger.error(f"{param} must be boolean")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return False