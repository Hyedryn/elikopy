"""
NODDI processing module for ElikoPy
=================================

This module provides functionality for NODDI (Neurite Orientation Dispersion and Density Imaging)
processing using dmipy, following the original implementation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from elikopy.core.base import ModelProcessor, ProcessingResult, ProcessingStatus


logger = logging.getLogger(__name__)


@dataclass
class NODDIConfig:
    """Configuration for NODDI processing"""
    lambda_iso_diff: float = 3.0e-9  # Isotropic diffusivity (m²/s)
    lambda_par_diff: float = 1.7e-9  # Parallel diffusivity (m²/s)
    use_amico: bool = False  # Use AMICO optimizer
    use_parallel_processing: bool = True
    number_of_processors: int = 4
    solver: str = "brute2fine"  # Solver method
    maxiter: int = 300  # Maximum iterations
    output_metrics: List[str] = field(default_factory=lambda: [
        "mu", "odi", "fiso", "fbundle", "fintra", "icvf", "fextra", "mse", "R2"
    ])
    save_quality_control: bool = True
    auto_mask: bool = True
    mask_median_radius: int = 4
    mask_numpass: int = 1


@dataclass
class NODDIResult:
    """Result of NODDI model fitting"""
    mu: np.ndarray  # Principal fiber direction
    odi: np.ndarray  # Orientation dispersion index
    fiso: np.ndarray  # Isotropic volume fraction
    fbundle: np.ndarray  # Bundle volume fraction
    fintra: np.ndarray  # Intra-cellular volume fraction
    icvf: np.ndarray  # Intra-cellular volume fraction (thresholded)
    fextra: np.ndarray  # Extra-cellular volume fraction
    mse: np.ndarray  # Mean squared error
    R2: np.ndarray  # R² coefficient of determination
    mask: Optional[np.ndarray] = None
    fitted_parameters: Optional[Dict[str, np.ndarray]] = None


class NODDIProcessor(ModelProcessor):
    """
    Processor for NODDI model fitting and metrics computation.
    
    This processor implements NODDI using dmipy following the original implementation.
    NODDI models tissue microstructure using a multi-compartment model with:
    - Isotropic compartment (CSF/free water)
    - Watson-distributed bundle (intra- and extra-cellular)
    """
    
    def __init__(self, config: Optional[NODDIConfig] = None):
        """Initialize NODDI processor."""
        self.config = config or NODDIConfig()
        self._noddi_model = None
        logger.info("NODDI processor initialized with lambda_iso=%.2e, lambda_par=%.2e", 
                   self.config.lambda_iso_diff, self.config.lambda_par_diff)
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the NODDI processor."""
        if 'lambda_iso_diff' in config:
            self.config.lambda_iso_diff = config['lambda_iso_diff']
        if 'lambda_par_diff' in config:
            self.config.lambda_par_diff = config['lambda_par_diff']
        if 'use_amico' in config:
            self.config.use_amico = config['use_amico']
        if 'number_of_processors' in config:
            self.config.number_of_processors = config['number_of_processors']
        if 'solver' in config:
            self.config.solver = config['solver']
        if 'maxiter' in config:
            self.config.maxiter = config['maxiter']
        if 'output_metrics' in config:
            self.config.output_metrics = config['output_metrics']
        
        # Reset model to force recreation with new parameters
        self._noddi_model = None
        logger.info("NODDI processor reconfigured")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default NODDI configuration."""
        return {
            'lambda_iso_diff': 3.0e-9,
            'lambda_par_diff': 1.7e-9,
            'use_amico': False,
            'use_parallel_processing': True,
            'number_of_processors': 4,
            'solver': 'brute2fine',
            'maxiter': 300,
            'output_metrics': [
                'mu', 'odi', 'fiso', 'fbundle', 'fintra', 'icvf', 'fextra', 'mse', 'R2'
            ],
            'save_quality_control': True,
            'auto_mask': True,
            'mask_median_radius': 4,
            'mask_numpass': 1
        }
    
    def fit_model(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Fit NODDI model to data (interface compliance method)."""
        noddi_result = self.fit_noddi(dwi_data, bvals, bvecs, mask)
        
        # Convert NODDIResult to dictionary format expected by interface
        model_params = {}
        for attr in ['mu', 'odi', 'fiso', 'fbundle', 'fintra', 'icvf', 'fextra', 'mse', 'R2']:
            if hasattr(noddi_result, attr):
                model_params[attr] = getattr(noddi_result, attr)
        
        if noddi_result.fitted_parameters:
            model_params.update(noddi_result.fitted_parameters)
        
        return model_params
    
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
            
            # Check for sufficient b-values for NODDI
            unique_bvals = np.unique(bvals)
            high_b_shells = unique_bvals[unique_bvals >= 1000]
            if len(high_b_shells) < 1:
                logger.error("NODDI requires at least one high b-value shell (>=1000 s/mm²)")
                return False
            
            # Check for sufficient directions
            high_b_indices = np.where(bvals >= 1000)[0]
            if len(high_b_indices) < 30:
                logger.warning("NODDI typically requires at least 30 high b-value directions for reliable results")
            
            logger.info("Input validation passed")
            return True
            
        except Exception as e:
            logger.error("Error during input validation: %s", str(e))
            return False
    
    def process(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                mask: Optional[np.ndarray] = None, affine: Optional[np.ndarray] = None,
                output_dir: Optional[Path] = None, subject_id: Optional[str] = None,
                session_id: Optional[str] = None, **kwargs) -> ProcessingResult:
        """Execute NODDI processing pipeline."""
        try:
            logger.info("Starting NODDI processing")
            
            if not self.validate_inputs(dwi_data, bvals, bvecs, mask):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Input validation failed"
                )
            
            if mask is None and self.config.auto_mask:
                logger.info("Creating automatic mask using median_otsu")
                from dipy.segment.mask import median_otsu
                _, mask = median_otsu(
                    dwi_data, 
                    median_radius=self.config.mask_median_radius,
                    numpass=self.config.mask_numpass
                )
            
            noddi_result = self.fit_noddi(dwi_data, bvals, bvecs, mask)
            
            output_files = []
            metadata = {
                "processing_method": "NODDI",
                "lambda_iso_diff": self.config.lambda_iso_diff,
                "lambda_par_diff": self.config.lambda_par_diff,
                "use_amico": self.config.use_amico,
                "solver": self.config.solver,
                "maxiter": self.config.maxiter,
                "mask_applied": mask is not None,
                "auto_mask": self.config.auto_mask
            }
            
            if output_dir is not None:
                output_files = self._save_outputs(
                    noddi_result, output_dir, 
                    subject_id, session_id, affine
                )
            
            logger.info("NODDI processing completed successfully")
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("NODDI processing failed: %s", str(e))
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def _create_noddi_model(self):
        """Create NODDI model using dmipy (following original implementation)."""
        try:
            # Initialize the compartments model
            from dmipy.signal_models import cylinder_models, gaussian_models
            ball = gaussian_models.G1Ball()
            stick = cylinder_models.C1Stick()
            zeppelin = gaussian_models.G2Zeppelin()
            
            # Watson distribution of stick and Zeppelin
            from dmipy.distributions.distribute_models import SD1WatsonDistributed
            watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
            watson_dispersed_bundle.set_tortuous_parameter(
                'G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0'
            )
            watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
            watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', self.config.lambda_par_diff)
            
            # Build the NODDI model
            from dmipy.core.modeling_framework import MultiCompartmentModel
            noddi_model = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
            
            # Fix the isotropic diffusivity
            noddi_model.set_fixed_parameter('G1Ball_1_lambda_iso', self.config.lambda_iso_diff)
            
            logger.info("NODDI model created successfully")
            return noddi_model
            
        except ImportError as e:
            logger.error("dmipy not available: %s", str(e))
            raise ImportError("dmipy is required for NODDI processing. Please install it with: pip install dmipy")
        except Exception as e:
            logger.error("Error creating NODDI model: %s", str(e))
            raise

    def fit_noddi(self, dwi_data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, 
                 mask: Optional[np.ndarray] = None) -> NODDIResult:
        """Fit NODDI model to data (following original implementation)."""
        logger.info("Fitting NODDI model")
        
        # Create NODDI model if not already created
        if self._noddi_model is None:
            self._noddi_model = self._create_noddi_model()
        
        # Transform bvals, bvecs to dmipy format
        b0_threshold = np.min(bvals) + 10
        b0_threshold = max(50, b0_threshold)
        gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
        
        try:
            from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
            acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)
        except ImportError:
            raise ImportError("dmipy is required for NODDI processing")
        
        # Fit the model
        if self.config.use_amico:
            logger.info("Using AMICO optimizer for NODDI fitting")
            try:
                from dmipy.optimizers import amico_cvxpy
                noddi_fit = amico_cvxpy.AmicoCvxpyOptimizer(acq_scheme_dmipy, dwi_data, mask=mask)
            except ImportError:
                logger.warning("AMICO optimizer not available, falling back to standard fitting")
                noddi_fit = self._noddi_model.fit(
                    acq_scheme_dmipy, dwi_data, mask=mask,
                    use_parallel_processing=self.config.use_parallel_processing,
                    number_of_processors=self.config.number_of_processors
                )
        else:
            logger.info("Using standard optimizer for NODDI fitting")
            noddi_fit = self._noddi_model.fit(
                acq_scheme_dmipy, dwi_data, mask=mask,
                use_parallel_processing=self.config.use_parallel_processing,
                number_of_processors=self.config.number_of_processors
            )
        
        # Extract the metrics (following original implementation)
        fitted_parameters = noddi_fit.fitted_parameters
        
        mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
        odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
        fiso = fitted_parameters["partial_volume_0"]
        fbundle = fitted_parameters["partial_volume_1"]
        fintra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * 
                 fitted_parameters['partial_volume_1'])
        icvf = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * 
               (fitted_parameters['partial_volume_1'] > 0.05))
        fextra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * 
                 fitted_parameters['partial_volume_1'])
        
        # Compute fit quality metrics
        mse = noddi_fit.mean_squared_error(dwi_data)
        R2 = noddi_fit.R2_coefficient_of_determination(dwi_data)
        
        logger.info("NODDI fitting completed successfully")
        
        return NODDIResult(
            mu=mu,
            odi=odi,
            fiso=fiso,
            fbundle=fbundle,
            fintra=fintra,
            icvf=icvf,
            fextra=fextra,
            mse=mse,
            R2=R2,
            mask=mask,
            fitted_parameters=fitted_parameters
        )
    
    def compute_metrics(self, noddi_result: NODDIResult) -> Dict[str, np.ndarray]:
        """Compute derived metrics from NODDI results."""
        metrics = {}
        
        # Basic NODDI metrics are already computed in fit_noddi
        for metric_name in self.config.output_metrics:
            if hasattr(noddi_result, metric_name):
                metric_data = getattr(noddi_result, metric_name)
                if noddi_result.mask is not None:
                    metric_data = metric_data.copy()
                    metric_data[noddi_result.mask == 0] = 0
                metrics[metric_name] = metric_data
        
        # Additional derived metrics
        if hasattr(noddi_result, 'fintra') and hasattr(noddi_result, 'fextra'):
            # Total tissue fraction (should be close to 1 - fiso)
            total_tissue = noddi_result.fintra + noddi_result.fextra
            if noddi_result.mask is not None:
                total_tissue[noddi_result.mask == 0] = 0
            metrics["total_tissue_fraction"] = total_tissue
        
        # Neurite density index (NDI) - same as ICVF
        if hasattr(noddi_result, 'icvf'):
            ndi = noddi_result.icvf.copy()
            if noddi_result.mask is not None:
                ndi[noddi_result.mask == 0] = 0
            metrics["ndi"] = ndi
        
        logger.info("Computed %d NODDI metrics", len(metrics))
        return metrics
    
    def _save_outputs(self, noddi_result: NODDIResult, output_dir: Path,
                     subject_id: Optional[str] = None,
                     session_id: Optional[str] = None, 
                     affine: Optional[np.ndarray] = None) -> List[Path]:
        """Save NODDI outputs in BIDS-compliant format."""
        output_files = []
        
        noddi_dir = output_dir / "noddi"
        noddi_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        prefix = "_".join(prefix_parts) if prefix_parts else "noddi"
        
        # Save all NODDI metrics
        metric_descriptions = {
            "mu": "Principal fiber direction from NODDI",
            "odi": "Orientation dispersion index from NODDI",
            "fiso": "Isotropic volume fraction from NODDI",
            "fbundle": "Bundle volume fraction from NODDI",
            "fintra": "Intra-cellular volume fraction from NODDI",
            "icvf": "Intra-cellular volume fraction (thresholded) from NODDI",
            "fextra": "Extra-cellular volume fraction from NODDI",
            "mse": "Mean squared error from NODDI fitting",
            "R2": "R² coefficient of determination from NODDI fitting"
        }
        
        for metric_name in self.config.output_metrics:
            if hasattr(noddi_result, metric_name):
                metric_data = getattr(noddi_result, metric_name)
                
                # Save metric
                metric_path = noddi_dir / f"{prefix}_model-NODDI_parameter-{metric_name}.nii.gz"
                metric_img = nib.Nifti1Image(metric_data.astype(np.float32), affine)
                nib.save(metric_img, metric_path)
                output_files.append(metric_path)
                
                # Save JSON metadata
                metric_json = {
                    "Description": metric_descriptions.get(metric_name, f"NODDI {metric_name}"),
                    "Model": "NODDI",
                    "Parameter": metric_name,
                    "Units": "dimensionless"
                }
                json_path = metric_path.with_suffix('').with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metric_json, f, indent=2)
                output_files.append(json_path)
        
        # Save mask
        if noddi_result.mask is not None:
            mask_path = noddi_dir / f"{prefix}_model-NODDI_mask.nii.gz"
            mask_img = nib.Nifti1Image(noddi_result.mask.astype(np.uint8), affine)
            nib.save(mask_img, mask_path)
            output_files.append(mask_path)
        
        logger.info("Saved %d NODDI output files", len(output_files))
        return output_files
    
    def process_noddi_solo(self, dwi_path: Path, bval_path: Path, bvec_path: Path,
                          mask_path: Path, output_dir: Path, subject_id: str,
                          mask_type: str = "brain_mask_dilated",
                          lambda_iso_diff: float = 3.0e-9, lambda_par_diff: float = 1.7e-9,
                          use_amico: bool = False, core_count: int = 1) -> ProcessingResult:
        """
        Process NODDI following the original implementation pattern.
        
        This method closely follows the original noddi_solo function.
        """
        try:
            logger.info("Starting NODDI processing for subject %s", subject_id)
            
            # Update config with provided parameters
            self.config.lambda_iso_diff = lambda_iso_diff
            self.config.lambda_par_diff = lambda_par_diff
            self.config.use_amico = use_amico
            self.config.number_of_processors = core_count
            
            # Load data
            dwi_data, affine = load_nifti(str(dwi_path))
            bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
            
            # Load mask
            if mask_path.exists():
                mask, _ = load_nifti(str(mask_path))
            else:
                # Fallback to brain mask
                fallback_mask = mask_path.parent / f"{subject_id}_brain_mask_dilated.nii.gz"
                if fallback_mask.exists():
                    mask, _ = load_nifti(str(fallback_mask))
                else:
                    logger.warning("No mask found, using auto-masking")
                    mask = None
            
            # Fit NODDI model
            noddi_result = self.fit_noddi(dwi_data, bvals, bvecs, mask)
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = []
            
            # Save outputs following original naming convention
            for metric_name in self.config.output_metrics:
                if hasattr(noddi_result, metric_name):
                    metric_data = getattr(noddi_result, metric_name)
                    metric_path = output_dir / f"{subject_id}_noddi_{metric_name}.nii.gz"
                    save_nifti(str(metric_path), metric_data.astype(np.float32), affine)
                    output_files.append(metric_path)
            
            # Generate quality control if enabled
            if self.config.save_quality_control:
                qc_files = self._generate_quality_control(
                    noddi_result, output_dir, subject_id, affine
                )
                output_files.extend(qc_files)
            
            metadata = {
                "processing_method": "NODDI",
                "implementation": "dmipy",
                "subject_id": subject_id,
                "lambda_iso_diff": lambda_iso_diff,
                "lambda_par_diff": lambda_par_diff,
                "use_amico": use_amico,
                "core_count": core_count,
                "mask_type": mask_type
            }
            
            logger.info("NODDI processing completed for subject %s", subject_id)
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("NODDI processing failed for subject %s: %s", subject_id, str(e))
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def _generate_quality_control(self, noddi_result: NODDIResult, output_dir: Path,
                                 subject_id: str, affine: np.ndarray) -> List[Path]:
        """Generate quality control plots (following original implementation)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            qc_dir = output_dir / "quality_control"
            qc_dir.mkdir(parents=True, exist_ok=True)
            
            output_files = []
            
            # Create quality control plots following original implementation
            metric1 = noddi_result.odi.copy()
            metric2 = noddi_result.fiso.copy()
            mse = noddi_result.mse
            R2 = noddi_result.R2
            
            # Title plot
            fig, axs = plt.subplots(2, 1, figsize=(2, 1))
            fig.suptitle('Elikopy : Quality control report - NODDI', fontsize=50)
            axs[0].set_axis_off()
            axs[1].set_axis_off()
            title_path = qc_dir / "title.jpg"
            plt.savefig(title_path, dpi=300, bbox_inches='tight')
            plt.close()
            output_files.append(title_path)
            
            # Error plots
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            sl = mse.shape[2] // 2
            
            # MSE plot
            plot_mse = np.zeros((mse.shape[0], mse.shape[1] * 5))
            plot_mse[:, 0:mse.shape[1]] = mse[..., max(sl - 10, 0)]
            plot_mse[:, mse.shape[1]:(mse.shape[1] * 2)] = mse[..., max(sl - 5, 0)]
            plot_mse[:, (mse.shape[1] * 2):(mse.shape[1] * 3)] = mse[..., sl]
            plot_mse[:, (mse.shape[1] * 3):(mse.shape[1] * 4)] = mse[..., min(sl + 5, 2*sl-1)]
            plot_mse[:, (mse.shape[1] * 4):(mse.shape[1] * 5)] = mse[..., min(sl + 10, 2*sl-1)]
            
            im0 = axs[0].imshow(plot_mse, cmap='gray')
            axs[0].set_title('MSE')
            axs[0].set_axis_off()
            fig.colorbar(im0, ax=axs[0], orientation='horizontal')
            
            # R2 plot
            plot_R2 = np.zeros((R2.shape[0], R2.shape[1] * 5))
            plot_R2[:, 0:R2.shape[1]] = R2[..., max(sl - 10, 0)]
            plot_R2[:, R2.shape[1]:(R2.shape[1] * 2)] = R2[..., max(sl - 5, 0)]
            plot_R2[:, (R2.shape[1] * 2):(R2.shape[1] * 3)] = R2[..., sl]
            plot_R2[:, (R2.shape[1] * 3):(R2.shape[1] * 4)] = R2[..., min(sl + 5, 2*sl-1)]
            plot_R2[:, (R2.shape[1] * 4):(R2.shape[1] * 5)] = R2[..., min(sl + 10, 2*sl-1)]
            
            im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
            axs[1].set_title('R2')
            axs[1].set_axis_off()
            fig.colorbar(im1, ax=axs[1], orientation='horizontal')
            
            plt.tight_layout()
            error_path = qc_dir / "error.jpg"
            plt.savefig(error_path, dpi=300, bbox_inches='tight')
            plt.close()
            output_files.append(error_path)
            
            # Metrics plots
            fig, axs = plt.subplots(2, 1, figsize=(12, 6))
            sl = metric1.shape[2] // 2
            
            # ODI plot
            plot_metric1 = np.zeros((metric1.shape[0], metric1.shape[1] * 5))
            plot_metric1[:, 0:metric1.shape[1]] = metric1[..., max(sl - 10, 0)]
            plot_metric1[:, metric1.shape[1]:(metric1.shape[1] * 2)] = metric1[..., max(sl - 5, 0)]
            plot_metric1[:, (metric1.shape[1] * 2):(metric1.shape[1] * 3)] = metric1[..., sl]
            plot_metric1[:, (metric1.shape[1] * 3):(metric1.shape[1] * 4)] = metric1[..., min(sl + 5, 2*sl-1)]
            plot_metric1[:, (metric1.shape[1] * 4):(metric1.shape[1] * 5)] = metric1[..., min(sl + 10, 2*sl-1)]
            
            axs[0].imshow(plot_metric1, cmap='gray')
            axs[0].set_title('Orientation dispersion index')
            axs[0].set_axis_off()
            
            # FISO plot
            plot_metric2 = np.zeros((metric2.shape[0], metric2.shape[1] * 5))
            plot_metric2[:, 0:metric2.shape[1]] = metric2[..., max(sl - 10, 0)]
            plot_metric2[:, metric2.shape[1]:(metric2.shape[1] * 2)] = metric2[..., max(sl - 5, 0)]
            plot_metric2[:, (metric2.shape[1] * 2):(metric2.shape[1] * 3)] = metric2[..., sl]
            plot_metric2[:, (metric2.shape[1] * 3):(metric2.shape[1] * 4)] = metric2[..., min(sl + 5, 2*sl-1)]
            plot_metric2[:, (metric2.shape[1] * 4):(metric2.shape[1] * 5)] = metric2[..., min(sl + 10, 2*sl-1)]
            
            axs[1].imshow(plot_metric2, cmap='gray')
            axs[1].set_title('Fraction iso')
            axs[1].set_axis_off()
            
            plt.tight_layout()
            metrics_path = qc_dir / "metrics.jpg"
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
            plt.close()
            output_files.append(metrics_path)
            
            logger.info("Generated %d quality control files", len(output_files))
            return output_files
            
        except Exception as e:
            logger.warning("Failed to generate quality control plots: %s", str(e))
            return []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate NODDI configuration parameters."""
        try:
            # Check lambda values
            lambda_iso = config.get("lambda_iso_diff", 3.0e-9)
            lambda_par = config.get("lambda_par_diff", 1.7e-9)
            
            if not 1e-10 < lambda_iso < 1e-8:
                logger.error("lambda_iso_diff should be between 1e-10 and 1e-8, got: %e", lambda_iso)
                return False
            
            if not 1e-10 < lambda_par < 1e-8:
                logger.error("lambda_par_diff should be between 1e-10 and 1e-8, got: %e", lambda_par)
                return False
            
            # Check solver
            valid_solvers = ["brute2fine", "mix", "Powell", "Nelder-Mead", "L-BFGS-B"]
            solver = config.get("solver", "brute2fine")
            if solver not in valid_solvers:
                logger.error("Invalid solver: %s", solver)
                return False
            
            # Check maxiter
            maxiter = config.get("maxiter", 300)
            if not isinstance(maxiter, int) or maxiter < 1:
                logger.error("maxiter must be a positive integer, got: %s", maxiter)
                return False
            
            # Check number of processors
            num_proc = config.get("number_of_processors", 4)
            if not isinstance(num_proc, int) or num_proc < 1:
                logger.error("number_of_processors must be a positive integer, got: %s", num_proc)
                return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating config: %s", str(e))
            return False