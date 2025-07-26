"""
Connectivity processing module for ElikoPy
=======================================

This module provides functionality for connectivity matrix extraction and analysis,
including atlas registration, streamline-atlas intersection computation, and
BIDS-compliant connectivity matrix export.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import nibabel as nib
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines
from dipy.io.streamline import load_trk, load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.align.reslice import reslice
from scipy.ndimage import binary_dilation
from scipy.io import savemat
import pandas as pd

from elikopy.core.base import ProcessingComponent, ProcessingResult, ProcessingStatus, ConfigurableComponent
from elikopy.data.derivatives import DerivativesManager


logger = logging.getLogger(__name__)


@dataclass
class ConnectivityConfig:
    """Configuration for connectivity processing"""
    atlas_name: str = "aal"
    atlas_path: Optional[Path] = None
    atlas_labels_path: Optional[Path] = None
    weighting: str = "count"  # count, density, length, fa, mean_fa
    symmetric: bool = True
    normalize: bool = False
    inclusive: bool = False  # Include streamlines that only touch one region
    dilation_radius: int = 0  # Dilate atlas regions by N voxels
    min_streamline_length: float = 20.0  # Minimum streamline length (mm)
    max_streamline_length: float = 200.0  # Maximum streamline length (mm)
    output_formats: List[str] = field(default_factory=lambda: ["csv", "mat", "json"])
    save_assignment_maps: bool = False  # Save streamline-to-region assignment maps
    save_filtered_streamlines: bool = False  # Save streamlines used for connectivity


@dataclass
class AtlasData:
    """Atlas data structure"""
    atlas_image: np.ndarray
    labels: List[str]
    label_indices: List[int]
    affine: np.ndarray
    mni_space: bool = True
    atlas_name: str = "unknown"


@dataclass
class ConnectivityMatrix:
    """Connectivity matrix data structure"""
    matrix: np.ndarray
    atlas_labels: List[str]
    label_indices: List[int]
    subject_id: str
    session_id: Optional[str]
    atlas_name: str
    weighting: str
    streamline_count: int
    processing_info: Dict[str, Any]


@dataclass
class RegistrationResult:
    """Result of atlas registration"""
    registered_atlas: np.ndarray
    transformation_matrix: Optional[np.ndarray]
    atlas_in_subject_space: bool
    registration_method: str
    registration_quality: Optional[float] = None


class ConnectivityProcessor(ProcessingComponent, ConfigurableComponent):
    """
    Processor for connectivity matrix extraction and analysis.
    
    This processor implements atlas registration, streamline-atlas intersection
    computation, and connectivity matrix generation with BIDS-compliant output.
    """
    
    def __init__(self, config: Optional[ConnectivityConfig] = None):
        """
        Initialize connectivity processor.
        
        Parameters
        ----------
        config : ConnectivityConfig, optional
            Configuration for connectivity processing
        """
        self.config = config or ConnectivityConfig()
        self.derivatives_manager = None
    
    def validate_inputs(self, streamlines_file: Optional[Path] = None,
                       atlas_data: Optional[AtlasData] = None,
                       reference_image: Optional[Path] = None) -> bool:
        """
        Validate inputs before processing.
        
        Parameters
        ----------
        streamlines_file : Path, optional
            Path to streamlines file
        atlas_data : AtlasData, optional
            Atlas data
        reference_image : Path, optional
            Reference image for registration
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if streamlines_file is None or not streamlines_file.exists():
            logger.error(f"Streamlines file not found: {streamlines_file}")
            return False
            
        if atlas_data is None:
            if self.config.atlas_path is None or not self.config.atlas_path.exists():
                logger.error(f"Atlas file not found: {self.config.atlas_path}")
                return False
        
        if reference_image is None or not reference_image.exists():
            logger.error(f"Reference image not found: {reference_image}")
            return False
            
        return True
    
    def process(self, streamlines_file: Path, reference_image: Path,
                output_dir: Path, subject_id: str, session_id: Optional[str] = None,
                atlas_data: Optional[AtlasData] = None,
                fa_map: Optional[Path] = None) -> ProcessingResult:
        """
        Execute connectivity processing.
        
        Parameters
        ----------
        streamlines_file : Path
            Path to streamlines file (TRK or TCK format)
        reference_image : Path
            Reference image for registration (typically DWI or FA map)
        output_dir : Path
            Output directory for results
        subject_id : str
            Subject identifier
        session_id : str, optional
            Session identifier
        atlas_data : AtlasData, optional
            Pre-loaded atlas data
        fa_map : Path, optional
            FA map for weighted connectivity (if weighting includes FA)
            
        Returns
        -------
        ProcessingResult
            Processing result with output files and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validate_inputs(streamlines_file, atlas_data, reference_image):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Input validation failed"
                )
            
            logger.info(f"Starting connectivity analysis for subject {subject_id}")
            
            # Load streamlines
            streamlines, affine = self._load_streamlines(streamlines_file, reference_image)
            if streamlines is None:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    output_files=[],
                    metadata={},
                    error_message="Failed to load streamlines"
                )
            
            # Load or prepare atlas
            if atlas_data is None:
                atlas_data = self._load_atlas_data()
                if atlas_data is None:
                    return ProcessingResult(
                        status=ProcessingStatus.FAILED,
                        output_files=[],
                        metadata={},
                        error_message="Failed to load atlas data"
                    )
            
            # Register atlas to subject space
            registration_result = self.register_to_subject_space(
                atlas_data, reference_image, affine
            )
            
            # Apply atlas to get labeled regions
            atlas_result = self.apply_atlas(registration_result, atlas_data)
            
            # Compute connectivity matrix
            connectivity_matrix = self.compute_connectivity_matrix(
                streamlines, atlas_result, affine, fa_map
            )
            
            # Export connectivity matrices
            output_files = self.export_connectivity_matrices(
                connectivity_matrix, output_dir, subject_id, session_id
            )
            
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                "atlas_name": self.config.atlas_name,
                "weighting": self.config.weighting,
                "streamline_count": len(streamlines),
                "matrix_size": connectivity_matrix.matrix.shape,
                "processing_time": processing_time,
                "config": self._config_to_dict()
            }
            
            logger.info(f"Connectivity analysis completed in {processing_time:.2f} seconds")
            
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                output_files=output_files,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Connectivity processing failed: {str(e)}")
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
        # Convert dict to ConnectivityConfig
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
            "atlas_name": "aal",
            "atlas_path": None,
            "atlas_labels_path": None,
            "weighting": "count",
            "symmetric": True,
            "normalize": False,
            "inclusive": False,
            "dilation_radius": 0,
            "min_streamline_length": 20.0,
            "max_streamline_length": 200.0,
            "output_formats": ["csv", "mat", "json"]
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
        valid_weightings = ["count", "density", "length", "fa", "mean_fa"]
        valid_formats = ["csv", "mat", "json", "npy"]
        
        if "weighting" in config and config["weighting"] not in valid_weightings:
            logger.error(f"Invalid weighting: {config['weighting']}")
            return False
            
        if "output_formats" in config:
            for fmt in config["output_formats"]:
                if fmt not in valid_formats:
                    logger.error(f"Invalid output format: {fmt}")
                    return False
                    
        if "dilation_radius" in config and config["dilation_radius"] < 0:
            logger.error(f"Dilation radius must be non-negative: {config['dilation_radius']}")
            return False
            
        return True
    
    def register_to_subject_space(self, atlas_data: AtlasData, 
                                 reference_image: Path,
                                 reference_affine: np.ndarray) -> RegistrationResult:
        """
        Register atlas to subject space.
        
        Parameters
        ----------
        atlas_data : AtlasData
            Atlas data to register
        reference_image : Path
            Reference image in subject space
        reference_affine : np.ndarray
            Affine transformation of reference image
            
        Returns
        -------
        RegistrationResult
            Registration result with transformed atlas
        """
        try:
            logger.info(f"Registering {atlas_data.atlas_name} atlas to subject space")
            
            # Load reference image
            ref_img = nib.load(reference_image)
            ref_data = ref_img.get_fdata()
            
            # For now, implement a simple reslicing approach
            # In a full implementation, this would use proper registration (ANTs, FSL, etc.)
            
            # Reslice atlas to match reference image dimensions
            atlas_resliced, atlas_affine_new = reslice(
                atlas_data.atlas_image,
                atlas_data.affine,
                atlas_data.atlas_image.shape[:3],
                ref_data.shape[:3],
                reference_affine,
                order=0  # Use nearest neighbor for label preservation
            )
            
            # Round to nearest integer for label preservation
            atlas_resliced = np.round(atlas_resliced).astype(np.int32)
            
            # Apply dilation if requested
            if self.config.dilation_radius > 0:
                atlas_resliced = self._dilate_atlas_regions(
                    atlas_resliced, self.config.dilation_radius
                )
            
            logger.info(f"Atlas registered to subject space with shape {atlas_resliced.shape}")
            
            return RegistrationResult(
                registered_atlas=atlas_resliced,
                transformation_matrix=reference_affine,
                atlas_in_subject_space=True,
                registration_method="reslicing"
            )
            
        except Exception as e:
            logger.error(f"Atlas registration failed: {str(e)}")
            raise
    
    def apply_atlas(self, registration_result: RegistrationResult,
                   atlas_data: AtlasData) -> Dict[str, Any]:
        """
        Apply registered atlas to create labeled regions.
        
        Parameters
        ----------
        registration_result : RegistrationResult
            Result from atlas registration
        atlas_data : AtlasData
            Original atlas data
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing atlas result with labeled regions
        """
        try:
            atlas_labels = registration_result.registered_atlas
            
            # Get unique labels (excluding background)
            unique_labels = np.unique(atlas_labels)
            unique_labels = unique_labels[unique_labels > 0]  # Remove background
            
            logger.info(f"Atlas contains {len(unique_labels)} regions")
            
            return {
                "atlas_labels": atlas_labels,
                "unique_labels": unique_labels,
                "label_names": atlas_data.labels,
                "label_indices": atlas_data.label_indices,
                "atlas_name": atlas_data.atlas_name,
                "affine": registration_result.transformation_matrix
            }
            
        except Exception as e:
            logger.error(f"Atlas application failed: {str(e)}")
            raise
    
    def compute_connectivity_matrix(self, streamlines: Streamlines,
                                  atlas_result: Dict[str, Any],
                                  affine: np.ndarray,
                                  fa_map: Optional[Path] = None) -> ConnectivityMatrix:
        """
        Compute connectivity matrix from streamlines and atlas.
        
        Parameters
        ----------
        streamlines : Streamlines
            Tractography streamlines
        atlas_result : Dict[str, Any]
            Result from atlas application
        affine : np.ndarray
            Affine transformation matrix
        fa_map : Path, optional
            FA map for weighted connectivity
            
        Returns
        -------
        ConnectivityMatrix
            Computed connectivity matrix
        """
        try:
            logger.info("Computing connectivity matrix")
            
            atlas_labels = atlas_result["atlas_labels"]
            unique_labels = atlas_result["unique_labels"]
            
            # Filter streamlines by length if specified
            filtered_streamlines = self._filter_streamlines_by_length(streamlines)
            
            logger.info(f"Using {len(filtered_streamlines)} streamlines for connectivity")
            
            # Compute connectivity matrix using DIPY
            if self.config.weighting == "count":
                # Simple streamline count
                connectivity_matrix, mapping = utils.connectivity_matrix(
                    filtered_streamlines,
                    affine,
                    atlas_labels,
                    inclusive=self.config.inclusive,
                    return_mapping=True,
                    mapping_as_streamlines=False
                )
            elif self.config.weighting == "length":
                # Length-weighted connectivity
                connectivity_matrix = self._compute_length_weighted_matrix(
                    filtered_streamlines, atlas_labels, affine, unique_labels
                )
            elif self.config.weighting in ["fa", "mean_fa"] and fa_map is not None:
                # FA-weighted connectivity
                connectivity_matrix = self._compute_fa_weighted_matrix(
                    filtered_streamlines, atlas_labels, affine, unique_labels, fa_map
                )
            else:
                # Default to count
                connectivity_matrix, mapping = utils.connectivity_matrix(
                    filtered_streamlines,
                    affine,
                    atlas_labels,
                    inclusive=self.config.inclusive,
                    return_mapping=True,
                    mapping_as_streamlines=False
                )
            
            # Remove background (label 0) if present
            if connectivity_matrix.shape[0] > len(unique_labels):
                connectivity_matrix = connectivity_matrix[1:, 1:]
            
            # Make symmetric if requested
            if self.config.symmetric:
                connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
            
            # Normalize if requested
            if self.config.normalize:
                connectivity_matrix = self._normalize_matrix(connectivity_matrix)
            
            logger.info(f"Connectivity matrix computed with shape {connectivity_matrix.shape}")
            
            return ConnectivityMatrix(
                matrix=connectivity_matrix,
                atlas_labels=atlas_result["label_names"],
                label_indices=list(unique_labels),
                subject_id="",  # Will be set by caller
                session_id=None,  # Will be set by caller
                atlas_name=atlas_result["atlas_name"],
                weighting=self.config.weighting,
                streamline_count=len(filtered_streamlines),
                processing_info={
                    "symmetric": self.config.symmetric,
                    "normalized": self.config.normalize,
                    "inclusive": self.config.inclusive,
                    "dilation_radius": self.config.dilation_radius
                }
            )
            
        except Exception as e:
            logger.error(f"Connectivity matrix computation failed: {str(e)}")
            raise
    
    def export_connectivity_matrices(self, connectivity_matrix: ConnectivityMatrix,
                                   output_dir: Path, subject_id: str,
                                   session_id: Optional[str] = None) -> List[Path]:
        """
        Export connectivity matrices in various formats.
        
        Parameters
        ----------
        connectivity_matrix : ConnectivityMatrix
            Connectivity matrix to export
        output_dir : Path
            Output directory
        subject_id : str
            Subject identifier
        session_id : str, optional
            Session identifier
            
        Returns
        -------
        List[Path]
            List of output file paths
        """
        output_files = []
        
        # Update connectivity matrix with subject info
        connectivity_matrix.subject_id = subject_id
        connectivity_matrix.session_id = session_id
        
        # Create derivatives manager for BIDS-compliant output
        if self.derivatives_manager is None:
            self.derivatives_manager = DerivativesManager(output_dir.parent, "elikopy")
        
        try:
            # Get output directory for connectivity
            connectivity_dir = self.derivatives_manager.get_modality_dir(
                subject_id, "connectivity", session_id
            )
            
            # Create base filename
            entities = {
                "atlas": self.config.atlas_name,
                "desc": self.config.weighting
            }
            
            # Export in requested formats
            for fmt in self.config.output_formats:
                if fmt == "csv":
                    csv_file = self.derivatives_manager.get_output_path(
                        subject_id, "connectivity", "connectivity",
                        extension=".csv", session_id=session_id, **entities
                    )
                    self._export_csv(connectivity_matrix, csv_file)
                    output_files.append(csv_file)
                    
                elif fmt == "mat":
                    mat_file = self.derivatives_manager.get_output_path(
                        subject_id, "connectivity", "connectivity",
                        extension=".mat", session_id=session_id, **entities
                    )
                    self._export_mat(connectivity_matrix, mat_file)
                    output_files.append(mat_file)
                    
                elif fmt == "json":
                    json_file = self.derivatives_manager.get_output_path(
                        subject_id, "connectivity", "connectivity",
                        extension=".json", session_id=session_id, **entities
                    )
                    self._export_json(connectivity_matrix, json_file)
                    output_files.append(json_file)
                    
                elif fmt == "npy":
                    npy_file = self.derivatives_manager.get_output_path(
                        subject_id, "connectivity", "connectivity",
                        extension=".npy", session_id=session_id, **entities
                    )
                    self._export_npy(connectivity_matrix, npy_file)
                    output_files.append(npy_file)
            
            # Create metadata sidecar
            metadata_file = self.derivatives_manager.get_output_path(
                subject_id, "connectivity", "connectivity",
                extension=".json", session_id=session_id, **entities
            )
            
            # Only create if not already created above
            if metadata_file not in output_files:
                self._create_metadata_sidecar(connectivity_matrix, metadata_file)
                output_files.append(metadata_file)
            
            logger.info(f"Exported connectivity matrices to {len(output_files)} files")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to export connectivity matrices: {str(e)}")
            return []
    
    def _load_streamlines(self, streamlines_file: Path, 
                         reference_image: Path) -> Tuple[Optional[Streamlines], Optional[np.ndarray]]:
        """Load streamlines from file"""
        try:
            logger.info(f"Loading streamlines from {streamlines_file}")
            
            if streamlines_file.suffix == '.trk':
                tractogram = load_trk(str(streamlines_file), str(reference_image))
                streamlines = tractogram.streamlines
                affine = tractogram.affine
            elif streamlines_file.suffix == '.tck':
                # For TCK files, we need to load differently
                # This is a simplified approach
                logger.warning("TCK format support is limited")
                return None, None
            else:
                logger.error(f"Unsupported streamlines format: {streamlines_file.suffix}")
                return None, None
            
            logger.info(f"Loaded {len(streamlines)} streamlines")
            return streamlines, affine
            
        except Exception as e:
            logger.error(f"Failed to load streamlines: {str(e)}")
            return None, None
    
    def _load_atlas_data(self) -> Optional[AtlasData]:
        """Load atlas data from configuration"""
        try:
            if self.config.atlas_path is None:
                logger.error("Atlas path not specified")
                return None
            
            logger.info(f"Loading atlas from {self.config.atlas_path}")
            
            # Load atlas image
            atlas_img = nib.load(self.config.atlas_path)
            atlas_data = atlas_img.get_fdata().astype(np.int32)
            atlas_affine = atlas_img.affine
            
            # Load labels if available
            labels = []
            label_indices = []
            
            if self.config.atlas_labels_path and self.config.atlas_labels_path.exists():
                # Load labels from file (assuming simple text format)
                with open(self.config.atlas_labels_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            label_indices.append(int(parts[0]))
                            labels.append(' '.join(parts[1:]))
            else:
                # Generate default labels
                unique_labels = np.unique(atlas_data)
                unique_labels = unique_labels[unique_labels > 0]
                for i, label in enumerate(unique_labels):
                    label_indices.append(int(label))
                    labels.append(f"Region_{label}")
            
            return AtlasData(
                atlas_image=atlas_data,
                labels=labels,
                label_indices=label_indices,
                affine=atlas_affine,
                atlas_name=self.config.atlas_name
            )
            
        except Exception as e:
            logger.error(f"Failed to load atlas data: {str(e)}")
            return None
    
    def _dilate_atlas_regions(self, atlas_labels: np.ndarray, 
                             radius: int) -> np.ndarray:
        """Dilate atlas regions by specified radius"""
        try:
            logger.info(f"Dilating atlas regions by {radius} voxels")
            
            # Create structuring element
            from scipy.ndimage import generate_binary_structure
            struct_elem = generate_binary_structure(3, 1)  # 6-connectivity
            
            # Dilate each region separately
            dilated_atlas = np.zeros_like(atlas_labels)
            unique_labels = np.unique(atlas_labels)
            unique_labels = unique_labels[unique_labels > 0]
            
            for label in unique_labels:
                mask = atlas_labels == label
                dilated_mask = binary_dilation(mask, structure=struct_elem, iterations=radius)
                dilated_atlas[dilated_mask] = label
            
            # Restore original labels where they overlap with dilated regions
            dilated_atlas[atlas_labels > 0] = atlas_labels[atlas_labels > 0]
            
            return dilated_atlas
            
        except Exception as e:
            logger.error(f"Atlas dilation failed: {str(e)}")
            return atlas_labels
    
    def _filter_streamlines_by_length(self, streamlines: Streamlines) -> Streamlines:
        """Filter streamlines by length criteria"""
        try:
            filtered = []
            
            for streamline in streamlines:
                # Calculate streamline length
                if len(streamline) < 2:
                    continue
                    
                # Calculate length as sum of distances between consecutive points
                diffs = np.diff(streamline, axis=0)
                length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
                
                if self.config.min_streamline_length <= length <= self.config.max_streamline_length:
                    filtered.append(streamline)
            
            logger.info(f"Filtered streamlines: {len(streamlines)} -> {len(filtered)}")
            return Streamlines(filtered)
            
        except Exception as e:
            logger.error(f"Streamline filtering failed: {str(e)}")
            return streamlines
    
    def _compute_length_weighted_matrix(self, streamlines: Streamlines,
                                       atlas_labels: np.ndarray,
                                       affine: np.ndarray,
                                       unique_labels: np.ndarray) -> np.ndarray:
        """Compute length-weighted connectivity matrix"""
        try:
            n_regions = len(unique_labels)
            connectivity_matrix = np.zeros((n_regions, n_regions))
            
            for streamline in streamlines:
                # Calculate streamline length
                if len(streamline) < 2:
                    continue
                    
                diffs = np.diff(streamline, axis=0)
                length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
                
                # Get regions intersected by streamline
                labels_intersected = utils.connectivity_matrix(
                    [streamline], affine, atlas_labels, return_mapping=False
                )
                
                # Add length to connectivity matrix
                nonzero_indices = np.nonzero(labels_intersected)
                if len(nonzero_indices[0]) > 0:
                    for i in range(len(nonzero_indices[0])):
                        for j in range(len(nonzero_indices[1])):
                            if nonzero_indices[0][i] > 0 and nonzero_indices[1][j] > 0:
                                connectivity_matrix[nonzero_indices[0][i]-1, nonzero_indices[1][j]-1] += length
            
            return connectivity_matrix
            
        except Exception as e:
            logger.error(f"Length-weighted matrix computation failed: {str(e)}")
            # Fall back to count-based matrix
            matrix, _ = utils.connectivity_matrix(
                streamlines, affine, atlas_labels, return_mapping=False
            )
            return matrix[1:, 1:] if matrix.shape[0] > len(unique_labels) else matrix
    
    def _compute_fa_weighted_matrix(self, streamlines: Streamlines,
                                   atlas_labels: np.ndarray,
                                   affine: np.ndarray,
                                   unique_labels: np.ndarray,
                                   fa_map: Path) -> np.ndarray:
        """Compute FA-weighted connectivity matrix"""
        try:
            # Load FA map
            fa_img = nib.load(fa_map)
            fa_data = fa_img.get_fdata()
            
            n_regions = len(unique_labels)
            connectivity_matrix = np.zeros((n_regions, n_regions))
            
            for streamline in streamlines:
                if len(streamline) < 2:
                    continue
                
                # Sample FA values along streamline
                fa_values = []
                for point in streamline:
                    # Convert to voxel coordinates
                    vox_coords = np.round(
                        np.linalg.solve(affine, np.append(point, 1))[:3]
                    ).astype(int)
                    
                    # Check bounds
                    if (0 <= vox_coords[0] < fa_data.shape[0] and
                        0 <= vox_coords[1] < fa_data.shape[1] and
                        0 <= vox_coords[2] < fa_data.shape[2]):
                        fa_values.append(fa_data[vox_coords[0], vox_coords[1], vox_coords[2]])
                
                if len(fa_values) > 0:
                    mean_fa = np.mean(fa_values)
                    
                    # Get regions intersected by streamline
                    labels_intersected = utils.connectivity_matrix(
                        [streamline], affine, atlas_labels, return_mapping=False
                    )
                    
                    # Add FA weight to connectivity matrix
                    nonzero_indices = np.nonzero(labels_intersected)
                    if len(nonzero_indices[0]) > 0:
                        for i in range(len(nonzero_indices[0])):
                            for j in range(len(nonzero_indices[1])):
                                if nonzero_indices[0][i] > 0 and nonzero_indices[1][j] > 0:
                                    connectivity_matrix[nonzero_indices[0][i]-1, nonzero_indices[1][j]-1] += mean_fa
            
            return connectivity_matrix
            
        except Exception as e:
            logger.error(f"FA-weighted matrix computation failed: {str(e)}")
            # Fall back to count-based matrix
            matrix, _ = utils.connectivity_matrix(
                streamlines, affine, atlas_labels, return_mapping=False
            )
            return matrix[1:, 1:] if matrix.shape[0] > len(unique_labels) else matrix
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize connectivity matrix"""
        try:
            # Normalize by maximum value
            max_val = np.max(matrix)
            if max_val > 0:
                return matrix / max_val
            return matrix
        except Exception as e:
            logger.error(f"Matrix normalization failed: {str(e)}")
            return matrix
    
    def _export_csv(self, connectivity_matrix: ConnectivityMatrix, output_file: Path):
        """Export connectivity matrix as CSV"""
        try:
            # Create DataFrame with proper labels
            df = pd.DataFrame(
                connectivity_matrix.matrix,
                index=connectivity_matrix.atlas_labels[:len(connectivity_matrix.matrix)],
                columns=connectivity_matrix.atlas_labels[:len(connectivity_matrix.matrix)]
            )
            df.to_csv(output_file)
            logger.info(f"Exported CSV: {output_file}")
        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")
    
    def _export_mat(self, connectivity_matrix: ConnectivityMatrix, output_file: Path):
        """Export connectivity matrix as MATLAB file"""
        try:
            savemat(str(output_file), {
                'connectivity_matrix': connectivity_matrix.matrix,
                'atlas_labels': connectivity_matrix.atlas_labels,
                'label_indices': connectivity_matrix.label_indices,
                'atlas_name': connectivity_matrix.atlas_name,
                'weighting': connectivity_matrix.weighting,
                'subject_id': connectivity_matrix.subject_id,
                'session_id': connectivity_matrix.session_id or '',
                'streamline_count': connectivity_matrix.streamline_count
            })
            logger.info(f"Exported MAT: {output_file}")
        except Exception as e:
            logger.error(f"MAT export failed: {str(e)}")
    
    def _export_json(self, connectivity_matrix: ConnectivityMatrix, output_file: Path):
        """Export connectivity matrix metadata as JSON"""
        try:
            metadata = {
                'matrix_shape': connectivity_matrix.matrix.shape,
                'atlas_labels': connectivity_matrix.atlas_labels,
                'label_indices': connectivity_matrix.label_indices,
                'atlas_name': connectivity_matrix.atlas_name,
                'weighting': connectivity_matrix.weighting,
                'subject_id': connectivity_matrix.subject_id,
                'session_id': connectivity_matrix.session_id,
                'streamline_count': connectivity_matrix.streamline_count,
                'processing_info': connectivity_matrix.processing_info
            }
            
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Exported JSON: {output_file}")
        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
    
    def _export_npy(self, connectivity_matrix: ConnectivityMatrix, output_file: Path):
        """Export connectivity matrix as NumPy array"""
        try:
            np.save(output_file, connectivity_matrix.matrix)
            logger.info(f"Exported NPY: {output_file}")
        except Exception as e:
            logger.error(f"NPY export failed: {str(e)}")
    
    def _create_metadata_sidecar(self, connectivity_matrix: ConnectivityMatrix, 
                                metadata_file: Path):
        """Create BIDS metadata sidecar"""
        try:
            metadata = {
                'Description': f'Structural connectivity matrix derived from tractography',
                'Atlas': connectivity_matrix.atlas_name,
                'Weighting': connectivity_matrix.weighting,
                'MatrixSize': connectivity_matrix.matrix.shape,
                'NumberOfStreamlines': connectivity_matrix.streamline_count,
                'ProcessingParameters': connectivity_matrix.processing_info,
                'RegionLabels': connectivity_matrix.atlas_labels,
                'RegionIndices': connectivity_matrix.label_indices
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Created metadata sidecar: {metadata_file}")
        except Exception as e:
            logger.error(f"Metadata sidecar creation failed: {str(e)}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "atlas_name": self.config.atlas_name,
            "atlas_path": str(self.config.atlas_path) if self.config.atlas_path else None,
            "weighting": self.config.weighting,
            "symmetric": self.config.symmetric,
            "normalize": self.config.normalize,
            "inclusive": self.config.inclusive,
            "dilation_radius": self.config.dilation_radius,
            "min_streamline_length": self.config.min_streamline_length,
            "max_streamline_length": self.config.max_streamline_length,
            "output_formats": self.config.output_formats
        }