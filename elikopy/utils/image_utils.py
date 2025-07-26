"""
Image processing utilities for ElikoPy
=====================================

This module contains utility functions for image processing operations including
diffusion tensor operations, spherical harmonics conversions, mask processing,
acquisition view detection, and image normalization.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Union, Literal

import numpy as np
import nibabel as nib
import scipy.ndimage
from skimage.morphology import flood

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sf_to_sh

from elikopy.infrastructure.exceptions import ElikopyError, DataValidationError


class ImageProcessingError(ElikopyError):
    """Exception raised for image processing errors"""
    pass


def deltas_to_diffusion_tensor(dx: float, dy: float, dz: float,
                               lamb: np.ndarray = None, vec_len: float = 500) -> np.ndarray:
    """
    Create a diffusion tensor from three orthogonal components.

    Parameters
    ----------
    dx : float
        X component of the diffusion direction
    dy : float
        Y component of the diffusion direction
    dz : float
        Z component of the diffusion direction
    lamb : np.ndarray, optional
        3x3 diagonal matrix containing diffusion eigenvalues.
        Default is np.diag([1, 0, 0])
    vec_len : float, optional
        Value for scaling the diffusion. Default is 500

    Returns
    -------
    np.ndarray
        3x3 diffusion tensor matrix

    Raises
    ------
    ImageProcessingError
        If the matrix inversion fails (singular matrix)
    """
    if lamb is None:
        lamb = np.diag([1, 0, 0])

    # Validate inputs
    if not all(isinstance(x, (int, float)) for x in [dx, dy, dz, vec_len]):
        raise DataValidationError("All direction components and vec_len must be numeric")

    if not isinstance(lamb, np.ndarray) or lamb.shape != (3, 3):
        raise DataValidationError("lamb must be a 3x3 numpy array")

    e = np.array([[dx, -dz-dy, dy*dx-dx*dz],
                  [dy, dx, -dx**2-(dz+dy)*dz],
                  [dz, dx, dx**2+(dy+dz)*dy]])

    try:
        e_inv = np.linalg.inv(e)
    except np.linalg.LinAlgError as err:
        raise ImageProcessingError(f"Failed to invert matrix for diffusion tensor calculation: {err}") from err

    diffusion_tensor = (e.dot(lamb)).dot(e_inv) / vec_len
    return diffusion_tensor


def peak_to_tensor(peaks: np.ndarray, norm: Optional[np.ndarray] = None,
                   pixdim: list = None) -> np.ndarray:
    """
    Convert peaks to tensor format used in DIAMOND.

    Parameters
    ----------
    peaks : np.ndarray
        4D array containing peaks of shape (x, y, z, 3)
    norm : np.ndarray, optional
        Normalization array of shape (x, y, z)
    pixdim : list, optional
        Pixel dimensions [x, y, z]. Default is [2, 2, 2]

    Returns
    -------
    np.ndarray
        5D tensor array of shape (x, y, z, 1, 6)

    Raises
    ------
    DataValidationError
        If input arrays have incorrect shapes
    """
    if pixdim is None:
        pixdim = [2, 2, 2]

    # Validate inputs
    if not isinstance(peaks, np.ndarray) or len(peaks.shape) != 4 or peaks.shape[3] != 3:
        raise DataValidationError("peaks must be a 4D array with shape (x, y, z, 3)")

    if norm is not None:
        if not isinstance(norm, np.ndarray) or norm.shape != peaks.shape[:3]:
            raise DataValidationError("norm must have shape matching peaks[:3]")

    if len(pixdim) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in pixdim):
        raise DataValidationError("pixdim must be a list of 3 positive numbers")

    tensor = np.zeros(peaks.shape[:3] + (1, 6))
    scale_factor = 1000 / min(pixdim)

    for xyz in np.ndindex(peaks.shape[:3]):
        if np.all(peaks[xyz] == 0):
            continue

        dx, dy, dz = peaks[xyz]

        try:
            if norm is not None:
                diffusion_tensor = deltas_to_diffusion_tensor(dx, dy, dz, vec_len=scale_factor/norm[xyz])
            else:
                diffusion_tensor = deltas_to_diffusion_tensor(dx, dy, dz, vec_len=scale_factor)
        except ImageProcessingError:
            # Skip voxels where tensor calculation fails
            continue

        # Store tensor components in DIAMOND format
        tensor[xyz + (0, 0)] = diffusion_tensor[0, 0]
        tensor[xyz + (0, 1)] = diffusion_tensor[0, 1]
        tensor[xyz + (0, 2)] = diffusion_tensor[1, 1]
        tensor[xyz + (0, 3)] = diffusion_tensor[0, 2]
        tensor[xyz + (0, 4)] = diffusion_tensor[1, 2]
        tensor[xyz + (0, 5)] = diffusion_tensor[2, 2]

    return tensor


def tensor_to_peak(tensor: np.ndarray) -> np.ndarray:
    """
    Convert tensor from DIAMOND format to peaks format for Microstructure Fingerprinting.

    Parameters
    ----------
    tensor : np.ndarray
        5D tensor array of shape (x, y, z, 1, 6) or 4D array of shape (x, y, z, 6)

    Returns
    -------
    np.ndarray
        4D array containing peaks of shape (x, y, z, 3)

    Raises
    ------
    DataValidationError
        If input tensor has incorrect shape
    """
    if not isinstance(tensor, np.ndarray):
        raise DataValidationError("Input tensor must be a numpy array")

    if len(tensor.shape) == 4:
        if tensor.shape[3] != 6:
            raise DataValidationError("4D tensor must have shape (x, y, z, 6)")
        
        # For 4D input, the tensor components are arranged differently
        diffusion_tensor = np.array([[tensor[:, :, :, 0], tensor[:, :, :, 1], tensor[:, :, :, 3]],
                                     [tensor[:, :, :, 1], tensor[:, :, :, 2], tensor[:, :, :, 4]],
                                     [tensor[:, :, :, 3], tensor[:, :, :, 4], tensor[:, :, :, 5]]])
    elif len(tensor.shape) == 5:
        if tensor.shape[3:] != (1, 6):
            raise DataValidationError("5D tensor must have shape (x, y, z, 1, 6)")

        diffusion_tensor = np.array([[tensor[:, :, :, 0, 0], tensor[:, :, :, 0, 1], tensor[:, :, :, 0, 3]],
                                     [tensor[:, :, :, 0, 1], tensor[:, :, :, 0, 2], tensor[:, :, :, 0, 4]],
                                     [tensor[:, :, :, 0, 3], tensor[:, :, :, 0, 4], tensor[:, :, :, 0, 5]]])
    else:
        raise DataValidationError("Tensor must be 4D or 5D array")

    diffusion_tensor = np.transpose(diffusion_tensor, (2, 3, 4, 0, 1))

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(diffusion_tensor)

    vol_shape = tensor.shape[0] * tensor.shape[1] * tensor.shape[2]
    eigenvectors = eigenvectors.reshape((vol_shape, 3, 3))
    eigenvectors = np.transpose(eigenvectors, (0, 2, 1))

    # Get principal eigenvector (largest eigenvalue)
    idx = np.argmax(eigenvalues.reshape((vol_shape, 3)), axis=1)
    peaks = eigenvectors[range(vol_shape), idx].reshape(tensor.shape[:3] + (3,)).real

    return peaks


def _flip_negative_m_values(sh: np.ndarray, sh_order: int, full_basis: bool = False) -> np.ndarray:
    """
    Flip negative m values in spherical harmonics coefficients.

    Parameters
    ----------
    sh : np.ndarray
        4D spherical harmonics coefficient array of shape (x, y, z, coeff)
    sh_order : int
        Order of the spherical harmonics
    full_basis : bool, optional
        If True, uses full basis. Default is False

    Returns
    -------
    np.ndarray
        Modified spherical harmonics array

    Raises
    ------
    DataValidationError
        If input arrays have incorrect shapes or types
    """
    if not isinstance(sh, np.ndarray) or len(sh.shape) != 4:
        raise DataValidationError("sh must be a 4D numpy array")

    if not isinstance(sh_order, (int, np.integer)) or sh_order < 0:
        raise DataValidationError("sh_order must be a non-negative integer")

    sh = sh.copy()  # Avoid modifying input array
    counter = 0

    for l_order in range(int(sh_order)):
        n = 1 + 2 * l_order
        m_list = np.linspace((n-1)/2, -(n-1)/2, n)

        for m in m_list:
            if full_basis:
                if m % 2 == 0 and m < 0:
                    sh[:, :, :, counter] *= -1
                counter += 1
            else:
                if l_order % 2 == 0:
                    if m % 2 == 0 and m < 0:
                        sh[:, :, :, counter] *= -1
                    counter += 1

    return sh


def convert_dipy_fod_to_mrtrix(sh: np.ndarray) -> np.ndarray:
    """
    Convert spherical harmonics from DIPY format to MRtrix format.
    Only works with symmetrical SH, not full basis.

    Parameters
    ----------
    sh : np.ndarray
        4D spherical harmonics coefficient array of shape (x, y, z, coeff)

    Returns
    -------
    np.ndarray
        Converted spherical harmonics array in MRtrix format

    Raises
    ------
    DataValidationError
        If input array has incorrect shape
    ImageProcessingError
        If conversion fails
    """
    if not isinstance(sh, np.ndarray) or len(sh.shape) != 4:
        raise DataValidationError("sh must be a 4D numpy array")

    try:
        sh_order = int((np.sqrt(sh.shape[3] * 8 + 1) - 3) / 2)

        if sh_order < 0:
            raise ValueError("Invalid SH coefficient count")

        sh = _flip_negative_m_values(sh, sh_order)

        default_sphere = get_sphere(name='repulsion724')

        # Convert to signal function
        temp = sh_to_sf(sh, sphere=default_sphere, sh_order_max=sh_order,
                       basis_type='descoteaux07', legacy=False)

        # Convert back to SH with MRtrix basis
        sh = sf_to_sh(temp, sphere=default_sphere, sh_order_max=sh_order,
                     basis_type='tournier07', legacy=False)

        return sh

    except Exception as e:
        raise ImageProcessingError(f"Failed to convert DIPY FOD to MRtrix format: {e}") from e


def convert_mrtrix_fod_to_dipy(sh: np.ndarray) -> np.ndarray:
    """
    Convert spherical harmonics from MRtrix format to DIPY format.
    Only works with symmetrical SH, not full basis.

    Parameters
    ----------
    sh : np.ndarray
        4D spherical harmonics coefficient array of shape (x, y, z, coeff)

    Returns
    -------
    np.ndarray
        Converted spherical harmonics array in DIPY format

    Raises
    ------
    DataValidationError
        If input array has incorrect shape
    ImageProcessingError
        If conversion fails
    """
    if not isinstance(sh, np.ndarray) or len(sh.shape) != 4:
        raise DataValidationError("sh must be a 4D numpy array")

    try:
        sh_order = int((np.sqrt(sh.shape[3] * 8 + 1) - 3) / 2)

        if sh_order < 0:
            raise ValueError("Invalid SH coefficient count")

        default_sphere = get_sphere(name='repulsion724')

        # Convert to signal function
        temp = sh_to_sf(sh, sphere=default_sphere, sh_order_max=sh_order,
                       basis_type='tournier07', legacy=False)

        # Convert back to SH with DIPY basis
        sh = sf_to_sh(temp, sphere=default_sphere, sh_order_max=sh_order,
                     basis_type='descoteaux07', legacy=False)

        sh = _flip_negative_m_values(sh, sh_order)

        return sh

    except Exception as e:
        raise ImageProcessingError(f"Failed to convert MRtrix FOD to DIPY format: {e}") from e


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean binary mask using morphological operations to fill holes and remove islands.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask array

    Returns
    -------
    np.ndarray
        Cleaned binary mask

    Raises
    ------
    DataValidationError
        If input mask has incorrect shape or type
    """
    if not isinstance(mask, np.ndarray):
        raise DataValidationError("mask must be a numpy array")
    
    if len(mask.shape) != 3:
        raise DataValidationError("mask must be a 3D array")
    
    # Work on a copy to avoid modifying input
    mask = mask.copy().astype(bool)
    
    # Pad mask to handle edge cases
    mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    
    # Fill holes from each axis direction
    mask_filled = mask.copy()
    
    # Fill from x-axis
    for x in range(mask.shape[0]):
        try:
            mask_filled[x, :, :] = flood(mask[x, :, :], (0, 0))
        except (IndexError, ValueError):
            continue
    mask = np.where(mask_filled == 0, 1, mask)

    # Fill from y-axis
    mask_filled = mask.copy()
    for y in range(mask.shape[1]):
        try:
            mask_filled[:, y, :] = flood(mask[:, y, :], (0, 0))
        except (IndexError, ValueError):
            continue
    mask = np.where(mask_filled == 0, 1, mask)

    # Fill from z-axis
    mask_filled = mask.copy()
    for z in range(mask.shape[2]):
        try:
            mask_filled[:, :, z] = flood(mask[:, :, z], (0, 0))
        except (IndexError, ValueError):
            continue
    mask = np.where(mask_filled == 0, 1, mask)

    # Find center of mass and flood fill from center
    indices = np.where(mask == 1)
    if len(indices[0]) > 0:
        center = tuple([int(np.average(idx)) for idx in indices])
        try:
            mask = flood(mask, center, connectivity=1)
        except (IndexError, ValueError):
            # If flood fill fails, return original mask
            pass
    
    # Create final cleaned mask
    mask_cleaned = np.zeros(mask.shape, dtype=bool)
    mask_cleaned[mask] = 1

    # Remove padding
    mask_cleaned = mask_cleaned[tuple(slice(1, dim - 1) for dim in mask_cleaned.shape)]

    return mask_cleaned.astype(np.uint8)


def get_acquisition_view(affine: np.ndarray) -> Literal['axial', 'sagittal', 'coronal', 'oblique']:
    """
    Determine the acquisition view from the affine transformation matrix.
    
    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix
        
    Returns
    -------
    str
        Acquisition view: 'axial', 'sagittal', 'coronal', or 'oblique'
        
    Raises
    ------
    DataValidationError
        If affine matrix has incorrect shape
    """
    if not isinstance(affine, np.ndarray):
        raise DataValidationError("affine must be a numpy array")
    
    if affine.shape != (4, 4):
        raise DataValidationError("affine must be a 4x4 matrix")

    # Extract rotation/scaling part
    affine = affine[:3, :3].copy()

    sum_whole_aff = np.sum(np.abs(affine))
    sum_diag_aff = np.sum(np.diag(np.abs(affine)))
    sum_extra_diag_aff = sum_whole_aff - sum_diag_aff

    # Extract matrix elements
    a, b, c = affine[0, 0], affine[1, 1], affine[2, 2]
    d, e, f = affine[0, 2], affine[1, 0], affine[2, 1]
    g, h, i = affine[0, 1], affine[1, 2], affine[2, 0]

    # Determine acquisition view based on non-zero elements
    if (a != 0 and b != 0 and c != 0 and sum_extra_diag_aff == 0):
        return "axial"
    elif (d != 0 and e != 0 and f != 0 and sum_diag_aff == 0):
        return "sagittal"
    elif (g != 0 and h != 0 and i != 0 and sum_diag_aff == 0):
        return "coronal"
    else:
        return "oblique"


def load_nifti_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI image and return data and affine.
    
    Parameters
    ----------
    image_path : str or Path
        Path to NIfTI image file
        
    Returns
    -------
    tuple
        (image_data, affine) where image_data is np.ndarray and affine is 4x4 matrix
        
    Raises
    ------
    DataValidationError
        If file doesn't exist or can't be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise DataValidationError(f"Image file does not exist: {image_path}")
    
    try:
        img = nib.load(str(image_path))
        return img.get_fdata(), img.affine
    except Exception as e:
        raise DataValidationError(f"Failed to load NIfTI image {image_path}: {e}")


def save_nifti_image(image_data: np.ndarray, affine: np.ndarray, 
                     output_path: Union[str, Path]) -> None:
    """
    Save image data as NIfTI file.
    
    Parameters
    ----------
    image_data : np.ndarray
        Image data to save
    affine : np.ndarray
        4x4 affine transformation matrix
    output_path : str or Path
        Output file path
        
    Raises
    ------
    DataValidationError
        If inputs are invalid or saving fails
    """
    if not isinstance(image_data, np.ndarray):
        raise DataValidationError("image_data must be a numpy array")
    
    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise DataValidationError("affine must be a 4x4 numpy array")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        img = nib.Nifti1Image(image_data, affine)
        nib.save(img, str(output_path))
    except Exception as e:
        raise DataValidationError(f"Failed to save NIfTI image to {output_path}: {e}")


# Legacy class for backward compatibility
class ImageUtils:
    """Legacy utility class for image processing operations"""
    
    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        """Load image from file - legacy method"""
        data, _ = load_nifti_image(image_path)
        return data
    
    @staticmethod
    def save_image(image_data: np.ndarray, output_path: Path, 
                   affine: np.ndarray = None) -> None:
        """Save image data to file - legacy method"""
        if affine is None:
            affine = np.eye(4)
        save_nifti_image(image_data, affine, output_path)
    
    @staticmethod
    def apply_mask(image_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image data"""
        if not isinstance(image_data, np.ndarray) or not isinstance(mask, np.ndarray):
            raise DataValidationError("Both image_data and mask must be numpy arrays")
        
        if image_data.shape[:3] != mask.shape[:3]:
            raise DataValidationError("Image and mask must have compatible spatial dimensions")
        
        # Apply mask
        masked_data = image_data.copy()
        if len(image_data.shape) == 3:
            masked_data[mask == 0] = 0
        else:
            masked_data[mask == 0, :] = 0
            
        return masked_data


def normalize_image(img: np.ndarray, max_img: float, min_img: float, 
                   max_val: float, min_val: float) -> np.ndarray:
    """
    Normalize image values to a specified range.
    
    Parameters
    ----------
    img : np.ndarray
        Input image array
    max_img : float
        Maximum value in the input image
    min_img : float
        Minimum value in the input image
    max_val : float
        Target maximum value
    min_val : float
        Target minimum value
        
    Returns
    -------
    np.ndarray
        Normalized image array
        
    Raises
    ------
    DataValidationError
        If input parameters are invalid
    """
    if not isinstance(img, np.ndarray):
        raise DataValidationError("img must be a numpy array")
    
    if max_img <= min_img:
        raise DataValidationError("max_img must be greater than min_img")
    
    if max_val <= min_val:
        raise DataValidationError("max_val must be greater than min_val")
    
    # Scale to [0, 1]
    normalized = (img - min_img) / (max_img - min_img)
    
    # Scale to [min_val, max_val]
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def unnormalize_image(img: np.ndarray, max_img: float, min_img: float,
                     max_val: float, min_val: float) -> np.ndarray:
    """
    Reverse the normalization applied by normalize_image.
    
    Parameters
    ----------
    img : np.ndarray
        Normalized image array
    max_img : float
        Original maximum value in the image
    min_img : float
        Original minimum value in the image
    max_val : float
        Current maximum value
    min_val : float
        Current minimum value
        
    Returns
    -------
    np.ndarray
        Unnormalized image array
        
    Raises
    ------
    DataValidationError
        If input parameters are invalid
    """
    if not isinstance(img, np.ndarray):
        raise DataValidationError("img must be a numpy array")
    
    if max_img <= min_img:
        raise DataValidationError("max_img must be greater than min_img")
    
    if max_val <= min_val:
        raise DataValidationError("max_val must be greater than min_val")
    
    # Reverse the normalization
    unnormalized = (img - min_val) / (max_val - min_val) * (max_img - min_img) + min_img
    
    return unnormalized


def nifti_to_torch_format(nii_img: np.ndarray) -> np.ndarray:
    """
    Convert NIfTI image format to PyTorch tensor format.
    
    Parameters
    ----------
    nii_img : np.ndarray
        Input image of shape (x, y, z, channels)
        
    Returns
    -------
    np.ndarray
        Output array of shape (1, channels, z, x, y)
        
    Raises
    ------
    DataValidationError
        If input array has incorrect shape
    """
    if not isinstance(nii_img, np.ndarray):
        raise DataValidationError("nii_img must be a numpy array")
    
    if len(nii_img.shape) != 4:
        raise DataValidationError("nii_img must be a 4D array with shape (x, y, z, channels)")
    
    # Expand dims => (1, x, y, z, channels)
    torch_img = np.expand_dims(nii_img, axis=0)
    
    # Permute dimensions => (1, channels, z, x, y)
    torch_img = np.transpose(torch_img, axes=(0, 4, 3, 1, 2))
    
    return torch_img


def torch_to_nifti_format(torch_img: np.ndarray) -> np.ndarray:
    """
    Convert PyTorch tensor format to NIfTI image format.
    
    Parameters
    ----------
    torch_img : np.ndarray
        Input tensor of shape (1, channels, z, x, y)
        
    Returns
    -------
    np.ndarray
        Output array of shape (x, y, z, channels)
        
    Raises
    ------
    DataValidationError
        If input array has incorrect shape
    """
    if not isinstance(torch_img, np.ndarray):
        raise DataValidationError("torch_img must be a numpy array")
    
    if len(torch_img.shape) != 5:
        raise DataValidationError("torch_img must be a 5D array with shape (1, channels, z, x, y)")
    
    # Remove first dim => (channels, z, x, y)
    nii_img = torch_img[0, :, :, :, :]
    
    # Permute dimensions => (x, y, z, channels)
    nii_img = np.transpose(nii_img, axes=(2, 3, 1, 0))
    
    return nii_img


def generate_random_unit_vector() -> np.ndarray:
    """
    Generate a random unit vector on the unit sphere.
    
    Returns
    -------
    np.ndarray
        Random unit vector of shape (3,)
    """
    theta = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(-1, 1)
    
    x = np.sqrt(1 - z ** 2) * np.cos(theta)
    y = np.sqrt(1 - z ** 2) * np.sin(theta)
    
    return np.array([x, y, z])


def rodrigues_to_rotation_matrix(k: np.ndarray, theta: float) -> np.ndarray:
    """
    Convert Rodrigues rotation vector to rotation matrix.
    
    Parameters
    ----------
    k : np.ndarray
        Unit rotation axis vector of shape (3,)
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
        
    Raises
    ------
    DataValidationError
        If input parameters are invalid
    """
    if not isinstance(k, np.ndarray) or k.shape != (3,):
        raise DataValidationError("k must be a 3D numpy array")
    
    if not isinstance(theta, (int, float)):
        raise DataValidationError("theta must be a numeric value")
    
    # Get cross product matrix
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],    0]])
    
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)


def rotation_translation_to_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix and translation vector to 4x4 transformation matrix.
    
    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix
    t : np.ndarray
        3x1 translation vector
        
    Returns
    -------
    np.ndarray
        4x4 transformation matrix
        
    Raises
    ------
    DataValidationError
        If input matrices have incorrect shapes
    """
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise DataValidationError("R must be a 3x3 numpy array")
    
    if not isinstance(t, np.ndarray) or t.shape not in [(3,), (3, 1)]:
        raise DataValidationError("t must be a 3D numpy array or 3x1 matrix")
    
    # Ensure t is column vector
    if t.shape == (3,):
        t = t.reshape(3, 1)
    
    # Concatenate R and t
    Rt = np.concatenate((R, t), axis=1)
    
    # Concatenate [0, 0, 0, 1] to form affine matrix
    return np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)


def apply_transform_to_volume(xform: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 transformation matrix to a 3D volume.
    
    Parameters
    ----------
    xform : np.ndarray
        4x4 transformation matrix
    vol : np.ndarray
        3D volume to transform
        
    Returns
    -------
    np.ndarray
        Transformed volume
        
    Raises
    ------
    DataValidationError
        If input arrays have incorrect shapes
    """
    if not isinstance(xform, np.ndarray) or xform.shape != (4, 4):
        raise DataValidationError("xform must be a 4x4 numpy array")
    
    if not isinstance(vol, np.ndarray) or len(vol.shape) != 3:
        raise DataValidationError("vol must be a 3D numpy array")
    
    # Get voxel coordinates
    coords = np.meshgrid(np.arange(vol.shape[1]),
                        np.arange(vol.shape[0]),
                        np.arange(vol.shape[2]))
    
    xyz = np.vstack([coords[0].reshape(-1) - float(vol.shape[1] - 1) / 2,
                     coords[1].reshape(-1) - float(vol.shape[0] - 1) / 2,
                     coords[2].reshape(-1) - float(vol.shape[2] - 1) / 2,
                     np.ones(vol.shape).reshape(-1)])
    
    xyz_xform = np.matmul(xform, xyz)
    
    x = xyz_xform[0, :] + float(vol.shape[1] - 1) / 2
    y = xyz_xform[1, :] + float(vol.shape[0] - 1) / 2
    z = xyz_xform[2, :] + float(vol.shape[2] - 1) / 2
    
    x = x.reshape(vol.shape)
    y = y.reshape(vol.shape)
    z = z.reshape(vol.shape)
    
    return scipy.ndimage.map_coordinates(vol, [y, x, z], order=3)