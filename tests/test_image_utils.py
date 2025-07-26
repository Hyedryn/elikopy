"""
Unit tests for image processing utilities
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import nibabel as nib

from elikopy.utils.image_utils import (
    deltas_to_diffusion_tensor,
    peak_to_tensor,
    tensor_to_peak,
    _flip_negative_m_values,
    convert_dipy_fod_to_mrtrix,
    convert_mrtrix_fod_to_dipy,
    clean_binary_mask,
    get_acquisition_view,
    load_nifti_image,
    save_nifti_image,
    normalize_image,
    unnormalize_image,
    nifti_to_torch_format,
    torch_to_nifti_format,
    generate_random_unit_vector,
    rodrigues_to_rotation_matrix,
    rotation_translation_to_transform,
    apply_transform_to_volume,
    ImageUtils,
    ImageProcessingError
)
from elikopy.infrastructure.exceptions import DataValidationError


class TestDiffusionTensorOperations:
    """Test diffusion tensor related functions"""
    
    def test_deltas_to_diffusion_tensor_default(self):
        """Test diffusion tensor creation with default parameters"""
        dx, dy, dz = 1.0, 0.0, 0.0
        tensor = deltas_to_diffusion_tensor(dx, dy, dz)
        
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (3, 3)
        assert np.allclose(tensor, tensor.T)  # Should be symmetric
    
    def test_deltas_to_diffusion_tensor_custom_lambda(self):
        """Test diffusion tensor creation with custom eigenvalues"""
        dx, dy, dz = 1.0, 0.0, 0.0
        lamb = np.diag([2, 1, 0.5])
        tensor = deltas_to_diffusion_tensor(dx, dy, dz, lamb=lamb)
        
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (3, 3)
    
    def test_deltas_to_diffusion_tensor_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        with pytest.raises(DataValidationError):
            deltas_to_diffusion_tensor("invalid", 0.0, 0.0)
        
        with pytest.raises(DataValidationError):
            deltas_to_diffusion_tensor(1.0, 0.0, 0.0, lamb=np.array([1, 2]))
    
    def test_peak_to_tensor_basic(self):
        """Test peak to tensor conversion"""
        peaks = np.zeros((10, 10, 10, 3))
        peaks[5, 5, 5, :] = [1.0, 0.0, 0.0]  # Single peak
        
        tensor = peak_to_tensor(peaks)
        
        assert tensor.shape == (10, 10, 10, 1, 6)
        assert not np.allclose(tensor[5, 5, 5, 0, :], 0)  # Should have non-zero values
    
    def test_peak_to_tensor_with_norm(self):
        """Test peak to tensor conversion with normalization"""
        peaks = np.zeros((5, 5, 5, 3))
        peaks[2, 2, 2, :] = [1.0, 0.0, 0.0]
        norm = np.ones((5, 5, 5))
        
        tensor = peak_to_tensor(peaks, norm=norm)
        
        assert tensor.shape == (5, 5, 5, 1, 6)
    
    def test_peak_to_tensor_invalid_inputs(self):
        """Test error handling for peak_to_tensor"""
        with pytest.raises(DataValidationError):
            peak_to_tensor(np.zeros((5, 5)))  # Wrong dimensions
        
        with pytest.raises(DataValidationError):
            peak_to_tensor(np.zeros((5, 5, 5, 3)), norm=np.zeros((3, 3)))  # Wrong norm shape
    
    def test_tensor_to_peak_4d(self):
        """Test tensor to peak conversion for 4D input"""
        tensor = np.random.rand(5, 5, 5, 6)
        peaks = tensor_to_peak(tensor)
        
        assert peaks.shape == (5, 5, 5, 3)
    
    def test_tensor_to_peak_5d(self):
        """Test tensor to peak conversion for 5D input"""
        tensor = np.random.rand(5, 5, 5, 1, 6)
        peaks = tensor_to_peak(tensor)
        
        assert peaks.shape == (5, 5, 5, 3)
    
    def test_tensor_to_peak_invalid_inputs(self):
        """Test error handling for tensor_to_peak"""
        with pytest.raises(DataValidationError):
            tensor_to_peak(np.zeros((5, 5, 5)))  # Wrong dimensions
        
        with pytest.raises(DataValidationError):
            tensor_to_peak(np.zeros((5, 5, 5, 5)))  # Wrong last dimension


class TestSphericalHarmonics:
    """Test spherical harmonics related functions"""
    
    def test_flip_negative_m_values(self):
        """Test flipping negative m values"""
        sh = np.random.rand(5, 5, 5, 15)  # Order 4 SH
        sh_flipped = _flip_negative_m_values(sh, 4)
        
        assert sh_flipped.shape == sh.shape
        assert isinstance(sh_flipped, np.ndarray)
    
    def test_flip_negative_m_values_invalid_inputs(self):
        """Test error handling for _flip_negative_m_values"""
        with pytest.raises(DataValidationError):
            _flip_negative_m_values(np.zeros((5, 5)), 4)  # Wrong dimensions
        
        with pytest.raises(DataValidationError):
            _flip_negative_m_values(np.zeros((5, 5, 5, 15)), -1)  # Negative order
    
    def test_convert_dipy_fod_to_mrtrix(self):
        """Test DIPY to MRtrix FOD conversion"""
        sh = np.random.rand(5, 5, 5, 15)  # Order 4 SH
        sh_mrtrix = convert_dipy_fod_to_mrtrix(sh)
        
        assert sh_mrtrix.shape == sh.shape
        assert isinstance(sh_mrtrix, np.ndarray)
    
    def test_convert_mrtrix_fod_to_dipy(self):
        """Test MRtrix to DIPY FOD conversion"""
        sh = np.random.rand(5, 5, 5, 15)  # Order 4 SH
        sh_dipy = convert_mrtrix_fod_to_dipy(sh)
        
        assert sh_dipy.shape == sh.shape
        assert isinstance(sh_dipy, np.ndarray)
    
    def test_fod_conversion_roundtrip(self):
        """Test that conversion roundtrip preserves data approximately"""
        sh_original = np.random.rand(3, 3, 3, 15)
        sh_mrtrix = convert_dipy_fod_to_mrtrix(sh_original)
        sh_back = convert_mrtrix_fod_to_dipy(sh_mrtrix)
        
        # Should be approximately equal (allowing for numerical precision)
        assert np.allclose(sh_original, sh_back, rtol=1e-5)


class TestMaskProcessing:
    """Test mask processing functions"""
    
    def test_clean_binary_mask_basic(self):
        """Test basic mask cleaning"""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True  # Create a cube
        
        cleaned = clean_binary_mask(mask)
        
        assert cleaned.shape == mask.shape
        assert cleaned.dtype == np.uint8
        assert np.sum(cleaned) > 0
    
    def test_clean_binary_mask_with_holes(self):
        """Test mask cleaning with holes"""
        mask = np.ones((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = False  # Create a hole
        
        cleaned = clean_binary_mask(mask)
        
        assert cleaned.shape == mask.shape
        assert cleaned[5, 5, 5] == 1  # Hole should be filled
    
    def test_clean_binary_mask_invalid_inputs(self):
        """Test error handling for clean_binary_mask"""
        with pytest.raises(DataValidationError):
            clean_binary_mask(np.zeros((5, 5)))  # Wrong dimensions
        
        with pytest.raises(DataValidationError):
            clean_binary_mask("invalid")  # Wrong type


class TestAcquisitionView:
    """Test acquisition view detection"""
    
    def test_get_acquisition_view_axial(self):
        """Test axial acquisition detection"""
        affine = np.diag([2, 2, 2, 1])  # Standard axial
        view = get_acquisition_view(affine)
        assert view == "axial"
    
    def test_get_acquisition_view_oblique(self):
        """Test oblique acquisition detection"""
        affine = np.array([[1, 0.5, 0, 0],
                          [0.5, 1, 0, 0],
                          [0, 0, 2, 0],
                          [0, 0, 0, 1]])
        view = get_acquisition_view(affine)
        assert view == "oblique"
    
    def test_get_acquisition_view_invalid_inputs(self):
        """Test error handling for get_acquisition_view"""
        with pytest.raises(DataValidationError):
            get_acquisition_view(np.zeros((3, 3)))  # Wrong shape
        
        with pytest.raises(DataValidationError):
            get_acquisition_view("invalid")  # Wrong type


class TestNiftiIO:
    """Test NIfTI I/O functions"""
    
    def test_load_save_nifti_roundtrip(self):
        """Test loading and saving NIfTI files"""
        # Create test data
        data = np.random.rand(10, 10, 10)
        affine = np.eye(4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.nii.gz"
            
            # Save and load
            save_nifti_image(data, affine, filepath)
            loaded_data, loaded_affine = load_nifti_image(filepath)
            
            assert np.allclose(data, loaded_data)
            assert np.allclose(affine, loaded_affine)
    
    def test_load_nifti_nonexistent_file(self):
        """Test error handling for nonexistent file"""
        with pytest.raises(DataValidationError):
            load_nifti_image("nonexistent.nii.gz")
    
    def test_save_nifti_invalid_inputs(self):
        """Test error handling for save_nifti_image"""
        with pytest.raises(DataValidationError):
            save_nifti_image("invalid", np.eye(4), "test.nii.gz")
        
        with pytest.raises(DataValidationError):
            save_nifti_image(np.zeros((5, 5, 5)), np.zeros((3, 3)), "test.nii.gz")


class TestImageNormalization:
    """Test image normalization functions"""
    
    def test_normalize_image_basic(self):
        """Test basic image normalization"""
        img = np.array([0, 50, 100])
        normalized = normalize_image(img, 100, 0, 1, 0)
        
        expected = np.array([0, 0.5, 1])
        assert np.allclose(normalized, expected)
    
    def test_unnormalize_image_basic(self):
        """Test basic image unnormalization"""
        img = np.array([0, 0.5, 1])
        unnormalized = unnormalize_image(img, 100, 0, 1, 0)
        
        expected = np.array([0, 50, 100])
        assert np.allclose(unnormalized, expected)
    
    def test_normalize_unnormalize_roundtrip(self):
        """Test normalization roundtrip"""
        original = np.random.rand(5, 5, 5) * 100
        normalized = normalize_image(original, 100, 0, 1, 0)
        unnormalized = unnormalize_image(normalized, 100, 0, 1, 0)
        
        assert np.allclose(original, unnormalized)
    
    def test_normalize_image_invalid_inputs(self):
        """Test error handling for normalize_image"""
        with pytest.raises(DataValidationError):
            normalize_image("invalid", 100, 0, 1, 0)
        
        with pytest.raises(DataValidationError):
            normalize_image(np.zeros((5, 5)), 0, 100, 1, 0)  # max < min


class TestTorchConversion:
    """Test PyTorch format conversion functions"""
    
    def test_nifti_to_torch_format(self):
        """Test NIfTI to PyTorch format conversion"""
        nii_img = np.random.rand(10, 10, 10, 3)
        torch_img = nifti_to_torch_format(nii_img)
        
        assert torch_img.shape == (1, 3, 10, 10, 10)
    
    def test_torch_to_nifti_format(self):
        """Test PyTorch to NIfTI format conversion"""
        torch_img = np.random.rand(1, 3, 10, 10, 10)
        nii_img = torch_to_nifti_format(torch_img)
        
        assert nii_img.shape == (10, 10, 10, 3)
    
    def test_torch_conversion_roundtrip(self):
        """Test PyTorch conversion roundtrip"""
        original = np.random.rand(5, 5, 5, 2)
        torch_format = nifti_to_torch_format(original)
        back_to_nifti = torch_to_nifti_format(torch_format)
        
        assert np.allclose(original, back_to_nifti)
    
    def test_torch_conversion_invalid_inputs(self):
        """Test error handling for torch conversions"""
        with pytest.raises(DataValidationError):
            nifti_to_torch_format(np.zeros((5, 5, 5)))  # Wrong dimensions
        
        with pytest.raises(DataValidationError):
            torch_to_nifti_format(np.zeros((5, 5, 5, 5)))  # Wrong dimensions


class TestGeometricOperations:
    """Test geometric transformation functions"""
    
    def test_generate_random_unit_vector(self):
        """Test random unit vector generation"""
        vector = generate_random_unit_vector()
        
        assert vector.shape == (3,)
        assert np.allclose(np.linalg.norm(vector), 1.0)
    
    def test_rodrigues_to_rotation_matrix(self):
        """Test Rodrigues to rotation matrix conversion"""
        k = np.array([0, 0, 1])  # Rotation around z-axis
        theta = np.pi / 2  # 90 degrees
        
        R = rodrigues_to_rotation_matrix(k, theta)
        
        assert R.shape == (3, 3)
        assert np.allclose(np.linalg.det(R), 1.0)  # Proper rotation
        assert np.allclose(R @ R.T, np.eye(3))  # Orthogonal
    
    def test_rotation_translation_to_transform(self):
        """Test rotation and translation to transform matrix"""
        R = np.eye(3)
        t = np.array([1, 2, 3])
        
        T = rotation_translation_to_transform(R, t)
        
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3], R)
        assert np.allclose(T[:3, 3], t)
        assert np.allclose(T[3, :], [0, 0, 0, 1])
    
    def test_apply_transform_to_volume(self):
        """Test volume transformation"""
        vol = np.random.rand(10, 10, 10)
        xform = np.eye(4)  # Identity transform
        
        transformed = apply_transform_to_volume(xform, vol)
        
        assert transformed.shape == vol.shape
        # Identity transform should preserve the volume approximately
        assert np.allclose(transformed, vol, rtol=1e-1)
    
    def test_geometric_operations_invalid_inputs(self):
        """Test error handling for geometric operations"""
        with pytest.raises(DataValidationError):
            rodrigues_to_rotation_matrix(np.array([1, 2]), 1.0)  # Wrong k shape
        
        with pytest.raises(DataValidationError):
            rotation_translation_to_transform(np.eye(2), np.array([1, 2, 3]))  # Wrong R shape
        
        with pytest.raises(DataValidationError):
            apply_transform_to_volume(np.eye(3), np.zeros((5, 5, 5)))  # Wrong xform shape


class TestLegacyImageUtils:
    """Test legacy ImageUtils class"""
    
    def test_legacy_load_save_image(self):
        """Test legacy image loading and saving"""
        data = np.random.rand(5, 5, 5)
        affine = np.eye(4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.nii.gz"
            
            # Save using new method
            save_nifti_image(data, affine, filepath)
            
            # Load using legacy method
            loaded_data = ImageUtils.load_image(filepath)
            
            assert np.allclose(data, loaded_data)
    
    def test_legacy_apply_mask(self):
        """Test legacy mask application"""
        image_data = np.ones((5, 5, 5))
        mask = np.zeros((5, 5, 5))
        mask[2, 2, 2] = 1
        
        masked = ImageUtils.apply_mask(image_data, mask)
        
        assert masked.shape == image_data.shape
        assert masked[2, 2, 2] == 1
        assert masked[0, 0, 0] == 0
    
    def test_legacy_apply_mask_4d(self):
        """Test legacy mask application on 4D data"""
        image_data = np.ones((5, 5, 5, 3))
        mask = np.zeros((5, 5, 5))
        mask[2, 2, 2] = 1
        
        masked = ImageUtils.apply_mask(image_data, mask)
        
        assert masked.shape == image_data.shape
        assert np.all(masked[2, 2, 2, :] == 1)
        assert np.all(masked[0, 0, 0, :] == 0)


class TestErrorHandling:
    """Test error handling across all functions"""
    
    def test_image_processing_error_inheritance(self):
        """Test that ImageProcessingError inherits from ElikopyError"""
        from elikopy.infrastructure.exceptions import ElikopyError
        
        error = ImageProcessingError("test error")
        assert isinstance(error, ElikopyError)
    
    def test_data_validation_error_messages(self):
        """Test that DataValidationError provides meaningful messages"""
        with pytest.raises(DataValidationError) as exc_info:
            deltas_to_diffusion_tensor("invalid", 0.0, 0.0)
        
        assert "numeric" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__])