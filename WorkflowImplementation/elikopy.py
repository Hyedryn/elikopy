"""
 
"""
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

def dicom_to_nifti(folder_path):
    """Convert dicom data into nifti. Converted dicom are then moved to a sub-folder named original_data
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    Returns
    -------
    /
    Notes
    -----
    Dicom can be organised into sub folder or in the root folder
    Examples
    --------
    dicom_to_nifti("C:\Memoire\example_data\")
    """


def patient_list(folder_path):
    """Verify the validity of all the nifti present in the root folder. If some nifti does not posses an associated bval
     and bvec file, they are discarded and the user is notified by the mean of a summary file named patient_error.txt generated
     in the out sub-directory. All the valid patient are stored in a file named patient_list.json
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    Notes
    -----
    /
    Examples
    --------
    patient_list("C:\Memoire\example_data\")
    """


def preproc(folder_path, eddy=False, denoising=False):
    """Perform bet and optionnaly eddy and denoising. Generated data are stored in bet, eddy, denoising and final directory
    located in the folder out/preproc
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    eddy: If True, eddy is called
    denoising: If True, denoising is called
    Notes
    -----
    All the function executed after this function MUST take input data from folder_path/out/preproc/final
    Examples
    --------
    preproc("C:\Memoire\example_data\")
    """


def dti(folder_path):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """
    
    # load the data======================================
    data, affine = load_nifti(hardi_fname)
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    # create the model===================================
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)
    # FA ================================================
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    save_nifti('name_fa.nii.gz', FA.astype(np.float32), affine)
    # colored FA ========================================
    RGB = dti.color_fa(FA, tenfit.evecs)
    save_nifti('name_fargb.nii.gz', np.array(255 * RGB, 'uint8'), affine)
    # Mean diffusivity ==================================
    MD = dti.mean_diffusivity(tenfit.evals)
    save_nifti('name_md.nii.gz', MD.astype(np.float32), affine)
    # Radial diffusivity ==================================
    RD = dti.radial_diffusivity(tenfit.evals)
    save_nifti('name_rd.nii.gz', RD.astype(np.float32), affine)
    # Axial diffusivity ==================================
    AD = dti.axial_diffusivity(tenfit.evals)
    save_nifti('name_ad.nii.gz', AD.astype(np.float32), affine)
    # eigen vectors =====================================
    save_nifti('name_evecs.nii.gz', tenfit.evecs.astype(np.float32), affine)
    # eigen values ======================================
    save_nifti('name_evals.nii.gz', tenfit.evals.astype(np.float32), affine)
    # diffusion tensor ====================================
    save_nifti('name_tensor.nii.gz', tenfit.quadratic_form.astype(np.float32), affine)


def total_workflow(folder_path, dicomToNifti=False, eddy=False, denoising=False, dti=False):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """
