import elikopy
import os
import subprocess
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti
import dipy.tracking
from dipy.io.streamline import load_tractogram
from scipy.ndimage import binary_dilation
from skimage.morphology import ball

def connectivityMatrix(folder_path, p, label_fname, input="TCKGEN", inclusive=False, dilation_radius=0, longitudinal=False, tractogram_filename="tractogram", suffix=""):

    assert input in ["TCKGEN", "SIFT", "SIFT2"], "input must be either TCKGEN, SIFT or SIFT2"

    dwi_path = folder_path + '/subjects/' + p + '/dMRI/preproc/' + p + '_dmri_preproc.nii.gz'

    odf_wmfod = folder_path + '/subjects/' + p + "/dMRI/ODF/MSMT-CSD/" + p + "_MSMT-CSD_WM_ODF.nii.gz"
    odf_csd_mrtrix = folder_path + '/subjects/' + p + "/dMRI/ODF/CSD/" + p + "_CSD_SH_ODF_mrtrix.nii.gz"
    if os.path.exists(odf_wmfod):
        img = nib.load(odf_wmfod)
    elif os.path.exists(odf_csd_mrtrix):
        img = nib.load(odf_csd_mrtrix)
    else:
        raise Exception("No ODF file found in " + folder_path + '/subjects/' + p + "/dMRI/ODF/")
        return
        

    reg_path = folder_path + '/subjects/' + p + '/reg/'
    if longitudinal:
        label_fpath = os.path.join(reg_path, p + "_Atlas_" + label_fname + "_longitudinal.nii.gz")
    else:
        label_fpath = os.path.join(reg_path, p + "_Atlas_" + label_fname + ".nii.gz")
    labels_nii = nib.load(label_fpath)
    labels = np.round(labels_nii.get_fdata()).astype(int)
    
    tracking_path = folder_path + '/subjects/' + p + "/dMRI/tractography/"
    
    if dilation_radius>0:
        # Create a spherical structuring element for dilation
        #struct_el = ball(dilation_radius)
        
        n_dim = len(labels.shape)
        struct_el = np.zeros((3,)*n_dim)
        for d in range(n_dim):
            idc = tuple([slice(None) if d == i else 1 for i in range(n_dim)])
            struct_el[idc] = 1

        # Dilate each label separately (assuming labels are integer values starting from 1)
        dilated_label_data = np.zeros_like(labels)
        solo_labels = np.unique(labels)[1:]  # Exclude background (0)
        for label in solo_labels:
            mask = labels == label
            dilated_mask = binary_dilation(mask, structure=struct_el)
            dilated_label_data[dilated_mask] = label
        
        for label in solo_labels:
            mask = labels == label
            dilated_label_data[mask] = label
        
        labels = np.round(dilated_label_data).astype(int)
        print("Shape:", dilated_label_data.shape)
        dilated_label_img = nib.Nifti1Image(dilated_label_data, labels_nii.affine, dtype=np.uint16)
        nib.save(dilated_label_img, tracking_path + f'dilated_{dilation_radius}_labels.nii.gz')

    
    if input == "TCKGEN":
        tractogram_path = tracking_path + p + f'_{tractogram_filename}.tck'
    elif input == "SIFT":
        tractogram_path = tracking_path + p + f'_{tractogram_filename}_sift.tck'
    elif input == "SIFT2":
        raise Exception("SIFT2 not implemented yet")
        pass # TODO
    else:
        raise Exception("Invalid input type")

    if os.path.exists(tractogram_path):
        tractogram = load_tractogram(tractogram_path, dwi_path)
    else:
        raise Exception("No tractogram found in " + tracking_path)
        return

    # Compute the connectivity matrix
    MV, grouping = dipy.tracking.utils.connectivity_matrix(tractogram.streamlines, img.affine, labels, inclusive=inclusive, return_mapping=True, mapping_as_streamlines=True)

    MV = np.delete(MV, 0, 0)
    MV = np.delete(MV, 0, 1)


    import matplotlib.pyplot as plt
    plt.imshow(np.log1p(MV), interpolation='nearest')

    if longitudinal:
        np.save(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_time-longitudinal_connectivityMatrix{suffix}.npy", MV)
        plt.savefig(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_time-longitudinal_connectivityMatrix{suffix}.png")
    else:
        np.save(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_connectivityMatrix{suffix}.npy", MV)
        plt.savefig(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_connectivityMatrix{suffix}.png")
