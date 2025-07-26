import os
import subprocess

import nibabel as nib
from elikopy.utils import dipy_fod_to_mrtrix

def siftComputation(folder_path, p, SIFT2=False, msmtCSD:bool=True, core_count=1):
    """
    Computes the SIFT2 algorithm using the fod and tractogram.

    Warning: DWI volumes often have a non-negligible B1 bias field, mostly due to high-density receiver coils.
    If left uncorrected, SIFT will incorrectly interpret this as a spatially-varying fibre density.
    Therefore bias field correction during the preprocessing is HIGHLY recommended when using SIFT.

    :param folder_path:
    :param p:
    :param SIFT2:
    :param msmtCSD:
    :param core_count:
    :return:
    """
    patient_path = p

    tracking_path = folder_path + '/subjects/' + patient_path + "/dMRI/tractography/"
    in_tracks = tracking_path + patient_path + '_tractogram.tck'

    if SIFT2:
        out_tracks = tracking_path + patient_path + '_tractogram_weight_sift2.txt'
        cmd = "tcksift2"
    else:
        out_tracks = tracking_path + patient_path + '_tractogram_sift.tck'
        cmd = "tcksift"

    if msmtCSD:
        if not os.path.isdir(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD/"):
            raise Exception("No MSMT-CSD folder found")
        in_fod = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD/"+patient_path + "_MSMT-CSD_WM_ODF.nii.gz"
    else:
        if not os.path.isdir(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"):
            raise Exception("No CSD folder found")
        if not os.path.isfile(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF_mrtrix.nii.gz"):
            img = nib.load(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF.nii.gz")
            data = dipy_fod_to_mrtrix(img.get_fdata())
            out = nib.Nifti1Image(data, img.affine, img.header)
            out.to_filename(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF_mrtrix.nii.gz")
        in_fod = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF_mrtrix.nii.gz"

    bashCommand = (cmd + " -nthreads " + str(core_count) + " " + in_tracks + " " + in_fod + " " + out_tracks + " -force ")

    tracking_log = open(tracking_path + "tractography_logs.txt", "a+")
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tracking_log,
                               stderr=subprocess.STDOUT)

    process.communicate()

    tracking_log.close()

    from dipy.io.streamline import load_tractogram, save_trk
    dwi_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz'
    tract = load_tractogram(out_tracks, dwi_path)
    save_trk(tract, out_tracks[:-3]+'trk')
