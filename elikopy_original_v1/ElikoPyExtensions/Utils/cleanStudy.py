import os
import json
import shutil


def cleanPreproc(folder_path, p):

    preproc_path = os.path.join(folder_path, "subjects", p, "dMRI", "preproc")

    # Gibbs
    gibbs_corrected = os.path.join(preproc_path, "gibbs", p + "_gibbscorrected.nii.gz")
    if os.path.exists(gibbs_corrected):
        os.remove(gibbs_corrected)

    # Mppca
    denoised = os.path.join(preproc_path, "mppca", p + "_mppca.nii.gz")
    if os.path.exists(denoised):
        os.remove(denoised)

    # Topup
    topup_corr = os.path.join(preproc_path, "topup", p + "_topup_corr.nii.gz")
    if os.path.exists(topup_corr):
        os.remove(topup_corr)
    synb0DISCO_folder = os.path.join(preproc_path, "topup", "synb0-DisCo")
    if os.path.exists(synb0DISCO_folder):
        shutil.rmtree(synb0DISCO_folder)

    # Eddy
    eddy_corr = os.path.join(preproc_path, "eddy", p + "_eddy_corr.nii.gz")
    if os.path.exists(eddy_corr):
        os.remove(eddy_corr)
    eddy_residuals = os.path.join(preproc_path, "eddy", p + "_eddy_corr.eddy_residuals.nii.gz")
    if os.path.exists(eddy_residuals):
        os.remove(eddy_residuals)
    eddy_outlier_free_data = os.path.join(preproc_path, "eddy", p + "_eddy_corr.eddy_outlier_free_data.nii.gz")
    if os.path.exists(eddy_outlier_free_data):
        os.remove(eddy_outlier_free_data)
    eddy_range_cnr_maps = os.path.join(preproc_path, "eddy", p + "_eddy_corr.eddy_range_cnr_maps.nii.gz")
    if os.path.exists(eddy_range_cnr_maps):
        os.remove(eddy_range_cnr_maps)
    eddy_range_cnr_maps = os.path.join(preproc_path, "eddy", p + "_eddy_corr.eddy_cnr_maps.nii.gz")
    if os.path.exists(eddy_range_cnr_maps):
        os.remove(eddy_range_cnr_maps)

def cleanTractography(folder_path, p):

    # Delete all Trk files in folder
    tractography = os.path.join(folder_path, "subjects", p, "dMRI", "tractography")
    if os.path.exists(tractography):
        for f in os.listdir(tractography):
            if f.endswith(".trk"):
                os.remove(os.path.join(tractography, f))

def cleanStudy(folder_path, p):

    cleanPreproc(folder_path, p)
    cleanTractography(folder_path, p)