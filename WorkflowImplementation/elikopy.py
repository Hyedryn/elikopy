"""
 
"""
import os
import json
from WorkflowImplementation.utils import preproc_solo


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
    bashCommand = 'dcm2niix -f "%i_%p_%z" -p y -z y -o ' + folder_path + ' ' + folder_path + ''
    import subprocess
    bashcmd = bashCommand.split()
    #print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)

    #wait until mricron finish
    output, error = process.communicate()


    #Move all old dicom to dicom folder
    import shutil
    import os

    dest = folder_path + "/dicom"
    files = os.listdir(folder_path)
    if not(os.path.exists(dest)):
        try:
            os.mkdir(dest)
        except OSError:
            print ("Creation of the directory %s failed" % dest)
        else:
            print ("Successfully created the directory %s " % dest)

    for f in files:
        if "mrdc" in f or "MRDC" in f:
            shutil.move(folder_path + '/' + f, dest)


def patient_list(folder_path):
    """Verify the validity of all the nifti present in the root folder. If some nifti does not posses an associated bval
     and bvec file, they are discarded and the user is notified by the mean of a summary file named patient_error.json generated
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

    import os

    error = []
    success = []

    for file in os.listdir(folder_path):

        if file.endswith(".nii"):
            name = os.path.splitext(file)[0]
            bvec = os.path.splitext(file)[0] + ".bvec"
            bval = os.path.splitext(file)[0] + ".bval"
            if bvec not in os.listdir(folder_path) or bval not in os.listdir(folder_path):
                error.append(name)
            else:
                success.append(name)

        if file.endswith(".nii.gz"):
            name = os.path.splitext(os.path.splitext(file)[0])[0]
            bvec = os.path.splitext(os.path.splitext(file)[0])[0] + ".bvec"
            bval = os.path.splitext(os.path.splitext(file)[0])[0] + ".bval"
            if bvec not in os.listdir(folder_path) or bval not in os.listdir(folder_path):
                error.append(name)
            else:
                success.append(name)

    error = list(dict.fromkeys(error))
    success = list(dict.fromkeys(success))

    dest = folder_path + '/out'
    if not (os.path.exists(dest)):
        try:
            os.mkdir(dest)
        except OSError:
            print("Creation of the directory %s failed" % dest)
        else:
            print("Successfully created the directory %s " % dest)

    import json
    dest_error = folder_path + "/out/patient_error.json"
    with open(dest_error, 'w') as f:
        json.dump(error, f)

    dest_success = folder_path + "/out/patient_list.json"
    with open(dest_success, 'w') as f:
        json.dump(success, f)


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



    dest_success = folder_path + "/out/patient_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    preproc_path = folder_path + "/out/preproc"
    if not (os.path.exists(preproc_path)):
        try:
            os.mkdir(preproc_path)
        except OSError:
            print("Creation of the directory %s failed" % preproc_path)
        else:
            print("Successfully created the directory %s " % preproc_path)

    bet_path = folder_path + "/out/preproc/bet"
    if not (os.path.exists(bet_path)):
        try:
            os.mkdir(bet_path)
        except OSError:
            print("Creation of the directory %s failed" % bet_path)
        else:
            print("Successfully created the directory %s " % bet_path)

    final_path = folder_path + "/out/preproc/final"
    try:
        os.mkdir(final_path)
    except OSError:
        print("Creation of the directory %s failed" % final_path)
    else:
        print("Successfully created the directory %s " % final_path)

    for p in patient_list:
        preproc_solo(folder_path,p,eddy,denoising,)

def dti(folder_path):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """
    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    import dipy.reconst.dti as dti
    import os
    import json

    dti_path = folder_path + "/out/dti"
    try:
        os.mkdir(dti_path)
    except OSError:
        print("Creation of the directory %s failed" % dti_path)
    else:
        print("Successfully created the directory %s " % dti_path)

    dest_success = folder_path + "/out/patient_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    for p in patient_list:
        patient_path = os.path.splitext(p)[0]
        # load the data======================================
        data, affine = load_nifti(folder_path + "/out/preproc/final" + "/" + patient_path + ".nii.gz")
        mask, _ = load_nifti(folder_path + "/out/preproc/final" + "/" + patient_path + "_binary_mask.nii.gz")
        bvals, bvecs = read_bvals_bvecs(folder_path + "/out/preproc/final" + "/" + patient_path + ".bval", folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")
        # create the model===================================
        gtab = gradient_table(bvals, bvecs)
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask=mask)
        # FA ================================================
        FA = dti.fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        FA = np.clip(FA, 0, 1)
        save_nifti(folder_path + "/out/dti/" + patient_path + "_fa.nii.gz", FA.astype(np.float32), affine)
        # colored FA ========================================
        RGB = dti.color_fa(FA, tenfit.evecs)
        save_nifti(folder_path + "/out/dti/" + patient_path + "_fargb.nii.gz", np.array(255 * RGB, 'uint8'), affine)
        # Mean diffusivity ==================================
        MD = dti.mean_diffusivity(tenfit.evals)
        save_nifti(folder_path + "/out/dti/" + patient_path + "_md.nii.gz", MD.astype(np.float32), affine)
        # Radial diffusivity ==================================
        RD = dti.radial_diffusivity(tenfit.evals)
        save_nifti(folder_path + "/out/dti/" + patient_path + "_rd.nii.gz", RD.astype(np.float32), affine)
        # Axial diffusivity ==================================
        AD = dti.axial_diffusivity(tenfit.evals)
        save_nifti(folder_path + "/out/dti/" + patient_path + "_ad.nii.gz", AD.astype(np.float32), affine)
        # eigen vectors =====================================
        save_nifti(folder_path + "/out/dti/" + patient_path + "_evecs.nii.gz", tenfit.evecs.astype(np.float32), affine)
        # eigen values ======================================
        save_nifti(folder_path + "/out/dti/" + patient_path + "_evals.nii.gz", tenfit.evals.astype(np.float32), affine)
        # diffusion tensor ====================================
        save_nifti(folder_path + "/out/dti/" + patient_path + "_dtensor.nii.gz", tenfit.quadratic_form.astype(np.float32), affine)


def fingerprinting(folder_path):
    """Perform fingerprinting and store the data in the out/fingerprinting folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """

    import os

    fingerprinting_path = ""
    os.path.join(fingerprinting_path, folder_path, "/out/fingerprinting")
    try:
        os.mkdir(fingerprinting_path)
    except OSError:
        print("Creation of the directory %s failed" % fingerprinting_path)
    else:
        print("Successfully created the directory %s " % fingerprinting_path)


    import os
    import sys
    import json

    import microstructure_fingerprinting as mf
    import microstructure_fingerprinting.mf_utils as mfu

    dictionary_file = 'mf_dictionary.mat'

    # Instantiate model:
    mf_model = mf.MFModel(dictionary_file)

    patient_list = json.load(folder_path + "/out/patient_list.json")

    for p in patient_list:

        patient_path = os.path.splitext(p)[0]

        # Fit to data:
        MF_fit = mf_model.fit(folder_path + "/out/preproc/final" + "/" + patient_path + ".nii.gz",  # help(mf_model.fit)
                              maskfile,
                              numfasc,  # all arguments after this MUST be named: argname=argvalue
                              peaks=peaks,
                              bvals=folder_path + "/out/preproc/final" + "/" + patient_path + ".bval",
                              bvecs=folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec",
                              csf_mask=csf_mask,
                              ear_mask=ear_mask,
                              verbose=3,
                              parallel=False
                              )

        # Save estimated parameter maps as NIfTI files:
        outputbasename = 'MF_' + patient_path
        MF_fit.write_nifti(outputbasename)


def total_workflow(folder_path, dicomToNifti=False, eddy=False, denoising=False, dti=False):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """