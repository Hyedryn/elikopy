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

    bashCommand = 'dcm2niix -f "%n_%p_%z_%i" -p y -z y -ba n ' + folder_path
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

    #wait until mricron finish
    output, error = process.communicate()


    #Move all old dicom to dicom folder
    import shutil
    import os

    dest = ""
    os.path.join(dest, folder_path, "/dicom")
    files = os.listdir(folder_path)

    for f in files:
        if f.find("mrdc") or f.find("MRDC"):
            shutil.move(f, dest)






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

    directory = os.fsencode(folder_path)

    error = []
    success = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".nii"):
            # print(os.path.join(directory, filename))
            name = os.path.splitext(filename)[0]
            bvec = os.path.splitext(filename)[0] + ".bvec"
            bval = os.path.splitext(filename)[0] + ".bval"
            if bvec not in os.listdir(directory) or bval not in os.listdir(directory):
                error = error.append(filename)
            else:
                success = success.append(name)

        if filename.endswith(".nii.gz"):
            # print(os.path.join(directory, filename))
            name = os.path.splitext(filename)[1]
            bvec = os.path.splitext(filename)[1] + ".bvec"
            bval = os.path.splitext(filename)[1] + ".bval"
            if bvec not in os.listdir(directory) or bval not in os.listdir(directory):
                error = error.append(filename)
            else:
                success = success.append(name)

    import json
    dest_error = ""
    os.path.join(dest_error, folder_path, "/out/patient_error.json")
    with open(dest_error, 'w') as f:
        json.dump(error, f)

    dest_success = ""
    os.path.join(dest_success, folder_path, "/out/patient_list.json")
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

    import os
    import json
    with open("JSON Directory") as BOB:
        patient_list = json.load(folder_path + "/out/patient_list.json")

    from os.path import join as pjoin
    import numpy as np
    from dipy.data import get_fnames
    from dipy.io.image import load_nifti, save_nifti
    from dipy.segment.mask import median_otsu

    for p in patient_list:
        patient_path = os.path.splitext(p)[0]

        data, affine = load_nifti(folder_path + '/' + patient_path + '.nii')
        data = np.squeeze(data)

        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
        save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
        save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_mask.nii.gz', b0_mask.astype(np.float32), affine)

    if denoising:
        denoising_path = ""
        os.path.join(denoising_path, folder_path, "/out/preproc/denoising")
        try:
            os.mkdir(denoising_path)
        except OSError:
            print ("Creation of the directory %s failed" % denoising_path)
        else:
            print ("Successfully created the directory %s " % denoising_path)

    if eddy:
        eddy_path = ""
        os.path.join(eddy_path, folder_path, "/out/preproc/eddy")
        try:
            os.mkdir(eddy_path)
        except OSError:
            print ("Creation of the directory %s failed" % eddy_path)
        else:
            print ("Successfully created the directory %s " % eddy_path)

        for p in patient_list:
            patient_path = os.path.splitext(p)[0]
            bashCommand = 'eddy --imain=' + folder_path  + '/' + patient_path + '.nifti --mask=' + folder_path  + '/out/preproc/bet/' +  patient_path + '_bet.nifti --acqp="' + folder_path + '/acqparams.txt" --index="' + folder_path + '/index.txt" --bvecs="' + folder_path + '/' + patient_path + '.bvec" --bvals="' + folder_path + '/' + patient_path + '.bval" --out="' + folder_path + '/out/preproc/eddy/' + patient_path + '_mfc" --verbose'
            import subprocess
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

            #wait until eddy finish
            output, error = process.communicate()

    final_path = ""
    os.path.join(final_path, folder_path, "/out/preproc/final")
    try:
        os.mkdir(final_path)
    except OSError:
        print ("Creation of the directory %s failed" % final_path)
    else:
        print ("Successfully created the directory %s " % final_path)

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
