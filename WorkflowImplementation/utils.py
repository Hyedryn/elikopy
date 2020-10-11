
import os
import os
import shutil
import json
import numpy as np
import math


def preproc_solo(folder_path, p, eddy=False, denoising=False):
    from dipy.io.image import load_nifti, save_nifti
    from dipy.segment.mask import median_otsu
    from dipy.denoise.localpca import mppca

    patient_path = os.path.splitext(p)[0]

    nifti_path = folder_path + '/' + patient_path + '.nii.gz'
    data, affine = load_nifti(nifti_path)

    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]))
    save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
    save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_mask.nii.gz', b0_mask.astype(np.float32), affine)
    if not(denoising) and not(eddy):
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '.nii.gz', b0_mask.astype(np.float32),affine)
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32),affine)
        shutil.copyfile(folder_path + "/" + patient_path + ".bval", folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
        shutil.copyfile(folder_path + "/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")


    if denoising:
        denoising_path = folder_path + "/out/preproc/denoising"
        if not(os.path.exists(denoising_path)):
            try:
                os.mkdir(denoising_path)
            except OSError:
                print ("Creation of the directory %s failed" % denoising_path)
            else:
                print ("Successfully created the directory %s " % denoising_path)

        pr = math.ceil((np.shape(b0_mask)[3] ** (1 / 3) - 1) / 2)
        denoised = mppca(b0_mask, patch_radius=pr)
        save_nifti(denoising_path + '/' + patient_path + '_mask_denoised.nii.gz', denoised.astype(np.float32), affine)
        if not eddy:
            save_nifti(folder_path + '/out/preproc/final/' + patient_path + '.nii.gz', denoised.astype(np.float32),affine)
            save_nifti(folder_path + '/out/preproc/final/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
            shutil.copyfile(folder_path + "/" + patient_path + ".bval",folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
            shutil.copyfile(folder_path + "/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")

    if eddy:
        eddy_path = folder_path + "/out/preproc/eddy"
        if not(os.path.exists(eddy_path)):
            try:
                os.mkdir(eddy_path)
            except OSError:
                print ("Creation of the directory %s failed" % eddy_path)
            else:
                print ("Successfully created the directory %s " % eddy_path)

        if denoising:
            bashCommand = 'eddy --imain=' + folder_path  + '/out/preproc/denoising/' + patient_path + '_mask_denoised.nii.gz --mask=' + folder_path  + '/out/preproc/bet/' +  patient_path + '_bet.nii.gz --acqp="' + folder_path + '/acqparams.txt" --index="' + folder_path + '/index.txt" --bvecs="' + folder_path + '/' + patient_path + '.bvec" --bvals="' + folder_path + '/' + patient_path + '.bval" --out="' + folder_path + '/out/preproc/eddy/' + patient_path + '_mfc" --verbose'
        else:
            bashCommand = 'eddy --imain=' + folder_path  + '/' + patient_path + '.nii.gz --mask=' + folder_path  + '/out/preproc/bet/' +  patient_path + '_bet.nii.gz --acqp="' + folder_path + '/acqparams.txt" --index="' + folder_path + '/index.txt" --bvecs="' + folder_path + '/' + patient_path + '.bvec" --bvals="' + folder_path + '/' + patient_path + '.bval" --out="' + folder_path + '/out/preproc/eddy/' + patient_path + '_mfc" --verbose'

        import subprocess
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)

        #wait until eddy finish
        output, error = process.communicate()

        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + ".bval",folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")
        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + "_mfc.nii.gz",folder_path + "/out/preproc/final" + "/" + patient_path + ".nii.gz")
        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + "_bet_mfc.nii.gz",folder_path + "/out/preproc/final" + "/" + patient_path + "_binary_mask.nii.gz")
