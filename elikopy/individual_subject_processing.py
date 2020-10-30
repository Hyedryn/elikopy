import datetime
import os
import os
import shutil
import json
import numpy as np
import math

from future.utils import iteritems
import subprocess

from dipy.denoise.gibbs import gibbs_removal


def preproc_solo(folder_path, p, eddy=False, denoising=False, reslice=False, gibbs=False):

    patient_path = os.path.splitext(p)[0]
    preproc_path = folder_path + '/' + patient_path + "/dMRI/preproc/bet"
    if not(os.path.exists(preproc_path)):
        try:
            os.makedirs(preproc_path)
        except OSError:
            print ("Creation of the directory %s failed" % preproc_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % preproc_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % preproc_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % preproc_path)
            f.close()

    mask_path = folder_path + '/' + patient_path + "/masks"
    if not(os.path.exists(mask_path)):
        try:
            os.makedirs(mask_path)
        except OSError:
            print ("Creation of the directory %s failed" % mask_path)
            f=open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % mask_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % mask_path)
            f=open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % mask_path)
            f.close()

    print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f.close()
    from dipy.io.image import load_nifti, save_nifti
    from dipy.segment.mask import median_otsu
    from dipy.denoise.localpca import mppca

    nifti_path = folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.nii.gz'
    data, affine, voxel_size = load_nifti(nifti_path, return_voxsize=True)

    if reslice:
        reslice_path = folder_path + '/' + patient_path + "/dMRI/preproc/reslice"
        if not(os.path.exists(reslice_path)):
            try:
                os.makedirs(reslice_path)
            except OSError:
                print ("Creation of the directory %s failed" % reslice_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % reslice_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % reslice_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % reslice_path)
                f.close()

        from dipy.align.reslice import reslice
        new_voxel_size = (2., 2., 2.)
        data, affine = reslice(data, affine, voxel_size, new_voxel_size)
        save_nifti(reslice_path + '/' + patient_path + '_reslice.nii.gz', data, affine)
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f.close()

    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]))
    save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz', b0_mask.astype(np.float32), affine)
    print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
    f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
    f.close()

    if not(denoising) and not(eddy) and not(gibbs):
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz', b0_mask.astype(np.float32),affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz', mask.astype(np.float32),affine)
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval", folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")


    if denoising:
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of denoising for patient %s \n" % p)
        f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Denoising launched for patient %s \n" % p)
        f.close()

        denoising_path = folder_path + '/' + patient_path + '/dMRI/preproc/mppca'
        if not(os.path.exists(denoising_path)):
            try:
                os.mkdir(denoising_path)
            except OSError:
                print ("Creation of the directory %s failed" % denoising_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % denoising_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % denoising_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % denoising_path)
                f.close()

        pr = math.ceil((np.shape(b0_mask)[3] ** (1 / 3) - 1) / 2)
        denoised = mppca(b0_mask, patch_radius=pr)
        save_nifti(denoising_path + '/' + patient_path + '_mppca.nii.gz', denoised.astype(np.float32), affine)
        #save_nifti(denoising_path + '/' + patient_path + '_mppca_mask.nii.gz', mask.astype(np.float32),affine)

        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of denoising for patient %s \n" % p)

        #nifti_path = folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz'
        #data, affine = load_nifti(nifti_path)
        b0_mask = denoised

        if not eddy and not gibbs:
            save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz', b0_mask.astype(np.float32),affine)
            save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz', mask.astype(np.float32),affine)
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval", folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    if gibbs:

        gibbs_path = folder_path + '/' + patient_path + '/dMRI/preproc/gibbs'
        if not(os.path.exists(gibbs_path)):
            try:
                os.mkdir(gibbs_path)
            except OSError:
                print ("Creation of the directory %s failed" % gibbs_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % gibbs_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % gibbs_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % gibbs_path)
                f.close()

        data = gibbs_removal(b0_mask)
        corrected_path = folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/" + patient_path + '_gibbscorrected.nii.gz'
        save_nifti(corrected_path, data.astype(np.float32), affine)

        if not eddy:
            save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz', data.astype(np.float32),affine)
            save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz', mask.astype(np.float32),affine)
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval", folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")


    #Explicitly freeing memory
    import gc
    denoised = None
    b0_mask = None
    mask = None
    data= None
    affine = None
    gc.collect()


    if eddy:
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of eddy for patient %s \n" % p)

        eddy_path = folder_path + '/' + patient_path + "/dMRI/preproc/eddy"
        if not(os.path.exists(eddy_path)):
            try:
                os.makedirs(eddy_path)
            except OSError:
                print ("Creation of the directory %s failed" % eddy_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % eddy_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % eddy_path)
                f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % eddy_path)
                f.close()

        if gibbs:
            bashCommand = 'eddy --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' +  patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose'
        elif denoising:
            bashCommand = 'eddy --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' +  patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose'
        else:
            bashCommand = 'eddy --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' +  patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose'

        import subprocess
        bashcmd = bashCommand.split()
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f.close()

        eddy_log = open(folder_path + '/' + patient_path + "/dMRI/preproc/eddy/eddy_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=eddy_log, stderr=subprocess.STDOUT)

        #wait until eddy finish
        output, error = process.communicate()
        eddy_log.close()

        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f.close()

        data, affine = load_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + "_eddy_corr.nii.gz")
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]))
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz', b0_mask.astype(np.float32),affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz', mask.astype(np.float32),affine)
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval", folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + "_eddy_corr.eddy_rotated_bvecs",folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f=open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def dti_solo(folder_path, p):
    print("[DTI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual DTI processing for patient %s \n" % p)

    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    import dipy.reconst.dti as dti

    patient_path = os.path.splitext(p)[0]

    dti_path = folder_path + '/' + patient_path + "/dMRI/microstructure/dti"
    if not(os.path.exists(dti_path)):
        try:
            os.makedirs(dti_path)
        except OSError:
            print ("Creation of the directory %s failed" % dti_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dti_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % dti_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dti_path)
            f.close()

    # load the data======================================
    data, affine = load_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")
    bvals, bvecs = read_bvals_bvecs(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval", folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    # create the model===================================
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    # FA ================================================
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz", FA.astype(np.float32), affine)
    # colored FA ========================================
    RGB = dti.color_fa(FA, tenfit.evecs)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_fargb.nii.gz", np.array(255 * RGB, 'uint8'), affine)
    # Mean diffusivity ==================================
    MD = dti.mean_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_MD.nii.gz", MD.astype(np.float32), affine)
    # Radial diffusivity ==================================
    RD = dti.radial_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_RD.nii.gz", RD.astype(np.float32), affine)
    # Axial diffusivity ==================================
    AD = dti.axial_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_AD.nii.gz", AD.astype(np.float32), affine)
    # eigen vectors =====================================
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_evecs.nii.gz", tenfit.evecs.astype(np.float32), affine)
    # eigen values ======================================
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_evals.nii.gz", tenfit.evals.astype(np.float32), affine)
    # diffusion tensor ====================================
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_dtensor.nii.gz", tenfit.quadratic_form.astype(np.float32), affine)

    print("[DTI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
    f.write("[DTI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def white_mask_solo(folder_path, p):

    print("[White mask solo] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual white mask processing for patient %s \n" % p)

    from dipy.align.imaffine import (AffineMap, MutualInformationMetric, AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
    from dipy.segment.tissue import TissueClassifierHMRF
    from dipy.io.image import load_nifti, save_nifti
    import subprocess
    from dipy.denoise.gibbs import gibbs_removal
    from dipy.data import get_sphere
    import dipy.reconst.shm as shm
    import dipy.direction.peaks as dp
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table

    patient_path = os.path.splitext(p)[0]
    anat_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
    if os.path.isfile(anat_path):
        # Read the moving image ====================================
        anat_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
        data_gibbs, affine_gibbs = load_nifti(anat_path)
        data_gibbs = gibbs_removal(data_gibbs)
        corrected_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_gibbscorrected.nii.gz'
        save_nifti(corrected_path, data_gibbs.astype(np.float32), affine_gibbs)
        #anat_path = folder_path + '/anat/' + patient_path + '_T1.nii.gz'
        bet_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
        bashCommand = 'bet2 ' + corrected_path + ' ' + bet_path +' -f 1 -g -3'
        bashcmd = bashCommand.split()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)
        output, error = process.communicate()
        anat_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
        bet_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_brain_brain.nii.gz'
        bashCommand = 'bet2 ' + anat_path + ' ' + bet_path + ' -f 0.4 -g -0.2'
        bashcmd = bashCommand.split()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)
        output, error = process.communicate()
        moving_data, moving_affine = load_nifti(bet_path)
        moving = moving_data
        moving_grid2world = moving_affine
        # Read the static image ====================================
        static_data, static_affine = load_nifti(folder_path + "/" + patient_path + "/dMRI/preproc/" + patient_path + "_dmri_preproc.nii.gz")
        static = np.squeeze(static_data)[..., 0]
        static_grid2world = static_affine
        # Reslice the moving image ====================================
        identity = np.eye(4)
        affine_map = AffineMap(identity, static.shape, static_grid2world, moving.shape, moving_grid2world)
        # translation the moving image ====================================
        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)
        level_iters = [10000, 1000, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]
        affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
        transform = TranslationTransform3D()
        params0 = None
        starting_affine = affine_map.affine
        translation = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)
        # Rigid transform the moving image ====================================
        transform = RigidTransform3D()
        params0 = None
        starting_affine = translation.affine
        rigid = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)
        # affine transform the moving image ====================================
        transform = AffineTransform3D()
        params0 = None
        starting_affine = rigid.affine
        affine = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)
        transformed = affine.transform(moving)
        # final result of registration ==========================================
        anat = transformed
        anat_affine = static_grid2world
        # make the white matter segmentation ===================================
        nclass = 3
        beta = 0.1
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(anat, nclass, beta)
        # save the white matter mask ============================================
        white_mask = np.where(final_segmentation == 3, 1, 0)
    else:
        # compute the white matter mask with the Anisotropic power map
        data, affine = load_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
        mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")
        bvals, bvecs = read_bvals_bvecs(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
        gtab = gradient_table(bvals, bvecs)
        sphere = get_sphere('symmetric724')
        qball_model = shm.QballModel(gtab, 8)
        peaks = dp.peaks_from_model(model=qball_model, data=data, relative_peak_threshold=.5, min_separation_angle=25, sphere=sphere, mask=mask)
        ap = shm.anisotropic_power(peaks.shm_coeff)
        nclass = 3
        beta = 0.1
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
        white_mask = np.where(final_segmentation == 3, 1, 0)
        anat_affine = affine

    mask_path = folder_path + '/' + patient_path + "/masks"
    if not(os.path.exists(mask_path)):
        try:
            os.makedirs(mask_path)
        except OSError:
            print ("Creation of the directory %s failed" % mask_path)
            f=open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % mask_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % mask_path)
            f=open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % mask_path)
            f.close()

    out_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    save_nifti(out_path, white_mask.astype(np.float32), anat_affine)

    print("[White mask solo] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
    f.write("[White mask solo] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def noddi_solo(folder_path, p):
    print("[NODDI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual NODDI processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs

    patient_path = os.path.splitext(p)[0]

    noddi_path = folder_path + '/' + patient_path + "/dMRI/microstructure/noddi"
    if not(os.path.exists(noddi_path)):
        try:
            os.makedirs(noddi_path)
        except OSError:
            print ("Creation of the directory %s failed" % noddi_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % noddi_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % noddi_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
            f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % noddi_path)
            f.close()

    # initialize the compartments model
    from dmipy.signal_models import cylinder_models, gaussian_models
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    # watson distribution of stick and Zepelin
    from dmipy.distributions.distribute_models import SD1WatsonDistributed
    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
    watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
    watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

    # build the NODDI model
    from dmipy.core.modeling_framework import MultiCompartmentModel
    NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])

    # fix the isotropic diffusivity
    NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

    # load the data
    data, affine = load_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    wm_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    if os.path.isfile(wm_path):
        mask, _ = load_nifti(wm_path)
    else:
        mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")

    # transform the bval, bvecs in a form suited for NODDI
    from dipy.core.gradients import gradient_table
    from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
    gtab_dipy = gradient_table(bvals, bvecs)
    acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy)

    # fit the model to the data
    NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask)

    # exctract the metrics
    fitted_parameters = NODDI_fit.fitted_parameters
    mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
    odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
    f_iso = fitted_parameters["partial_volume_0"]
    f_bundle = fitted_parameters["partial_volume_1"]
    f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
    f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']
    f_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters['partial_volume_1'])
    mse = NODDI_fit.mean_squared_error(data)

    # save the nifti
    save_nifti(noddi_path + '/' + patient_path + '_noddi_mu.nii.gz', mu.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_odi.nii.gz', odi.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fiso.nii.gz', f_iso.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fbundle.nii.gz', f_bundle.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fintra.nii.gz', f_intra.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_icvf.nii.gz', f_icvf.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fextra.nii.gz', f_extra.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_mse.nii.gz', mse.astype(np.float32), affine)

    print("[NODDI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
    f.write("[NODDI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def diamond_solo(folder_path, p, box):
    print("[DIAMOND SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual DIAMOND processing for patient %s \n" % p)
    patient_path = os.path.splitext(p)[0]

    diamond_path = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond"
    if not(os.path.exists(diamond_path)):
        try:
            os.makedirs(diamond_path)
        except OSError:
            print ("Creation of the directory %s failed" % diamond_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
            f.write("[DIAMOND SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % diamond_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % diamond_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
            f.write("[DIAMOND SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % diamond_path)
            f.close()

    #'--bbox 0,0,38,128,128,1'
    if box is not None:
        bashCommand = 'crlDCIEstimate -i ' + folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz' + ' -m ' + folder_path + '/' + patient_path + '/dMRI/masks/' + patient_path + '_brain_mask.nii.gz' + ' -n 3 --automose aicu --fascicle diamondcyl -o ' + folder_path + '/' + patient_path + '/dMRI/microstructure/diamond/' + patient_path + '_diamond.nii.gz' + ' -p 4'
    else:
        bashCommand = 'crlDCIEstimate -i ' + folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz' + ' -m ' + folder_path + '/' + patient_path + '/dMRI/masks/' + patient_path + '_brain_mask.nii.gz' + ' -n 3 --automose aicu --fascicle diamondcyl -o ' + folder_path + '/' + patient_path + '/dMRI/microstructure/diamond/' + patient_path + '_diamond.nii.gz' + ' -p 4'

    import subprocess
    bashcmd = bashCommand.split()
    print("[DIAMOND SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": crlDCIEstimate launched for patient %s \n" % p + " with bash command " + bashCommand)
    f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    f.write("[DIAMOND SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": crlDCIEstimate launched for patient %s \n" % p + " with bash command " + bashCommand)
    f.close()

    diamond_log = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=diamond_log, stderr=subprocess.STDOUT)

    output, error = process.communicate()
    diamond_log.close()

    print("[DIAMOND SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    f.write("[DIAMOND SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def mf_solo(folder_path, p, dictionary_path, CSD_bvalue = None):
    print("[MF SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual microstructure fingerprinting processing for patient %s \n" % p)
    patient_path = os.path.splitext(p)[0]

    mf_path = folder_path + '/' + patient_path + "/dMRI/microstructure/mf"
    if not(os.path.exists(mf_path)):
        try:
            os.makedirs(mf_path)
        except OSError:
            print ("Creation of the directory %s failed" % mf_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")
            f.write("[MF SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % mf_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % mf_path)
            f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")
            f.write("[MF SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % mf_path)
            f.close()

    # imports
    import microstructure_fingerprinting as mf
    import microstructure_fingerprinting.mf_utils as mfu
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response)
    from dipy.direction import peaks_from_model
    from dipy.data import default_sphere

    # load the data
    data, affine = load_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    wm_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    if os.path.isfile(wm_path):
        mask, _ = load_nifti(wm_path)
    else:
        mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask=.nii.gz")

    # compute numfasc and peaks
    diamond_path = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond"
    if os.path.exists(diamond_path):
        tensor_files0 = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t0.nrrd'
        tensor_files1 = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t1.nrrd'
        fracs_file = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_fractions.nrrd'
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=None, frac2=None, mu1=tensor_files0, mu2=tensor_files1,peakmode='tensor', mask=mask, frac12=fracs_file)
    else:
        sel_b = np.logical_or(bvals == 0, np.logical_and((CSD_bvalue-5) <= bvals, bvals <= (CSD_bvalue+5)))
        data_CSD = data[..., sel_b]
        gtab_CSD = gradient_table(bvals[sel_b], bvecs[sel_b])
        response, ratio = auto_response(gtab_CSD, data_CSD, roi_radius=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response, sh_order=6)
        csd_peaks = peaks_from_model(npeaks=2, model=csd_model, data=data_CSD, sphere=default_sphere,relative_peak_threshold=.15, min_separation_angle=25, parallel=True, mask=mask,normalize_peaks=True)
        save_nifti(mf_path + '/' + patient_path + '_mf_CSDpeaks.nii.gz', csd_peaks.peak_dirs, affine)
        save_nifti(mf_path + '/' + patient_path + '_mf_CSDvalues.nii.gz', csd_peaks.peak_values, affine)
        normPeaks0 = csd_peaks.peak_dirs[..., 0, :]
        normPeaks1 = csd_peaks.peak_dirs[..., 1, :]
        for i in range(np.shape(csd_peaks.peak_dirs)[0]):
            for j in range(np.shape(csd_peaks.peak_dirs)[1]):
                for k in range(np.shape(csd_peaks.peak_dirs)[2]):
                    norm = np.sqrt(np.sum(normPeaks0[i, j, k, :] ** 2))
                    normPeaks0[i, j, k, :] = normPeaks0[i, j, k, :] / norm
                    norm = np.sqrt(np.sum(normPeaks1[i, j, k, :] ** 2))
                    normPeaks1[i, j, k, :] = normPeaks1[i, j, k, :] / norm
        mu1 = normPeaks0
        mu2 = normPeaks1
        frac1 = csd_peaks.peak_values[..., 0]
        frac2 = csd_peaks.peak_values[..., 1]
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=frac1, frac2=frac2, mu1=mu1, mu2=mu2, peakmode='peaks',mask=mask, frac12=None)

    # get the dictionary
    mf_model = mf.MFModel(dictionary_path)

    # compute csf_mask and ear_mask
    csf_mask = False
    ear_mask = (numfasc >= 1)  # (numfasc == 1)

    # Fit to data:
    MF_fit = mf_model.fit(data, mask, numfasc, peaks=peaks, bvals=bvals, bvecs=bvecs, csf_mask=csf_mask, ear_mask=ear_mask, verbose=3, parallel=True)

    # extract info
    M0 = MF_fit.M0
    frac_f0 = MF_fit.frac_f0
    DIFF_ex_f0 = MF_fit.DIFF_ex_f0
    fvf_f0 = MF_fit.fvf_f0
    frac_f1 = MF_fit.frac_f1
    DIFF_ex_f1 = MF_fit.DIFF_ex_f1
    fvf_f1 = MF_fit.fvf_f1
    fvf_tot = MF_fit.fvf_tot
    frac_ear = MF_fit.frac_ear
    D_ear = MF_fit.D_ear
    MSE = MF_fit.MSE
    R2 = MF_fit.R2

    # Save nifti
    save_nifti(mf_path + '/' + patient_path + '_mf_peaks.nii.gz', peaks.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_M0.nii.gz', M0.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_frac_f0.nii.gz', frac_f0.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_DIFF_ex_f0.nii.gz', DIFF_ex_f0.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_fvf_f0.nii.gz', fvf_f0.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_frac_f1.nii.gz', frac_f1.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_DIFF_ex_f1.nii.gz', DIFF_ex_f1.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_fvf_f1.nii.gz', fvf_f1.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_fvf_tot.nii.gz', fvf_tot.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_frac_ear.nii.gz', frac_ear.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_D_ear.nii.gz', D_ear.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_MSE.nii.gz', MSE.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_R2.nii.gz', R2.astype(np.float32), affine)

    print("[MF SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f=open(folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")
    f.write("[MF SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def tbss_utils(folder_path, corrected = False):

    # create the output directory
    outputdir = folder_path + "/TBSS"
    if not (os.path.exists(outputdir)):
        try:
            os.makedirs(outputdir)
        except OSError:
            print("Creation of the directory %s failed" % outputdir)
            f = open(folder_path + "/logs.txt", "a+")
            f.write("[TBSS] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % outputdir)
            f.close()
        else:
            print("Successfully created the directory %s " % outputdir)
            f = open(folder_path + "/logs.txt", "a+")
            f.write("[TBSS] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % outputdir)
            f.close()

    # open the subject and is_control lists
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)
    dest_iscontrol = folder_path + "/subjects/is_control.json"
    with open(dest_iscontrol, 'r') as f:
        is_control = json.load(f)

    # transfer the FA files to the TBSS directory
    numpatient = 0
    numcontrol = 0
    for index, p in patient_list:
        patient_path = os.path.splitext(p)[0]
        control_info = is_control[index]
        if control_info:
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz", outputdir + "/control" + numcontrol + "_" + patient_path + "_fa.nii.gz")
            numcontrol += 1
        else:
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz", outputdir + "/case" + numpatient + "_" + patient_path + "_fa.nii.gz")
            numpatient += 1


    import subprocess

    bashCommand = 'cd ' + outputdir + ' && tbss_1_preproc \"*_fa.nii.gz\"'
    bashcmd = bashCommand.split()
    print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = 'cd ' + outputdir + ' && tbss_2_reg -T'
    bashcmd = bashCommand.split()
    print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = 'cd ' + outputdir + ' && tbss_3_postreg -S'
    bashcmd = bashCommand.split()
    print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = 'cd ' + outputdir + ' && tbss_4_prestats 0.2'
    bashcmd = bashCommand.split()
    print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = 'cd ' + outputdir + '/stats ' + ' && design_ttest2 design ' + numcontrol + ' ' + numpatient
    bashcmd = bashCommand.split()
    print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    if corrected:
        bashCommand = 'cd ' + outputdir + '/stats ' + ' && randomise -i all_FA_skeletonised -o tbss -m mean_FA_skeleton_mask -d design.mat -t design.con -n 5000 --T2'
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
        output, error = process.communicate()

        bashCommand = 'cd ' + outputdir + '/stats ' + ' && autoaq -i tbss_tfce_corrp_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report1_subcortical.txt && autoaq -i tbss_tfce_corrp_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report2_subcortical.txt && autoaq -i tbss_tfce_corrp_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report1_cortical.txt && autoaq -i tbss_tfce_corrp_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report2_cortical.txt'
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
        output, error = process.communicate()
    else:
        bashCommand = 'cd ' + outputdir + '/stats ' + ' && randomise -i all_FA_skeletonised -o tbss -m mean_FA_skeleton_mask -d design.mat -t design.con -n 5000 --T2 --uncorrp'
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
        output, error = process.communicate()

        bashCommand = 'cd ' + outputdir + '/stats ' + ' && autoaq -i tbss_tfce_p_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report1_subcortical.txt && autoaq -i tbss_tfce_p_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report2_subcortical.txt && autoaq -i tbss_tfce_p_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report1_cortical.txt && autoaq -i tbss_tfce_p_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report2_cortical.txt'
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
        output, error = process.communicate()