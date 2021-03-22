import datetime
import os
import os
import shutil
import json
import numpy as np
import math

import subprocess
from elikopy.utils import makedir

from dipy.denoise.gibbs import gibbs_removal


def preproc_solo(folder_path, p, reslice=False, denoising=False,gibbs=False, topup=False,  eddy=False, starting_state=None, bet_median_radius=2, bet_numpass=1, bet_dilate=2, cuda=False, cuda_name="eddy_cuda10.1", s2v=[0,5,1,'trilinear'], olrep=[False, 4, 250, 'sw']):
    """
    Perform bet and optionnaly denoising, gibbs, topup and eddy. Generated data are stored in bet, eddy, denoising and final directory
    located in the folder out/preproc. All the function executed after this function MUST take input data from folder_path/out/preproc/final.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param reslice: If true, data will be resliced with a new voxel resolution of 2*2*2.
    :param denoising: If true, PCA-based denoising using the Marcenko-Pastur distribution will be performed.
    :param gibbs: If true, Gibbs ringing artefacts of images volumes will be suppressed.
    :param topup: If true, topup will estimate and correct susceptibility induced distortions.
    :param eddy: If true, eddy will correct eddy currents and movements in diffusion data.
    :param starting_state: Manually set which step of the preprocessing to execute first. Could either be None, denoising, gibbs, topup or eddy.
    :param bet_median_radius: Radius (in voxels) of the applied median filter during bet.
    :param bet_numpass: Number of pass of the median filter during bet.
    :param bet_dilate: Number of iterations for binary dilation during bet.
    :param cuda: If true, eddy will run on cuda with the command name specified in cuda_name.
    :param cuda_name: name of the eddy command to run when cuda==True.
    :param s2v: list of parameters eddy for slice-to-volume correction (see Eddy FSL documentation): [mporder,s2v_niter,s2v_lambda,s2v_interp].
    :param olrep: list of parameters eddy outlier replacement (see Eddy FSL documentation): [repol,ol_nstd,ol_nvox,ol_type].
    """

    assert starting_state in (None, "denoising", "gibbs", "topup", "eddy"), 'invalid starting state!'
    if starting_state == "denoising":
        assert denoising == True, 'if starting_state is denoising, denoising must be True!'
    if starting_state == "gibbs":
        assert gibbs == True, 'if starting_state is gibbs, gibbs must be True!'
    if starting_state == "topup":
        assert topup == True, 'if starting_state is topup, topup must be True!'
    if starting_state == "eddy":
        assert eddy == True, 'if starting_state is eddy, eddy must be True!'

    log_prefix = "PREPROC SOLO"
    patient_path = os.path.splitext(p)[0]
    preproc_path = folder_path + '/' + patient_path + "/dMRI/preproc/bet"
    makedir(preproc_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

    mask_path = folder_path + '/' + patient_path + "/masks"
    makedir(mask_path, folder_path + '/' + patient_path + "/masks/wm_logs.txt", log_prefix)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f.close()
    from dipy.io.image import load_nifti, save_nifti
    from dipy.segment.mask import median_otsu
    from dipy.denoise.localpca import mppca

    nifti_path = folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.nii.gz'
    if (starting_state == None):
        data, affine, voxel_size = load_nifti(nifti_path, return_voxsize=True)

    reslice_path = folder_path + '/' + patient_path + "/dMRI/preproc/reslice"
    if reslice and starting_state == None:
        makedir(reslice_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        from dipy.align.reslice import reslice
        new_voxel_size = (2., 2., 2.)
        data, affine = reslice(data, affine, voxel_size, new_voxel_size)
        save_nifti(reslice_path + '/' + patient_path + '_reslice.nii.gz', data, affine)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f.close()

    if starting_state == None:
        b0_mask, mask = median_otsu(data, median_radius=bet_median_radius, numpass=bet_numpass, vol_idx=range(0, np.shape(data)[3]), dilate=bet_dilate)
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz',
                   mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz',
                   b0_mask.astype(np.float32), affine)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
        f.close()

    if not denoising and not eddy and not gibbs and not topup:
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                   b0_mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                   mask.astype(np.float32), affine)
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    denoising_path = folder_path + '/' + patient_path + '/dMRI/preproc/mppca'
    if denoising and starting_state!="gibbs" and starting_state!="eddy" and starting_state!="topup":
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of denoising for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Denoising launched for patient %s \n" % p)
        f.close()

        makedir(denoising_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if (starting_state == denoising):
            mask_path = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz'
            b0_mask_path = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz'
            b0_mask, affine, voxel_size = load_nifti(b0_mask_path, return_voxsize=True)
            mask, _ = load_nifti(mask_path)

        pr = math.ceil((np.shape(b0_mask)[3] ** (1 / 3) - 1) / 2)
        denoised, sigma = mppca(b0_mask, patch_radius=pr, return_sigma=True, mask = mask)

        #mean_sigma = np.mean(sigma[b0_mask])
        #mean_signal = np.mean(denoised[b0_mask])
        #snr = mean_signal/mean_sigma
        #f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        #f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        #    "%d.%b %Y %H:%M:%S") + ": Denoising mean sigma:"+ str(mean_sigma)+ ", snr:" + str(snr) + " for patient %s \n" % p)
        #print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        #    "%d.%b %Y %H:%M:%S") + ": Denoising mean sigma" + str(mean_sigma)+ ", snr:" + str(snr) + " for patient %s \n" % p)
        #f.close()
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Denoising finished for patient %s \n" % p)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Denoising finished for patient %s \n" % p)
        f.close()

        save_nifti(denoising_path + '/' + patient_path + '_sigmaNoise.nii.gz', sigma.astype(np.float32), affine)
        save_nifti(denoising_path + '/' + patient_path + '_mppca.nii.gz', denoised.astype(np.float32), affine)

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of denoising for patient %s \n" % p)

        b0_mask = denoised

        if not eddy and not gibbs and not topup:
            save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                       b0_mask.astype(np.float32), affine)
            save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                       mask.astype(np.float32), affine)
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    if gibbs and starting_state!="eddy" and starting_state!="topup":
        gibbs_path = folder_path + '/' + patient_path + '/dMRI/preproc/gibbs'
        makedir(gibbs_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if (starting_state == denoising):
            mask_path = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz'
            if not denoising:
                b0_mask_path = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz'
            else:
                b0_mask_path = denoising_path + '/' + patient_path + '_mppca.nii.gz'
            b0_mask, affine, voxel_size = load_nifti(b0_mask_path, return_voxsize=True)
            mask, _ = load_nifti(mask_path)

        data = gibbs_removal(b0_mask)
        corrected_path = folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/" + patient_path + '_gibbscorrected.nii.gz'
        save_nifti(corrected_path, data.astype(np.float32), affine)

        if not eddy and not topup:
            save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                       data.astype(np.float32), affine)
            save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                       mask.astype(np.float32), affine)
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    # Explicitly freeing memory
    import gc
    denoised = None
    b0_mask = None
    mask = None
    data = None
    affine = None
    gc.collect()

    topup_path = folder_path + '/' + patient_path + "/dMRI/preproc/topup"
    if topup and starting_state!="eddy":

        import subprocess
        #cmd = 'topup --imain=all_my_b0_images.nii --datain=acquisition_parameters.txt --config=b02b0.cnf --out=my_output"'
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of topup for patient %s \n" % p)

        topup_path = folder_path + '/' + patient_path + "/dMRI/preproc/topup"
        makedir(topup_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
        makedir(topup_path+"/synb0-DisCo", folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if gibbs:
            imain_tot = folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz'
        elif denoising:
            imain_tot = folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz'
        else:
            imain_tot = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz'

        multiple_encoding=False
        topup_log = open(folder_path + '/' + patient_path + "/dMRI/preproc/topup/topup_logs.txt", "a+")

        with open(folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt') as f:
            line = f.read()
            topup_index = [int(s) for s in line.split(' ')]
        with open(folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt') as f:
            line = f.read()
            topup_acq = [[int(x) for x in line.split()] for line in f]

        #Find all the bo to extract.
        current_index = 0
        i=1
        roi=[]
        for ind in topup_index:
            if ind!=current_index:
                roi.append(i)
                fslroi = "fslroi " + imain_tot + " " + topup_path + "/b0_"+str(i)+".nii.gz "+str(i)+" 1"
                process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=topup_log,
                                           stderr=subprocess.STDOUT)
                output, error = process.communicate()
                print("B0 of index" + str(i) + " extracted!")
            current_index=ind
            i=i+1

        #Merge b0
        if len(roi) == 1:
            shutil.copyfile(topup_path + "/b0_"+str(roi[0])+".nii.gz", topup_path + "/b0.nii.gz")
        else:
            roi_to_merge=""
            for r in roi:
                roi_to_merge = roi_to_merge + " b0_" + str(r) + ".nii.gz"
            print("The following roi will be merged: " + roi_to_merge)
            cmd = "fslmerge -t " + topup_path + "/b0.nii.gz " + roi_to_merge
            process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=topup_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()

        #Check if multiple or single encoding direction
        curr_x=0
        curr_y=0
        curr_z=0
        first=True
        for acq in topup_acq:
            if not first and (curr_x!=acq[1] or curr_y!=acq[2] or curr_z!=acq[3]):
                multiple_encoding=True
            first=False
            curr_x=acq[1]
            curr_y=acq[2]
            curr_z=acq[3]

        if multiple_encoding:
            f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Patient %s \n" % p + " has multiple direction of gradient encoding, launching topup directly ")
            bashCommand = 'topup --imain="' + topup_path + '/b0.nii.gz" --config="b02b0.cnf" --datain="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --verbose'
            bashcmd = bashCommand.split()
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Topup launched for patient %s \n" % p + " with bash command " + bashCommand)

            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Topup launched for patient %s \n" % p + " with bash command " + bashCommand)
            f.close()

            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=topup_log,
                                       stderr=subprocess.STDOUT)
            # wait until topup finish
            output, error = process.communicate()
        else:
            f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Patient %s \n" % p + " has a single direction of gradient encoding, launching synb0DisCo ")
            f.close()
            from elikopy.utils import synb0DisCo
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt',
                            topup_path + '/synb0-DisCo/' + 'acqparams.txt')

            shutil.copyfile(folder_path + '/' + patient_path + '/T1/' + patient_path + '_T1.nii.gz',
                            topup_path + '/synb0-DisCo/' + 'T1.nii.gz')

            shutil.copyfile(topup_path + "/b0.nii.gz",topup_path + "/synb0-DisCo/b0.nii.gz")

            process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=topup_log,stderr=subprocess.STDOUT)
            output, error = process.communicate()
            synb0DisCo(topup_path,starting_step=None,topup=True,gpu=False)

        bashCommand2 = 'applytopup --imain="' + imain_tot + '" --inindex=1 --datatin="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --topup="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr"'

        process2 = subprocess.Popen(bashCommand2, universal_newlines=True, shell=True, stdout=topup_log,
                                    stderr=subprocess.STDOUT)
        # wait until apply topup finish
        output, error = process2.communicate()

        topup_log.close()

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of topup for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of topup for patient %s \n" % p)
        f.close()

        if not eddy:
            data, affine = load_nifti(
                folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + "_unwarped.nii.gz")
            b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]), dilate=2)
            save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                       b0_mask.astype(np.float32), affine)
            save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                       mask.astype(np.float32), affine)
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    if eddy:
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of eddy for patient %s \n" % p)

        eddy_path = folder_path + '/' + patient_path + "/dMRI/preproc/eddy"
        makedir(eddy_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if cuda:
            eddycmd = cuda_name
        else:
            eddycmd = "eddy"

        slspec_path = folder_path + '/' + patient_path + '/dMRI/raw/' + 'slspec.txt'
        if os.path.isfile(slspec_path):
            if topup:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_unwarped.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]
            elif gibbs:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]
            elif denoising:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]
            else:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]
        else:
            if topup:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_unwarped.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]
            elif gibbs:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]
            elif denoising:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]
            else:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3]



        import subprocess
        bashcmd = bashCommand.split()
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f.close()

        eddy_log = open(folder_path + '/' + patient_path + "/dMRI/preproc/eddy/eddy_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=eddy_log,
                                   stderr=subprocess.STDOUT)

        # wait until eddy finish
        output, error = process.communicate()
        eddy_log.close()

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f.close()

        data, affine = load_nifti(
            folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + "_eddy_corr.nii.gz")
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]), dilate=2)
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                   b0_mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                   mask.astype(np.float32), affine)
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(
            folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + "_eddy_corr.eddy_rotated_bvecs",
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def dti_solo(folder_path, p):
    """
    Tensor reconstruction and computation of DTI metrics using Weighted Least-Squares.
    Performs a tensor reconstruction and saves the DTI metrics.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    """
    log_prefix = "DTI SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual DTI processing for patient %s \n" % p)

    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    import dipy.reconst.dti as dti

    patient_path = os.path.splitext(p)[0]

    dti_path = folder_path + '/' + patient_path + "/dMRI/microstructure/dti"
    makedir(dti_path, folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", log_prefix)

    # load the data======================================
    data, affine = load_nifti(
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    # create the model===================================
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    # FA ================================================
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
               FA.astype(np.float32), affine)
    # colored FA ========================================
    RGB = dti.color_fa(FA, tenfit.evecs)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_fargb.nii.gz",
               np.array(255 * RGB, 'uint8'), affine)
    # Mean diffusivity ==================================
    MD = dti.mean_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_MD.nii.gz",
               MD.astype(np.float32), affine)
    # Radial diffusivity ==================================
    RD = dti.radial_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_RD.nii.gz",
               RD.astype(np.float32), affine)
    # Axial diffusivity ==================================
    AD = dti.axial_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_AD.nii.gz",
               AD.astype(np.float32), affine)
    # eigen vectors =====================================
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_evecs.nii.gz",
               tenfit.evecs.astype(np.float32), affine)
    # eigen values ======================================
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_evals.nii.gz",
               tenfit.evals.astype(np.float32), affine)
    # diffusion tensor ====================================
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_dtensor.nii.gz",
               tenfit.quadratic_form.astype(np.float32), affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def white_mask_solo(folder_path, p):
    """ Compute a white matter mask of the diffusion data for each patient based on T1 volumes or on diffusion data if
    T1 is not available. The T1 images must have the same name as the patient it corresponds to with _T1 at the end and must be in
    a folder named anat in the root folder.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    """

    log_prefix = "White mask solo"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual white mask processing for patient %s \n" % p)

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
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from T1 %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from T1 %s \n" % p)
        f.close()
        # Read the moving image ====================================
        anat_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
        data_gibbs, affine_gibbs = load_nifti(anat_path)
        data_gibbs = gibbs_removal(data_gibbs)
        corrected_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_gibbscorrected.nii.gz'
        save_nifti(corrected_path, data_gibbs.astype(np.float32), affine_gibbs)
        # anat_path = folder_path + '/anat/' + patient_path + '_T1.nii.gz'
        bet_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
        bashCommand = 'bet2 ' + corrected_path + ' ' + bet_path + ' -f 1 -g -3'
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
        static_data, static_affine = load_nifti(
            folder_path + "/" + patient_path + "/dMRI/preproc/" + patient_path + "_dmri_preproc.nii.gz")
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
        translation = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                      starting_affine=starting_affine)
        # Rigid transform the moving image ====================================
        transform = RigidTransform3D()
        params0 = None
        starting_affine = translation.affine
        rigid = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                starting_affine=starting_affine)
        # affine transform the moving image ====================================
        transform = AffineTransform3D()
        params0 = None
        starting_affine = rigid.affine
        affine = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                 starting_affine=starting_affine)
        """"
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
        """
        # make the white matter segmentation ===================================
        anat = moving
        nclass = 3
        beta = 0.1
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(anat, nclass, beta)
        # save the white matter mask ============================================
        white_mask = PVE[..., 2]
        white_mask[white_mask >= 0.01] = 1
        white_mask[white_mask < 0.01] = 0
        # transform the white matter mask ======================================
        white_mask = affine.transform(white_mask)
        white_mask[white_mask != 0] = 1
        anat_affine = static_grid2world
        segmentation = affine.transform(final_segmentation)
    else:
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from AP %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from AP %s \n" % p)
        f.close()
        f = open(folder_path + "/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Warning: Mask done from AP for patient %s \n" % p)
        f.close()
        # compute the white matter mask with the Anisotropic power map
        data, affine = load_nifti(
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
        mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")
        bvals, bvecs = read_bvals_bvecs(
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
        gtab = gradient_table(bvals, bvecs)
        sphere = get_sphere('symmetric724')
        qball_model = shm.QballModel(gtab, 8)
        peaks = dp.peaks_from_model(model=qball_model, data=data, relative_peak_threshold=.5, min_separation_angle=25,
                                    sphere=sphere, mask=mask)
        ap = shm.anisotropic_power(peaks.shm_coeff)
        nclass = 3
        beta = 0.1
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
        white_mask = PVE[..., 2]
        white_mask[white_mask >= 0.01] = 1
        white_mask[white_mask < 0.01] = 0
        anat_affine = affine
        segmentation = np.copy(final_segmentation)

    mask_path = folder_path + '/' + patient_path + "/masks"
    makedir(mask_path, folder_path + '/' + patient_path + "/masks/wm_logs.txt", log_prefix)

    out_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    save_nifti(out_path, white_mask.astype(np.float32), anat_affine)
    save_nifti(folder_path + '/' + patient_path + "/masks/" + patient_path + '_segmentation.nii.gz', segmentation.astype(np.float32), anat_affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def noddi_solo(folder_path, p, force_brain_mask=False, lambda_iso_diff=3.e-9, lambda_par_diff=1.7e-9, use_amico=False):
    """ Perform noddi and store the data in the subjID/dMRI/microstructure/noddi folder.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param force_brain_mask:
    :param lambda_iso_diff:
    :param lambda_par_diff:
    :param use_amico:
    """
    print("[NODDI SOLO] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual NODDI processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs

    patient_path = os.path.splitext(p)[0]
    log_prefix = "NODDI SOLO"

    noddi_path = folder_path + '/' + patient_path + "/dMRI/microstructure/noddi"
    makedir(noddi_path, folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", log_prefix)

    # initialize the compartments model
    from dmipy.signal_models import cylinder_models, gaussian_models
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    # watson distribution of stick and Zepelin
    from dmipy.distributions.distribute_models import SD1WatsonDistributed
    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
    watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par',
                                                   'partial_volume_0')
    watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', lambda_par_diff)

    # build the NODDI model
    from dmipy.core.modeling_framework import MultiCompartmentModel
    NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])

    # fix the isotropic diffusivity
    NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso_diff)

    # load the data
    data, affine = load_nifti(
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    wm_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    if os.path.isfile(wm_path) and not force_brain_mask:
        mask, _ = load_nifti(wm_path)
    else:
        mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")

    # transform the bval, bvecs in a form suited for NODDI
    from dipy.core.gradients import gradient_table
    from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
    gtab_dipy = gradient_table(bvals, bvecs)
    acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy)

    if use_amico:
        # fit the model to the data using noddi amico
        from dmipy.optimizers import amico_cvxpy
        NODDI_fit = amico_cvxpy.AmicoCvxpyOptimizer(acq_scheme_dmipy, data, mask=mask)
    else:
        # fit the model to the data
        NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask)
        # NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask, solver='mix', maxiter=300)

    # exctract the metrics
    fitted_parameters = NODDI_fit.fitted_parameters
    mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
    odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
    f_iso = fitted_parameters["partial_volume_0"]
    f_bundle = fitted_parameters["partial_volume_1"]
    f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
    f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1']>0.05
    f_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters[
        'partial_volume_1'])
    mse = NODDI_fit.mean_squared_error(data)
    R2 = NODDI_fit.R2_coefficient_of_determination(data)

    # save the nifti
    save_nifti(noddi_path + '/' + patient_path + '_noddi_mu.nii.gz', mu.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_odi.nii.gz', odi.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fiso.nii.gz', f_iso.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fbundle.nii.gz', f_bundle.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fintra.nii.gz', f_intra.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_icvf.nii.gz', f_icvf.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_fextra.nii.gz', f_extra.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_mse.nii.gz', mse.astype(np.float32), affine)
    save_nifti(noddi_path + '/' + patient_path + '_noddi_R2.nii.gz', R2.astype(np.float32), affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def noddi_amico_solo(folder_path, p):
    """ Perform noddi and store the data in the subjID/dMRI/microstructure/noddi_amico folder.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    """
    print("[NODDI AMICO SOLO] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual NODDI AMICO processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs

    log_prefix = "NODDI AMICO SOLO"

    patient_path = os.path.splitext(p)[0]

    noddi_path = folder_path + '/' + patient_path + "/dMRI/microstructure/noddi_amico"
    makedir(noddi_path, folder_path + '/' + patient_path + "/dMRI/microstructure/noddi_amico/noddi_amico_logs.txt",
            log_prefix)

    wm_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    if os.path.isfile(wm_path):
        mask, _ = load_nifti(wm_path)
    else:
        mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")

    import amico
    amico.core.setup()
    ae = amico.Evaluation(study_path=folder_path + '/noddi_AMICO/',
                          subject=folder_path + '/' + patient_path + '/dMRI/microstructure/noddi_amico/',
                          output_path=folder_path + '/' + patient_path + '/dMRI/microstructure/noddi_amico/')

    schemeFile = folder_path + '/' + patient_path + '/dMRI/microstructure/noddi_amico/' + patient_path + "_NODDI_protocol.scheme"
    dwi_preproc = folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc"

    amico.util.fsl2scheme(dwi_preproc + ".bval", dwi_preproc + ".bvec", schemeFilename=schemeFile)

    ae.load_data(dwi_filename=dwi_preproc + ".nii.gz", scheme_filename=schemeFile, mask_filename=wm_path, b0_thr=0)
    ae.set_model("NODDI")
    ae.generate_kernels()
    ae.load_kernels()
    ae.fit()
    ae.save_results(path_suffix=patient_path)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi_amico/noddi_amico_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def diamond_solo(folder_path, p, box=None):
    """Perform diamond and store the data in the subjID/dMRI/microstructure/diamond folder.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param box:
    """
    log_prefix = "DIAMOND SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual DIAMOND processing for patient %s \n" % p)
    patient_path = os.path.splitext(p)[0]

    diamond_path = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond"
    makedir(diamond_path, folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt",
            log_prefix)
    if not (os.path.exists(diamond_path)):
        try:
            os.makedirs(diamond_path)
        except OSError:
            print("Creation of the directory %s failed" % diamond_path)
            f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % diamond_path)
            f.close()
        else:
            print("Successfully created the directory %s " % diamond_path)
            f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % diamond_path)
            f.close()

    # '--bbox 0,0,38,128,128,1'
    # if box is not None:
    #    bashCommand = 'crlDCIEstimate -i ' + folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz' + ' -m ' + folder_path + '/' + patient_path + '/dMRI/masks/' + patient_path + '_brain_mask.nii.gz' + ' -n 3 --automose aicu --fascicle diamondcyl -o ' + folder_path + '/' + patient_path + '/dMRI/microstructure/diamond/' + patient_path + '_diamond.nii.gz' + ' -p 4'
    # else:
    wm_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    if os.path.isfile(wm_path):
        mask = wm_path
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": white matter mask based on T1 is used \n")
        f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": white matter mask based on T1 is used \n")
        f.close()
    else:
        mask = folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz'
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": brain mask based on diffusion data is used \n")
        f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": brain mask based on diffusion data is used \n")
        f.close()

    bashCommand = 'crlDCIEstimate --input "' + folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz' + '" --output "' + folder_path + '/' + patient_path + '/dMRI/microstructure/diamond/' + patient_path + '_diamond.nii.gz' + '" --mask "' + mask + '" --proc 4 --ntensors 2 --reg 1.0 --estimb0 1 --automose aicu --mosemodels --fascicle diamondcyl --waterfraction 1 --waterDiff 0.003 --omtm 1 --residuals --fractions_sumto1 0 --verbose 1 --log'

    import subprocess
    bashcmd = bashCommand.split()
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": crlDCIEstimate launched for patient %s \n" % p + " with bash command " + bashCommand)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": crlDCIEstimate launched for patient %s \n" % p + " with bash command " + bashCommand)
    f.close()

    diamond_log = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=diamond_log,
                               stderr=subprocess.STDOUT)

    output, error = process.communicate()
    diamond_log.close()

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def mf_solo(folder_path, p, dictionary_path, CSD_bvalue=None):
    """Perform microstructure fingerprinting and store the data in the subjID/dMRI/microstructure/mf folder.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param dictionary_path:
    :param CSD_bvalue:
    """
    log_prefix = "MF SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual microstructure fingerprinting processing for patient %s \n" % p)
    patient_path = os.path.splitext(p)[0]

    mf_path = folder_path + '/' + patient_path + "/dMRI/microstructure/mf"
    makedir(mf_path, folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", log_prefix)

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
    data, affine = load_nifti(
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    wm_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    if os.path.isfile(wm_path):
        mask, _ = load_nifti(wm_path)
    else:
        mask, _ = load_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz")

    # compute numfasc and peaks
    diamond_path = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond"
    if os.path.exists(diamond_path):
        tensor_files0 = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t0.nii.gz'
        tensor_files1 = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t1.nii.gz'
        fracs_file = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_fractions.nii.gz'
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=None, frac2=None, mu1=tensor_files0, mu2=tensor_files1,
                                                 peakmode='tensor', mask=mask, frac12=fracs_file)
    else:
        sel_b = np.logical_or(bvals == 0, np.logical_and((CSD_bvalue - 5) <= bvals, bvals <= (CSD_bvalue + 5)))
        data_CSD = data[..., sel_b]
        gtab_CSD = gradient_table(bvals[sel_b], bvecs[sel_b])
        response, ratio = auto_response(gtab_CSD, data_CSD, roi_radius=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response, sh_order=6)
        csd_peaks = peaks_from_model(npeaks=2, model=csd_model, data=data_CSD, sphere=default_sphere,
                                     relative_peak_threshold=.15, min_separation_angle=25, parallel=False, mask=mask,
                                     normalize_peaks=True)
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
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=frac1, frac2=frac2, mu1=mu1, mu2=mu2, peakmode='peaks',
                                                 mask=mask, frac12=None)

    # get the dictionary
    mf_model = mf.MFModel(dictionary_path)

    # compute csf_mask and ear_mask
    csf_mask = True
    ear_mask = False  # (numfasc == 1)

    # Fit to data:
    MF_fit = mf_model.fit(data, mask, numfasc, peaks=peaks, bvals=bvals, bvecs=bvecs, csf_mask=csf_mask,
                          ear_mask=ear_mask, verbose=3, parallel=False)

    # extract info
    M0 = MF_fit.M0
    frac_f0 = MF_fit.frac_f0
    DIFF_ex_f0 = MF_fit.DIFF_ex_f0
    fvf_f0 = MF_fit.fvf_f0
    frac_f1 = MF_fit.frac_f1
    DIFF_ex_f1 = MF_fit.DIFF_ex_f1
    fvf_f1 = MF_fit.fvf_f1
    fvf_tot = MF_fit.fvf_tot
    # frac_ear = MF_fit.frac_ear
    # D_ear = MF_fit.D_ear
    frac_csf = MF_fit.frac_csf
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
    # save_nifti(mf_path + '/' + patient_path + '_mf_frac_ear.nii.gz', frac_ear.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_D_ear.nii.gz', D_ear.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_frac_csf.nii.gz', frac_csf.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_MSE.nii.gz', MSE.astype(np.float32), affine)
    save_nifti(mf_path + '/' + patient_path + '_mf_R2.nii.gz', R2.astype(np.float32), affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def report_solo(folder_path,patient_path):
    """

    -x 0.4 slicesdir/grota.png -x 0.5 slicesdir/grotb.png -x 0.6 slicesdir/grotc.png -y 0.4 slicesdir/grotd.png -y 0.5
    slicesdir/grote.png -y 0.6 slicesdir/grotf.png -z 0.4 slicesdir/grotg.png -z 0.5 slicesdir/groth.png -z 0.6 slicesdir/groti.png



    """

    report_path = folder_path + '/' + patient_path + "/report/raw/"
    log_prefix="Individual Report"
    makedir(report_path, folder_path + "/logs.txt", log_prefix)
    report_log = open(report_path + "report_logs.txt", "a+")

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 18)
    pdf.cell(0, 10, '', 0, 1, 'C')
    pdf.cell(0, 10, "Individual report for subject "+patient_path, 0, 2, 'C')
    pdf.cell(0, 5, '', 0, 1, 'C')
    pdf.set_font('arial', 'B', 12)



    image=[]
    image.append((folder_path + '/' + patient_path + "/dMRI/raw/"+patient_path+"_raw_dmri","raw_drmi","Raw dMRI ("+patient_path+"_raw_drmi.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc","drmi_preproc","dMRI preprocessed ("+patient_path+"_drmi_preproc.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/bet/"+patient_path+"_mask" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/bet/"+patient_path+"_mask","drmi_preproc_bet","dMRI BET preprocessing ("+patient_path+"_mask.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/mppca/"+patient_path+"_mppca" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/mppca/"+patient_path+"_mppca","drmi_preproc_mppca","dMRI Denoised preprocessing ("+patient_path+"_mppca.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected","drmi_preproc_gibbs","dMRI Gibbs preprocessing ("+patient_path+"_gibbscorrected.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/topup/"+patient_path+"_topup" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/topup/"+patient_path+"_topup","drmi_preproc_topup","dMRI Topup preprocessing ("+patient_path+"_topup.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/eddy/" + patient_path + "_eddy_corr" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/eddy/" + patient_path + "_eddy_corr","drmi_preproc_eddy_corr", "dMRI Eddy preprocessing (" + patient_path + "_eddy_corr.nii.gz)"))

    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_FA" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_FA","dti_FA", "Microstructure: FA of dti (" + patient_path + "_FA.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_AD" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_AD","dti_AD", "Microstructure: AD of dti (" + patient_path + "_AD.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_MD" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_MD","dti_MD", "Microstructure: MD of dti (" + patient_path + "_MD.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_RD" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_RD","dti_RD", "Microstructure: RD of dti (" + patient_path + "_RD.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_dtensor" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_dtensor","dti_dtensor", "Microstructure: Dtensor of dti (" + patient_path + "_dtensor.nii.gz)"))

    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/" + patient_path + "_dtensor" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/" + patient_path + "_dtensor","noddi_dtensor", "Microstructure: ICVF of Noddi (" + patient_path + "_dtensor.nii.gz)"))

    for nifti,pre,texte in image:
        slices_info = "-x 0.4 "+report_path+pre+"a.png -x 0.5 "+report_path+pre+"b.png -x 0.6 "+report_path+pre+"c.png " \
                      "-y 0.4 "+report_path+pre+"d.png -y 0.5 "+report_path+pre+"e.png -y 0.6 "+report_path+pre+"f.png " \
                      "-z 0.4 "+report_path+pre+"g.png -z 0.5 "+report_path+pre+"h.png -z 0.6 "+report_path+pre+"i.png"
        slices_merge_info_1 = ""+report_path+pre+"a.png + "+report_path+pre+"b.png + "+report_path+pre+"c.png "
        slices_merge_info_2 = ""+report_path+pre+"d.png + "+report_path+pre+"e.png + "+report_path+pre+"f.png "
        slices_merge_info_3 = ""+report_path+pre+"g.png + "+report_path+pre+"h.png + "+report_path+pre+"i.png "

        cmd1 = "slicer " + nifti + "  -s 1 " + slices_info
        cmd2 = "pngappend " + slices_merge_info_1 + " " + report_path + pre + "_x.png"
        cmd3 = "pngappend " + slices_merge_info_2 + " " + report_path + pre + "_y.png"
        cmd4 = "pngappend " + slices_merge_info_3 + " " + report_path + pre + "_z.png"

        bashCommand = 'cd ' + report_path + '; ' + cmd1 + '; ' + cmd2 + '; ' + cmd3 + '; ' + cmd4
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        report_log.write(bashCommand + "\n")
        report_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=report_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        pdf.cell(0, 10, texte, 0, 1,'C')
        pdf.image(report_path + pre + "_x.png", x=None, y=None, w=190, h=0, type='', link='')
        pdf.image(report_path + pre + "_y.png", x=None, y=None, w=190, h=0, type='', link='')
        pdf.image(report_path + pre + "_z.png", x=None, y=None, w=190, h=0, type='', link='')
        pdf.add_page()


    pdf.output(folder_path + '/' + patient_path + "/report/report_"+patient_path+".pdf", 'F')

