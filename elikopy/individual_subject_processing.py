import datetime
import os
import shutil
import json
import numpy as np
import math

import subprocess
from elikopy.utils import makedir

from dipy.denoise.gibbs import gibbs_removal


def preproc_solo(folder_path, p, reslice=False, reslice_addSlice=False, denoising=False,gibbs=False, topup=False, topupConfig=None, forceSynb0DisCo=False, useGPUsynb0DisCo=False, eddy=False, biasfield=False, biasfield_bsplineFitting=[100,3], biasfield_convergence=[1000,0.001], starting_state=None, bet_median_radius=2, bet_numpass=1, bet_dilate=2, cuda=False, cuda_name="eddy_cuda10.1", s2v=[0,5,1,'trilinear'], olrep=[False, 4, 250, 'sw'], qc_reg=True, core_count=1, niter=5, report=True, slspec_gc_path=None):
    """
    Perform brain extraction and optionally reslicing, denoising, gibbs correction, susceptibility field estimation using topup, movement correction using eddy and biasfield correction. Generated data are stored in bet, reslice, mppca, gibbs, topup, eddy, biasfield and final directory
    located in the folder folder_path/subjects/<subjects_ID>/dMRI/preproc.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param reslice: If true, data will be resliced with a new voxel resolution of 2*2*2.
    :param reslice_addSlice: If true, an additional slice will be added to each volume to allow gradient cycling eddy motion correction.
    :param denoising: If true, PCA-based denoising using the Marcenko-Pastur distribution will be performed.
    :param gibbs: If true, Gibbs ringing artifacts of images volumes will be suppressed.
    :param topup: If true, topup will estimate and correct susceptibility induced distortions.
    :param topupConfig: If not None, topup will use these additionnal parameters based on the supplied config file.
    :param forceSynb0DisCo: If true, topup will always estimate the susceptibility field using the T1 structural image.
    :param eddy: If true, eddy will correct eddy currents and movements in diffusion data.
    :param biasfield: If true, low frequency intensity non-uniformity present in MRI image data known as a bias or gain field will be corrected.
    :param biasfield_bsplineFitting: Define the initial mesh resolution in mm and the bspline order of the biasfield correction tool.
    :param biasfield_convergence: Define the maximum number of iteration and the convergences threshold of the biasfield correction tool.
    :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
    :param starting_state: Manually set which step of the preprocessing to execute first. Could either be None, denoising, gibbs, topup, eddy, biasfield, report or post_report.
    :param bet_median_radius: Radius (in voxels) of the applied median filter during bet.
    :param bet_numpass: Number of pass of the median filter during bet.
    :param bet_dilate: Number of iterations for binary dilation during bet.
    :param cuda: If true, eddy will run on cuda with the command name specified in cuda_name.
    :param cuda_name: name of the eddy command to run when cuda==True.
    :param s2v: list of parameters eddy for slice-to-volume correction (see Eddy FSL documentation): [mporder,s2v_niter,s2v_lambda,s2v_interp].
    :param olrep: list of parameters eddy outlier replacement (see Eddy FSL documentation): [repol,ol_nstd,ol_nvox,ol_type].
    :param slurm: Whether to use the Slurm Workload Manager or not.
    :param slurm_email: Email adress to send notification if a task fails.
    :param slurm_timeout: Replace the default slurm timeout by a custom timeout.
    :param cpus: Replace the default number of slurm cpus by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing
    :param slurm_mem: Replace the default amount of ram allocated to the slurm task by a custom amount of ram.
    :param qc_reg: If true, the motion registration step of the quality control will be performed.
    :param niter: Define the number of iterations for eddy volume-to-volume
    :param slspec_gc_path: Path to the folder containing volume specific slice-specification for eddy. If not None, eddy motion correction with gradient cycling will be performed.
    :param report: If False, no quality report will be generated.
    """


    in_reslice = reslice
    assert starting_state in (None,"None", "denoising", "gibbs", "topup", "eddy", "biasfield", "report"), 'invalid starting state!'
    if starting_state == "denoising":
        assert denoising == True, 'if starting_state is denoising, denoising must be True!'
    if starting_state == "gibbs":
        assert gibbs == True, 'if starting_state is gibbs, gibbs must be True!'
    if starting_state == "topup":
        assert topup == True, 'if starting_state is topup, topup must be True!'
    if starting_state == "eddy":
        assert eddy == True, 'if starting_state is eddy, eddy must be True!'
    if starting_state == "biasfield":
        assert biasfield == True, 'if starting_state is biasfield, biasfield must be True!'
    if starting_state == "None":
        starting_state = None

    if topupConfig == "None":
        topupConfig = None

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
        data, affine = reslice(data, affine, voxel_size, new_voxel_size, num_processes=core_count)

        if reslice_addSlice:
            data = np.insert(data, np.size(data,2), 0, axis=2)

        save_nifti(reslice_path + '/' + patient_path + '_reslice.nii.gz', data, affine)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f.close()

    if starting_state == None:
        b0_mask, mask = median_otsu(data, median_radius=bet_median_radius, numpass=bet_numpass, vol_idx=range(0, np.shape(data)[3]), dilate=bet_dilate)
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz',mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz',b0_mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                   mask.astype(np.float32), affine)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
        f.close()

    if not denoising and not eddy and not gibbs and not topup and not biasfield:
        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz', b0_mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz', mask.astype(np.float32), affine)
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    denoising_path = folder_path + '/' + patient_path + '/dMRI/preproc/mppca'
    if denoising and starting_state!="gibbs" and starting_state!="eddy" and starting_state!="topup" and starting_state!="biasfield" and starting_state!="report":
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of denoising for patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Denoising launched for patient %s \n" % p)
        f.close()

        makedir(denoising_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if (starting_state == "denoising"):
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

        if not eddy and not gibbs and not topup and not biasfield:
            save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                       b0_mask.astype(np.float32), affine)
            save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                       mask.astype(np.float32), affine)
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    if gibbs and starting_state!="eddy" and starting_state!="topup" and starting_state!="biasfield" and starting_state!="report":
        gibbs_path = folder_path + '/' + patient_path + '/dMRI/preproc/gibbs'
        makedir(gibbs_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if (starting_state == "gibbs"):
            mask_path = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz'
            if not denoising:
                b0_mask_path = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz'
            else:
                b0_mask_path = denoising_path + '/' + patient_path + '_mppca.nii.gz'
            b0_mask, affine, voxel_size = load_nifti(b0_mask_path, return_voxsize=True)
            mask, _ = load_nifti(mask_path)

        data = gibbs_removal(b0_mask, num_threads=core_count)
        corrected_path = folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/" + patient_path + '_gibbscorrected.nii.gz'
        save_nifti(corrected_path, data.astype(np.float32), affine)

        if not eddy and not topup and not biasfield:
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
    if topup and starting_state!="eddy" and starting_state!="biasfield" and starting_state!="report":

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
            line = " ".join(line.split())
            topup_index = [int(s) for s in line.split(' ')]

        with open(folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt') as f:
            topup_acq = [[float(x) for x in line2.split()] for line2 in f]

        #Find all the bo to extract.
        current_index = 0
        all_index ={}
        i=1
        roi=[]
        for ind in topup_index:
            if ind!=current_index and ind not in all_index:
                roi.append(i)
                fslroi = "fslroi " + imain_tot + " " + topup_path + "/b0_"+str(i)+".nii.gz "+str(i-1)+" 1"
                process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=topup_log,
                                           stderr=subprocess.STDOUT)
                output, error = process.communicate()
                print("B0 of index" + str(i) + " extracted!")
            current_index=ind
            all_index[ind] = all_index.get(ind,0) + 1
            i=i+1

        #Merge b0
        if len(roi) == 1:
            shutil.copyfile(topup_path + "/b0_"+str(roi[0])+".nii.gz", topup_path + "/b0.nii.gz")
        else:
            roi_to_merge=""
            for r in roi:
                roi_to_merge = roi_to_merge + " " + topup_path +"/b0_" + str(r) + ".nii.gz"
            print("The following roi will be merged: " + roi_to_merge)
            cmd = "fslmerge -t " + topup_path + "/b0.nii.gz" + roi_to_merge
            process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=topup_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()

        #Check if multiple or single encoding direction
        curr_x=0.0
        curr_y=0.0
        curr_z=0.0
        first=True
        print("Topup acq parameters:")
        print(topup_acq)
        for acq in topup_acq:
            if not first and (curr_x!=acq[1] or curr_y!=acq[2] or curr_z!=acq[3]):
                multiple_encoding=True
            first=False
            curr_x=acq[1]
            curr_y=acq[2]
            curr_z=acq[3]

        if multiple_encoding and not forceSynb0DisCo:
            makedir(topup_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
            f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Patient %s \n" % p + " has multiple direction of gradient encoding, launching topup directly ")
            topupConfig = 'b02b0.cnf' if topupConfig is None else topupConfig
            bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; topup --imain="' + topup_path + '/b0.nii.gz" --config="' + topupConfig + '" --datain="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --fout="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_fout_estimate" --iout="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_iout_estimate" --verbose'
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

            if not eddy:
                inindex=""
                first=True
                for r in roi:
                    if first:
                        inindex = str(topup_index[r-1])
                    else:
                        inindex = inindex + "," + str(topup_index[r-1])

                bashCommand2 = 'applytopup --imain="' + imain_tot + '" --inindex='+inindex+' --datain="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --topup="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr"'

                process2 = subprocess.Popen(bashCommand2, universal_newlines=True, shell=True, stdout=topup_log,
                                            stderr=subprocess.STDOUT)
                # wait until apply topup finish
                output, error = process2.communicate()

        else:
            f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Patient %s \n" % p + " has a single direction of gradient encoding, launching synb0DisCo ")
            f.close()
            from elikopy.utils import synb0DisCo
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt',
                            topup_path + '/synb0-DisCo/' + 'acqparams_topup.txt')

            shutil.copyfile(folder_path + '/' + patient_path + '/T1/' + patient_path + '_T1.nii.gz',
                            topup_path + '/synb0-DisCo/' + 'T1.nii.gz')

            shutil.copyfile(topup_path + "/b0.nii.gz",topup_path + "/synb0-DisCo/b0.nii.gz")

            process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=topup_log,stderr=subprocess.STDOUT)
            output, error = process.communicate()
            synb0DisCo(topup_path,patient_path,starting_step=None,topup=True,gpu=useGPUsynb0DisCo)

            if not eddy:
                bashCommand2 = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; applytopup --imain="' + imain_tot + '" --inindex=1 --datain="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --topup="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --method=jac --interp=spline --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr"'

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

        if not eddy and not biasfield:
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

    if eddy and starting_state!="biasfield" and starting_state!="report":
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of eddy for patient %s \n" % p)

        eddy_path = folder_path + '/' + patient_path + "/dMRI/preproc/eddy"
        makedir(eddy_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if cuda:
            eddycmd = "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; " + cuda_name
        else:
            eddycmd = "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; eddy"

        fwhm = '10'
        for _ in range(niter-1):
            fwhm = fwhm + ',0'

        if s2v[0] != 0:
            slspec_path = folder_path + '/' + patient_path + '/dMRI/raw/' + 'slspec.txt'
            if slspec_gc_path is not None and os.path.isdir(slspec_gc_path):
                if gibbs:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --ge_slspecs="' + slspec_gc_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                elif denoising:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --ge_slspecs="' + slspec_gc_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                else:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --ge_slspecs="' + slspec_gc_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            elif os.path.isfile(slspec_path):
                if gibbs:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                elif denoising:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                else:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            else:
                if gibbs:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                elif denoising:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                else:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
        else:
            if gibbs:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            elif denoising:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            else:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'

        if topup:
            bashCommand = bashCommand + ' --topup="' + folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate"'

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

        if not biasfield:
            save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                       b0_mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                   mask.astype(np.float32), affine)
        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                        folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(
            folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + "_eddy_corr.eddy_rotated_bvecs",
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")


    if biasfield and starting_state!="report":

        #import SimpleITK as sitk
        makedir(folder_path + '/' + patient_path + "/dMRI/preproc/biasfield/", folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if eddy:
            inputImage = folder_path + '/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr.nii.gz'
        elif topup:
            inputImage = folder_path + '/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr.nii.gz'
        elif gibbs:
            inputImage = folder_path + '/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz'
        elif denoising:
            inputImage = folder_path + '/' + patient_path + '/dMRI/preproc/mppca/' + patient_path + '_mppca.nii.gz'
        else:
            inputImage = folder_path + '/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_mask.nii.gz'

        #bashCommand= "N4BiasFieldCorrection -i " + inputImage + " -o [" + folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_corr.nii.gz, " + folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_est.nii.gz] -d 4"


        bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; dwibiascorrect ants {} {} -fslgrad {} {} -mask {} -bias {} -scratch {} -force -info -nthreads {}'.format(
            inputImage, folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_corr.nii.gz",
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec",
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
            folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
            folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_est.nii.gz",
            folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/tmp',
            core_count)

        '''bashCommand = 'export OMP_NUM_THREADS=' + str(
            core_count) + ' ; dwibiascorrect ants {} {} -fslgrad {} {} -mask {} -bias {} -scratch {} -force -info -nthreads {} -ants.b {} -ants.c {} '.format(
            inputImage,
            folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_corr.nii.gz",
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec",
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
            folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
            folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_est.nii.gz",
            folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/tmp',
            core_count,
            biasfield_bsplineFitting,
            biasfield_convergence)'''

        import subprocess
        bashcmd = bashCommand.split()
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Bias Field launched for patient %s \n" % p + " with bash command " + bashCommand)
        f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Bias Field launched for patient %s \n" % p + " with bash command " + bashCommand)
        f.close()

        biasfield_log = open(folder_path + '/' + patient_path + "/dMRI/preproc/biasfield/biasfield_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=biasfield_log,
                                   stderr=subprocess.STDOUT)

        # wait until biasfield finish
        output, error = process.communicate()
        biasfield_log.close()

        #shutil.rmtree(folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/tmp', ignore_errors=True)

        shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + '_biasfield_corr.nii.gz',
            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz')

        data, affine = load_nifti(
            folder_path + '/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + '_biasfield_corr.nii.gz')
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]), dilate=2)

        save_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
                       b0_mask.astype(np.float32), affine)
        save_nifti(folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
                   mask.astype(np.float32), affine)

        if not eddy:
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    if not report:
        return

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting QC %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting QC %s \n" % p)
    f.close()
    # ==================================================================================================================

    """Imports"""
    from dipy.io.image import load_nifti, load_nifti_data
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dipy.reconst.dti as dti
    from dipy.align.imaffine import (AffineMap, MutualInformationMetric, AffineRegistration)
    from dipy.align.transforms import RigidTransform3D
    from dipy.segment.mask import segment_from_cfa
    from dipy.segment.mask import bounding_box
    from scipy.ndimage.morphology import binary_dilation
    from os.path import isdir
    from skimage import measure
    from fpdf import FPDF

    preproc_path = folder_path + '/' + patient_path + '/dMRI/preproc/'
    raw_path = folder_path + '/' + patient_path + '/dMRI/raw/'
    mask_path = folder_path + '/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz'
    qc_path = preproc_path + 'quality_control'
    makedir(qc_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

    """Open the data"""

    # original data
    raw_data, raw_affine = load_nifti(raw_path + patient_path + "_raw_dmri.nii.gz")
    bvals, bvecs = read_bvals_bvecs(raw_path + patient_path + "_raw_dmri.bval",
                                    raw_path + patient_path + "_raw_dmri.bvec")
    gtab_raw = gradient_table(bvals, bvecs)

    # reslice data
    bool_reslice = isdir(preproc_path + "reslice")
    if bool_reslice:
        reslice_data, reslice_affine = load_nifti(preproc_path + "reslice/" + patient_path + "_reslice.nii.gz")

    # bet data (stage to compare with final)
    bet_data, bet_affine = load_nifti(preproc_path + "bet/" + patient_path + "_mask.nii.gz")
    mask_raw, mask_raw_affine = load_nifti(preproc_path + "bet/" + patient_path + "_binary_mask.nii.gz")

    # mppca data
    bool_mppca = isdir(preproc_path + "mppca")
    if bool_mppca:
        mppca_data, mppca_affine = load_nifti(preproc_path + "mppca/" + patient_path + "_mppca.nii.gz")
        sigma, sigma_affine = load_nifti(preproc_path + "mppca/" + patient_path + "_sigmaNoise.nii.gz")

    # gibbs data
    bool_gibbs = isdir(preproc_path + "gibbs")
    if bool_gibbs:
        gibbs_data, gibbs_affine = load_nifti(preproc_path + "gibbs/" + patient_path + "_gibbscorrected.nii.gz")

    # topup data
    bool_topup = isdir(preproc_path + "topup")
    if bool_topup:
        if not eddy:
            topup_data, topup_affine = load_nifti(preproc_path + "topup/" + patient_path + "_topup_corr.nii.gz")
        field_data, field_affine = load_nifti(
            preproc_path + "topup/" + patient_path + "_topup_estimate_fieldcoef.nii.gz")

    # eddy data (=preproc total)
    bool_eddy = isdir(preproc_path + "eddy")
    preproc_data, preproc_affine = load_nifti(preproc_path + patient_path + "_dmri_preproc.nii.gz")
    mask_preproc, mask_preproc_affine = load_nifti(mask_path)
    bvals, bvecs = read_bvals_bvecs(preproc_path + patient_path + "_dmri_preproc.bval",
                                    preproc_path + patient_path + "_dmri_preproc.bvec")
    gtab_preproc = gradient_table(bvals, bvecs)

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - Preprocessing', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    rows = ["patient id", "reslice", "denoising", "gibbs", "topup", "eddy", "bet_median_radius", "bet_numpass",
            "bet_dilate", "cuda", "s2v", "olrep"]
    cell_text = [[p], [in_reslice], [denoising], [gibbs], [topup], [eddy], [bet_median_radius], [bet_numpass],
                 [bet_dilate], [cuda], [s2v], [olrep]]

    fig, ax = plt.subplots()
    ax.axis('off')
    fig.tight_layout()
    the_table = plt.table(cellText=cell_text, rowLabels=rows, rowColours=['lightsteelblue'] * 12,
                          colColours=['lightsteelblue'], loc='center', colLabels=["Input parameters"])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(14)
    the_table.scale(0.8, 1.5)
    plt.savefig(qc_path + "/inputs.jpg", dpi=300, bbox_inches='tight');

    """Raw data""";

    bval = np.copy(gtab_raw.bvals)
    list_bval = []
    for i in range(np.shape(bval)[0]):
        test = bval[i]
        flag = True
        for j in range(len(list_bval)):
            if list_bval[j] - 50 < test < list_bval[j] + 50:
                flag = False
        if flag == True:
            list_bval.append(test)

    sl = np.shape(raw_data)[2] // 2
    fig, axs = plt.subplots(len(list_bval), 1, figsize=(14, 3 * len(list_bval)))
    for i in range(len(list_bval)):
        shell_index = np.where(np.logical_and(bval > list_bval[i] - 50, bval < list_bval[i] + 50))[0]
        # plot the shell
        plot_shell = np.zeros((np.shape(raw_data)[0], np.shape(raw_data)[1] * 5))
        plot_shell[:, 0:np.shape(raw_data)[1]] = raw_data[..., sl - 10, shell_index[0]]
        plot_shell[:, np.shape(raw_data)[1]:(np.shape(raw_data)[1] * 2)] = raw_data[..., sl - 5, shell_index[0]]
        plot_shell[:, (np.shape(raw_data)[1] * 2):(np.shape(raw_data)[1] * 3)] = raw_data[..., sl, shell_index[0]]
        plot_shell[:, (np.shape(raw_data)[1] * 3):(np.shape(raw_data)[1] * 4)] = raw_data[..., sl + 5, shell_index[0]]
        plot_shell[:, (np.shape(raw_data)[1] * 4):(np.shape(raw_data)[1] * 5)] = raw_data[..., sl + 10, shell_index[0]]
        axs[i].imshow(plot_shell, cmap='gray')
        axs[i].set_axis_off()
        axs[i].set_title('Raw data at b=' + str(list_bval[i]))
    plt.savefig(qc_path + "/raw.jpg", dpi=300, bbox_inches='tight')

    list_images = [[qc_path + "/title.jpg", qc_path + "/inputs.jpg"],[qc_path + "/raw.jpg"]]

    """data each step Plot + brain mask""";

    def mask_to_coor(mask):
        contours = measure.find_contours(mask.astype(float), 0)
        X, Y = [], []
        for i in range(0, len(contours[0])):
            X.append(int(contours[0][i][1]))
            Y.append(int(contours[0][i][0]))
        return X, Y

    sl = np.shape(bet_data)[2] // 2
    for i in range(len(list_bval)):
        shell_index = np.where(np.logical_and(bval > list_bval[i] - 50, bval < list_bval[i] + 50))[0]
        numstep = 1 + bool_mppca + bool_gibbs  + bool_eddy
        if not eddy:
            numstep = numstep + bool_topup

        current_subplot = 0
        fig, axs = plt.subplots(numstep, 1, figsize=(14, 3 * numstep))
        fig.suptitle('Overview of processing steps for b=' + str(list_bval[i]), y=0.95, fontsize=16)

        # plot bet
        X1, Y1 = mask_to_coor(mask_raw[..., sl - 10])
        X2, Y2 = mask_to_coor(mask_raw[..., sl - 5])
        X3, Y3 = mask_to_coor(mask_raw[..., sl])
        X4, Y4 = mask_to_coor(mask_raw[..., sl + 5])
        X5, Y5 = mask_to_coor(mask_raw[..., sl + 10])
        Y = Y1 + Y2 + Y3 + Y4 + Y5
        X = X1 + [x + np.shape(mask_raw)[1] for x in X2] + [x + np.shape(mask_raw)[1] * 2 for x in X3] + [
            x + np.shape(mask_raw)[1] * 3 for x in X4] + [x + np.shape(mask_raw)[1] * 4 for x in X5]
        if numstep == 1:
            fig_scat = plt.scatter(X, Y, marker='.', s=1, c='red')
        else:
            axs[current_subplot].scatter(X, Y, marker='.', s=1, c='red')
        plot_bet = np.zeros((np.shape(bet_data)[0], np.shape(bet_data)[1] * 5))
        plot_bet[:, 0:np.shape(bet_data)[1]] = bet_data[..., sl - 10, shell_index[0]]
        plot_bet[:, np.shape(bet_data)[1]:(np.shape(bet_data)[1] * 2)] = bet_data[..., sl - 5, shell_index[0]]
        plot_bet[:, (np.shape(bet_data)[1] * 2):(np.shape(bet_data)[1] * 3)] = bet_data[..., sl, shell_index[0]]
        plot_bet[:, (np.shape(bet_data)[1] * 3):(np.shape(bet_data)[1] * 4)] = bet_data[..., sl + 5, shell_index[0]]
        plot_bet[:, (np.shape(bet_data)[1] * 4):(np.shape(bet_data)[1] * 5)] = bet_data[..., sl + 10, shell_index[0]]
        if numstep==1:
            plt.imshow(plot_bet, cmap='gray')
            #fig_scat.set_axis_off()
            #plt.set_title('brain extraction')
        else:
            axs[current_subplot].imshow(plot_bet, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('brain extraction')
        current_subplot = current_subplot + 1
        # plot mppca
        if bool_mppca:
            plot_mppca = np.zeros((np.shape(mppca_data)[0], np.shape(mppca_data)[1] * 5))
            plot_mppca[:, 0:np.shape(mppca_data)[1]] = mppca_data[..., sl - 10, shell_index[0]]
            plot_mppca[:, np.shape(mppca_data)[1]:(np.shape(mppca_data)[1] * 2)] = mppca_data[
                ..., sl - 5, shell_index[0]]
            plot_mppca[:, (np.shape(mppca_data)[1] * 2):(np.shape(mppca_data)[1] * 3)] = mppca_data[
                ..., sl, shell_index[0]]
            plot_mppca[:, (np.shape(mppca_data)[1] * 3):(np.shape(mppca_data)[1] * 4)] = mppca_data[
                ..., sl + 5, shell_index[0]]
            plot_mppca[:, (np.shape(mppca_data)[1] * 4):(np.shape(mppca_data)[1] * 5)] = mppca_data[
                ..., sl + 10, shell_index[0]]
            axs[current_subplot].imshow(plot_mppca, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Denoising')
            current_subplot = current_subplot + 1
        # plot gibbs
        if bool_gibbs:
            plot_gibbs = np.zeros((np.shape(gibbs_data)[0], np.shape(gibbs_data)[1] * 5))
            plot_gibbs[:, 0:np.shape(gibbs_data)[1]] = gibbs_data[..., sl - 10, shell_index[0]]
            plot_gibbs[:, np.shape(gibbs_data)[1]:(np.shape(gibbs_data)[1] * 2)] = gibbs_data[
                ..., sl - 5, shell_index[0]]
            plot_gibbs[:, (np.shape(gibbs_data)[1] * 2):(np.shape(gibbs_data)[1] * 3)] = gibbs_data[
                ..., sl, shell_index[0]]
            plot_gibbs[:, (np.shape(gibbs_data)[1] * 3):(np.shape(gibbs_data)[1] * 4)] = gibbs_data[
                ..., sl + 5, shell_index[0]]
            plot_gibbs[:, (np.shape(gibbs_data)[1] * 4):(np.shape(gibbs_data)[1] * 5)] = gibbs_data[
                ..., sl + 10, shell_index[0]]
            axs[current_subplot].imshow(plot_gibbs, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Gibbs ringing correction')
            current_subplot = current_subplot + 1
        # plot topup
        if bool_topup and not eddy:
            plot_topup = np.zeros((np.shape(topup_data)[0], np.shape(topup_data)[1] * 5))
            plot_topup[:, 0:np.shape(topup_data)[1]] = topup_data[..., sl - 10, shell_index[0]]
            plot_topup[:, np.shape(topup_data)[1]:(np.shape(topup_data)[1] * 2)] = topup_data[
                ..., sl - 5, shell_index[0]]
            plot_topup[:, (np.shape(topup_data)[1] * 2):(np.shape(topup_data)[1] * 3)] = topup_data[
                ..., sl, shell_index[0]]
            plot_topup[:, (np.shape(topup_data)[1] * 3):(np.shape(topup_data)[1] * 4)] = topup_data[
                ..., sl + 5, shell_index[0]]
            plot_topup[:, (np.shape(topup_data)[1] * 4):(np.shape(topup_data)[1] * 5)] = topup_data[
                ..., sl + 10, shell_index[0]]
            axs[current_subplot].imshow(plot_topup, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Susceptibility induced distortions correction')
            current_subplot = current_subplot + 1
        # plot eddy
        if bool_eddy:
            plot_eddy = np.zeros((np.shape(preproc_data)[0], np.shape(preproc_data)[1] * 5))
            plot_eddy[:, 0:np.shape(preproc_data)[1]] = preproc_data[..., sl - 10, shell_index[0]]
            plot_eddy[:, np.shape(preproc_data)[1]:(np.shape(preproc_data)[1] * 2)] = preproc_data[
                ..., sl - 5, shell_index[0]]
            plot_eddy[:, (np.shape(preproc_data)[1] * 2):(np.shape(preproc_data)[1] * 3)] = preproc_data[
                ..., sl, shell_index[0]]
            plot_eddy[:, (np.shape(preproc_data)[1] * 3):(np.shape(preproc_data)[1] * 4)] = preproc_data[
                ..., sl + 5, shell_index[0]]
            plot_eddy[:, (np.shape(preproc_data)[1] * 4):(np.shape(preproc_data)[1] * 5)] = preproc_data[
                ..., sl + 10, shell_index[0]]
            axs[current_subplot].imshow(plot_eddy, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Eddy and motion correction')
            current_subplot = current_subplot + 1

        plt.savefig(qc_path + "/processing" + str(i) + ".jpg", dpi=300, bbox_inches='tight')
        list_images.append([qc_path + "/processing" + str(i) + ".jpg"])

    """Reslice data""";

    if bool_reslice:
        fig, axs = plt.subplots(2, 3, figsize=(8, 6))
        fig.suptitle('Data reslice', y=1, fontsize=16)
        # plot the raw and reslice data
        axs[0, 0].imshow(np.rot90(raw_data[:, :, np.shape(raw_data)[2] // 2, 0]), cmap='gray')
        axs[0, 0].set_axis_off()
        axs[0, 1].imshow(np.rot90(raw_data[:, np.shape(raw_data)[1] // 2, :, 0]), cmap='gray')
        axs[0, 1].set_axis_off()
        axs[0, 1].set_title('Raw data')
        axs[0, 2].imshow(np.rot90(raw_data[np.shape(raw_data)[0] // 2, :, :, 0]), cmap='gray')
        axs[0, 2].set_axis_off()
        axs[1, 0].imshow(np.rot90(reslice_data[:, :, np.shape(reslice_data)[2] // 2, 0]), cmap='gray')
        axs[1, 0].set_axis_off()
        axs[1, 1].imshow(np.rot90(reslice_data[:, np.shape(reslice_data)[1] // 2, :, 0]), cmap='gray')
        axs[1, 1].set_axis_off()
        axs[1, 1].set_title('Reslice data')
        axs[1, 2].imshow(np.rot90(reslice_data[np.shape(reslice_data)[0] // 2, :, :, 0]), cmap='gray')
        axs[1, 2].set_axis_off()
        plt.savefig(qc_path + "/reslice.jpg", dpi=300, bbox_inches='tight')
        list_images.append([qc_path + "/reslice.jpg"])

    """Gibbs ringing""";

    if bool_gibbs:

        if bool_mppca:
            previous = mppca_data
        else:
            previous = bet_data

        fig, axs = plt.subplots(len(list_bval), 3, figsize=(9, 3 * len(list_bval)))
        #fig.suptitle('Gibbs ringing correction', y=1, fontsize=16)
        for i in range(len(list_bval)):
            shell_index = np.where(np.logical_and(bval > list_bval[i] - 50, bval < list_bval[i] + 50))[0]
            # plot the gibbs before, after and residual
            axs[i, 0].imshow(previous[..., sl, shell_index[0]], cmap='gray')
            axs[i, 0].set_axis_off()
            axs[i, 0].set_title('Gibbs uncorrected at b=' + str(list_bval[i]))
            axs[i, 1].imshow(gibbs_data[..., sl, shell_index[0]], cmap='gray')
            axs[i, 1].set_axis_off()
            axs[i, 1].set_title('Gibbs corrected at b=' + str(list_bval[i]))
            axs[i, 2].imshow(np.abs(previous[..., sl, shell_index[0]] - gibbs_data[..., sl, shell_index[0]]),
                             cmap='gray')
            axs[i, 2].set_axis_off()
            axs[i, 2].set_title('Residual at b=' + str(list_bval[i]))
        plt.savefig(qc_path + "/gibbs.jpg", dpi=300, bbox_inches='tight')
        list_images.append([qc_path + "/gibbs.jpg"])

    """Noise correction""";

    if bool_mppca:

        # 1) DIPY SNR estimation =========================================================================

        tenmodel = dti.TensorModel(gtab_raw)
        _, maskSNR = median_otsu(raw_data, vol_idx=[0])
        tensorfit = tenmodel.fit(raw_data, mask=maskSNR)

        threshold = (0.5, 1, 0, 0.3, 0, 0.3)
        CC_box = np.zeros_like(raw_data[..., 0])
        mins, maxs = bounding_box(maskSNR)
        mins = np.array(mins)
        maxs = np.array(maxs)
        diff = (maxs - mins) // 4
        bounds_min = mins + diff
        bounds_max = maxs - diff
        CC_box[bounds_min[0]:bounds_max[0], bounds_min[1]:bounds_max[1], bounds_min[2]:bounds_max[2]] = 1
        mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box, threshold, return_cfa=True)

        mean_signal = np.mean(raw_data[mask_cc_part], axis=0)

        mask_noise = binary_dilation(maskSNR, iterations=20)
        mask_noise[..., :mask_noise.shape[-1] // 2] = 1
        mask_noise = ~mask_noise
        noise_std = np.std(raw_data[mask_noise, :])

        idx = np.sum(gtab_raw.bvecs, axis=-1) == 0
        gtab_raw.bvecs[idx] = np.inf
        axis_X = np.argmin(np.sum((gtab_raw.bvecs - np.array([1, 0, 0])) ** 2, axis=-1))
        axis_Y = np.argmin(np.sum((gtab_raw.bvecs - np.array([0, 1, 0])) ** 2, axis=-1))
        axis_Z = np.argmin(np.sum((gtab_raw.bvecs - np.array([0, 0, 1])) ** 2, axis=-1))

        stock = []
        for direction in [0, axis_X, axis_Y, axis_Z]:
            SNR = mean_signal[direction] / noise_std
            if direction == 0:
                SNRb0 = int(SNR)
            else:
                stock.append(SNR)
        stock = np.array(stock)

        rows = ["SNR of the b0 image", "Estimated SNR range"]
        cell_text = [[SNRb0], [str(int(np.min(stock))) + ' - ' + str(int(np.max(stock)))]]
        region = np.shape(raw_data)[0] // 2
        fig = plt.figure('Corpus callosum segmentation', figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Corpus callosum (CC)")
        plt.axis('off')
        red = cfa[..., 0]
        plt.imshow(np.rot90(red[region, ...]))
        plt.subplot(1, 2, 2)
        plt.title("CC mask used for SNR computation")
        plt.axis('off')
        plt.imshow(np.rot90(mask_cc_part[region, ...]))
        the_table = plt.table(cellText=cell_text, rowLabels=rows, rowColours=['lightsteelblue'] * 2,
                              bbox=[-0.5, -0.6, 1.5, 0.5])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        the_table.scale(2, 4)
        plt.savefig(qc_path + "/dipyNoise.jpg", dpi=300, bbox_inches='tight')

        # 2) MPPCA sigma + SNR estimation + before/after residual ==========================================

        fig, axs = plt.subplots(len(list_bval), 3, figsize=(9, 3 * len(list_bval)))
        #fig.suptitle('MPPCA denoising', y=1, fontsize=16)
        for i in range(len(list_bval)):
            shell_index = np.where(np.logical_and(bval > list_bval[i] - 50, bval < list_bval[i] + 50))[0]
            # plot the gibbs before, after and residual
            axs[i, 0].imshow(bet_data[..., sl, shell_index[0]], cmap='gray')
            axs[i, 0].set_axis_off()
            axs[i, 0].set_title('Original at b=' + str(list_bval[i]))
            axs[i, 1].imshow(mppca_data[..., sl, shell_index[0]], cmap='gray')
            axs[i, 1].set_axis_off()
            axs[i, 1].set_title('MPPCA denoised at b=' + str(list_bval[i]))
            axs[i, 2].imshow(np.abs(bet_data[..., sl, shell_index[0]] - mppca_data[..., sl, shell_index[0]]),
                             cmap='gray')
            axs[i, 2].set_axis_off()
            axs[i, 2].set_title('Residual at b=' + str(list_bval[i]))
        plt.savefig(qc_path + "/mppcaResidual.jpg", dpi=300, bbox_inches='tight')

        masked_sigma = np.ma.array(np.nan_to_num(sigma), mask=1 - mask_raw)
        mean_sigma = masked_sigma.mean()
        b0 = np.ma.array(mppca_data[..., 0], mask=1 - mask_raw)
        mean_signal = b0.mean()
        snr = mean_signal / mean_sigma
        sl = np.shape(sigma)[2] // 2
        plot_sigma = np.zeros((np.shape(sigma)[0], np.shape(sigma)[1] * 5))
        plot_sigma[:, 0:np.shape(sigma)[1]] = sigma[..., sl - 10]
        plot_sigma[:, np.shape(sigma)[1]:(np.shape(sigma)[1] * 2)] = sigma[..., sl - 5]
        plot_sigma[:, (np.shape(sigma)[1] * 2):(np.shape(sigma)[1] * 3)] = sigma[..., sl]
        plot_sigma[:, (np.shape(sigma)[1] * 3):(np.shape(sigma)[1] * 4)] = sigma[..., sl + 5]
        plot_sigma[:, (np.shape(sigma)[1] * 4):(np.shape(sigma)[1] * 5)] = sigma[..., sl + 10]
        rows = ["MPPCA SNR estimation"]
        cell_text = [[snr]]
        fig = plt.figure(figsize=(14, 4))
        plt.title("PCA Noise standard deviation estimation")
        plt.axis('off')
        plt.imshow(plot_sigma, cmap='gray')
        the_table = plt.table(cellText=cell_text, rowLabels=rows, rowColours=['lightsteelblue'] * 2,
                              bbox=[0.25, -0.3, 0.4, 0.2])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        the_table.scale(1, 1)
        plt.savefig(qc_path + "/mppcaSigma.jpg", dpi=300, bbox_inches='tight')

        list_images.append([qc_path + "/dipyNoise.jpg", qc_path + "/mppcaSigma.jpg"])
        list_images.append([qc_path + "/mppcaResidual.jpg"])

        # 3) tSNR estimation ===============================================================================

        for i in range(len(list_bval)):
            shell_mask = np.logical_and(bval > list_bval[i] - 50, bval < list_bval[i] + 50)
            if np.sum(shell_mask) > 3:
                # Compute the tSNR for raw and preproc
                mean_vox = np.mean(bet_data[..., shell_mask], axis=-1)
                std_vox = np.std(bet_data[..., shell_mask], axis=-1)
                tsnr_raw = np.nan_to_num(mean_vox / std_vox) * mask_raw
                mean_vox = np.mean(preproc_data[..., shell_mask], axis=-1)
                std_vox = np.std(preproc_data[..., shell_mask], axis=-1)
                tsnr_preproc = np.nan_to_num(mean_vox / std_vox) * mask_preproc

                # Make the Plot
                sl = np.shape(bet_data)[2] // 2
                # image of raw
                plot_raw = np.zeros((np.shape(bet_data)[0], np.shape(bet_data)[1] * 5))
                plot_raw[:, 0:np.shape(bet_data)[1]] = tsnr_raw[..., sl - 10]
                plot_raw[:, np.shape(bet_data)[1]:(np.shape(bet_data)[1] * 2)] = tsnr_raw[..., sl - 5]
                plot_raw[:, (np.shape(bet_data)[1] * 2):(np.shape(bet_data)[1] * 3)] = tsnr_raw[..., sl]
                plot_raw[:, (np.shape(bet_data)[1] * 3):(np.shape(bet_data)[1] * 4)] = tsnr_raw[..., sl + 5]
                plot_raw[:, (np.shape(bet_data)[1] * 4):(np.shape(bet_data)[1] * 5)] = tsnr_raw[..., sl + 10]
                # image of preproc
                plot_preproc = np.zeros((np.shape(preproc_data)[0], np.shape(preproc_data)[1] * 5))
                plot_preproc[:, 0:np.shape(preproc_data)[1]] = tsnr_preproc[..., sl - 10]
                plot_preproc[:, np.shape(preproc_data)[1]:(np.shape(preproc_data)[1] * 2)] = tsnr_preproc[..., sl - 5]
                plot_preproc[:, (np.shape(preproc_data)[1] * 2):(np.shape(preproc_data)[1] * 3)] = tsnr_preproc[..., sl]
                plot_preproc[:, (np.shape(preproc_data)[1] * 3):(np.shape(preproc_data)[1] * 4)] = tsnr_preproc[
                    ..., sl + 5]
                plot_preproc[:, (np.shape(preproc_data)[1] * 4):(np.shape(preproc_data)[1] * 5)] = tsnr_preproc[
                    ..., sl + 10]
                # image of difference
                plot_diff = np.zeros((np.shape(preproc_data)[0], np.shape(preproc_data)[1] * 5))
                plot_diff[:, 0:np.shape(preproc_data)[1]] = tsnr_raw[..., sl - 10] - tsnr_preproc[..., sl - 10]
                plot_diff[:, np.shape(preproc_data)[1]:(np.shape(preproc_data)[1] * 2)] = tsnr_raw[..., sl - 5] - \
                                                                                          tsnr_preproc[..., sl - 5]
                plot_diff[:, (np.shape(preproc_data)[1] * 2):(np.shape(preproc_data)[1] * 3)] = tsnr_raw[..., sl] - \
                                                                                                tsnr_preproc[..., sl]
                plot_diff[:, (np.shape(preproc_data)[1] * 3):(np.shape(preproc_data)[1] * 4)] = tsnr_raw[..., sl + 5] - \
                                                                                                tsnr_preproc[
                                                                                                    ..., sl + 5]
                plot_diff[:, (np.shape(preproc_data)[1] * 4):(np.shape(preproc_data)[1] * 5)] = tsnr_raw[..., sl + 10] - \
                                                                                                tsnr_preproc[
                                                                                                    ..., sl + 10]

                masked_tsnr_preproc = np.ma.array(tsnr_preproc, mask=1 - mask_preproc)
                masked_tsnr_raw = np.ma.array(tsnr_raw, mask=1 - mask_raw)
                masked_tsnr_diff = np.ma.array(tsnr_raw - tsnr_preproc, mask=1 - mask_preproc)
                max_plot = max(masked_tsnr_preproc.mean() + 2 * masked_tsnr_preproc.std(),
                               masked_tsnr_raw.mean() + 2 * masked_tsnr_raw.std())
                min_plot = min(masked_tsnr_preproc.mean() - 2 * masked_tsnr_preproc.std(),
                               masked_tsnr_raw.mean() - 2 * masked_tsnr_raw.std())

                fig, axs = plt.subplots(3, 1, figsize=(14, 10))
                fig.suptitle('tSNR for shell b=' + str(list_bval[i]), y=0.95, fontsize=16)
                axs[0].imshow(plot_raw, cmap='hot', vmax=max_plot, vmin=min_plot)
                axs[0].set_axis_off()
                axs[0].set_title('Raw data tSNR')
                axs[1].imshow(plot_preproc, cmap='hot', vmax=max_plot, vmin=min_plot)
                axs[1].set_axis_off()
                axs[1].set_title('Processed data tSNR')
                axs[2].imshow(plot_diff, cmap='jet', vmax=0, vmin=masked_tsnr_diff.mean() - 3 * masked_tsnr_diff.std())
                axs[2].set_axis_off()
                axs[2].set_title('difference: raw_tSNR - processed_tSNR')
                fig.colorbar(axs[2].imshow(plot_diff, cmap='jet', vmax=0,
                                           vmin=masked_tsnr_diff.mean() - 3 * masked_tsnr_diff.std()), ax=axs,
                             orientation='horizontal', pad=0.02, shrink=0.7)
                plt.savefig(qc_path + "/tsnr" + str(i) + ".jpg", dpi=300, bbox_inches='tight')
                list_images.append([qc_path + "/tsnr" + str(i) + ".jpg"])

    """Topup (synb0 + field)""";
    if bool_topup:
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        fig.suptitle('Topup estimated field coefficients', y=1.1, fontsize=16)

        sl = np.shape(field_data)[2] // 2
        plot_field = np.zeros((np.shape(field_data)[0], np.shape(field_data)[1] * 5))
        plot_field[:, 0:np.shape(field_data)[1]] = field_data[..., sl - 10]
        plot_field[:, np.shape(field_data)[1]:(np.shape(field_data)[1] * 2)] = field_data[..., sl - 5]
        plot_field[:, (np.shape(field_data)[1] * 2):(np.shape(field_data)[1] * 3)] = field_data[..., sl]
        plot_field[:, (np.shape(field_data)[1] * 3):(np.shape(field_data)[1] * 4)] = field_data[..., sl + 5]
        plot_field[:, (np.shape(field_data)[1] * 4):(np.shape(field_data)[1] * 5)] = field_data[..., sl + 10]
        axs[0].imshow(plot_field, cmap='gray')
        axs[0].set_axis_off()

        sl = np.shape(field_data)[1] // 2
        plot_field = np.zeros((np.shape(field_data)[2], np.shape(field_data)[0] * 5))
        plot_field[:, 0:np.shape(field_data)[0]] = np.rot90(field_data[..., sl - 10, :])
        plot_field[:, np.shape(field_data)[0]:(np.shape(field_data)[0] * 2)] = np.rot90(field_data[..., sl - 5, :])
        plot_field[:, (np.shape(field_data)[0] * 2):(np.shape(field_data)[0] * 3)] = np.rot90(field_data[..., sl, :])
        plot_field[:, (np.shape(field_data)[0] * 3):(np.shape(field_data)[0] * 4)] = np.rot90(
            field_data[..., sl + 5, :])
        plot_field[:, (np.shape(field_data)[0] * 4):(np.shape(field_data)[0] * 5)] = np.rot90(
            field_data[..., sl + 10, :])
        axs[1].imshow(plot_field, cmap='gray')
        axs[1].set_axis_off()

        sl = np.shape(field_data)[0] // 2
        plot_field = np.zeros((np.shape(field_data)[2], np.shape(field_data)[1] * 5))
        plot_field[:, 0:np.shape(field_data)[1]] = np.rot90(field_data[sl - 10, ...])
        plot_field[:, np.shape(field_data)[1]:(np.shape(field_data)[1] * 2)] = np.rot90(field_data[sl - 5, ...])
        plot_field[:, (np.shape(field_data)[1] * 2):(np.shape(field_data)[1] * 3)] = np.rot90(field_data[sl, ...])
        plot_field[:, (np.shape(field_data)[1] * 3):(np.shape(field_data)[1] * 4)] = np.rot90(field_data[sl + 5, ...])
        plot_field[:, (np.shape(field_data)[1] * 4):(np.shape(field_data)[1] * 5)] = np.rot90(field_data[sl + 10, ...])
        axs[2].imshow(plot_field, cmap='gray')
        axs[2].set_axis_off()
        plt.tight_layout()
        plt.savefig(qc_path + "/topup_field.jpg", dpi=300, bbox_inches='tight')
        list_images.append([qc_path + "/topup_field.jpg"])

    """Motion registration""";
    if bool_eddy and qc_reg:
        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)
        level_iters = [10000, 1000, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]
        affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors, verbosity=0)
        params0 = None
        transform = RigidTransform3D()

        # ===========================================================
        S0s_raw = bet_data[:, :, :, gtab_raw.b0s_mask]
        S0s_preproc = preproc_data[:, :, :, gtab_preproc.b0s_mask]
        volume = []
        motion_raw = []
        motion_proc = []
        for i in range(np.shape(preproc_data)[3]):
            #print('current iteration : ', i, end="\r")
            volume.append(i)

            rigid = affreg.optimize(np.copy(S0s_raw[..., 0]), np.copy(bet_data[..., i]), transform, params0, bet_affine,
                                    bet_affine, ret_metric=True)
            motion_raw.append(rigid[1])

            rigid = affreg.optimize(np.copy(S0s_preproc[..., 0]), np.copy(preproc_data[..., i]), transform, params0,
                                    preproc_affine, preproc_affine, ret_metric=True)
            motion_proc.append(rigid[1])
        # ============================================================

        motion_unproc = np.array(motion_raw)
        motion_proc = np.array(motion_proc)

        fig, (ax1, ax2) = plt.subplots(2, sharey=True, figsize=(10, 6))
        ax1.bar(volume, np.abs(motion_unproc[:, 3]) + np.abs(motion_unproc[:, 4]) + np.abs(motion_unproc[:, 5]),
                label='z')
        ax1.bar(volume, np.abs(motion_unproc[:, 3]) + np.abs(motion_unproc[:, 4]), label='y')
        ax1.bar(volume, np.abs(motion_unproc[:, 3]), label='x')
        ax1.legend(title='Translation', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_title('raw data translation')
        ax2.bar(volume, np.abs(motion_proc[:, 3]) + np.abs(motion_proc[:, 4]) + np.abs(motion_proc[:, 5]),
                label='z translation')
        ax2.bar(volume, np.abs(motion_proc[:, 3]) + np.abs(motion_proc[:, 4]), label='y translation')
        ax2.bar(volume, np.abs(motion_proc[:, 3]), label='x translation')
        ax2.set_title('processed data translation')
        plt.savefig(qc_path + "/motion1.jpg", dpi=300, bbox_inches='tight')

        fig, (ax1, ax2) = plt.subplots(2, sharey=True, figsize=(10, 6))
        ax1.bar(volume, np.abs(motion_unproc[:, 0]) + np.abs(motion_unproc[:, 1]) + np.abs(motion_unproc[:, 2]),
                label='z')
        ax1.bar(volume, np.abs(motion_unproc[:, 0]) + np.abs(motion_unproc[:, 1]), label='y')
        ax1.bar(volume, np.abs(motion_unproc[:, 0]), label='x')
        ax1.legend(title='Rotation', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_title('raw data rotation')
        ax2.bar(volume, np.abs(motion_proc[:, 0]) + np.abs(motion_proc[:, 1]) + np.abs(motion_proc[:, 2]),
                label='z rotation')
        ax2.bar(volume, np.abs(motion_proc[:, 0]) + np.abs(motion_proc[:, 2]), label='y rotation')
        ax2.bar(volume, np.abs(motion_proc[:, 0]), label='x rotation')
        ax2.set_title('processed data rotation');
        plt.savefig(qc_path + "/motion2.jpg", dpi=300, bbox_inches='tight');

        list_images.append([qc_path + "/motion1.jpg", qc_path + "/motion2.jpg"])

    """Save as a pdf"""

    # list_images = [qc_path+'/'+i for i in os.listdir(qc_path) if i.endswith(".jpg")]

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.WIDTH = 210
            self.HEIGHT = 297

        def header(self):
            # self.image('assets/logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 11)
            self.cell(self.WIDTH - 80)
            self.cell(60, 1, 'Quality control report - Preprocessing', 0, 0, 'R')
            self.ln(20)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

        def page_body(self, images):
            # Determine how many plots there are per page and set positions
            # and margins accordingly
            if len(images) == 3:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, 25 + 20, self.WIDTH - 30)
                self.image(images[2], 15, self.WIDTH / 2 + 40, self.WIDTH - 30)
            elif len(images) == 2:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 40, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)

        def print_page(self, images):
            # Generates the report
            self.add_page()
            self.page_body(images)

    pdf = PDF()

    for elem in list_images:
        pdf.print_page(elem)

    pdf.output(qc_path + '/qc_report.pdf', 'F');

    """Eddy quad + SNR/CNR""";

    if bool_eddy:
        # Do Eddy quad for the subject
        slspec_path = folder_path + '/' + patient_path + '/dMRI/raw/' + 'slspec.txt'
        if os.path.isfile(slspec_path):
            if topup:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" -s "' + slspec_path + '" -f "' + preproc_path + 'topup/' + patient_path + '_topup_estimate_fieldcoef.nii.gz"'
            else:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" -s "' + slspec_path + '"'
        else:
            if topup:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" -f "' + preproc_path + 'topup/' + patient_path + '_topup_estimate_fieldcoef.nii.gz"'
            else:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval"'

        import subprocess
        bashcmd = bashCommand.split()
        qc_log = open(qc_path + "/qc_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=qc_log,
                                   stderr=subprocess.STDOUT)
        output, error = process.communicate()
        qc_log.close()

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
    # Residual ============================================
    reconstructed = tenfit.predict(gtab, S0=data[...,0])
    residual = data - reconstructed
    save_nifti(folder_path + '/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_residual.nii.gz",
               residual.astype(np.float32), affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting QC %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting QC %s \n" % p)
    f.close()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metric1 = np.array(255 * RGB, 'uint8')
    metric2 = np.copy(MD)
    qc_path = folder_path + '/' + patient_path + "/dMRI/microstructure/dti/quality_control"
    makedir(qc_path, folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", log_prefix)

    mse = np.mean(residual ** 2, axis=-1)
    R2 = np.zeros_like(mse)
    (itot, jtot, ktot) = np.shape(R2)
    for i in range(itot):
        for j in range(jtot):
            for k in range(ktot):
                R2[i, j, k] = np.corrcoef(data[i, j, k, :], reconstructed[i, j, k, :])[0, 1] ** 2

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - DTI', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sl = np.shape(mse)[2] // 2
    masked_mse = np.ma.array(mse, mask=1 - mask)
    max_plot = masked_mse.mean() + 0.5 * masked_mse.std()
    plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., sl - 10]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., sl - 5]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., sl + 5]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., sl + 10]
    im0 = axs[0].imshow(plot_mse, cmap='gray', vmax=max_plot)
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., sl - 10]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., sl - 5]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., sl + 5]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., sl + 10]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sl = np.shape(metric1)[2] // 2
    plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5, 3), dtype=np.int16)
    plot_metric1[:, 0:np.shape(metric1)[1], :] = metric1[..., sl - 10, :]
    plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2), :] = metric1[..., sl - 5, :]
    plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3), :] = metric1[..., sl, :]
    plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4), :] = metric1[..., sl + 5, :]
    plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5), :] = metric1[..., sl + 10, :]
    axs[0].imshow(plot_metric1)
    axs[0].set_title('Fractional anisotropy')
    axs[0].set_axis_off()
    sl = np.shape(metric2)[2] // 2
    plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
    plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., sl - 10]
    plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., sl - 5]
    plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl]
    plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., sl + 5]
    plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., sl + 10]
    im1 = axs[1].imshow(plot_metric2, cmap='gray')
    axs[1].set_title('Mean diffusivity')
    axs[1].set_axis_off()
    plt.tight_layout();
    plt.savefig(qc_path + "/metrics.jpg", dpi=300, bbox_inches='tight');

    """Save as a pdf"""

    elem = [qc_path + "/title.jpg", qc_path + "/error.jpg", qc_path + "/metrics.jpg"]

    from fpdf import FPDF

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.WIDTH = 210
            self.HEIGHT = 297

        def header(self):
            # self.image('assets/logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 11)
            self.cell(self.WIDTH - 80)
            self.cell(60, 1, 'Quality control report - DTI', 0, 0, 'R')
            self.ln(20)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

        def page_body(self, images):
            # Determine how many plots there are per page and set positions
            # and margins accordingly
            if len(images) == 3:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, 25 + 20, self.WIDTH - 30)
                self.image(images[2], 15, self.WIDTH / 2 + 75, self.WIDTH - 30)
            elif len(images) == 2:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 40, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)

        def print_page(self, images):
            # Generates the report
            self.add_page()
            self.page_body(images)

    pdf = PDF()
    pdf.print_page(elem)
    pdf.output(qc_path + '/qc_report.pdf', 'F');

    if not os.path.exists(folder_path + '/' + patient_path + '/quality_control.pdf'):
        shutil.copyfile(qc_path + '/qc_report.pdf', folder_path + '/' + patient_path + '/quality_control.pdf')
    else:
        """Merge with QC of preproc""";
        from PyPDF2 import PdfFileMerger
        pdfs = [folder_path + '/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(folder_path + '/' + patient_path + '/quality_control_dti.pdf')
        merger.close()
        os.remove(folder_path + '/' + patient_path + '/quality_control.pdf')
        os.rename(folder_path + '/' + patient_path + '/quality_control_dti.pdf',folder_path + '/' + patient_path + '/quality_control.pdf')

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f.close()


def white_mask_solo(folder_path, p, corr_gibbs=True, core_count=1, forceUsePowerMap=False, debug=False):
    """ Compute a white matter mask of the diffusion data for each patient based on T1 volumes or on diffusion data if
    T1 is not available. Otherwise, compute the whitematter mask based on an anisotropic power map.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param corr_gibbs: Correct for gibbs oscillation.
    :param core_count: Number of allocated cpu cores.
    :param forceUsePowerMap: Force the use of an AnisotropicPower map for the white matter mask generation.
    :param debug: If true, additional intermediate output will be saved.
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
    if os.path.isfile(anat_path) and not forceUsePowerMap:
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from T1 %s \n" % p)
        f = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from T1 %s \n" % p)
        f.close()
        # Read the moving image ====================================
        anat_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'

        wm_log = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")

        # Correct for gibbs ringing
        if corr_gibbs:
            wm_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Beginning of gibbs for patient %s \n" % p)
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Beginning of gibbs for patient %s \n" % p)

            data_gibbs, affine_gibbs = load_nifti(anat_path)
            data_gibbs = gibbs_removal(data_gibbs,num_threads=core_count)
            corrected_gibbs_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_gibbscorrected.nii.gz'
            save_nifti(corrected_gibbs_path, data_gibbs.astype(np.float32), affine_gibbs)


            wm_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of gibbs for patient %s \n" % p)
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of gibbs for patient %s \n" % p)

        if corr_gibbs:
            input_bet_path = corrected_gibbs_path
        else:
            input_bet_path = anat_path

        # anat_path = folder_path + '/anat/' + patient_path + '_T1.nii.gz'
        bet_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
        bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; bet ' + input_bet_path + ' ' + bet_path + ' -B'
        if debug:
            bashCommand = bashCommand + " -d"
        bashcmd = bashCommand.split()

        wm_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Bet launched for patient %s \n" % p + " with bash command " + bashCommand)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Bet launched for patient %s \n" % p + " with bash command " + bashCommand)

        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=wm_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        wm_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of BET for patient %s \n" % p)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of BET for patient %s \n" % p)

        wm_log.close()

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

        # Save whitemask in T1 space
        out_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_whitemask.nii.gz'
        save_nifti(out_path, white_mask.astype(np.float32), moving_affine)

        # Save segmentation in T1 space
        out_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_segmentation.nii.gz'
        save_nifti(out_path, final_segmentation.astype(np.float32), moving_affine)

        white_mask[white_mask >= 0.01] = 1
        white_mask[white_mask < 0.01] = 0



        # transform the white matter mask ======================================
        white_mask = affine.transform(white_mask)
        white_mask[white_mask != 0] = 1
        anat_affine = static_grid2world
        segmentation = affine.transform(final_segmentation)

        # Save corrected projected T1
        out_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_corr_projected.nii.gz'
        save_nifti(out_path, affine.transform(moving_data).astype(np.float32), anat_affine)


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
        if core_count > 1:
            peaks = dp.peaks_from_model(model=qball_model, data=data, relative_peak_threshold=.5, min_separation_angle=25,
                                    sphere=sphere, mask=mask, parallel=True, nbr_processes=core_count)
        else:
            peaks = dp.peaks_from_model(model=qball_model, data=data, relative_peak_threshold=.5,
                                        min_separation_angle=25,
                                        sphere=sphere, mask=mask, parallel=False)

        ap = shm.anisotropic_power(peaks.shm_coeff)
        save_nifti(folder_path + '/' + patient_path + "/masks/" + patient_path + '_ap.nii.gz', ap.astype(np.float32), affine)
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
        "%d.%b %Y %H:%M:%S") + ": Starting quality control  %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()

    """Imports"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    from PyPDF2 import PdfFileMerger

    wm_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    seg_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_segmentation.nii.gz'
    T1_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
    T1gibbs_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_gibbscorrected.nii.gz'
    T1brain_path = folder_path + '/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
    ap_path = folder_path + '/' + patient_path + "/masks/" + patient_path + '_ap.nii.gz'
    preproc_path = folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz'
    qc_path = folder_path + '/' + patient_path + '/masks/' + 'quality_control'
    makedir(qc_path, folder_path + '/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - White matter mask - ' + patient_path, fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    # fig.suptitle('White matter segmentation', y=1.05,fontsize=16)
    seg_data, seg_affine = load_nifti(seg_path)
    sl = np.shape(seg_data)[2] // 2
    plot_seg = np.zeros((np.shape(seg_data)[0], np.shape(seg_data)[1] * 3))
    plot_seg[:, 0:np.shape(seg_data)[1]] = seg_data[..., sl - 10]
    plot_seg[:, np.shape(seg_data)[1]:(np.shape(seg_data)[1] * 2)] = seg_data[..., sl]
    plot_seg[:, (np.shape(seg_data)[1] * 2):(np.shape(seg_data)[1] * 3)] = seg_data[..., sl + 10]
    axs[0].imshow(plot_seg)
    axs[0].set_axis_off()
    seg_data, seg_affine = load_nifti(preproc_path)
    sl = np.shape(seg_data)[2] // 2
    plot_seg = np.zeros((np.shape(seg_data)[0], np.shape(seg_data)[1] * 3))
    plot_seg[:, 0:np.shape(seg_data)[1]] = seg_data[..., sl - 10, 0]
    plot_seg[:, np.shape(seg_data)[1]:(np.shape(seg_data)[1] * 2)] = seg_data[..., sl, 0]
    plot_seg[:, (np.shape(seg_data)[1] * 2):(np.shape(seg_data)[1] * 3)] = seg_data[..., sl + 10, 0]
    axs[1].imshow(plot_seg, cmap='gray')
    axs[1].set_axis_off()
    seg_data, seg_affine = load_nifti(wm_path)
    sl = np.shape(seg_data)[2] // 2
    plot_seg = np.zeros((np.shape(seg_data)[0], np.shape(seg_data)[1] * 3))
    plot_seg[:, 0:np.shape(seg_data)[1]] = seg_data[..., sl - 10]
    plot_seg[:, np.shape(seg_data)[1]:(np.shape(seg_data)[1] * 2)] = seg_data[..., sl]
    plot_seg[:, (np.shape(seg_data)[1] * 2):(np.shape(seg_data)[1] * 3)] = seg_data[..., sl + 10]
    test = np.ma.masked_where(plot_seg < 0.9, plot_seg)
    axs[1].imshow(test, cmap='hsv', interpolation='none')
    axs[1].set_axis_off()
    plt.tight_layout()
    plt.savefig(qc_path + "/segmentation.jpg", dpi=300, bbox_inches='tight')

    if os.path.isfile(T1_path):
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        anat_data, anat_affine = load_nifti(T1_path)
        sl = np.shape(anat_data)[2] // 2 + 15
        plot_anat = np.zeros((np.shape(anat_data)[0], np.shape(anat_data)[1] * 5))
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., sl - 10]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., sl - 5]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., sl + 5]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., sl + 10]
        axs[0].imshow(plot_anat, cmap='gray')
        axs[0].set_title('T1')
        axs[0].set_axis_off()
        anat_data, anat_affine = load_nifti(T1gibbs_path)
        sl = np.shape(anat_data)[2] // 2 + 15
        plot_anat = np.zeros((np.shape(anat_data)[0], np.shape(anat_data)[1] * 5))
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., sl - 10]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., sl - 5]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., sl + 5]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., sl + 10]
        axs[1].imshow(plot_anat, cmap='gray')
        axs[1].set_title('T1 gibbs ringing corrected')
        axs[1].set_axis_off()
        anat_data, anat_affine = load_nifti(T1brain_path)
        sl = np.shape(anat_data)[2] // 2 + 15
        plot_anat = np.zeros((np.shape(anat_data)[0], np.shape(anat_data)[1] * 5))
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., sl - 10]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., sl - 5]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., sl + 5]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., sl + 10]
        axs[2].imshow(plot_anat, cmap='gray')
        axs[2].set_title('T1 brain extracted')
        axs[2].set_axis_off()
        plt.tight_layout()
        plt.savefig(qc_path + "/origin.jpg", dpi=300, bbox_inches='tight')
    else:
        plt.figure(figsize=(10, 6))
        anat_data, anat_affine = load_nifti(ap_path)
        sl = np.shape(anat_data)[2] // 2
        plot_anat = np.zeros((np.shape(anat_data)[0], np.shape(anat_data)[1] * 5))
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., sl - 10]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., sl - 5]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., sl + 5]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., sl + 10]
        plt.imshow(plot_anat, cmap='gray')
        plt.title('Anisotropic power map')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(qc_path + "/origin.jpg", dpi=300, bbox_inches='tight')

    elem = [qc_path + "/title.jpg", qc_path + "/segmentation.jpg", qc_path + "/origin.jpg", ]

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.WIDTH = 210
            self.HEIGHT = 297

        def header(self):
            # self.image('assets/logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 11)
            self.cell(self.WIDTH - 80)
            self.cell(60, 1, 'Quality control report - White matter mask - ' + patient_path, 0, 0, 'R')
            self.ln(20)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

        def page_body(self, images):
            # Determine how many plots there are per page and set positions
            # and margins accordingly
            if len(images) == 3:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, 25 + 20, self.WIDTH - 30)
                self.image(images[2], 15, self.WIDTH / 2 + 75, self.WIDTH - 30)
            elif len(images) == 2:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 40, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)

        def print_page(self, images):
            # Generates the report
            self.add_page()
            self.page_body(images)

    pdf = PDF()
    pdf.print_page(elem)
    pdf.output(qc_path + '/qc_report.pdf', 'F');

    """Merge with QC of preproc""";

    pdfs = [folder_path + '/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(folder_path + '/' + patient_path + '/quality_control_wm.pdf')
    merger.close()

    os.remove(folder_path + '/' + patient_path + '/quality_control.pdf')
    os.rename(folder_path + '/' + patient_path + '/quality_control_wm.pdf',folder_path + '/' + patient_path + '/quality_control.pdf')


    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/masks/wm_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def noddi_solo(folder_path, p, force_brain_mask=False, lambda_iso_diff=3.e-9, lambda_par_diff=1.7e-9, use_amico=False,core_count=1):
    """ Perform noddi and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param force_brain_mask: Force the use of a brain mask even if a whitematter mask exist.
    :param lambda_iso_diff: Define the noddi lambda_iso_diff parameters.
    :param lambda_par_diff: Define the noddi lambda_par_diff parameters.
    :param use_amico: If true, use the amico optimizer.
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
        NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask,use_parallel_processing=True,number_of_processors=core_count)
        # NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask, solver='mix', maxiter=300)

    # exctract the metrics
    fitted_parameters = NODDI_fit.fitted_parameters
    mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
    odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
    f_iso = fitted_parameters["partial_volume_0"]
    f_bundle = fitted_parameters["partial_volume_1"]
    f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
    f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * (fitted_parameters['partial_volume_1']>0.05)
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
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metric1 = np.copy(odi)
    metric2 = np.copy(f_iso)
    qc_path = folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/quality_control"
    makedir(qc_path, folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", log_prefix)

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - NODDI', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sl = np.shape(mse)[2] // 2
    plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., sl - 10]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., sl - 5]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., sl + 5]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., sl + 10]
    im0 = axs[0].imshow(plot_mse, cmap='gray')
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., sl - 10]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., sl - 5]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., sl + 5]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., sl + 10]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sl = np.shape(metric1)[2] // 2
    plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5))
    plot_metric1[:, 0:np.shape(metric1)[1]] = metric1[..., sl - 10]
    plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2)] = metric1[..., sl - 5]
    plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3)] = metric1[..., sl]
    plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4)] = metric1[..., sl + 5]
    plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5)] = metric1[..., sl + 10]
    axs[0].imshow(plot_metric1, cmap='gray')
    axs[0].set_title('Orientation dispersion index')
    axs[0].set_axis_off()
    sl = np.shape(metric2)[2] // 2
    plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
    plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., sl - 10]
    plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., sl - 5]
    plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl]
    plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., sl + 5]
    plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., sl + 10]
    axs[1].imshow(plot_metric2, cmap='gray')
    axs[1].set_title('Fraction iso')
    axs[1].set_axis_off()
    plt.tight_layout();
    plt.savefig(qc_path + "/metrics.jpg", dpi=300, bbox_inches='tight');

    """Save as a pdf"""

    elem = [qc_path + "/title.jpg", qc_path + "/error.jpg", qc_path + "/metrics.jpg"]

    from fpdf import FPDF

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.WIDTH = 210
            self.HEIGHT = 297

        def header(self):
            # self.image('assets/logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 11)
            self.cell(self.WIDTH - 80)
            self.cell(60, 1, 'Quality control report - NODDI', 0, 0, 'R')
            self.ln(20)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

        def page_body(self, images):
            # Determine how many plots there are per page and set positions
            # and margins accordingly
            if len(images) == 3:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, 25 + 20, self.WIDTH - 30)
                self.image(images[2], 15, self.WIDTH / 2 + 75, self.WIDTH - 30)
            elif len(images) == 2:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 40, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)

        def print_page(self, images):
            # Generates the report
            self.add_page()
            self.page_body(images)

    pdf = PDF()
    pdf.print_page(elem)
    pdf.output(qc_path + '/qc_report.pdf', 'F');

    """Merge with QC of preproc""";
    from PyPDF2 import PdfFileMerger
    pdfs = [folder_path + '/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(folder_path + '/' + patient_path + '/quality_control_noddi.pdf')
    merger.close()
    os.remove(folder_path + '/' + patient_path + '/quality_control.pdf')
    os.rename(folder_path + '/' + patient_path + '/quality_control_noddi.pdf',folder_path + '/' + patient_path + '/quality_control.pdf')

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def noddi_amico_solo(folder_path, p):
    """ Perform noddi and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi_amico/.

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


def diamond_solo(folder_path, p, core_count=4, reportOnly=False):
    """Perform diamond and store the data in <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/diamond/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param core_count: Number of allocated cpu core.
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

    if not reportOnly:
        bashCommand = 'export OMP_NUM_THREADS=' +str(core_count)+ ' ; crlDCIEstimate --input "' + folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz' + '" --output "' + folder_path + '/' + patient_path + '/dMRI/microstructure/diamond/' + patient_path + '_diamond.nii.gz' + '" --mask "' + mask + '" --proc ' + str(core_count) + ' --ntensors 2 --reg 1.0 --estimb0 1 --automose aicu --mosemodels --fascicle diamondcyl --waterfraction 1 --waterDiff 0.003 --omtm 1 --residuals --fractions_sumto1 0 --verbose 1 --log'

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
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dipy.io.image import load_nifti

    mosemap, _ = load_nifti(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + "_diamond_mosemap.nii.gz")
    fractions, _ = load_nifti(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + "_diamond_fractions.nii.gz")
    residual, _ = load_nifti(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + "_diamond_residuals.nii.gz")
    data, _ = load_nifti(folder_path + '/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz')
    residual = np.squeeze(residual)
    reconstructed = data - residual

    metric1 = np.copy(mosemap)
    metric2 = np.copy(fractions[...,0])
    qc_path = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/quality_control"
    makedir(qc_path, folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", log_prefix)

    mse = np.mean(residual ** 2, axis=3)
    R2 = np.zeros_like(mse)
    (itot, jtot, ktot) = np.shape(R2)
    for i in range(itot):
        for j in range(jtot):
            for k in range(ktot):
                R2[i, j, k] = np.corrcoef(data[i, j, k, :], reconstructed[i, j, k, :])[0, 1] ** 2

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - DIAMOND', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sl = np.shape(mse)[2] // 2
    plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., sl - 10]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., sl - 5]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., sl + 5]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., sl + 10]
    im0 = axs[0].imshow(plot_mse, cmap='gray')#, vmin=0, vmax=np.min([np.max(residual),np.max(mse)]))
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., sl - 10]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., sl - 5]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., sl + 5]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., sl + 10]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sl = np.shape(metric1)[2] // 2
    plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5))
    plot_metric1[:, 0:np.shape(metric1)[1]] = metric1[..., sl - 10]
    plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2)] = metric1[..., sl - 5]
    plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3)] = metric1[..., sl]
    plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4)] = metric1[..., sl + 5]
    plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5)] = metric1[..., sl + 10]
    axs[0].imshow(plot_metric1, cmap='gray')
    axs[0].set_title('Mosemap')
    axs[0].set_axis_off()
    sl = np.shape(metric2)[2] // 2
    plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
    plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., sl - 10, 0]
    plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., sl - 5, 0]
    plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl, 0]
    plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., sl + 5, 0]
    plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., sl + 10, 0]
    axs[1].imshow(plot_metric2, cmap='gray')
    axs[1].set_title('Fraction of the first compartment')
    axs[1].set_axis_off()
    plt.tight_layout();
    plt.savefig(qc_path + "/metrics.jpg", dpi=300, bbox_inches='tight');

    """Save as a pdf"""

    elem = [qc_path + "/title.jpg", qc_path + "/error.jpg", qc_path + "/metrics.jpg"]

    from fpdf import FPDF

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.WIDTH = 210
            self.HEIGHT = 297

        def header(self):
            # self.image('assets/logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 11)
            self.cell(self.WIDTH - 80)
            self.cell(60, 1, 'Quality control report - DIAMOND', 0, 0, 'R')
            self.ln(20)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

        def page_body(self, images):
            # Determine how many plots there are per page and set positions
            # and margins accordingly
            if len(images) == 3:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, 25 + 20, self.WIDTH - 30)
                self.image(images[2], 15, self.WIDTH / 2 + 75, self.WIDTH - 30)
            elif len(images) == 2:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 40, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)

        def print_page(self, images):
            # Generates the report
            self.add_page()
            self.page_body(images)

    pdf = PDF()
    pdf.print_page(elem)
    pdf.output(qc_path + '/qc_report.pdf', 'F');

    """Merge with QC of preproc""";
    from PyPDF2 import PdfFileMerger
    pdfs = [folder_path + '/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(folder_path + '/' + patient_path + '/quality_control_diamond.pdf')
    merger.close()
    os.remove(folder_path + '/' + patient_path + '/quality_control.pdf')
    os.rename(folder_path + '/' + patient_path + '/quality_control_diamond.pdf', folder_path + '/' + patient_path + '/quality_control.pdf')

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def mf_solo(folder_path, p, dictionary_path, CSD_bvalue=None,core_count=1):
    """Perform microstructure fingerprinting and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/mf/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param dictionary_path: Path to the dictionary to use
    :param CSD_bvalue: Define a csd value.
    :param core_count: Define the number of available core.
    """
    log_prefix = "MF SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual microstructure fingerprinting processing for patient %s \n" % p)
    patient_path = os.path.splitext(p)[0]

    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual microstructure fingerprinting processing for patient %s \n" % p)

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
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Diamond Path found! MF will be based on diamond \n")
        tensor_files0 = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t0.nii.gz'
        tensor_files1 = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t1.nii.gz'
        fracs_file = folder_path + '/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_fractions.nii.gz'
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=None, frac2=None, mu1=tensor_files0, mu2=tensor_files1,
                                                 peakmode='tensor', mask=mask, frac12=fracs_file)
    else:
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Diamond Path not found! MF will be based on CSD \n")
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

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Loading of MF dic\n")
    # get the dictionary
    mf_model = mf.MFModel(dictionary_path)

    # compute csf_mask and ear_mask
    csf_mask = True
    ear_mask = False  # (numfasc == 1)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of fitting\n")

    # Fit to data:
    MF_fit = mf_model.fit(data, mask, numfasc, peaks=peaks, bvals=bvals, bvecs=bvecs, csf_mask=csf_mask,
                          ear_mask=ear_mask, verbose=3, parallel=True)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": End of fitting\n")

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

    # # Save nifti
    # save_nifti(mf_path + '/' + patient_path + '_mf_peaks.nii.gz', peaks.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_M0.nii.gz', M0.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_frac_f0.nii.gz', frac_f0.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_DIFF_ex_f0.nii.gz', DIFF_ex_f0.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_fvf_f0.nii.gz', fvf_f0.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_frac_f1.nii.gz', frac_f1.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_DIFF_ex_f1.nii.gz', DIFF_ex_f1.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_fvf_f1.nii.gz', fvf_f1.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_fvf_tot.nii.gz', fvf_tot.astype(np.float32), affine)
    # # save_nifti(mf_path + '/' + patient_path + '_mf_frac_ear.nii.gz', frac_ear.astype(np.float32), affine)
    # # save_nifti(mf_path + '/' + patient_path + '_mf_D_ear.nii.gz', D_ear.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_frac_csf.nii.gz', frac_csf.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_MSE.nii.gz', MSE.astype(np.float32), affine)
    # save_nifti(mf_path + '/' + patient_path + '_mf_R2.nii.gz', R2.astype(np.float32), affine)

    MF_fit.write_nifti(mf_path + '/' + patient_path + '_mf.nii.gz', affine=affine)

    #All the outputed metrics can be obtrained using MF_fit.param_names
    # Code used in MF:
    #for p in self.param_names:
    #    nii = nib.Nifti1Image(getattr(self, p), affine)
    #    nii_fname = '%s_%s%s' % (basename, p, ext)
    #    nib.save(nii, nii_fname)
    #    fnames.append(nii_fname)



    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mse = np.copy(MSE)
    metric1 = np.copy(fvf_tot)
    metric2 = np.copy(frac_f0)
    qc_path = folder_path + '/' + patient_path + "/dMRI/microstructure/mf/quality_control"
    makedir(qc_path, folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", log_prefix)

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - Microstructure fingerprinting', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sl = np.shape(mse)[2] // 2
    plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., sl - 10]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., sl - 5]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., sl + 5]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., sl + 10]
    im0 = axs[0].imshow(plot_mse, cmap='gray')
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., sl - 10]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., sl - 5]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., sl + 5]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., sl + 10]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sl = np.shape(metric1)[2] // 2
    plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5))
    plot_metric1[:, 0:np.shape(metric1)[1]] = metric1[..., sl - 10]
    plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2)] = metric1[..., sl - 5]
    plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3)] = metric1[..., sl]
    plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4)] = metric1[..., sl + 5]
    plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5)] = metric1[..., sl + 10]
    axs[0].imshow(plot_metric1, cmap='gray')
    axs[0].set_title('fvf_tot')
    axs[0].set_axis_off()
    sl = np.shape(metric2)[2] // 2
    plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
    plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., sl - 10]
    plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., sl - 5]
    plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl]
    plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., sl + 5]
    plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., sl + 10]
    axs[1].imshow(plot_metric2, cmap='gray')
    axs[1].set_title('frac_f0')
    axs[1].set_axis_off()
    plt.tight_layout();
    plt.savefig(qc_path + "/metrics.jpg", dpi=300, bbox_inches='tight');

    """Save as a pdf"""

    elem = [qc_path + "/title.jpg", qc_path + "/error.jpg", qc_path + "/metrics.jpg"]

    from fpdf import FPDF

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.WIDTH = 210
            self.HEIGHT = 297

        def header(self):
            # self.image('assets/logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 11)
            self.cell(self.WIDTH - 80)
            self.cell(60, 1, 'Quality control report - MF', 0, 0, 'R')
            self.ln(20)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

        def page_body(self, images):
            # Determine how many plots there are per page and set positions
            # and margins accordingly
            if len(images) == 3:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, 25 + 20, self.WIDTH - 30)
                self.image(images[2], 15, self.WIDTH / 2 + 75, self.WIDTH - 30)
            elif len(images) == 2:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 40, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)

        def print_page(self, images):
            # Generates the report
            self.add_page()
            self.page_body(images)

    pdf = PDF()
    pdf.print_page(elem)
    pdf.output(qc_path + '/qc_report.pdf', 'F');

    """Merge with QC of preproc""";
    from PyPDF2 import PdfFileMerger
    pdfs = [folder_path + '/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(folder_path + '/' + patient_path + '/quality_control_mf.pdf')
    merger.close()
    os.remove(folder_path + '/' + patient_path + '/quality_control.pdf')
    os.rename(folder_path + '/' + patient_path + '/quality_control_mf.pdf', folder_path + '/' + patient_path + '/quality_control.pdf')

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()


def report_solo(folder_path,patient_path, slices=None, short=False):
    """ Legacy report function.

    :param folder_path: path to the root directory.
    :param patient_path: Name of the subject.
    :param slices: Add additional slices cut to specific volumes
    :param short: Only output raw data, preprocessed data and FA data.
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
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/raw/"+patient_path+"_raw_dmri" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/raw/"+patient_path+"_raw_dmri","raw_drmi","Raw dMRI ("+patient_path+"_raw_drmi.nii.gz)"))
    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc","drmi_preproc","dMRI preprocessed ("+patient_path+"_drmi_preproc.nii.gz)"))

        if slices:
            i = slices
            fslroi = "fslroi " + folder_path + '/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc" + ".nii.gz" + " " + report_path + "/preproc_" + str(i) + ".nii.gz " + str(i - 1) + " 1"
            process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=report_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()
            image.append((report_path + "/preproc_" + str(i),
                          "drmi_preproc_" + str(i), "dMRI preprocessed slice "+ str(i) + " (" + patient_path + "_drmi_preproc.nii.gz)"))

    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/bet/"+patient_path+"_mask" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/bet/"+patient_path+"_mask","drmi_preproc_bet","dMRI BET preprocessing ("+patient_path+"_mask.nii.gz)"))
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/mppca/"+patient_path+"_mppca" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/mppca/"+patient_path+"_mppca","drmi_preproc_mppca","dMRI Denoised preprocessing ("+patient_path+"_mppca.nii.gz)"))
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected","drmi_preproc_gibbs","dMRI Gibbs preprocessing ("+patient_path+"_gibbscorrected.nii.gz)"))

        if slices:
            i = slices
            fslroi = "fslroi " + folder_path + '/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected" + ".nii.gz" + " " + report_path + "/preproc_gibbs_" + str(
                i) + ".nii.gz " + str(i - 1) + " 1"
            process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=report_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()
            image.append((report_path + "/preproc_gibbs_" + str(i),
                          "drmi_preproc_gibbs_" + str(i),
                          "dMRI Gibbs preprocessing " + str(i) + " ("+patient_path+"_gibbscorrected.nii.gz)"))

    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/topup/"+patient_path+"_topup" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/topup/"+patient_path+"_topup","drmi_preproc_topup","dMRI Topup preprocessing ("+patient_path+"_topup.nii.gz)"))
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/preproc/eddy/" + patient_path + "_eddy_corr" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/preproc/eddy/" + patient_path + "_eddy_corr","drmi_preproc_eddy_corr", "dMRI Eddy preprocessing (" + patient_path + "_eddy_corr.nii.gz)"))

    if os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_FA" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_FA","dti_FA", "Microstructure: FA of dti (" + patient_path + "_FA.nii.gz)"))
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_AD" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_AD","dti_AD", "Microstructure: AD of dti (" + patient_path + "_AD.nii.gz)"))
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_MD" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_MD","dti_MD", "Microstructure: MD of dti (" + patient_path + "_MD.nii.gz)"))
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_RD" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_RD","dti_RD", "Microstructure: RD of dti (" + patient_path + "_RD.nii.gz)"))
    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_dtensor" + ".nii.gz"):
        image.append((folder_path + '/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_dtensor","dti_dtensor", "Microstructure: Dtensor of dti (" + patient_path + "_dtensor.nii.gz)"))

    if not short and os.path.exists(folder_path + '/' + patient_path + "/dMRI/microstructure/noddi/" + patient_path + "_dtensor" + ".nii.gz"):
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

