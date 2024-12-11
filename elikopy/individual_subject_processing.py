import datetime
import os
import shutil
import json
import sys
import numpy as np
import math
from scipy.ndimage.morphology import binary_dilation

import subprocess
from elikopy.utils import makedir, update_status

import functools
print = functools.partial(print, flush=True)


def preproc_solo(folder_path, p, reslice=False, reslice_addSlice=False, denoising=False, gibbs=False, topup=False, topupConfig=None, forceSynb0DisCo=False, useGPUsynb0DisCo=False, eddy=False, biasfield=False, biasfield_bsplineFitting=[100,3], biasfield_convergence=[1000,0.001], static_files_path=None, starting_state=None, bet_median_radius=2, bet_numpass=1, bet_dilate=2, cuda=False, cuda_name="eddy_cuda10.1", s2v=[0,5,1,'trilinear'], olrep=[False, 4, 250, 'sw'], eddy_additional_arg="", qc_reg=True, core_count=1, niter=5, report=True, slspec_gc_path=None):
    """ Performs data preprocessing on a single subject. By default only the brain extraction is enabled. Optional preprocessing steps include : reslicing,
    denoising, gibbs ringing correction, susceptibility field estimation, EC-induced distortions and motion correction, bias field correction.
    The results are stored in the preprocessing subfolder of the study subject <folder_path>/subjects/<subjects_ID>/dMRI/preproc.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param reslice: If true, data will be resliced with a new voxel resolution of 2*2*2. default=False
    :param reslice_addSlice: If true, an additional empty slice will be added to each volume (might be useful for motion correction if one slice is dropped during the acquisition and the user still wants to perform easily the slice-to-volume motion correction). default=False
    :param denoising: If true, Patch-to-self/MPPCA-denoising is performed on the data. default=False
    :param mppca_legacy_denoising: If true, MPPCA-denoising is performed instead of Patch-to-self denoising. default=False
    :param gibbs: If true, Gibbs ringing correction is performed. We do not advise to use this correction unless the data suffers from a lot of Gibbs ringing artifacts. default=False
    :param topup: If true, Topup will estimate the susceptibility induced distortions. These distortions are corrected at the same time as EC-induced distortions if eddy=True. In the absence of images acquired with a reverse phase encoding direction, a T1 structural image is required. default=False
    :param topupConfig: If not None, Topup will use additionnal parameters based on the supplied config file located at <topupConfig>. default=None
    :param forceSynb0DisCo: If true, Topup will always estimate the susceptibility field using the T1 structural image. default=False
    :param useGPUsynb0DisCo: If true, Topup will estimate the susceptibility field with the T1 structural image using cuda. default=FALSE
    :param eddy: If true, Eddy corrects the EC-induced (+ susceptibility, if estimated) distortions and the motion. If these corrections are performed the acquparam and index files are required (see documentation). To perform the slice-to-volume motion correction the slspec file is also needed. default=False
    :param biasfield: If true, low frequency intensity non-uniformity present in MRI image data known as a bias or gain field will be corrected. default=False
    :param biasfield_bsplineFitting: Define the initial mesh resolution in mm and the bspline order of the biasfield correction tool. default=[100,3]
    :param biasfield_convergence: Define the maximum number of iteration and the convergences threshold of the biasfield correction tool. default=[1000,0.001]
    :param starting_state: Manually set which step of the preprocessing to execute first. Could either be None, denoising, gibbs, topup, eddy, biasfield, report or post_report. default=None
    :param bet_median_radius: Radius (in voxels) of the applied median filter during brain extraction. default=2
    :param bet_numpass: Number of pass of the median filter during brain extraction. default=1
    :param bet_dilate: Number of iterations for binary dilation during brain extraction. default=2
    :param cuda: If true, eddy will run on cuda with the command name specified in cuda_name. default=False
    :param cuda_name: name of the eddy command to run when cuda==True. default="eddy_cuda10.1"
    :param s2v: list of parameters of Eddy for slice-to-volume motion correction (see Eddy FSL documentation): [mporder,s2v_niter,s2v_lambda,s2v_interp]. The slice-to-volume motion correction is performed if mporder>0, cuda is used and a slspec file is provided during the patient_list command. default=[0,5,1,'trilinear']
    :param olrep: list of parameters of Eddy for outlier replacement (see Eddy FSL documentation): [repol,ol_nstd,ol_nvox,ol_type]. The outlier replacement is performed if repol==True. default=[False, 4, 250, 'sw']
    :param qc_reg: If true, the motion registration step of the quality control will be performed. We do not advise to use this argument as it increases the computation time. default=False
    :param niter: Define the number of iterations for eddy volume-to-volume. default=5
    :param slspec_gc_path: Path to the folder containing volume specific slice-specification for eddy. If not None, eddy motion correction with gradient cycling will be performed.
    :param report: If False, no quality report will be generated. default=True
    :param core_count: Number of allocated cpu cores. default=1
    """

    in_reslice = reslice
    assert starting_state in (None,"None", "denoising", "gibbs", "topup", "eddy", "biasfield", "report", "topup_synb0DisCo_Registration", "topup_synb0DisCo_Inference", "topup_synb0DisCo_Apply", "topup_synb0DisCo_topup"), 'invalid starting state!'
    if starting_state == "denoising":
        assert denoising == True, 'if starting_state is denoising, denoising must be True!'
    if starting_state == "gibbs":
        assert gibbs == True, 'if starting_state is gibbs, gibbs must be True!'
    if starting_state in ("topup", "topup_synb0DisCo_Registration", "topup_synb0DisCo_Inference", "topup_synb0DisCo_Apply", "topup_synb0DisCo_topup"):
        assert topup == True, 'if starting_state is topup, topup must be True!'
    if starting_state == "eddy":
        assert eddy == True, 'if starting_state is eddy, eddy must be True!'
    if starting_state == "biasfield":
        assert biasfield == True, 'if starting_state is biasfield, biasfield must be True!'
    if starting_state == "None":
        starting_state = None

    if topupConfig == "None":
        topupConfig = None

    if static_files_path == "None":
        static_files_path = None

    log_prefix = "PREPROC SOLO"
    patient_path = p
    preproc_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/bet"
    makedir(preproc_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

    mask_path = folder_path + '/subjects/' + patient_path + "/masks"
    makedir(mask_path, folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", log_prefix)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f.close()

    anat_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
    if os.path.exists(anat_path):
        brain_extracted_T1_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
        brain_extracted_mask_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_brain_mask.nii.gz'
        if not os.path.exists(brain_extracted_T1_path):
            cmd = f"mri_synth_strip -i {anat_path} -o {brain_extracted_T1_path} -m {brain_extracted_mask_path} "
            import subprocess
            process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)

            if not os.path.exists(brain_extracted_T1_path):
                print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                    "%d.%b %Y %H:%M:%S") + ": T1 Brain extraction failed for patient %s \n" % p)
                # print to std err
                print("T1 Brain extraction failed for patient %s" % p, file=sys.stderr)

    from dipy.io.image import load_nifti, save_nifti
    from dipy.segment.mask import median_otsu

    nifti_path = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.nii.gz'
    if (starting_state == None):
        data, affine, voxel_size = load_nifti(nifti_path, return_voxsize=True)
        curr_dmri = data
    reslice_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/reslice"
    if reslice and starting_state == None:
        makedir(reslice_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        from dipy.align.reslice import reslice
        new_voxel_size = (2., 2., 2.)
        data, affine = reslice(data, affine, voxel_size, new_voxel_size, num_processes=core_count)

        if reslice_addSlice:
            data = np.insert(data, np.size(data,2), 0, axis=2)

        curr_dmri = data
        nifti_path = reslice_path + '/' + patient_path + '_reslice.nii.gz'
        save_nifti(reslice_path + '/' + patient_path + '_reslice.nii.gz', data, affine)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f.close()
    if reslice:
        nifti_path = reslice_path + '/' + patient_path + '_reslice.nii.gz'

    if starting_state == None:
        from elikopy.utils import clean_mask

        _, mask = median_otsu(curr_dmri, median_radius=bet_median_radius, numpass=bet_numpass, vol_idx=range(0, np.shape(curr_dmri)[3]), dilate=bet_dilate)
        mask = clean_mask(mask)

        save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz',mask.astype(np.float32), affine)

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
        f.close()

    if not denoising and not eddy and not gibbs and not topup and not biasfield:
        save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz', curr_dmri.astype(np.float32), affine)
        shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    denoising_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/mppca'
    denoising_ext = '_mppca.nii.gz'


    if denoising and starting_state!="gibbs" and starting_state!="eddy" and (starting_state not in ("topup", "topup_synb0DisCo_Registration", "topup_synb0DisCo_Inference", "topup_synb0DisCo_Apply", "topup_synb0DisCo_topup")) and starting_state!="biasfield" and starting_state!="report":
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of denoising for patient %s \n" % p)

        makedir(denoising_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt",
                log_prefix)

        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Denoising launched for patient %s \n" % p)

        import subprocess
        bashCommand = "dwidenoise -nthreads " + str(core_count) + " " + nifti_path + \
              " " + denoising_path + '/' + patient_path + '_mppca.nii.gz' +\
              " -noise " + denoising_path + '/' + patient_path + '_sigmaNoise.nii.gz -force'

        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=f,
                                   stderr=subprocess.STDOUT)

        output, error = process.communicate()

        b0_reverse_raw_path= os.path.join(folder_path, 'subjects', patient_path, 'dMRI', 'raw', patient_path + '_b0_reverse.nii.gz')
        if os.path.exists(b0_reverse_raw_path):
            bashCommand = "dwidenoise -nthreads " + str(core_count) + " " + b0_reverse_raw_path + \
                          " " + denoising_path + '/' + patient_path + '_b0_reverse_mppca.nii.gz' + \
                          " -noise " + denoising_path + '/' + patient_path + '_b0_reverse_sigmaNoise.nii.gz -force'

            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=f,
                                       stderr=subprocess.STDOUT)

            output, error = process.communicate()
        denoised, _ = load_nifti(denoising_path + '/' + patient_path + '_mppca.nii.gz')


        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Denoising finished for patient %s \n" % p)
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Denoising finished for patient %s \n" % p)
        f.close()


        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of denoising for patient %s \n" % p)

        curr_dmri = denoised

        if not eddy and not gibbs and not topup and not biasfield:
            save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz',
                       curr_dmri.astype(np.float32), affine)
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    if gibbs and starting_state!="eddy" and (starting_state not in ("topup", "topup_synb0DisCo_Registration", "topup_synb0DisCo_Inference", "topup_synb0DisCo_Apply", "topup_synb0DisCo_topup"))  and starting_state!="biasfield" and starting_state!="report":
        from dipy.denoise.gibbs import gibbs_removal
        gibbs_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/gibbs'
        makedir(gibbs_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if (starting_state == "gibbs"):
            if not denoising:
                curr_dmri_path = nifti_path
            else:
                curr_dmri_path = denoising_path + '/' + patient_path + denoising_ext
            curr_dmri, affine, voxel_size = load_nifti(curr_dmri_path, return_voxsize=True)

        data = gibbs_removal(curr_dmri, num_processes=core_count)
        corrected_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/gibbs/" + patient_path + '_gibbscorrected.nii.gz'
        save_nifti(corrected_path, data.astype(np.float32), affine)

        curr_dmri = data
        if not eddy and not topup and not biasfield:
            save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz',
                       curr_dmri.astype(np.float32), affine)
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    # Explicitly freeing memory
    import gc
    denoised = None
    mask = None
    data = None
    affine = None
    gc.collect()

    topup_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/topup"
    if topup and starting_state!="eddy" and starting_state!="biasfield" and starting_state!="report":

        import subprocess
        #cmd = 'topup --imain=all_my_b0_images.nii --datain=acquisition_parameters.txt --config=b02b0.cnf --out=my_output"'
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of topup for patient %s \n" % p)

        topup_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/topup"
        makedir(topup_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
        makedir(topup_path+"/synb0-DisCo", folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if gibbs:
            imain_tot = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz'
        elif denoising:
            imain_tot = denoising_path + '/' + patient_path + denoising_ext
        else:
            imain_tot = nifti_path

        if denoising:
            b0_reverse_path = denoising_path + '/' + patient_path + '_b0_reverse_mppca.nii.gz'
        else:
            b0_reverse_path = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_b0_reverse.nii.gz'

        multiple_encoding=False
        topup_log = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/topup/topup_logs.txt", "a+")

        with open(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt') as f:
            line = f.read()
            line = " ".join(line.split())
            topup_index = [int(s) for s in line.split(' ')]

        with open(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt') as f:
            topup_acq = [[float(x) for x in line2.split()] for line2 in f]

        #Find all the b0 to extract.
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
            makedir(topup_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Patient %s \n" % p + " has multiple direction of gradient encoding, launching topup directly ")
            topupConfig = 'b02b0.cnf' if topupConfig is None else topupConfig


            # Extract b0s from main file:

            # Read bval file directly
            path_bval = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval"
            with open(path_bval, "r") as bval_file:
                bvals = list(map(float, bval_file.read().strip().split()))

            # Identify indices of b0 volumes
            b0_indices = [idx for idx, bval in enumerate(bvals) if bval == 0]

            if not b0_indices:
                print("No b0 volumes found in dMRI.")
                b0_indices = [0]

            # Extract and concatenate non-sequential b0 volumes
            temp_b0_files = []
            for i, b0_idx in enumerate(b0_indices):
                temp_b0_file = f"{folder_path}/subjects/{patient_path}/dMRI/preproc/topup/temp_b0_{i}.nii.gz"
                temp_b0_files.append(temp_b0_file)
                fslroi_cmd = f"fslroi {imain_tot} {temp_b0_file} {b0_idx} 1"
                try:
                    output = subprocess.check_output(fslroi_cmd, universal_newlines=True,
                                                     shell=True, stderr=subprocess.STDOUT)
                    print(output)
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting b0 volume at index {b0_idx}")
                    exit()

            # Merge all temporary b0 files into a single file
            b0_part1_path = f"{topup_path}/b0_part1.nii.gz"
            b0_part2_path = f"{topup_path}/b0_part2.nii.gz"
            shutil.copyfile(b0_reverse_path, b0_part2_path)
            merge_b0_cmd = f"fslmerge -t {b0_part1_path} " + " ".join(temp_b0_files)
            try:
                output = subprocess.check_output(merge_b0_cmd, universal_newlines=True, shell=True,
                                                 stderr=subprocess.STDOUT)
                print(output)
            except subprocess.CalledProcessError as e:
                print("Error when merging b0 volumes")
                exit()

            # Cleanup temporary files
            for temp_file in temp_b0_files:
                os.remove(temp_file)

            #  Merge extracted b0 volumes with the original DW-MRI file
            if os.path.exists(f"{topup_path}/b0.nii.gz"):
                os.remove(f"{topup_path}/b0.nii.gz")
            b0_path = f"{topup_path}/b0.nii.gz"
            merge_b0_cmd = f"fslmerge -t {b0_path} {b0_part1_path} {b0_part2_path} "
            try:
                print(merge_b0_cmd)
                output = subprocess.check_output(merge_b0_cmd, universal_newlines=True, shell=True,
                                                 stderr=subprocess.STDOUT)
                print(output)
            except subprocess.CalledProcessError as e:
                print("Error when merging b0 volumes with the original DW-MRI")
                exit()

            # Generate acqparams_alL.txt file
            with open(folder_path + "/subjects/" + patient_path + '/dMRI/raw/' + 'acqparams_original.txt') as f1:
                original_acq = [[float(x) for x in line2.split()] for line2 in f1]

            with open(folder_path + "/subjects/" + patient_path + '/dMRI/raw/' + 'acqparams_reverse.txt') as f2:
                reverse_acq = [[float(x) for x in line2.split()] for line2 in f2]

            # Get number of b0 in b0_part2 using index_reverse.txt
            with open(folder_path + "/subjects/" + patient_path + '/dMRI/raw/' + 'index_reverse.txt') as f3:
                b0_num_reverse = len([int(x) for x in f3.read().strip().split()])

            for i in range(len(b0_indices)-1):
                original_acq.append([original_acq[0][0], original_acq[0][1], original_acq[0][2], original_acq[0][3]])
            for i in range(b0_num_reverse):
                original_acq.append([reverse_acq[0][0], reverse_acq[0][1], reverse_acq[0][2], reverse_acq[0][3]])

            with open(f"{topup_path}/acqparams_all.txt", "w") as f4:
                f4.writelines(' '.join(str(j) for j in i) + '\n' for i in original_acq)

            bashCommand = ('export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+
                           ' ; topup --imain="' + topup_path + '/b0.nii.gz" --config="' + topupConfig +
                           '" --datain="' + topup_path + '/acqparams_all.txt" '+
                           '--out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" '+
                           '--fout="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_fout_estimate" '+
                           '--iout="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_iout_estimate" '+
                           '--verbose')
            bashcmd = bashCommand.split()
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Topup launched for patient %s \n" % p + " with bash command " + bashCommand)

            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Topup launched for patient %s \n" % p + " with bash command " + bashCommand)


            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=topup_log,
                                       stderr=subprocess.STDOUT)
            # wait until topup finish
            output, error = process.communicate()

            # Merge the two parts if necessary (optional)
            imain_tot_merged = f"{imain_tot},{b0_part2_path}"

            bashCommand2 = 'applytopup --imain="' + imain_tot_merged + '" --inindex=1,2 --datain="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --topup="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr"'

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": applytopup launched for patient %s \n" % p + " with bash command " + bashCommand2)

            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": applytopup launched for patient %s \n" % p + " with bash command " + bashCommand2)
            f.close()

            process2 = subprocess.Popen(bashCommand2, universal_newlines=True, shell=True, stdout=topup_log,
                                        stderr=subprocess.STDOUT)
            # wait until apply topup finish
            output, error = process2.communicate()

        else:
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Patient %s \n" % p + " has a single direction of gradient encoding, launching synb0DisCo ")
            f.close()
            from elikopy.utils import synb0DisCo
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt',
                            topup_path + '/synb0-DisCo/' + 'acqparams_topup.txt')

            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/T1/' + patient_path + '_T1.nii.gz',
                            topup_path + '/synb0-DisCo/' + 'T1.nii.gz')

            shutil.copyfile(topup_path + "/b0.nii.gz",topup_path + "/synb0-DisCo/b0.nii.gz")

            process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=topup_log,stderr=subprocess.STDOUT)
            output, error = process.communicate()
            synb0DisCo_starting_step = None
            if starting_state=="topup_synb0DisCo_Registration":
                synb0DisCo_starting_step = "Registration"
            elif starting_state=="topup_synb0DisCo_Inference":
                synb0DisCo_starting_step = "Inference"
            elif starting_state=="topup_synb0DisCo_Apply":
                synb0DisCo_starting_step = "Apply"
            elif starting_state =="topup_synb0DisCo_topup":
                synb0DisCo_starting_step = "topup"
            synb0DisCo(folder_path,topup_path,patient_path,starting_step=synb0DisCo_starting_step,topup=True,gpu=useGPUsynb0DisCo, static_files_path=static_files_path)

            bashCommand2 = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; applytopup --imain="' + imain_tot + '" --inindex=1 --datain="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --topup="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --method=jac --interp=spline --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr"'

            process2 = subprocess.Popen(bashCommand2, universal_newlines=True, shell=True, stdout=topup_log,
                                        stderr=subprocess.STDOUT)
            # wait until apply topup finish
            output, error = process2.communicate()


        topup_log.close()

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of topup for patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of topup for patient %s \n" % p)
        f.close()


        ## Compute pre eddy/biasfield mask
        gc.collect()
        from elikopy.utils import clean_mask

        topup_corr_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr.nii.gz'

        topup_corr, affine = load_nifti(topup_corr_path)
        topup_corr_b0_ref = topup_corr[..., 0]
        dwiref_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr_dwiref.nii.gz'
        save_nifti(dwiref_path, topup_corr_b0_ref.astype(np.float32), affine)
        topup_corr_b0_ref = None
        topup_corr = None
        gc.collect()


        # Step 1 : dwi2mask
        bvec_path = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec'
        bval_path = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval'
        dwi2mask_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_space-dwi_type-dwi2mask_brainmask.nii.gz'
        cmd = f"dwi2mask -fslgrad {bvec_path} {bval_path} {topup_corr_path} {dwi2mask_path} -force "
        import subprocess
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": dwi2mask launched for patient %s \n" % p + " with bash command " + cmd)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": dwi2mask launched for patient %s \n" % p + " with bash command " + cmd)
        f.flush()
        process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=f, stderr=subprocess.STDOUT)
        # wait until dwi2mask finish
        output, error = process.communicate()
        f.flush()
        f.close()

        gc.collect()
        # Step 2 : mri_synth_strip on dwiref
        mrisynthstrip_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_space-dwi_type-mrisynthstrip_brainmask.nii.gz'
        cmd = f"mri_synth_strip -i {dwiref_path} -m {mrisynthstrip_path}"
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": mri_synth_strip launched for patient %s \n" % p + " with bash command " + cmd)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": mri_synth_strip launched for patient %s \n" % p + " with bash command " + cmd)
        f.flush()
        process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=f, stderr=subprocess.STDOUT)
        output, error = process.communicate()
        f.flush()
        f.close()

        # Step 3 : median otsu on preprocess data
        topup_corr, affine = load_nifti(topup_corr_path)
        _, mask = median_otsu(topup_corr, median_radius=2, numpass=1, vol_idx=range(0, np.shape(topup_corr)[3]),
                                           dilate=2)
        mask = clean_mask(mask)
        save_nifti(
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_type-otsu_dilate-2_brainmask.nii.gz',
            mask.astype(np.float32), affine)

        # Step 4: Apply all masks to preprocess data
        dwi2mask_mask, _ = load_nifti(dwi2mask_path)
        mrisynthstrip_mask, _ = load_nifti(mrisynthstrip_path)

        full_mask = np.logical_or(mask, np.logical_or(dwi2mask_mask, mrisynthstrip_mask))
        full_mask = binary_dilation(full_mask, iterations=1)

        topup_corr_full_mask_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_brain_mask.nii.gz'
        save_nifti(topup_corr_full_mask_path,
                   full_mask.astype(np.float32), affine)

        if not eddy and not biasfield:
            data, affine = load_nifti(
                folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + "_unwarped.nii.gz")
            save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz', data.astype(np.float32), affine)
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
            save_nifti(folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz', full_mask.astype(np.float32), affine)

    if topup:
        topup_corr_full_mask_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_brain_mask.nii.gz'
        processing_mask = topup_corr_full_mask_path
    else:
        processing_mask = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/bet/' + patient_path + '_binary_mask.nii.gz'

    if eddy and starting_state!="biasfield" and starting_state!="report":
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of eddy for patient %s \n" % p)

        eddy_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/eddy"
        makedir(eddy_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if cuda:
            eddycmd = "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; " + cuda_name
        else:
            eddycmd = "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; eddy"

        fwhm = '10'
        for _ in range(niter-1):
            fwhm = fwhm + ',0'

        if s2v[0] != 0:
            slspec_path = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'slspec.txt'
            if slspec_gc_path is not None and os.path.isdir(slspec_gc_path):
                if gibbs:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --ge_slspecs="' + slspec_gc_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                elif denoising:
                    bashCommand = eddycmd + ' --imain="' + denoising_path + '/' + patient_path + denoising_ext + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --ge_slspecs="' + slspec_gc_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                else:
                    bashCommand = eddycmd + ' --imain="' + nifti_path + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --ge_slspecs="' + slspec_gc_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            elif os.path.isfile(slspec_path):
                if gibbs:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                elif denoising:
                    bashCommand = eddycmd + ' --imain="' + denoising_path + '/' + patient_path + denoising_ext + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                else:
                    bashCommand = eddycmd + ' --imain="' + nifti_path + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --slspec="' + slspec_path + '" --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            else:
                if gibbs:
                    bashCommand = eddycmd + ' --imain="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                elif denoising:
                    bashCommand = eddycmd + ' --imain="' + denoising_path + '/' + patient_path + denoising_ext + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
                else:
                    bashCommand = eddycmd + ' --imain="' + nifti_path + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --mporder=' + str(s2v[0]) + ' --s2v_niter=' + str(s2v[1]) + ' --s2v_lambda=' + str(s2v[2]) + ' --s2v_interp=' + s2v[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
        else:
            if gibbs:
                bashCommand = eddycmd + ' --imain="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            elif denoising:
                bashCommand = eddycmd + ' --imain="' + denoising_path + '/' + patient_path + denoising_ext + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'
            else:
                bashCommand = eddycmd + ' --imain="' + nifti_path + '" --mask="' + processing_mask + '" --acqp="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --index="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" --bvecs="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bvec" --bvals="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr" --verbose --cnr_maps --residuals --repol=' + str(olrep[0]) + ' --ol_nstd=' + str(olrep[1]) + ' --ol_nvox=' + str(olrep[2]) + ' --ol_type=' + olrep[3] + ' --fwhm=' + fwhm + ' --niter=' + str(niter) + ' --slm=linear'

        if topup:
            bashCommand = bashCommand + ' --topup="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate"'

        import subprocess
        bashCommand = bashCommand + " " + eddy_additional_arg
        bashcmd = bashCommand.split()
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f.close()

        eddy_log = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/eddy/eddy_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=eddy_log,
                                   stderr=subprocess.STDOUT)

        # wait until eddy finish
        output, error = process.communicate()
        eddy_log.close()

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f.close()

        if not biasfield:
            data, affine = load_nifti(
                folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + "_eddy_corr.nii.gz")
            save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz',
                       data.astype(np.float32), affine)

        shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
        shutil.copyfile(
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + "_eddy_corr.eddy_rotated_bvecs",
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")


    if biasfield and starting_state!="report":

        #import SimpleITK as sitk
        makedir(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/biasfield/", folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        if eddy:
            inputImage = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr.nii.gz'
        elif topup:
            inputImage = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr.nii.gz'
        elif gibbs:
            inputImage = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/gibbs/' + patient_path + '_gibbscorrected.nii.gz'
        elif denoising:
            inputImage = denoising_path + '/' + patient_path + denoising_ext
        else:
            inputImage = nifti_path

        bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; dwibiascorrect ants {} {} -fslgrad {} {} -bias {} -scratch {} -force -info -nthreads {}'.format(
            inputImage, folder_path + '/subjects/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_corr.nii.gz",
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec",
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + "_biasfield_est.nii.gz",
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/biasfield/tmp',
            core_count)

        import subprocess
        bashcmd = bashCommand.split()
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Bias Field launched for patient %s \n" % p + " with bash command " + bashCommand)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Bias Field launched for patient %s \n" % p + " with bash command " + bashCommand)
        f.close()

        biasfield_log = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/biasfield/biasfield_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=biasfield_log,
                                   stderr=subprocess.STDOUT)

        # wait until biasfield finish
        output, error = process.communicate()
        biasfield_log.close()

        shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/biasfield/' + patient_path + '_biasfield_corr.nii.gz',
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz')


        if not eddy:
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval")
            shutil.copyfile(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec",
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")


    #replace windows newline by unix newline:
    with open(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval", "r+") as f:
        data = f.read()
        output = data.replace(r"\r\n", r"\n")
        f.seek(0)
        f.write(output)
        f.truncate()

    with open(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec", "r+") as f:
        data = f.read()
        output = data.replace(r"\r\n", r"\n")
        f.seek(0)
        f.write(output)
        f.truncate()

    # Generate b0 ref:
    preproc, affine = load_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz')
    b0_ref = preproc[..., 0]
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dwiref.nii.gz', b0_ref.astype(np.float32), affine)

    preproc = None
    b0_ref = None
    gc.collect()
    #### Generate final mask ####
    from elikopy.utils import clean_mask

    # Step 1 : dwi2mask
    preproc_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz'
    bvec_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.bvec'
    bval_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.bval'
    dwi2mask_path = folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_space-dwi_type-dwi2mask_brainmask.nii.gz'
    cmd = f"dwi2mask -fslgrad {bvec_path} {bval_path} {preproc_path} {dwi2mask_path} -force"
    import subprocess
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": dwi2mask launched for patient %s \n" % p + " with bash command " + cmd)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": dwi2mask launched for patient %s \n" % p + " with bash command " + cmd)
    f.flush()
    process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=f, stderr=subprocess.STDOUT)
    # wait until dwi2mask finish
    output, error = process.communicate()
    f.flush()
    f.close()

    # Step 2 : mri_synth_strip on dwiref
    dwiref_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dwiref.nii.gz'
    mrisynthstrip_path = folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_space-dwi_type-mrisynthstrip_brainmask.nii.gz'
    cmd = f"mri_synth_strip -i {dwiref_path} -m {mrisynthstrip_path}"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": mri_synth_strip launched for patient %s \n" % p + " with bash command " + cmd)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": mri_synth_strip launched for patient %s \n" % p + " with bash command " + cmd)
    f.flush()
    process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=f, stderr=subprocess.STDOUT)
    output, error = process.communicate()
    f.flush()
    f.close()

    # Step 3 : median otsu on preprocess data
    preproc, affine = load_nifti(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc_nomask.nii.gz')
    preproc_masked, mask = median_otsu(preproc, median_radius=2, numpass=1, vol_idx=range(0, np.shape(preproc)[3]), dilate=2)
    mask = clean_mask(mask)
    save_nifti(folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_type-otsu_dilate-2_brainmask.nii.gz',
               mask.astype(np.float32), affine)

    # Step 4: Apply all masks to preprocess data
    dwi2mask_mask, _ = load_nifti(dwi2mask_path)
    mrisynthstrip_mask, _ = load_nifti(mrisynthstrip_path)

    all_mask = mask + dwi2mask_mask + mrisynthstrip_mask
    full_mask = np.zeros_like(all_mask)
    full_mask[all_mask >= 2] = 1
    full_mask = clean_mask(full_mask)

    full_mask_inclusive = np.logical_or(mask, np.logical_or(dwi2mask_mask, mrisynthstrip_mask))
    full_mask_inclusive = clean_mask(full_mask_inclusive)
    # dilate mask
    full_mask_inclusive = binary_dilation(full_mask_inclusive, iterations=1)
    preproc_masked = preproc * full_mask_inclusive[..., np.newaxis]

    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz',
               preproc_masked.astype(np.float32), affine)
    save_nifti(folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_brain_mask.nii.gz',
               full_mask.astype(np.float32), affine)
    save_nifti(folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_brain_mask_dilated.nii.gz',
               full_mask_inclusive.astype(np.float32), affine)



    if not report:
        update_status(folder_path, patient_path, "preproc")
        return

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting QC %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
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
    from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
    from dipy.align.transforms import RigidTransform3D
    from dipy.segment.mask import segment_from_cfa
    from dipy.segment.mask import bounding_box
    from os.path import isdir
    from skimage import measure
    from fpdf import FPDF

    preproc_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/'
    raw_path = folder_path + '/subjects/' + patient_path + '/dMRI/raw/'
    mask_path = folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_brain_mask_dilated.nii.gz'
    qc_path = preproc_path + 'quality_control'
    makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

    """Open the data"""

    # original data
    raw_data, raw_affine = load_nifti(raw_path + patient_path + "_raw_dmri.nii.gz")
    bvals, bvecs = read_bvals_bvecs(raw_path + patient_path + "_raw_dmri.bval",
                                    raw_path + patient_path + "_raw_dmri.bvec")

    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab_raw = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    # reslice data
    bool_reslice = isdir(os.path.join(preproc_path, "reslice"))
    if bool_reslice:
        reslice_data, reslice_affine = load_nifti(preproc_path + "reslice/" + patient_path + "_reslice.nii.gz")

    # bet data (stage to compare with final)
    mask_raw, mask_raw_affine = load_nifti(preproc_path + "bet/" + patient_path + "_binary_mask.nii.gz")
    if bool_reslice:
        bet_data = reslice_data * mask_raw[..., np.newaxis]
        bet_affine = reslice_affine
    else:
        bet_data = raw_data * mask_raw[..., np.newaxis]
        bet_affine = raw_affine

    # mppca data
    bool_mppca = isdir(os.path.join(preproc_path, "mppca"))
    if bool_mppca:
        mppca_data, mppca_affine = load_nifti(preproc_path + "mppca/" + patient_path + "_mppca.nii.gz")
        sigma, sigma_affine = load_nifti(preproc_path + "mppca/" + patient_path + "_sigmaNoise.nii.gz")

    # patch2self data
    bool_patch2self = isdir(os.path.join(preproc_path, "patch2self"))
    if bool_patch2self:
        patch2self_data, patch2self_affine = load_nifti(preproc_path + "patch2self/" + patient_path + "_patch2self.nii.gz")

    # gibbs data
    bool_gibbs = isdir(os.path.join(preproc_path, "gibbs"))
    if bool_gibbs:
        gibbs_data, gibbs_affine = load_nifti(preproc_path + "gibbs/" + patient_path + "_gibbscorrected.nii.gz")

    # topup data
    bool_topup = isdir(os.path.join(preproc_path, "topup"))
    if bool_topup:
        if not eddy:
            topup_data, topup_affine = load_nifti(preproc_path + "topup/" + patient_path + "_topup_corr.nii.gz")
        field_data, field_affine = load_nifti(
            preproc_path + "topup/" + patient_path + "_topup_estimate_fieldcoef.nii.gz")

    # eddy data (=preproc total)
    bool_eddy = isdir(os.path.join(preproc_path, "eddy"))
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
        plot_shell[:, 0:np.shape(raw_data)[1]] = raw_data[..., max(sl - 10, 0), shell_index[0]]
        plot_shell[:, np.shape(raw_data)[1]:(np.shape(raw_data)[1] * 2)] = raw_data[..., max(sl - 5, 0), shell_index[0]]
        plot_shell[:, (np.shape(raw_data)[1] * 2):(np.shape(raw_data)[1] * 3)] = raw_data[..., sl, shell_index[0]]
        plot_shell[:, (np.shape(raw_data)[1] * 3):(np.shape(raw_data)[1] * 4)] = raw_data[..., min(sl + 5, 2*sl-1), shell_index[0]]
        plot_shell[:, (np.shape(raw_data)[1] * 4):(np.shape(raw_data)[1] * 5)] = raw_data[..., min(sl + 10, 2*sl-1), shell_index[0]]
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
        numstep = 1 + bool_mppca + bool_patch2self + bool_gibbs  + bool_eddy
        if not eddy:
            numstep = numstep + bool_topup

        current_subplot = 0
        fig, axs = plt.subplots(numstep, 1, figsize=(14, 3 * numstep))
        fig.suptitle('Overview of processing steps for b=' + str(list_bval[i]), y=0.95, fontsize=16)

        # plot bet
        X1, Y1 = mask_to_coor(mask_raw[..., max(sl - 10, 0)])
        X2, Y2 = mask_to_coor(mask_raw[..., max(sl - 5, 0)])
        X3, Y3 = mask_to_coor(mask_raw[..., sl])
        X4, Y4 = mask_to_coor(mask_raw[..., min(sl + 5, 2*sl-1)])
        X5, Y5 = mask_to_coor(mask_raw[..., min(sl + 10, 2*sl-1)])
        Y = Y1 + Y2 + Y3 + Y4 + Y5
        X = X1 + [x + np.shape(mask_raw)[1] for x in X2] + [x + np.shape(mask_raw)[1] * 2 for x in X3] + [
            x + np.shape(mask_raw)[1] * 3 for x in X4] + [x + np.shape(mask_raw)[1] * 4 for x in X5]
        if numstep == 1:
            fig_scat = plt.scatter(X, Y, marker='.', s=1, c='red')
        else:
            axs[current_subplot].scatter(X, Y, marker='.', s=1, c='red')
        plot_bet = np.zeros((np.shape(bet_data)[0], np.shape(bet_data)[1] * 5))
        plot_bet[:, 0:np.shape(bet_data)[1]] = bet_data[..., max(sl - 10, 0), shell_index[0]]
        plot_bet[:, np.shape(bet_data)[1]:(np.shape(bet_data)[1] * 2)] = bet_data[..., max(sl - 5, 0), shell_index[0]]
        plot_bet[:, (np.shape(bet_data)[1] * 2):(np.shape(bet_data)[1] * 3)] = bet_data[..., sl, shell_index[0]]
        plot_bet[:, (np.shape(bet_data)[1] * 3):(np.shape(bet_data)[1] * 4)] = bet_data[..., min(sl + 5, 2*sl-1), shell_index[0]]
        plot_bet[:, (np.shape(bet_data)[1] * 4):(np.shape(bet_data)[1] * 5)] = bet_data[..., min(sl + 10, 2*sl-1), shell_index[0]]
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
            plot_mppca[:, 0:np.shape(mppca_data)[1]] = mppca_data[..., max(sl - 10, 0), shell_index[0]]
            plot_mppca[:, np.shape(mppca_data)[1]:(np.shape(mppca_data)[1] * 2)] = mppca_data[
                ..., max(sl - 5, 0), shell_index[0]]
            plot_mppca[:, (np.shape(mppca_data)[1] * 2):(np.shape(mppca_data)[1] * 3)] = mppca_data[
                ..., sl, shell_index[0]]
            plot_mppca[:, (np.shape(mppca_data)[1] * 3):(np.shape(mppca_data)[1] * 4)] = mppca_data[
                ..., min(sl + 5, 2*sl-1), shell_index[0]]
            plot_mppca[:, (np.shape(mppca_data)[1] * 4):(np.shape(mppca_data)[1] * 5)] = mppca_data[
                ..., min(sl + 10, 2*sl-1), shell_index[0]]
            axs[current_subplot].imshow(plot_mppca, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Denoising MPPCA')
            current_subplot = current_subplot + 1

        # plot patch2self
        if bool_patch2self:
            plot_patch2self = np.zeros((np.shape(patch2self_data)[0], np.shape(patch2self_data)[1] * 5))
            plot_patch2self[:, 0:np.shape(patch2self_data)[1]] = patch2self_data[..., max(sl - 10, 0), shell_index[0]]
            plot_patch2self[:, np.shape(patch2self_data)[1]:(np.shape(patch2self_data)[1] * 2)] = patch2self_data[
                ..., max(sl - 5, 0), shell_index[0]]
            plot_patch2self[:, (np.shape(patch2self_data)[1] * 2):(np.shape(patch2self_data)[1] * 3)] = patch2self_data[
                ..., sl, shell_index[0]]
            plot_patch2self[:, (np.shape(patch2self_data)[1] * 3):(np.shape(patch2self_data)[1] * 4)] = patch2self_data[
                ..., min(sl + 5, 2*sl-1), shell_index[0]]
            plot_patch2self[:, (np.shape(patch2self_data)[1] * 4):(np.shape(patch2self_data)[1] * 5)] = patch2self_data[
                ..., min(sl + 10, 2*sl-1), shell_index[0]]
            axs[current_subplot].imshow(plot_patch2self, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Denoising patch2self')
            current_subplot = current_subplot + 1

        # plot gibbs
        if bool_gibbs:
            plot_gibbs = np.zeros((np.shape(gibbs_data)[0], np.shape(gibbs_data)[1] * 5))
            plot_gibbs[:, 0:np.shape(gibbs_data)[1]] = gibbs_data[..., max(sl - 10, 0), shell_index[0]]
            plot_gibbs[:, np.shape(gibbs_data)[1]:(np.shape(gibbs_data)[1] * 2)] = gibbs_data[
                ..., max(sl - 5, 0), shell_index[0]]
            plot_gibbs[:, (np.shape(gibbs_data)[1] * 2):(np.shape(gibbs_data)[1] * 3)] = gibbs_data[
                ..., sl, shell_index[0]]
            plot_gibbs[:, (np.shape(gibbs_data)[1] * 3):(np.shape(gibbs_data)[1] * 4)] = gibbs_data[
                ..., min(sl + 5, 2*sl-1), shell_index[0]]
            plot_gibbs[:, (np.shape(gibbs_data)[1] * 4):(np.shape(gibbs_data)[1] * 5)] = gibbs_data[
                ..., min(sl + 10, 2*sl-1), shell_index[0]]
            axs[current_subplot].imshow(plot_gibbs, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Gibbs ringing correction')
            current_subplot = current_subplot + 1
        # plot topup
        if bool_topup and not eddy:
            plot_topup = np.zeros((np.shape(topup_data)[0], np.shape(topup_data)[1] * 5))
            plot_topup[:, 0:np.shape(topup_data)[1]] = topup_data[..., max(sl - 10, 0), shell_index[0]]
            plot_topup[:, np.shape(topup_data)[1]:(np.shape(topup_data)[1] * 2)] = topup_data[
                ..., max(sl - 5, 0), shell_index[0]]
            plot_topup[:, (np.shape(topup_data)[1] * 2):(np.shape(topup_data)[1] * 3)] = topup_data[
                ..., sl, shell_index[0]]
            plot_topup[:, (np.shape(topup_data)[1] * 3):(np.shape(topup_data)[1] * 4)] = topup_data[
                ..., min(sl + 5, 2*sl-1), shell_index[0]]
            plot_topup[:, (np.shape(topup_data)[1] * 4):(np.shape(topup_data)[1] * 5)] = topup_data[
                ..., min(sl + 10, 2*sl-1), shell_index[0]]
            axs[current_subplot].imshow(plot_topup, cmap='gray')
            axs[current_subplot].set_axis_off()
            axs[current_subplot].set_title('Susceptibility induced distortions correction')
            current_subplot = current_subplot + 1
        # plot eddy
        if bool_eddy:
            plot_eddy = np.zeros((np.shape(preproc_data)[0], np.shape(preproc_data)[1] * 5))
            plot_eddy[:, 0:np.shape(preproc_data)[1]] = preproc_data[..., max(sl - 10, 0), shell_index[0]]
            plot_eddy[:, np.shape(preproc_data)[1]:(np.shape(preproc_data)[1] * 2)] = preproc_data[
                ..., max(sl - 5, 0), shell_index[0]]
            plot_eddy[:, (np.shape(preproc_data)[1] * 2):(np.shape(preproc_data)[1] * 3)] = preproc_data[
                ..., sl, shell_index[0]]
            plot_eddy[:, (np.shape(preproc_data)[1] * 3):(np.shape(preproc_data)[1] * 4)] = preproc_data[
                ..., min(sl + 5, 2*sl-1), shell_index[0]]
            plot_eddy[:, (np.shape(preproc_data)[1] * 4):(np.shape(preproc_data)[1] * 5)] = preproc_data[
                ..., min(sl + 10, 2*sl-1), shell_index[0]]
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
        elif bool_patch2self:
            previous = patch2self_data
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

    if bool_mppca or bool_patch2self:

        # 1) DIPY SNR estimation =========================================================================

        tenmodel = dti.TensorModel(gtab_raw)
        _, maskSNR = median_otsu(raw_data, vol_idx=[0])
        maskSNR = clean_mask(maskSNR)
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
        list_images.append([qc_path + "/dipyNoise.jpg"])

        # 2) MPPCA sigma + SNR estimation + before/after residual ==========================================

        if bool_patch2self:
            denoising_data = patch2self_data
        elif bool_mppca:
            denoising_data = mppca_data
        fig, axs = plt.subplots(len(list_bval), 3, figsize=(9, 3 * len(list_bval)))
        #fig.suptitle('MPPCA denoising', y=1, fontsize=16)
        for i in range(len(list_bval)):
            shell_index = np.where(np.logical_and(bval > list_bval[i] - 50, bval < list_bval[i] + 50))[0]
            # plot the gibbs before, after and residual
            axs[i, 0].imshow(bet_data[..., sl, shell_index[0]], cmap='gray')
            axs[i, 0].set_axis_off()
            axs[i, 0].set_title('Original at b=' + str(list_bval[i]))
            axs[i, 1].imshow(denoising_data[..., sl, shell_index[0]], cmap='gray')
            axs[i, 1].set_axis_off()
            axs[i, 1].set_title('MPPCA denoised at b=' + str(list_bval[i]))
            axs[i, 2].imshow(np.abs(bet_data[..., sl, shell_index[0]] - denoising_data[..., sl, shell_index[0]]),
                             cmap='gray')
            axs[i, 2].set_axis_off()
            axs[i, 2].set_title('Residual at b=' + str(list_bval[i]))
        plt.savefig(qc_path + "/denoisingResidual.jpg", dpi=300, bbox_inches='tight')
        list_images.append([qc_path + "/denoisingResidual.jpg"])

        if bool_mppca:
            masked_sigma = np.ma.array(np.nan_to_num(sigma), mask=1 - mask_raw)
            mean_sigma = masked_sigma.mean()
            b0 = np.ma.array(mppca_data[..., 0], mask=1 - mask_raw)
            mean_signal = b0.mean()
            snr = mean_signal / mean_sigma
            sl = np.shape(sigma)[2] // 2
            plot_sigma = np.zeros((np.shape(sigma)[0], np.shape(sigma)[1] * 5))
            plot_sigma[:, 0:np.shape(sigma)[1]] = sigma[..., max(sl - 10, 0)]
            plot_sigma[:, np.shape(sigma)[1]:(np.shape(sigma)[1] * 2)] = sigma[..., max(sl - 5, 0)]
            plot_sigma[:, (np.shape(sigma)[1] * 2):(np.shape(sigma)[1] * 3)] = sigma[..., sl]
            plot_sigma[:, (np.shape(sigma)[1] * 3):(np.shape(sigma)[1] * 4)] = sigma[..., min(sl + 5, 2*sl-1)]
            plot_sigma[:, (np.shape(sigma)[1] * 4):(np.shape(sigma)[1] * 5)] = sigma[..., min(sl + 10, 2*sl-1)]
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

            list_images.append([qc_path + "/mppcaSigma.jpg"])


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
                plot_raw[:, 0:np.shape(bet_data)[1]] = tsnr_raw[..., max(sl - 10, 0)]
                plot_raw[:, np.shape(bet_data)[1]:(np.shape(bet_data)[1] * 2)] = tsnr_raw[..., max(sl - 5, 0)]
                plot_raw[:, (np.shape(bet_data)[1] * 2):(np.shape(bet_data)[1] * 3)] = tsnr_raw[..., sl]
                plot_raw[:, (np.shape(bet_data)[1] * 3):(np.shape(bet_data)[1] * 4)] = tsnr_raw[..., min(sl + 5, 2*sl-1)]
                plot_raw[:, (np.shape(bet_data)[1] * 4):(np.shape(bet_data)[1] * 5)] = tsnr_raw[..., min(sl + 10, 2*sl-1)]
                # image of preproc
                plot_preproc = np.zeros((np.shape(preproc_data)[0], np.shape(preproc_data)[1] * 5))
                plot_preproc[:, 0:np.shape(preproc_data)[1]] = tsnr_preproc[..., max(sl - 10, 0)]
                plot_preproc[:, np.shape(preproc_data)[1]:(np.shape(preproc_data)[1] * 2)] = tsnr_preproc[..., max(sl - 5, 0)]
                plot_preproc[:, (np.shape(preproc_data)[1] * 2):(np.shape(preproc_data)[1] * 3)] = tsnr_preproc[..., sl]
                plot_preproc[:, (np.shape(preproc_data)[1] * 3):(np.shape(preproc_data)[1] * 4)] = tsnr_preproc[
                    ..., min(sl + 5, 2*sl-1)]
                plot_preproc[:, (np.shape(preproc_data)[1] * 4):(np.shape(preproc_data)[1] * 5)] = tsnr_preproc[
                    ..., min(sl + 10, 2*sl-1)]
                # image of difference
                plot_diff = np.zeros((np.shape(preproc_data)[0], np.shape(preproc_data)[1] * 5))
                plot_diff[:, 0:np.shape(preproc_data)[1]] = tsnr_raw[..., max(sl - 10, 0)] - tsnr_preproc[..., max(sl - 10, 0)]
                plot_diff[:, np.shape(preproc_data)[1]:(np.shape(preproc_data)[1] * 2)] = tsnr_raw[..., max(sl - 5, 0)] - \
                                                                                          tsnr_preproc[..., max(sl - 5, 0)]
                plot_diff[:, (np.shape(preproc_data)[1] * 2):(np.shape(preproc_data)[1] * 3)] = tsnr_raw[..., sl] - \
                                                                                                tsnr_preproc[..., sl]
                plot_diff[:, (np.shape(preproc_data)[1] * 3):(np.shape(preproc_data)[1] * 4)] = tsnr_raw[..., min(sl + 5, 2*sl-1)] - \
                                                                                                tsnr_preproc[
                                                                                                    ..., min(sl + 5, 2*sl-1)]
                plot_diff[:, (np.shape(preproc_data)[1] * 4):(np.shape(preproc_data)[1] * 5)] = tsnr_raw[..., min(sl + 10, 2*sl-1)] - \
                                                                                                tsnr_preproc[
                                                                                                    ..., min(sl + 10, 2*sl-1)]

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
        plot_field[:, 0:np.shape(field_data)[1]] = field_data[..., max(sl - 10, 0)]
        plot_field[:, np.shape(field_data)[1]:(np.shape(field_data)[1] * 2)] = field_data[..., max(sl - 5, 0)]
        plot_field[:, (np.shape(field_data)[1] * 2):(np.shape(field_data)[1] * 3)] = field_data[..., sl]
        plot_field[:, (np.shape(field_data)[1] * 3):(np.shape(field_data)[1] * 4)] = field_data[..., min(sl + 5, 2*sl-1)]
        plot_field[:, (np.shape(field_data)[1] * 4):(np.shape(field_data)[1] * 5)] = field_data[..., min(sl + 5, 2*sl-1)]
        axs[0].imshow(plot_field, cmap='gray')
        axs[0].set_axis_off()

        sl = np.shape(field_data)[1] // 2
        plot_field = np.zeros((np.shape(field_data)[2], np.shape(field_data)[0] * 5))
        plot_field[:, 0:np.shape(field_data)[0]] = np.rot90(field_data[..., max(sl - 10, 0), :])
        plot_field[:, np.shape(field_data)[0]:(np.shape(field_data)[0] * 2)] = np.rot90(field_data[..., max(sl - 5, 0), :])
        plot_field[:, (np.shape(field_data)[0] * 2):(np.shape(field_data)[0] * 3)] = np.rot90(field_data[..., sl, :])
        plot_field[:, (np.shape(field_data)[0] * 3):(np.shape(field_data)[0] * 4)] = np.rot90(
            field_data[..., min(sl + 5, 2*sl-1), :])
        plot_field[:, (np.shape(field_data)[0] * 4):(np.shape(field_data)[0] * 5)] = np.rot90(
            field_data[..., min(sl + 10, 2*sl-1), :])
        axs[1].imshow(plot_field, cmap='gray')
        axs[1].set_axis_off()

        sl = np.shape(field_data)[0] // 2
        plot_field = np.zeros((np.shape(field_data)[2], np.shape(field_data)[1] * 5))
        plot_field[:, 0:np.shape(field_data)[1]] = np.rot90(field_data[sl - 10, ...])
        plot_field[:, np.shape(field_data)[1]:(np.shape(field_data)[1] * 2)] = np.rot90(field_data[sl - 5, ...])
        plot_field[:, (np.shape(field_data)[1] * 2):(np.shape(field_data)[1] * 3)] = np.rot90(field_data[sl, ...])
        plot_field[:, (np.shape(field_data)[1] * 3):(np.shape(field_data)[1] * 4)] = np.rot90(field_data[min(sl + 5, 2*sl-1), ...])
        plot_field[:, (np.shape(field_data)[1] * 4):(np.shape(field_data)[1] * 5)] = np.rot90(field_data[min(sl + 10, 2*sl-1), ...])
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

        result = []

        if core_count > 2:
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Starting Eddy QC_REG for patient %s with multicore enabled\n" % p)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Starting Eddy QC_REG for patient %s with multicore enabled \n" % p)
            f.close()
            from concurrent.futures import ProcessPoolExecutor
            for i in range(np.shape(preproc_data)[3]):
                #print('current iteration : ', i, end="\r")
                volume.append(i)

            def motion_raw_transform(i):
                return affreg.optimize(np.copy(S0s_raw[..., 0]), np.copy(bet_data[..., i]), transform, params0, bet_affine,
                                        bet_affine, ret_metric=True)[1]
            with ProcessPoolExecutor(max_workers=int(core_count*0.8)) as executor:
                for r in executor.map(motion_raw_transform, range(np.shape(preproc_data)[3])):
                    motion_raw.append(r)

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of QC_REG motion_raw for patient %s" % p)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of QC_REG motion_raw for patient %s \n" % p)
            f.close()

            def motion_preproc_transform(i):
                return affreg.optimize(np.copy(S0s_preproc[..., 0]), np.copy(preproc_data[..., i]), transform, params0,
                                        preproc_affine, preproc_affine, ret_metric=True)[1]
            with ProcessPoolExecutor(max_workers=int(core_count*0.8)) as executor:
                for r in executor.map(motion_preproc_transform, range(np.shape(preproc_data)[3])):
                    motion_proc.append(r)

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of QC_REG motion_preproc for patient %s" % p)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of QC_REG motion_preproc for patient %s \n" % p)
            f.close()
        else:
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Starting Eddy QC_REG for patient %s with multicore disabled\n" % p)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Starting Eddy QC_REG for patient %s with multicore disabled \n" % p)
            f.close()
            for i in range(np.shape(preproc_data)[3]):
                #print('current iteration : ', i, end="\r")
                volume.append(i)

                rigid = affreg.optimize(np.copy(S0s_raw[..., 0]), np.copy(bet_data[..., i]), transform, params0, bet_affine,
                                        bet_affine, ret_metric=True)
                motion_raw.append(rigid[1])

                rigid = affreg.optimize(np.copy(S0s_preproc[..., 0]), np.copy(preproc_data[..., i]), transform, params0,
                                        preproc_affine, preproc_affine, ret_metric=True)
                motion_proc.append(rigid[1])

                if int(i/np.shape(preproc_data)[3])%5==0:
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": Eddy QC_REG : Volume " + str(i) + "/" + str(np.shape(preproc_data)[3]) + " \n")
                    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": Eddy QC_REG : Volume " + str(i) + "/" + str(np.shape(preproc_data)[3]) + " \n")
                    f.close()
        # ============================================================

        motion_raw = np.array(motion_raw)
        motion_proc = np.array(motion_proc)

        fig, (ax1, ax2) = plt.subplots(2, sharey=True, figsize=(10, 6))
        ax1.bar(volume, np.abs(motion_raw[:, 3]) + np.abs(motion_raw[:, 4]) + np.abs(motion_raw[:, 5]),
                label='z')
        ax1.bar(volume, np.abs(motion_raw[:, 3]) + np.abs(motion_raw[:, 4]), label='y')
        ax1.bar(volume, np.abs(motion_raw[:, 3]), label='x')
        ax1.legend(title='Translation', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_title('raw data translation')
        ax2.bar(volume, np.abs(motion_proc[:, 3]) + np.abs(motion_proc[:, 4]) + np.abs(motion_proc[:, 5]),
                label='z translation')
        ax2.bar(volume, np.abs(motion_proc[:, 3]) + np.abs(motion_proc[:, 4]), label='y translation')
        ax2.bar(volume, np.abs(motion_proc[:, 3]), label='x translation')
        ax2.set_title('processed data translation')
        plt.savefig(qc_path + "/motion1.jpg", dpi=300, bbox_inches='tight')

        fig, (ax1, ax2) = plt.subplots(2, sharey=True, figsize=(10, 6))
        ax1.bar(volume, np.abs(motion_raw[:, 0]) + np.abs(motion_raw[:, 1]) + np.abs(motion_raw[:, 2]),
                label='z')
        ax1.bar(volume, np.abs(motion_raw[:, 0]) + np.abs(motion_raw[:, 1]), label='y')
        ax1.bar(volume, np.abs(motion_raw[:, 0]), label='x')
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
        if 'multiple_encoding' not in locals():
            with open(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt') as f:
                topup_acq = [[float(x) for x in line2.split()] for line2 in f]
            # Check if multiple or single encoding direction
            multiple_encoding = False
            curr_x = 0.0
            curr_y = 0.0
            curr_z = 0.0
            first = True
            print("Topup acq parameters:")
            print(topup_acq)
            for acq in topup_acq:
                if not first and (curr_x != acq[1] or curr_y != acq[2] or curr_z != acq[3]):
                    multiple_encoding = True
                first = False
                curr_x = acq[1]
                curr_y = acq[2]
                curr_z = acq[3]

        # Do Eddy quad for the subject
        slspec_path = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'slspec.txt'
        if os.path.isfile(slspec_path):
            if topup and multiple_encoding and not forceSynb0DisCo:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" -s "' + slspec_path + '" -f "' + preproc_path + 'topup/' + patient_path + '_topup_estimate_fieldcoef.nii.gz"'
            else:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" -s "' + slspec_path + '"'
        else:
            if topup and multiple_encoding and not forceSynb0DisCo:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval" -f "' + preproc_path + 'topup/' + patient_path + '_topup_estimate_fieldcoef.nii.gz"'
            else:
                bashCommand = 'eddy_quad ' + preproc_path + 'eddy/' + patient_path + '_eddy_corr -idx "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt" -par "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" -m "' + mask_path + '" -b "' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.bval"'

        import subprocess
        bashcmd = bashCommand.split()
        qc_log = open(qc_path + "/qc_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=qc_log,
                                   stderr=subprocess.STDOUT)
        output, error = process.communicate()
        qc_log.close()

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()

    update_status(folder_path, patient_path, "preproc")


def dti_solo(folder_path, p, maskType="brain_mask_dilated",
             use_all_shells: bool = False, report=True):
    """
    Computes the DTI metrics for a single subject. The outputs are available in
    the directories <folder_path>/subjects/<subjects_ID>/dMRI/dti/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param use_wm_mask: If true a white matter mask is used. The white_matter()
    function needs to already be applied. default=False
    :param use_all_shells: Boolean. DTI will use all shells available, not just
    shells <= 2000, this will cause a more defined white matter at the cost of
    an erronous estimation of the CSF. The default is False.
    """
    log_prefix = "DTI SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual DTI processing for patient %s \n" % p)

    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    import dipy.reconst.dti as dti

    assert maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"

    patient_path = p

    dti_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti"
    makedir(dti_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", log_prefix)

    # load the data======================================
    data, affine = load_nifti(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")


    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'
    if os.path.isfile(mask_path):
        mask, _ = load_nifti(mask_path)
    else:
        mask, _ = load_nifti(folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")

    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    # Remove shells >2000================================
    if not use_all_shells:
        indexes = np.argwhere(bvals < 2000+10)
        indexes = indexes.squeeze()
        bvals = bvals[indexes]
        bvecs = bvecs[indexes]
        data = data[..., indexes]
        print('Warning: removing shells above b=2000 for DTI. To disable this, '
              + 'activate the use_all_shells option.')
    # create the model===================================
    b0_threshold = np.min(bvals)+10
    b0_threshold = max(50, b0_threshold)
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    # FA ================================================
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
               FA.astype(np.float32), affine)
    # colored FA ========================================
    RGB = dti.color_fa(FA, tenfit.evecs)
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_fargb.nii.gz",
               np.array(255 * RGB, 'uint8'), affine)
    # Mean diffusivity ==================================
    MD = dti.mean_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_MD.nii.gz",
               MD.astype(np.float32), affine)
    # Radial diffusivity ==================================
    RD = dti.radial_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_RD.nii.gz",
               RD.astype(np.float32), affine)
    # Axial diffusivity ==================================
    AD = dti.axial_diffusivity(tenfit.evals)
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_AD.nii.gz",
               AD.astype(np.float32), affine)
    # eigen vectors =====================================
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_evecs.nii.gz",
               tenfit.evecs.astype(np.float32), affine)
    # eigen values ======================================
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_evals.nii.gz",
               tenfit.evals.astype(np.float32), affine)
    # diffusion tensor ====================================
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_dtensor.nii.gz",
               tenfit.quadratic_form.astype(np.float32), affine)
    # Residual ============================================
    reconstructed = tenfit.predict(gtab, S0=data[...,0])
    residual = data - reconstructed
    save_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_residual.nii.gz",
               residual.astype(np.float32), affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting QC %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting QC %s \n" % p)
    f.close()


    if report:

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metric1 = np.array(255 * RGB, 'uint8')
        metric2 = np.copy(MD)
        qc_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/quality_control"
        makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", log_prefix)

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
        plot_mse[:, 0:np.shape(mse)[1]] = mse[..., max(sl - 10, 0)]
        plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., max(sl - 5, 0)]
        plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
        plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., min(sl + 5, 2*sl-1)]
        plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., min(sl + 10, 2*sl-1)]
        im0 = axs[0].imshow(plot_mse, cmap='gray', vmax=max_plot)
        axs[0].set_title('MSE')
        axs[0].set_axis_off()
        fig.colorbar(im0, ax=axs[0], orientation='horizontal')
        sl = np.shape(R2)[2] // 2
        plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
        plot_R2[:, 0:np.shape(R2)[1]] = R2[..., max(sl - 10, 0)]
        plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., max(sl - 5, 0)]
        plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
        plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., min(sl + 5, 2*sl-1)]
        plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., min(sl + 10, 2*sl-1)]
        im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
        axs[1].set_title('R2')
        axs[1].set_axis_off()
        fig.colorbar(im1, ax=axs[1], orientation='horizontal');
        plt.tight_layout()
        plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        sl = np.shape(metric1)[2] // 2
        plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5, 3), dtype=np.int16)
        plot_metric1[:, 0:np.shape(metric1)[1], :] = metric1[..., max(sl - 10, 0), :]
        plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2), :] = metric1[..., max(sl - 5, 0), :]
        plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3), :] = metric1[..., sl, :]
        plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4), :] = metric1[..., min(sl + 5, 2*sl-1), :]
        plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5), :] = metric1[..., min(sl + 10, 2*sl-1), :]
        axs[0].imshow(plot_metric1)
        axs[0].set_title('Fractional anisotropy')
        axs[0].set_axis_off()
        sl = np.shape(metric2)[2] // 2
        plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
        plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., max(sl - 10, 0)]
        plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., max(sl - 5, 0)]
        plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl]
        plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., min(sl + 5, 2*sl-1)]
        plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., min(sl + 10, 2*sl-1)]
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

        if not os.path.exists(folder_path + '/subjects/' + patient_path + '/quality_control.pdf'):
            shutil.copyfile(qc_path + '/qc_report.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        else:
            """Merge with QC of preproc""";
            from pypdf import PdfMerger
            pdfs = [folder_path + '/subjects/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
            merger = PdfMerger()
            for pdf in pdfs:
                merger.append(pdf)
            merger.write(folder_path + '/subjects/' + patient_path + '/quality_control_dti.pdf')
            merger.close()
            os.remove(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
            os.rename(folder_path + '/subjects/' + patient_path + '/quality_control_dti.pdf',folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
            f.close()

    update_status(folder_path, patient_path, "dti")


def white_mask_solo(folder_path, p, maskType, corr_gibbs=True, core_count=1, debug=False):
    """ Computes a white matter mask for a single subject based on the T1 structural image or on the anisotropic power map
    (obtained from the diffusion images) if the T1 image is not available. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/masks/.
    The T1 images can be gibbs ringing corrected.

    :param maskType: maskType must be either 'wm_mask_FSL_T1' or 'wm_mask_AP'.
    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param corr_gibbs: If true, Gibbs ringing correction is performed on the T1 image. default=True
    :param core_count: Number of allocated cpu cores. default=1
    :param debug: If true, additional intermediate output will be saved. default=False
    """

    assert maskType in ['wm_mask_FSL_T1', 'wm_mask_AP'], "maskType must be either 'wm_mask_FSL_T1' or 'wm_mask_AP'"

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

    patient_path = p
    anat_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
    if os.path.isfile(anat_path) and maskType=="wm_mask_FSL_T1":
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from T1 %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from T1 %s \n" % p)
        f.close()
        # Read the moving image ====================================
        anat_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
        if os.path.exists(anat_path):
            brain_extracted_T1_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
            brain_extracted_mask_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_brain_mask.nii.gz'
            if not os.path.exists(brain_extracted_T1_path):
                print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                    "%d.%b %Y %H:%M:%S") + ": T1 Brain extraction for patient %s (mri_synth_strip) \n" % p)
                cmd = f"mri_synth_strip -i {anat_path} -o {brain_extracted_T1_path} -m {brain_extracted_mask_path} "
                print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": " + cmd)
                import subprocess
                process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                output, error = process.communicate()
                print(output)
                print(error)

                if not os.path.exists(brain_extracted_T1_path):
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": T1 Brain extraction failed for patient %s \n" % p)
                    # print to std err
                    print("T1 Brain extraction failed for patient %s" % p, file=sys.stderr)

        wm_log = open(folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", "a+")

        # Correct for gibbs ringing
        if corr_gibbs:
            wm_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Beginning of gibbs for patient %s \n" % p)
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Beginning of gibbs for patient %s \n" % p)

            data_gibbs, affine_gibbs = load_nifti(anat_path)
            data_gibbs = gibbs_removal(data_gibbs,num_processes=core_count)
            corrected_gibbs_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_gibbscorrected.nii.gz'
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
        brain_extracted_T1_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
        wm_log.close()

        moving_data, moving_affine = load_nifti(brain_extracted_T1_path)
        moving = moving_data
        moving_grid2world = moving_affine
        # Read the static image ====================================
        static_data, static_affine = load_nifti(
            folder_path + "/subjects/" + patient_path + "/dMRI/preproc/" + patient_path + "_dmri_preproc.nii.gz")

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
        out_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_whitemask.nii.gz'
        save_nifti(out_path, white_mask.astype(np.float32), moving_affine)

        # Save segmentation in T1 space
        out_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_segmentation.nii.gz'
        save_nifti(out_path, final_segmentation.astype(np.float32), moving_affine)

        white_mask[white_mask >= 0.01] = 1
        white_mask[white_mask < 0.01] = 0



        # transform the white matter mask ======================================
        white_mask = affine.transform(white_mask)
        white_mask[white_mask != 0] = 1
        anat_affine = static_grid2world
        segmentation = affine.transform(final_segmentation)

        # Save corrected projected T1
        out_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_corr_projected.nii.gz'
        save_nifti(out_path, affine.transform(moving_data).astype(np.float32), anat_affine)

        mask_path = folder_path + '/subjects/' + patient_path + "/masks"
        makedir(mask_path, folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", log_prefix)

        out_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_wm_mask_FSL_T1.nii.gz'
        save_nifti(out_path, white_mask.astype(np.float32), anat_affine)
        save_nifti(folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_segmentation_FSL_T1.nii.gz',
                   segmentation.astype(np.float32), anat_affine)

        wm_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_wm_mask_FSL_T1.nii.gz'
        seg_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_segmentation_FSL_T1.nii.gz'
    elif maskType == "wm_mask_AP":
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from AP %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Mask done from AP %s \n" % p)
        f.close()
        f = open(folder_path + "/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Warning: Mask done from AP for patient %s \n" % p)
        f.close()
        # compute the white matter mask with the Anisotropic power map
        data, affine = load_nifti(
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
        mask, _ = load_nifti(folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")
        bvals, bvecs = read_bvals_bvecs(
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
        b0_threshold = np.min(bvals) + 10
        b0_threshold = max(50, b0_threshold)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
        sphere = get_sphere('symmetric724')
        qball_model = shm.QballModel(gtab, 8)
        if core_count > 1:
            peaks = dp.peaks_from_model(model=qball_model, data=data, relative_peak_threshold=.5, min_separation_angle=25,
                                    sphere=sphere, mask=mask, parallel=True, num_processes=core_count)
        else:
            peaks = dp.peaks_from_model(model=qball_model, data=data, relative_peak_threshold=.5,
                                        min_separation_angle=25,
                                        sphere=sphere, mask=mask, parallel=False)

        ap = shm.anisotropic_power(peaks.shm_coeff)
        save_nifti(folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_ap.nii.gz', ap.astype(np.float32), affine)
        nclass = 3
        beta = 0.1
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
        white_mask = PVE[..., 2]
        white_mask[white_mask >= 0.01] = 1
        white_mask[white_mask < 0.01] = 0
        anat_affine = affine
        segmentation = np.copy(final_segmentation)

        mask_path = folder_path + '/subjects/' + patient_path + "/masks"
        makedir(mask_path, folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", log_prefix)

        out_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_wm_mask_AP.nii.gz'
        save_nifti(out_path, white_mask.astype(np.float32), anat_affine)
        save_nifti(folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_segmentation_AP.nii.gz', segmentation.astype(np.float32), anat_affine)

        wm_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_wm_mask_AP.nii.gz'
        seg_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_segmentation_AP.nii.gz'
    else:
        print("ERROR")
        exit(-1)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control  %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()

    """Imports"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    from pypdf import PdfMerger

    T1_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1.nii.gz'
    T1gibbs_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_gibbscorrected.nii.gz'
    T1brain_path = folder_path + '/subjects/' + patient_path + "/T1/" + patient_path + '_T1_brain.nii.gz'
    ap_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_ap.nii.gz'
    preproc_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz'
    qc_path = folder_path + '/subjects/' + patient_path + '/masks/' + 'quality_control'
    makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

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
    plot_seg[:, 0:np.shape(seg_data)[1]] = seg_data[..., max(sl - 10, 0)]
    plot_seg[:, np.shape(seg_data)[1]:(np.shape(seg_data)[1] * 2)] = seg_data[..., sl]
    plot_seg[:, (np.shape(seg_data)[1] * 2):(np.shape(seg_data)[1] * 3)] = seg_data[..., min(sl + 10, 2*sl-1)]
    axs[0].imshow(plot_seg)
    axs[0].set_axis_off()
    seg_data, seg_affine = load_nifti(preproc_path)
    sl = np.shape(seg_data)[2] // 2
    plot_seg = np.zeros((np.shape(seg_data)[0], np.shape(seg_data)[1] * 3))
    plot_seg[:, 0:np.shape(seg_data)[1]] = seg_data[..., max(sl - 10, 0), 0]
    plot_seg[:, np.shape(seg_data)[1]:(np.shape(seg_data)[1] * 2)] = seg_data[..., sl, 0]
    plot_seg[:, (np.shape(seg_data)[1] * 2):(np.shape(seg_data)[1] * 3)] = seg_data[..., min(sl + 10, 2*sl-1), 0]
    axs[1].imshow(plot_seg, cmap='gray')
    axs[1].set_axis_off()
    seg_data, seg_affine = load_nifti(wm_path)
    sl = np.shape(seg_data)[2] // 2
    plot_seg = np.zeros((np.shape(seg_data)[0], np.shape(seg_data)[1] * 3))
    plot_seg[:, 0:np.shape(seg_data)[1]] = seg_data[..., max(sl - 10, 0)]
    plot_seg[:, np.shape(seg_data)[1]:(np.shape(seg_data)[1] * 2)] = seg_data[..., sl]
    plot_seg[:, (np.shape(seg_data)[1] * 2):(np.shape(seg_data)[1] * 3)] = seg_data[..., min(sl + 10, 2*sl-1)]
    test = np.ma.masked_where(plot_seg < 0.9, plot_seg)
    axs[1].imshow(test, cmap='hsv', interpolation='none')
    axs[1].set_axis_off()
    plt.tight_layout()
    plt.savefig(qc_path + "/segmentation.jpg", dpi=300, bbox_inches='tight')

    if os.path.isfile(T1_path) and os.path.isfile(T1gibbs_path):
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        anat_data, anat_affine = load_nifti(T1_path)
        sl = np.shape(anat_data)[2] // 2 + 15
        plot_anat = np.zeros((np.shape(anat_data)[0], np.shape(anat_data)[1] * 5))
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., max(sl - 10, 0)]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., max(sl - 5, 0)]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., min(sl + 5, 2*sl-1)]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., min(sl + 10, 2*sl-1)]
        axs[0].imshow(plot_anat, cmap='gray')
        axs[0].set_title('T1')
        axs[0].set_axis_off()
        anat_data, anat_affine = load_nifti(T1gibbs_path)
        sl = np.shape(anat_data)[2] // 2 + 15
        plot_anat = np.zeros((np.shape(anat_data)[0], np.shape(anat_data)[1] * 5))
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., max(sl - 10, 0)]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., max(sl - 5, 0)]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., min(sl + 5, 2*sl-1)]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., min(sl + 10, 2*sl-1)]
        axs[1].imshow(plot_anat, cmap='gray')
        axs[1].set_title('T1 gibbs ringing corrected')
        axs[1].set_axis_off()
        anat_data, anat_affine = load_nifti(T1brain_path)
        sl = np.shape(anat_data)[2] // 2 + 15
        plot_anat = np.zeros((np.shape(anat_data)[0], np.shape(anat_data)[1] * 5))
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., max(sl - 10, 0)]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., max(sl - 5, 0)]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., min(sl + 5, 2*sl-1)]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., min(sl + 10, 2*sl-1)]
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
        plot_anat[:, 0:np.shape(anat_data)[1]] = anat_data[..., max(sl - 10, 0)]
        plot_anat[:, np.shape(anat_data)[1]:(np.shape(anat_data)[1] * 2)] = anat_data[..., max(sl - 5, 0)]
        plot_anat[:, (np.shape(anat_data)[1] * 2):(np.shape(anat_data)[1] * 3)] = anat_data[..., sl]
        plot_anat[:, (np.shape(anat_data)[1] * 3):(np.shape(anat_data)[1] * 4)] = anat_data[..., min(sl + 5, 2*sl-1)]
        plot_anat[:, (np.shape(anat_data)[1] * 4):(np.shape(anat_data)[1] * 5)] = anat_data[..., min(sl + 10, 2*sl-1)]
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

    if not os.path.exists(folder_path + '/subjects/' + patient_path + '/quality_control.pdf'):
        shutil.copyfile(qc_path + '/qc_report.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
    else:
        pdfs = [folder_path + '/subjects/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(folder_path + '/subjects/' + patient_path + '/quality_control_wm.pdf')
        merger.close()

        os.remove(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        os.rename(folder_path + '/subjects/' + patient_path + '/quality_control_wm.pdf',folder_path + '/subjects/' + patient_path + '/quality_control.pdf')


    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/masks/wm_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()

    update_status(folder_path, patient_path, maskType)

def noddi_solo(folder_path, p, maskType="brain_mask_dilated", lambda_iso_diff=3.e-9, lambda_par_diff=1.7e-9, use_amico=False,core_count=1):
    """ Computes the NODDI metrics for a single. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param use_wm_mask: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
    :param lambda_iso_diff: Define the noddi lambda_iso_diff parameters. default=3.e-9
    :param lambda_par_diff: Define the noddi lambda_par_diff parameters. default=1.7e-9
    :param use_amico: If true, use the amico optimizer. default=FALSE
    :param core_count: Number of allocated cpu cores. default=1
    """
    print("[NODDI SOLO] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual NODDI processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs

    assert maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                        "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"

    patient_path = p
    log_prefix = "NODDI SOLO"

    noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi"
    makedir(noddi_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", log_prefix)

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
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")


    # load the mask
    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'
    if os.path.isfile(mask_path):
        mask, _ = load_nifti(mask_path)
    else:
        mask, _ = load_nifti(
            folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")

    # transform the bval, bvecs in a form suited for NODDI
    from dipy.core.gradients import gradient_table
    from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
    acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)

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
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metric1 = np.copy(odi)
    metric2 = np.copy(f_iso)
    qc_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/quality_control"
    makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", log_prefix)

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - NODDI', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sl = np.shape(mse)[2] // 2
    plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., max(sl - 10, 0)]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., max(sl - 5, 0)]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., min(sl + 5, 2*sl-1)]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., min(sl + 10, 2*sl-1)]
    im0 = axs[0].imshow(plot_mse, cmap='gray')
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., max(sl - 10, 0)]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., max(sl - 5, 0)]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., min(sl + 5, 2*sl-1)]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., min(sl + 10, 2*sl-1)]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sl = np.shape(metric1)[2] // 2
    plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5))
    plot_metric1[:, 0:np.shape(metric1)[1]] = metric1[..., max(sl - 10, 0)]
    plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2)] = metric1[..., max(sl - 5, 0)]
    plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3)] = metric1[..., sl]
    plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4)] = metric1[..., min(sl + 5, 2*sl-1)]
    plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5)] = metric1[..., min(sl + 10, 2*sl-1)]
    axs[0].imshow(plot_metric1, cmap='gray')
    axs[0].set_title('Orientation dispersion index')
    axs[0].set_axis_off()
    sl = np.shape(metric2)[2] // 2
    plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
    plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., max(sl - 10, 0)]
    plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., max(sl - 5, 0)]
    plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl]
    plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., min(sl + 5, 2*sl-1)]
    plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., min(sl + 10, 2*sl-1)]
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

    if not os.path.exists(folder_path + '/subjects/' + patient_path + '/quality_control.pdf'):
        shutil.copyfile(qc_path + '/qc_report.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
    else:
        """Merge with QC of preproc""";
        from pypdf import PdfMerger
        pdfs = [folder_path + '/subjects/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(folder_path + '/subjects/' + patient_path + '/quality_control_noddi.pdf')
        merger.close()
        os.remove(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        os.rename(folder_path + '/subjects/' + patient_path + '/quality_control_noddi.pdf',folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f.close()

    update_status(folder_path, patient_path, "noddi")

def noddi_amico_solo(folder_path, p, maskType="brain_mask_dilated"):
    """ Perform noddi amico on a single subject and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi_amico/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param use_wm_mask: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
    """
    print("[NODDI AMICO SOLO] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual NODDI AMICO processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs

    log_prefix = "NODDI AMICO SOLO"

    patient_path = p

    assert maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                        "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"

    noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi_amico"
    makedir(noddi_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi_amico/noddi_amico_logs.txt",
            log_prefix)

    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'
    if os.path.isfile(mask_path):
        mask, _ = load_nifti(mask_path)
    else:
        mask, _ = load_nifti(
            folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")

    import amico
    amico.core.setup()
    ae = amico.Evaluation(study_path=folder_path + '/static_files/noddi_AMICO/',
                          subject=folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi_amico/',
                          output_path=folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi_amico/')

    schemeFile = folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi_amico/' + patient_path + "_NODDI_protocol.scheme"
    dwi_preproc = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc"

    amico.util.fsl2scheme(dwi_preproc + ".bval", dwi_preproc + ".bvec", schemeFilename=schemeFile)

    ae.load_data(dwi_filename=dwi_preproc + ".nii.gz", scheme_filename=schemeFile, mask_filename=wm_path, b0_thr=0)
    ae.set_model("NODDI")
    ae.generate_kernels()
    ae.load_kernels()
    ae.fit()
    ae.save_results(path_suffix=patient_path)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi_amico/noddi_amico_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()

    update_status(folder_path, patient_path, "noddi_amico")


def diamond_solo(folder_path, p, core_count=4, reportOnly=False, maskType="brain_mask_dilated",customDiamond=""):
    """Computes the DIAMOND metrics for a single subject. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/diamond/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param core_count: Number of allocated cpu cores. default=4
    :param use_wm_mask: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
    :param customDiamond: If not empty, the string define custom value for --ntensors, -reg, --estimb0, --automose, --mosemodels, --fascicle, --waterfraction, --waterDiff, --omtm, --residuals, --fractions_sumto1, --verbose and --log
    """
    log_prefix = "DIAMOND SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual DIAMOND processing for patient %s \n" % p)
    patient_path = p

    assert maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                        "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"


    diamond_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond"
    makedir(diamond_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt",
            log_prefix)
    if not (os.path.exists(diamond_path)):
        try:
            os.makedirs(diamond_path)
        except OSError:
            print("Creation of the directory %s failed" % diamond_path)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % diamond_path)
            f.close()
        else:
            print("Successfully created the directory %s " % diamond_path)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % diamond_path)
            f.close()

    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'

    if os.path.isfile(mask_path):
        mask = mask_path
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": " + maskType + " is used \n")
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": " + maskType + " is used \n")
        f.close()
    else:
        mask = folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_brain_mask_dilated.nii.gz'
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": dilated brain mask based on diffusion data is used \n")
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": dilated brain mask based on diffusion data is used \n")
        f.close()

    if not reportOnly:
        bashCommand = 'export OMP_NUM_THREADS=' + str(
            core_count) + ' ; crlDCIEstimate --input "' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz' + '" --output "' + folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/diamond/' + patient_path + '_diamond.nii.gz' + '" --mask "' + mask + '" --proc ' + str(
            core_count)
        if customDiamond == "" or (customDiamond is None) or (isinstance(customDiamond, str) and len(customDiamond) < 3):
            bashCommand = bashCommand + ' --ntensors 2 --reg 1.0 --estimb0 1 --automose aicu --mosemodels --fascicle diamondcyl --waterfraction 1 --waterDiff 0.003 --omtm 1 --residuals --verbose 1 --log'
        else:
            bashCommand = bashCommand + customDiamond

        import subprocess
        bashcmd = bashCommand.split()
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": crlDCIEstimate launched for patient %s \n" % p + " with bash command " + bashCommand)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": crlDCIEstimate launched for patient %s \n" % p + " with bash command " + bashCommand)
        f.close()

        diamond_log = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=diamond_log,
                                   stderr=subprocess.STDOUT)

        output, error = process.communicate()
        diamond_log.close()


    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dipy.io.image import load_nifti

    mosemap, _ = load_nifti(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + "_diamond_mosemap.nii.gz")
    fractions, _ = load_nifti(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + "_diamond_fractions.nii.gz")
    residual, _ = load_nifti(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + "_diamond_residuals.nii.gz")
    data, _ = load_nifti(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz')
    residual = np.squeeze(residual)
    reconstructed = data - residual

    metric1 = np.copy(mosemap)
    metric2 = np.copy(fractions[...,0])
    qc_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/quality_control"
    makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", log_prefix)

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
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., max(sl - 10, 0)]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., max(sl - 5, 0)]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., min(sl + 5, 2*sl-1)]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., min(sl + 10, 2*sl-1)]
    im0 = axs[0].imshow(plot_mse, cmap='gray')#, vmin=0, vmax=np.min([np.max(residual),np.max(mse)]))
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., max(sl - 10, 0)]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., max(sl - 5, 0)]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., min(sl + 5, 2*sl-1)]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., min(sl + 10, 2*sl-1)]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sl = np.shape(metric1)[2] // 2
    plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5))
    plot_metric1[:, 0:np.shape(metric1)[1]] = metric1[..., max(sl - 10, 0)]
    plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2)] = metric1[..., max(sl - 5, 0)]
    plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3)] = metric1[..., sl]
    plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4)] = metric1[..., min(sl + 5, 2*sl-1)]
    plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5)] = metric1[..., min(sl + 10, 2*sl-1)]
    axs[0].imshow(plot_metric1, cmap='gray')
    axs[0].set_title('Mosemap')
    axs[0].set_axis_off()
    sl = np.shape(metric2)[2] // 2
    plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
    plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., max(sl - 10, 0), 0]
    plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., max(sl - 5, 0), 0]
    plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl, 0]
    plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., min(sl + 5, 2*sl-1), 0]
    plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., min(sl + 10, 2*sl-1), 0]
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

    if not os.path.exists(folder_path + '/subjects/' + patient_path + '/quality_control.pdf'):
        shutil.copyfile(qc_path + '/qc_report.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
    else:
        """Merge with QC of preproc""";
        from pypdf import PdfMerger
        pdfs = [folder_path + '/subjects/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(folder_path + '/subjects/' + patient_path + '/quality_control_diamond.pdf')
        merger.close()
        os.remove(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        os.rename(folder_path + '/subjects/' + patient_path + '/quality_control_diamond.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f.close()

    update_status(folder_path, patient_path, "diamond")

def mf_solo(folder_path, p, dictionary_path, core_count=1, maskType="brain_mask_dilated",
            report=True, csf_mask=True, ear_mask=False, peaksType="MSMT-CSD",
            mfdir=None, output_filename: str = ""):
    """Perform microstructure fingerprinting and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/mf/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param dictionary_path: Path to the dictionary of fingerprints (mandatory).
    :param CSD_bvalue: If the DIAMOND outputs are not available, the fascicles directions are estimated using a CSD with the images at the b-values specified in this argument. default=None
    :param core_count: Define the number of available core. default=1
    :param use_wm_mask: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
    :param output_filename: str. Specify output filename.
    """

    log_prefix = "MF SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual microstructure fingerprinting processing for patient %s \n" % p, flush = True)
    patient_path = p

    mfdir = "mf" if mfdir is None else mfdir
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/" + mfdir + "/mf_logs.txt", "a+")

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual microstructure fingerprinting processing for patient %s \n" % p)

    mf_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/" + mfdir
    makedir(mf_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/" + mfdir + "/mf_logs.txt", log_prefix)

    diamond_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond"
    odf_msmtcsd_path = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD"
    odf_csd_path = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD"

    assert maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                        "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"

    if peaksType == "DIAMOND":
        assert os.path.exists(diamond_path), "DIAMOND path does not exist"
    if peaksType == "MSMT-CSD":
        assert os.path.exists(odf_msmtcsd_path + '/' + patient_path + "_MSMT-CSD_peaks.nii.gz") and os.path.exists(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks_amp.nii.gz'), "MSMT-CSD path does not exist"
    if peaksType == "CSD":
        assert os.path.exists(odf_csd_path + '/' + patient_path + "_CSD_peaks.nii.gz") and os.path.exists(odf_csd_path + '/' + patient_path + '_CSD_values.nii.gz'), "CSD path does not exist"

    assert peaksType in ["MSMT-CSD", "CSD", "DIAMOND"], "peaksType must be one of the following: MSMT-CSD, CSD, DIAMOND"

    # imports
    import microstructure_fingerprinting as mf
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs

    # load the data
    data, affine = load_nifti(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'
    if os.path.isfile(mask_path):
        mask, _ = load_nifti(mask_path)
    else:
        mask, _ = load_nifti(
            folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")

    # compute numfasc and peaks
    if os.path.exists(diamond_path) and peaksType == "DIAMOND":
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Diamond Path found! MF will be based on diamond \n")
        tensor_files0 = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t0.nii.gz'
        tensor_files1 = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_t1.nii.gz'
        fracs_file = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/" + patient_path + '_diamond_fractions.nii.gz'
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=None, frac2=None,
                                                 mu1=tensor_files0, mu2=tensor_files1,
                                                 peakmode='tensor', mask=mask,
                                                 frac12=fracs_file)
        color_order = 'rgb'
    elif peaksType == "CSD":
        csd_peaks_peak_dirs, _ = load_nifti(odf_csd_path + '/' + patient_path + '_CSD_peaks.nii.gz')
        csd_peaks_peak_values, _ = load_nifti(odf_csd_path + '/' + patient_path + '_CSD_values.nii.gz')
        numfasc_2 = (np.sum(csd_peaks_peak_values[:, :, :, 0] > 0.15)
                     + np.sum(csd_peaks_peak_values[:, :, :, 1] > 0.15))
        print("Approximate number of non empty voxel: ", numfasc_2, flush=True)

        normPeaks0 = csd_peaks_peak_dirs[..., 0, :]
        normPeaks1 = csd_peaks_peak_dirs[..., 1, :]
        for i in range(np.shape(csd_peaks_peak_dirs)[0]):
            for j in range(np.shape(csd_peaks_peak_dirs)[1]):
                for k in range(np.shape(csd_peaks_peak_dirs)[2]):
                    norm = np.sqrt(np.sum(normPeaks0[i, j, k, :] ** 2))
                    normPeaks0[i, j, k, :] = normPeaks0[i, j, k, :] / norm
                    norm = np.sqrt(np.sum(normPeaks1[i, j, k, :] ** 2))
                    normPeaks1[i, j, k, :] = normPeaks1[i, j, k, :] / norm
        mu1 = normPeaks0
        mu2 = normPeaks1
        frac1 = csd_peaks_peak_values[..., 0]
        frac2 = csd_peaks_peak_values[..., 1]
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=frac1, frac2=frac2,
                                                 mu1=mu1, mu2=mu2,
                                                 peakmode='peaks',
                                                 mask=mask, frac12=None)
        color_order = 'rgb'
    elif peaksType == "MSMT-CSD":

        from elikopy.utils import get_acquisition_view
        from scipy.linalg import polar

        msmtcsd_peaks_peak_dirs, _ = load_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks.nii.gz')
        msmtcsd_peaks_peak_values, _ = load_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks_amp.nii.gz')

        numfasc_2 = (np.sum(msmtcsd_peaks_peak_values[:, :, :, 0] > 0.15)
                     + np.sum(msmtcsd_peaks_peak_values[:, :, :, 1] > 0.15))
        print("Approximate number of non empty voxel: ", numfasc_2, flush=True)

        # Polar decomposition
        u, _ = polar(affine[0:3, 0:3])

        # Number of fixels
        K = int(msmtcsd_peaks_peak_dirs.shape[-1]/3)

        # Rotate peaks to go from Mrtrix convention to Python
        for k in range(K):
            msmtcsd_peaks_peak_dirs[..., 3*k:3*k+3] = msmtcsd_peaks_peak_dirs[..., 3*k:3*k+3] @ u

        # TODO : Automate RGB selection to all views
        view = get_acquisition_view(affine)
        if view == 'axial':
            color_order = 'rgb'
        elif view == 'sagittal':
            color_order = 'brg'
        else:
            color_order = 'rgb'
            print("Warning: No correction found for the RGB colors of the current acquisition view. Defaulting to axial (RGB).")

        normPeaks0 = msmtcsd_peaks_peak_dirs[..., 0:3]
        normPeaks1 = msmtcsd_peaks_peak_dirs[..., 3:6]
        for i in range(np.shape(msmtcsd_peaks_peak_dirs)[0]):
            for j in range(np.shape(msmtcsd_peaks_peak_dirs)[1]):
                for k in range(np.shape(msmtcsd_peaks_peak_dirs)[2]):
                    norm = np.sqrt(np.sum(normPeaks0[i, j, k, :] ** 2))
                    normPeaks0[i, j, k, :] = normPeaks0[i, j, k, :] / norm
                    norm = np.sqrt(np.sum(normPeaks1[i, j, k, :] ** 2))
                    normPeaks1[i, j, k, :] = normPeaks1[i, j, k, :] / norm
        mu1 = normPeaks0
        mu2 = normPeaks1
        frac1 = msmtcsd_peaks_peak_values[..., 0]
        frac2 = msmtcsd_peaks_peak_values[..., 1]
        (peaks, numfasc) = mf.cleanup_2fascicles(frac1=frac1, frac2=frac2,
                                                 mu1=mu1, mu2=mu2,
                                                 peakmode='peaks',
                                                 mask=mask, frac12=None)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Loading of MF dic\n")
    # get the dictionary
    mf_model = mf.MFModel(dictionary_path)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of fitting\n")
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of fitting\n", flush=True)

    import time

    parallel = False if core_count == 1 else True
    start = time.time()
    # Fit to data:
    MF_fit = mf_model.fit(data, mask, numfasc, peaks=peaks, bvals=bvals,
                          bvecs=bvecs, csf_mask=csf_mask, ear_mask=ear_mask,
                          verbose=3, parallel=parallel)  # , cpus=core_count)

    end = time.time()
    stats_header = "patient_id, elapsed_time, core_count"
    stats_val = p + ", " + str(end - start) + ", " + str(core_count)
    print(stats_header, flush=True)
    print(stats_val, flush=True)
    f.write(stats_header)
    f.write(stats_val)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": End of fitting\n")
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": End of fitting\n", flush=True)
    # extract info
    frac_f0 = MF_fit.frac_f0
    fvf_tot = MF_fit.fvf_tot
    MSE = MF_fit.MSE
    R2 = MF_fit.R2

    if len(output_filename) > 0:
        filename = '_mf_'+output_filename
    else:
        filename = '_mf'

    MF_fit.write_nifti(mf_path + '/' + patient_path + filename+'.nii.gz', affine=affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Saving of unravel pseudotensor dic\n", flush = True)
    # Export pseudo tensor
    frac = 0
    frac_list = []
    peaks_list = []
    fvf_list = []
    import nibabel as nib
    from elikopy.utils import peak_to_tensor
    while os.path.exists(mf_path + '/' + patient_path + filename+'_peak_f'+str(frac)+'.nii.gz') and os.path.exists(mf_path + '/' + patient_path + filename+'_frac_f' + str(frac) + '.nii.gz'):
        peaks_path = mf_path + '/' + patient_path + filename+'_peak_f' + str(frac) + '.nii.gz'
        frac_path = mf_path + '/' + patient_path + filename+'_frac_f' + str(frac) + '.nii.gz'
        fvf_path = mf_path + '/' + patient_path + filename+'_fvf_f' + str(frac) + '.nii.gz'
        img_mf_peaks = nib.load(peaks_path)
        img_mf_frac = nib.load(frac_path)
        hdr = img_mf_peaks.header
        pixdim = hdr['pixdim'][1:4]
        t = peak_to_tensor(img_mf_peaks.get_fdata(),norm=None,pixdim=pixdim)
        t_normed = peak_to_tensor(img_mf_peaks.get_fdata(), norm=img_mf_frac.get_fdata(),pixdim=pixdim)
        hdr['dim'][0] = 5  # 4 scalar, 5 vector
        hdr['dim'][4] = 1  # 3
        hdr['dim'][5] = 6  # 1
        hdr['regular'] = b'r'
        hdr['intent_code'] = 1005
        save_nifti(mf_path + '/' + patient_path + filename+'_peak_f' + str(frac) + '_pseudoTensor.nii.gz', t, img_mf_peaks.affine, hdr)
        save_nifti(mf_path + '/' + patient_path + filename+'_peak_f' + str(frac) + '_pseudoTensor_normed.nii.gz', t_normed, img_mf_peaks.affine, hdr)

        import unravel.utils
        RGB_peak = unravel.utils.peaks_to_RGB(img_mf_peaks.get_fdata(), order=color_order)
        save_nifti(mf_path + '/' + patient_path + filename+'_peak_f' + str(frac) + '_RGB.nii.gz', RGB_peak, img_mf_frac.affine)
        peaks_list.append(img_mf_peaks.get_fdata())
        frac_list.append(img_mf_frac.get_fdata())

        if os.path.exists(fvf_path):
            img_mf_fvf = nib.load(fvf_path)
            fvf_list.append(img_mf_fvf.get_fdata())

        frac = frac + 1
    
    peaks=np.stack(peaks_list,axis=-1)
    frac=np.stack(frac_list,axis=-1)
    fvf=np.stack(fvf_list,axis=-1)
    if len(frac_list) > 0 and len(peaks_list) > 0:
        RGB_peaks_frac = unravel.utils.peaks_to_RGB(peaks, frac, order=color_order)
        save_nifti(mf_path + '/' + patient_path + filename+'_peak_tot_RGB_frac.nii.gz', RGB_peaks_frac, img_mf_frac.affine)

    if len(frac_list) > 0 and len(peaks_list) > 0 and len(fvf_list)>0:
        RGB_peaks_frac_fvf = unravel.utils.peaks_to_RGB(peaks, frac, fvf, order=color_order)
        save_nifti(mf_path + '/' + patient_path + filename+'_peak_tot_RGB_frac_fvf.nii.gz', RGB_peaks_frac_fvf, img_mf_frac.affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p, flush = True)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    if report:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mse = np.copy(MSE)
        metric1 = np.copy(fvf_tot)
        metric2 = np.copy(frac_f0)
        qc_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/" + mfdir + "/quality_control"
        makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/" + mfdir + "/mf_logs.txt",
                log_prefix)

        fig, axs = plt.subplots(2, 1, figsize=(2, 1))
        fig.suptitle('Elikopy : Quality control report - Microstructure fingerprinting', fontsize=50)
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        sl = np.shape(mse)[2] // 2
        plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
        plot_mse[:, 0:np.shape(mse)[1]] = mse[..., max(sl - 10, 0)]
        plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., max(sl - 5, 0)]
        plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
        plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., min(sl + 5, 2*sl-1)]
        plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., min(sl + 10, 2*sl-1)]
        im0 = axs[0].imshow(plot_mse, cmap='gray')
        axs[0].set_title('MSE')
        axs[0].set_axis_off()
        fig.colorbar(im0, ax=axs[0], orientation='horizontal')
        sl = np.shape(R2)[2] // 2
        plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
        plot_R2[:, 0:np.shape(R2)[1]] = R2[..., max(sl - 10, 0)]
        plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., max(sl - 5, 0)]
        plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
        plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., min(sl + 5, 2*sl-1)]
        plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., min(sl + 10, 2*sl-1)]
        im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
        axs[1].set_title('R2')
        axs[1].set_axis_off()
        fig.colorbar(im1, ax=axs[1], orientation='horizontal');
        plt.tight_layout()
        plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        sl = np.shape(metric1)[2] // 2
        plot_metric1 = np.zeros((np.shape(metric1)[0], np.shape(metric1)[1] * 5))
        plot_metric1[:, 0:np.shape(metric1)[1]] = metric1[..., max(sl - 10, 0)]
        plot_metric1[:, np.shape(metric1)[1]:(np.shape(metric1)[1] * 2)] = metric1[..., max(sl - 5, 0)]
        plot_metric1[:, (np.shape(metric1)[1] * 2):(np.shape(metric1)[1] * 3)] = metric1[..., sl]
        plot_metric1[:, (np.shape(metric1)[1] * 3):(np.shape(metric1)[1] * 4)] = metric1[..., min(sl + 5, 2*sl-1)]
        plot_metric1[:, (np.shape(metric1)[1] * 4):(np.shape(metric1)[1] * 5)] = metric1[..., min(sl + 10, 2*sl-1)]
        axs[0].imshow(plot_metric1, cmap='gray')
        axs[0].set_title('fvf_tot')
        axs[0].set_axis_off()
        sl = np.shape(metric2)[2] // 2
        plot_metric2 = np.zeros((np.shape(metric2)[0], np.shape(metric2)[1] * 5))
        plot_metric2[:, 0:np.shape(metric2)[1]] = metric2[..., max(sl - 10, 0)]
        plot_metric2[:, np.shape(metric2)[1]:(np.shape(metric2)[1] * 2)] = metric2[..., max(sl - 5, 0)]
        plot_metric2[:, (np.shape(metric2)[1] * 2):(np.shape(metric2)[1] * 3)] = metric2[..., sl]
        plot_metric2[:, (np.shape(metric2)[1] * 3):(np.shape(metric2)[1] * 4)] = metric2[..., min(sl + 5, 2*sl-1)]
        plot_metric2[:, (np.shape(metric2)[1] * 4):(np.shape(metric2)[1] * 5)] = metric2[..., min(sl + 10, 2*sl-1)]
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

        if not os.path.exists(folder_path + '/subjects/' + patient_path + '/quality_control.pdf'):
            shutil.copyfile(qc_path + '/qc_report.pdf',
                            folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        else:
            """Merge with QC of preproc""";
            from pypdf import PdfMerger
            pdfs = [folder_path + '/subjects/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
            merger = PdfMerger()
            for pdf in pdfs:
                merger.append(pdf)
            merger.write(folder_path + '/subjects/' + patient_path + '/quality_control_mf.pdf')
            merger.close()
            os.remove(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
            os.rename(folder_path + '/subjects/' + patient_path + '/quality_control_mf.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
            f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/"+mfdir+"/mf_logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
            f.close()


    update_status(folder_path, patient_path, "fingerprinting")

def odf_csd_solo(folder_path, p, num_peaks=2, peaks_threshold = .25, CSD_bvalue=None, core_count=1, maskType="brain_mask_dilated", report=True, CSD_FA_treshold=0.7, return_odf=False):
    """Perform microstructure fingerprinting and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/mf/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param dictionary_path: Path to the dictionary of fingerprints (mandatory).
    :param CSD_bvalue: If the DIAMOND outputs are not available, the fascicles directions are estimated using a CSD with the images at the b-values specified in this argument. default=None
    :param core_count: Define the number of available core. default=1
    :param use_wm_mask: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
    """
    log_prefix = "ODF CSD SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual ODF CSD for patient %s \n" % p, flush = True)
    patient_path = p

    assert maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                        "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"

    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/CSD_logs.txt", "a+")

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual ODF CSD processing for patient %s \n" % p)

    odf_csd_path = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD"
    makedir(odf_csd_path, folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/CSD_logs.txt", log_prefix)

    # imports
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    from dipy.direction import peaks_from_model
    from dipy.data import default_sphere

    # load the data
    data, affine = load_nifti(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'
    if os.path.isfile(mask_path):
        mask, _ = load_nifti(mask_path)
    else:
        mask, _ = load_nifti(
            folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")

    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)

    if CSD_bvalue is not None:
        print("Max CSD bvalue is", CSD_bvalue)
        sel_b = np.logical_or(bvals == 0, np.logical_and((CSD_bvalue - 5) <= bvals, bvals <= (CSD_bvalue + 5)))
        data_CSD = data[..., sel_b]
        gtab_CSD = gradient_table(bvals[sel_b], bvecs[sel_b], b0_threshold=b0_threshold)
    else:
        data_CSD = data
        gtab_CSD = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    from dipy.reconst.csdeconv import auto_response_ssst
    response, ratio = auto_response_ssst(gtab_CSD, data_CSD, roi_radii=10, fa_thr=CSD_FA_treshold)

    csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response, sh_order=8)
    csd_peaks = peaks_from_model(npeaks=num_peaks, model=csd_model, data=data_CSD, sphere=default_sphere,
                                 relative_peak_threshold=peaks_threshold, min_separation_angle=25, parallel=False, mask=mask,
                                 normalize_peaks=True,return_odf=return_odf,return_sh=True)

    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peaks.nii.gz', csd_peaks.peak_dirs, affine)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_values.nii.gz', csd_peaks.peak_values, affine)
    if return_odf:
        save_nifti(odf_csd_path + '/' + patient_path + '_CSD_ODF.nii.gz', csd_peaks.odf, affine)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_SH_ODF.nii.gz', csd_peaks.shm_coeff, affine)

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
    peaks1 = csd_peaks.peak_dirs[..., 0, :]
    peaks2 = csd_peaks.peak_dirs[..., 1, :]
    frac1 = csd_peaks.peak_values[..., 0]
    frac2 = csd_peaks.peak_values[..., 1]


    # Export pseudo tensor
    frac = 0
    frac_list = []
    peaks_list = []
    fvf_list = []
    import nibabel as nib
    from elikopy.utils import peak_to_tensor

    peaks_1_2 = np.concatenate((peaks1,peaks2))
    frac_1_2 = np.concatenate((frac1,frac2))


    img_mf_peaks = nib.load(odf_csd_path + '/' + patient_path + '_CSD_peaks.nii.gz')
    img_mf_frac = nib.load(odf_csd_path + '/' + patient_path + '_CSD_values.nii.gz')
    hdr = img_mf_peaks.header
    pixdim = hdr['pixdim'][1:4]

    t_p1 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 0, :], norm=None, pixdim=pixdim)
    t_normed_p1 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 0, :], norm=img_mf_frac.get_fdata()[..., 0], pixdim=pixdim)
    t_p2 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 1, :], norm=None, pixdim=pixdim)
    t_normed_p2 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 1, :], norm=img_mf_frac.get_fdata()[..., 1], pixdim=pixdim)

    hdr['dim'][0] = 5  # 4 scalar, 5 vector
    hdr['dim'][4] = 1  # 3
    hdr['dim'][5] = 6  # 1
    hdr['regular'] = b'r'
    hdr['intent_code'] = 1005

    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f1_pseudoTensor.nii.gz', t_p1, affine, hdr)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f1_pseudoTensor_normed.nii.gz', t_normed_p1, affine, hdr)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f2_pseudoTensor.nii.gz', t_p2, affine, hdr)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f2_pseudoTensor_normed.nii.gz', t_normed_p2, affine, hdr)

    import unravel.utils
    RGB_peak = unravel.utils.peaks_to_RGB(peaks1)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f1_RGB.nii.gz', RGB_peak, affine)
    RGB_peak_frac = unravel.utils.peaks_to_RGB(peaks1, frac1)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f1_RGB_frac.nii.gz', RGB_peak_frac, affine)

    RGB_peak = unravel.utils.peaks_to_RGB(peaks2)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f2_RGB.nii.gz', RGB_peak, affine)
    RGB_peak_frac = unravel.utils.peaks_to_RGB(peaks2, frac2)
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_f2_RGB_frac.nii.gz', RGB_peak_frac, affine)


    RGB_peaks = unravel.utils.peaks_to_RGB(np.stack([peaks1, peaks2],axis=-1))
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_tot_RGB.nii.gz', RGB_peaks, affine)
    RGB_peaks_frac = unravel.utils.peaks_to_RGB(np.stack([peaks1, peaks2],axis=-1),
                                                np.stack([frac1, frac2],axis=-1))
    save_nifti(odf_csd_path + '/' + patient_path + '_CSD_peak_tot_RGB_frac.nii.gz', RGB_peaks_frac, affine)


    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()

    update_status(folder_path, patient_path, "odf_csd")

def odf_msmtcsd_solo(folder_path, p, core_count=1, num_peaks=2, peaks_threshold = 0.25, report=True, maskType="brain_mask_dilated"):
    """Perform MSMT CSD odf computation and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/ODF/MSMT-CSD/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param core_count: Define the number of available core. default=1
    """
    log_prefix = "ODF MSMT-CSD SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual ODF MSMT-CSD for patient %s \n" % p, flush = True)
    patient_path = p

    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD/MSMT-CSD_logs.txt", "a+")

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual ODF MSMT-CSD processing for patient %s \n" % p)

    odf_msmtcsd_path = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD"
    makedir(odf_msmtcsd_path, folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD/MSMT-CSD_logs.txt", log_prefix)

    dwi2response_cmd = 'dwi2response dhollander -info ' + \
                       '-nthreads ' + str(core_count) + '  -fslgrad ' + \
                       folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec " + \
                       folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval " + \
                       folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz " + \
                       odf_msmtcsd_path + '/' + patient_path + '_dhollander_WM_response.txt ' + \
                       odf_msmtcsd_path + '/' + patient_path + '_dhollander_GM_response.txt ' + \
                       odf_msmtcsd_path + '/' + patient_path + '_dhollander_CSF_response.txt -force ; '

    dwi2fod_cmd = 'dwi2fod msmt_csd -info ' + \
                  '-nthreads ' + str(core_count) + ' -mask ' + \
                  folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_" + maskType + ".nii.gz " + \
                  folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz " + \
                  '-fslgrad ' + \
                  folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec " + \
                  folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval " + \
                  odf_msmtcsd_path + '/' + patient_path + '_dhollander_WM_response.txt ' + \
                  odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_WM_ODF.nii.gz ' + \
                  odf_msmtcsd_path + '/' + patient_path + '_dhollander_GM_response.txt ' + \
                  odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_GM.nii.gz ' + \
                  odf_msmtcsd_path + '/' + patient_path + '_dhollander_CSF_response.txt ' + \
                  odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_CSF.nii.gz -force ; '

    bashCommand = 'export OMP_NUM_THREADS=' + str(core_count) + ' ; ' + \
                  dwi2response_cmd + \
                  dwi2fod_cmd


    import subprocess
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": mrtrix ODF MSMT-CSD launched for patient %s \n" % p + " with bash command " + bashCommand)

    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=f,
                               stderr=subprocess.STDOUT)

    output, error = process.communicate()

    sh2peaks_cmd = "sh2peaks -force -nthreads " + str(core_count) + \
                    " -num " + str(num_peaks) + " " + \
                    odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_WM_ODF.nii.gz ' + \
                    odf_msmtcsd_path + '/' + patient_path + "_MSMT-CSD_peaks.nii.gz ; "

    peaks2amp_cmd = "peaks2amp -force -nthreads " + str(core_count) + " " + \
                     odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks.nii.gz ' + \
                     odf_msmtcsd_path + '/' + patient_path + "_MSMT-CSD_peaks_amp.nii.gz ; "

    bashCommand = 'export OMP_NUM_THREADS=' + str(core_count) + ' ; ' + sh2peaks_cmd + peaks2amp_cmd

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": mrtrix ODF MSMT-CSD postprocessing launched for patient %s \n" % p + " with bash command " + bashCommand)

    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=f,
                               stderr=subprocess.STDOUT)

    output, error = process.communicate()

    from dipy.io.image import load_nifti, save_nifti

    # Export pseudo tensor
    from elikopy.utils import peak_to_tensor

    peaks_1_2, affine = load_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks.nii.gz')
    frac_1_2, _ = load_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks_amp.nii.gz')

    import nibabel as nib

    img_mf_peaks = nib.load(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks.nii.gz')
    img_mf_frac = nib.load(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peaks_amp.nii.gz')
    hdr = img_mf_peaks.header
    pixdim = hdr['pixdim'][1:4]

    t_p1 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 0:3], norm=None, pixdim=pixdim)
    t_normed_p1 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 0:3], norm=img_mf_frac.get_fdata()[..., 0],
                                 pixdim=pixdim)
    t_p2 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 3:6], norm=None, pixdim=pixdim)
    t_normed_p2 = peak_to_tensor(img_mf_peaks.get_fdata()[..., 3:6], norm=img_mf_frac.get_fdata()[..., 1],
                                 pixdim=pixdim)

    hdr['dim'][0] = 5  # 4 scalar, 5 vector
    hdr['dim'][4] = 1  # 3
    hdr['dim'][5] = 6  # 1
    hdr['regular'] = b'r'
    hdr['intent_code'] = 1005

    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f1_pseudoTensor.nii.gz', t_p1, affine, hdr)
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f1_pseudoTensor_normed.nii.gz', t_normed_p1, affine,
               hdr)
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f2_pseudoTensor.nii.gz', t_p2, affine, hdr)
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f2_pseudoTensor_normed.nii.gz', t_normed_p2, affine,
               hdr)

    import unravel.utils

    RGB_peak = unravel.utils.peaks_to_RGB(peaks_1_2[:,:,:,0:3])
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f1_RGB.nii.gz', RGB_peak, affine)
    RGB_peak_frac = unravel.utils.peaks_to_RGB(peaks_1_2[:,:,:,0:3], frac_1_2[:, :, :, 0])
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f1_RGB_frac.nii.gz', RGB_peak_frac, affine)

    RGB_peak = unravel.utils.peaks_to_RGB(peaks_1_2[:,:,:,3:6])
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f2_RGB.nii.gz', RGB_peak, affine)
    RGB_peak_frac = unravel.utils.peaks_to_RGB(peaks_1_2[:,:,:,3:6], frac_1_2[:, :, :, 1])
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_f2_RGB_frac.nii.gz', RGB_peak_frac, affine)


    RGB_peaks = unravel.utils.peaks_to_RGB(np.stack([peaks_1_2[:,:,:,0:3], peaks_1_2[:,:,:,3:6]],axis=-1))
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_tot_RGB.nii.gz', RGB_peaks, affine)
    RGB_peaks_frac = unravel.utils.peaks_to_RGB(np.stack([peaks_1_2[:,:,:,0:3], peaks_1_2[:,:,:,3:6]],axis=-1),
                                                np.stack([frac_1_2[:, :, :, 0], frac_1_2[:, :, :, 1]],axis=-1))
    save_nifti(odf_msmtcsd_path + '/' + patient_path + '_MSMT-CSD_peak_tot_RGB_frac.nii.gz', RGB_peaks_frac, affine)


    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)

    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)

    f.close()

    update_status(folder_path, patient_path, "odf_msmtcsd")



def ivim_solo(folder_path, p, core_count=1, G1Ball_2_lambda_iso=7e-9, G1Ball_1_lambda_iso=[.5e-9, 6e-9], maskType="brain_mask_dilated"):
    """ Computes the IVIM metrics for a single. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/ivim/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param use_wm_mask: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
    :param use_amico: If true, use the amico optimizer. default=FALSE
    :param core_count: Number of allocated cpu cores. default=1
    """
    print("[IVIM SOLO] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual ivim processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs

    patient_path = p
    log_prefix = "IVIM SOLO"

    ivim_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/ivim"
    makedir(ivim_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/ivim/ivim_logs.txt", log_prefix)

    from dmipy.core.modeling_framework import MultiCompartmentModel
    from dmipy.signal_models.gaussian_models import G1Ball

    assert maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                        "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"

    # initialize the compartments model and build the ivim model
    from dmipy.core.modeling_framework import MultiCompartmentModel
    from dmipy.signal_models.gaussian_models import G1Ball
    ivim_mod = MultiCompartmentModel([G1Ball(), G1Ball()])

    # fix the isotropic diffusivity
    ivim_mod.set_fixed_parameter(
        'G1Ball_2_lambda_iso', G1Ball_2_lambda_iso)  # Following Gurney-Champion 2016
    ivim_mod.set_parameter_optimization_bounds(
        'G1Ball_1_lambda_iso', G1Ball_1_lambda_iso)  # Following Gurney-Champion 2016

    # load the data
    data, affine = load_nifti(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")

    # load data mask
    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'
    if os.path.isfile(mask_path):
        mask, _ = load_nifti(mask_path)
    else:
        mask, _ = load_nifti(
            folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")

    # transform the bval, bvecs in a form suited for ivim
    from dipy.core.gradients import gradient_table
    from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
    acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)

    print("Start of ivim_fit_Dfixed for patient %s \n" % p)
    # fit the model to the data
    ivim_fit_Dfixed = ivim_mod.fit(acquisition_scheme=acq_scheme_dmipy, data=data, mask=mask, use_parallel_processing=True, number_of_processors=core_count)

    #from dipy.reconst.ivim import IvimModel
    #ivimmodel_dipy = IvimModel(gtab_dipy)
    #ivim_fit_dipy = ivimmodel_dipy.fit(data)

    # exctract the metrics
    fitted_parameters = ivim_fit_Dfixed.fitted_parameters

    D_diffBall = fitted_parameters["G1Ball_1_lambda_iso"] # DiffusionBall diffusitivy
    f_BloodBall = fitted_parameters["partial_volume_0"] # BloodBall fraction
    f_DiffusionBall = fitted_parameters["partial_volume_1"] # DiffusionBall fraction
    mse = ivim_fit_Dfixed.mean_squared_error(data)
    R2 = ivim_fit_Dfixed.R2_coefficient_of_determination(data)

    # save the nifti
    save_nifti(ivim_path + '/' + patient_path + '_ivim_D_DiffBall.nii.gz', D_diffBall.astype(np.float32), affine)
    save_nifti(ivim_path + '/' + patient_path + '_ivim_f_BloodBall.nii.gz', f_BloodBall.astype(np.float32), affine)
    save_nifti(ivim_path + '/' + patient_path + '_ivim_f_DiffusionBall.nii.gz', f_DiffusionBall.astype(np.float32), affine)
    save_nifti(ivim_path + '/' + patient_path + '_ivim_mse.nii.gz', mse.astype(np.float32), affine)
    save_nifti(ivim_path + '/' + patient_path + '_ivim_R2.nii.gz', R2.astype(np.float32), affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/ivim/ivim_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    qc_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/ivim/quality_control"
    makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/ivim/ivim_logs.txt", log_prefix)

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - IVIM', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sl = np.shape(mse)[2] // 2
    plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., max(sl - 10, 0)]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., max(sl - 5, 0)]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., min(sl + 5, 2*sl-1)]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., min(sl + 10, 2*sl-1)]
    im0 = axs[0].imshow(plot_mse, cmap='gray')
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., max(sl - 10, 0)]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., max(sl - 5, 0)]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., min(sl + 5, 2*sl-1)]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., min(sl + 10, 2*sl-1)]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');


    """Save as a pdf"""

    elem = [qc_path + "/title.jpg", qc_path + "/error.jpg"]

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
            self.cell(60, 1, 'Quality control report - ivim', 0, 0, 'R')
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

    if not os.path.exists(folder_path + '/subjects/' + patient_path + '/quality_control.pdf'):
        shutil.copyfile(qc_path + '/qc_report.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
    else:
        """Merge with QC of preproc""";
        from pypdf import PdfMerger
        pdfs = [folder_path + '/subjects/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(folder_path + '/subjects/' + patient_path + '/quality_control_ivim.pdf')
        merger.close()
        os.remove(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        os.rename(folder_path + '/subjects/' + patient_path + '/quality_control_ivim.pdf',folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/ivim/ivim_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f.close()

    update_status(folder_path, patient_path, "ivim")


def tracking_solo(folder_path:str, p:str, streamline_number:int=100000,
                  max_angle:int=15, cutoff:float=0.1, msmtCSD:bool=True,
                  output_filename:str='tractogram',core_count:int=1,
                  maskType:str="brain_mask",save_as_trk=False):
    """ Computes the whole brain tractogram of a single patient based on the fod obtained from msmt-CSD.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param streamline_number: Number of streamlines in the final tractogram. default=100000
    :param max_angle: Maximum angle between two tractography steps. default=15
    :param cutoff: Value below which streamlines do not propagate. default=0.1
    :param msmtCSD: boolean. If True then uses ODF from msmt-CSD, if False from CSD. default=True
    :param output_filename: str. Specify output filename for tractogram.
    :param maskType: str. Specify a masking region of interest, streamlines exiting the mask will be truncated.
    """

    assert maskType in ["brain_mask_dilated","brain_mask"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask"

    import nibabel as nib
    from elikopy.utils import dipy_fod_to_mrtrix
    from dipy.io.streamline import load_tractogram, save_trk

    patient_path = p

    params={'Number of streamlines': streamline_number, 'Maximum angle': max_angle,
            'Cutoff value': cutoff, 'Mask type': maskType}

    if msmtCSD:
        if not os.path.isdir(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD/"):
            odf_msmtcsd_solo(folder_path, p)
        odf_file_path = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/MSMT-CSD/"+patient_path + "_MSMT-CSD_WM_ODF.nii.gz"
        params['Local modeling']='msmt-CSD'
    else:
        if not os.path.isdir(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"):
            odf_csd_solo(folder_path, p)
        if not os.path.isfile(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF_mrtrix.nii.gz"):
            img = nib.load(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF.nii.gz")
            data = dipy_fod_to_mrtrix(img.get_fdata())
            out = nib.Nifti1Image(data, img.affine, img.header)
            out.to_filename(folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF_mrtrix.nii.gz")
        odf_file_path = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD/"+patient_path + "_CSD_SH_ODF_mrtrix.nii.gz"
        params['Local modeling']='CSD'
    tracking_path = folder_path + '/subjects/' + patient_path + "/dMRI/tractography/"
    seed_path = folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask.nii.gz"
    mask_path = folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + '_' + maskType + '.nii.gz'
    dwi_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz'
    
    output_file = tracking_path+patient_path+'_'+output_filename+'.tck'

    if not os.path.isdir(tracking_path):
        os.mkdir(tracking_path)

    bashCommand=('tckgen -nthreads ' + str(core_count) + ' ' + odf_file_path +' '+ output_file+
                 ' -seed_image ' +seed_path+
                 ' -select ' +str(streamline_number)+
                 ' -angle ' +str(max_angle)+
                 ' -cutoff ' +str(cutoff)+
                 ' -mask ' +mask_path+
                 ' -force')

    tracking_log = open(tracking_path+"tractography_logs.txt", "a+")
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True,
                               stdout=tracking_log, stderr=subprocess.STDOUT)

    process.communicate()

    tracking_log.close()

    if save_as_trk:
        tract = load_tractogram(output_file, dwi_path)

        save_trk(tract, output_file[:-3]+'trk')

    with open(output_file[:-3]+'json', 'w') as outfile:
        json.dump(params, outfile)

    update_status(folder_path, patient_path, "tracking")


def sift_solo(folder_path: str, p: str, streamline_number: int = 100000,
              msmtCSD: bool = True, input_filename: str = 'tractogram',
              core_count: int = 1, save_as_trk=False):
    """ Computes the sifted whole brain tractogram of a single patient based on
    the fod obtained from msmt-CSD.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param streamline_number: Number of streamlines in the final tractogram. default=100000
    :param msmtCSD: boolean. If True then uses ODF from msmt-CSD, if False from CSD. default=True
    :param input_filename: str. Specify input filename for tractogram.
    """

    from dipy.io.streamline import load_tractogram, save_trk

    patient_path = p

    if msmtCSD:
        odf_file_path = (folder_path + '/subjects/' + patient_path
                         + "/dMRI/ODF/MSMT-CSD/"+patient_path
                         + "_MSMT-CSD_WM_ODF.nii.gz")
    else:
        odf_file_path = (folder_path + '/subjects/' + patient_path
                         + "/dMRI/ODF/CSD/"+patient_path
                         + "_CSD_SH_ODF_mrtrix.nii.gz")

    tracking_path = folder_path + '/subjects/' + patient_path + "/dMRI/tractography/"
    mask_path = folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz"
    dwi_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz'

    input_file = tracking_path+patient_path+'_'+input_filename+'.tck'
    output_file = tracking_path+patient_path+'_'+input_filename+'_sift.tck'

    bashCommand=('tcksift ' + input_file + ' ' + odf_file_path + ' ' +
                 output_file +
                 ' -nthreads ' + str(core_count) +
                 ' -term_number ' + str(streamline_number) +
                 ' -force')

    sift_log = open(tracking_path+"sift_logs.txt", "a+")
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True,
                               stdout=sift_log, stderr=subprocess.STDOUT)

    process.communicate()
    sift_log.close()

    if save_as_trk:
        tract = load_tractogram(output_file, dwi_path)
        save_trk(tract, output_file[:-3]+'trk')

    update_status(folder_path, patient_path, "siftComputation")


def verdict_solo(folder_path, p, core_count=1, small_delta=0.003, big_delta=0.035, G1Ball_1_lambda_iso=0.9e-9, C1Stick_1_lambda_par=[3.05e-9, 10e-9],TumorCells_Dconst=0.9e-9):
    """ Computes the verdict metrics for a single. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/verdict/.

    :param folder_path: the path to the root directory.
    :param p: The name of the patient.
    :param use_wm_mask: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
    :param use_amico: If true, use the amico optimizer. default=FALSE
    :param core_count: Number of allocated cpu cores. default=1
    """
    print("[verdict SOLO] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual verdict processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs

    patient_path = p
    log_prefix = "verdict SOLO"

    verdict_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/verdict"
    makedir(verdict_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/verdict/verdict_logs.txt", log_prefix)

    from dmipy.core.modeling_framework import MultiCompartmentModel
    from dmipy.signal_models.gaussian_models import G1Ball


    # initialize the compartments model and build the verdict model
    from dmipy.core.modeling_framework import MultiCompartmentModel
    from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models
    from dmipy.signal_models.gaussian_models import G1Ball

    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=TumorCells_Dconst)
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    verdict_mod = MultiCompartmentModel(models=[sphere, ball, stick])

    verdict_mod.set_fixed_parameter('G1Ball_1_lambda_iso', G1Ball_1_lambda_iso)
    verdict_mod.set_parameter_optimization_bounds('C1Stick_1_lambda_par', C1Stick_1_lambda_par)

    # load the data
    data, affine = load_nifti(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz")
    bvals, bvecs = read_bvals_bvecs(
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval",
        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec")
    wm_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_wm_mask.nii.gz'
    if os.path.isfile(wm_path):
        mask, _ = load_nifti(wm_path)
    else:
        mask, _ = load_nifti(folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")

    # transform the bval, bvecs in a form suited for verdict
    from dipy.core.gradients import gradient_table
    from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_threshold, small_delta=small_delta, big_delta=big_delta)
    acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)

    # fit the model to the data
    verdict_fit = verdict_mod.fit(acquisition_scheme=acq_scheme_dmipy, data=data, mask=mask, solver='mix', use_parallel_processing=True, number_of_processors=core_count)

    # extract the metrics
    fitted_parameters = verdict_fit.fitted_parameters

    diff_stick = fitted_parameters["C1Stick_1_lambda_par"] # Vascular Stick parallel diffusivity
    mu_vascular = fitted_parameters["C1Stick_1_mu"] #Vascular Stick Angle
    tumor_cells_diameter = fitted_parameters["S4SphereGaussianPhaseApproximation_1_diameter"] #Tumor Cells diameter

    f_tumor_cells = fitted_parameters["partial_volume_0"]  # Tumor Cells fraction
    f_hindered_extraCellular = fitted_parameters["partial_volume_1"] # Hindered Extra-Cellular fraction
    f_vascular = fitted_parameters["partial_volume_2"]  # Vascular fraction

    mse = verdict_fit.mean_squared_error(data)
    R2 = verdict_fit .R2_coefficient_of_determination(data)

    # save the nifti
    save_nifti(verdict_path + '/' + patient_path + '_verdict_diff_stick.nii.gz', diff_stick.astype(np.float32), affine)
    save_nifti(verdict_path + '/' + patient_path + '_verdict_mu_vascular.nii.gz', mu_vascular.astype(np.float32), affine)
    save_nifti(verdict_path + '/' + patient_path + '_verdict_tumor_cells_diameter.nii.gz', tumor_cells_diameter.astype(np.float32), affine)
    save_nifti(verdict_path + '/' + patient_path + '_verdict_f_tumor_cells.nii.gz', f_tumor_cells.astype(np.float32), affine)
    save_nifti(verdict_path + '/' + patient_path + '_verdict_f_hindered_extraCellular.nii.gz', f_hindered_extraCellular.astype(np.float32), affine)
    save_nifti(verdict_path + '/' + patient_path + '_verdict_f_vascular.nii.gz', f_vascular.astype(np.float32), affine)
    save_nifti(verdict_path + '/' + patient_path + '_verdict_mse.nii.gz', mse.astype(np.float32), affine)
    save_nifti(verdict_path + '/' + patient_path + '_verdict_R2.nii.gz', R2.astype(np.float32), affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/verdict/verdict_logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Starting quality control %s \n" % p)
    f.close()
    # ==================================================================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    qc_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/verdict/quality_control"
    makedir(qc_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/verdict/verdict_logs.txt", log_prefix)

    fig, axs = plt.subplots(2, 1, figsize=(2, 1))
    fig.suptitle('Elikopy : Quality control report - verdict', fontsize=50)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(qc_path + "/title.jpg", dpi=300, bbox_inches='tight');

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sl = np.shape(mse)[2] // 2
    plot_mse = np.zeros((np.shape(mse)[0], np.shape(mse)[1] * 5))
    plot_mse[:, 0:np.shape(mse)[1]] = mse[..., max(sl - 10, 0)]
    plot_mse[:, np.shape(mse)[1]:(np.shape(mse)[1] * 2)] = mse[..., max(sl - 5, 0)]
    plot_mse[:, (np.shape(mse)[1] * 2):(np.shape(mse)[1] * 3)] = mse[..., sl]
    plot_mse[:, (np.shape(mse)[1] * 3):(np.shape(mse)[1] * 4)] = mse[..., min(sl + 5, 2*sl-1)]
    plot_mse[:, (np.shape(mse)[1] * 4):(np.shape(mse)[1] * 5)] = mse[..., min(sl + 10, 2*sl-1)]
    im0 = axs[0].imshow(plot_mse, cmap='gray')
    axs[0].set_title('MSE')
    axs[0].set_axis_off()
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')
    sl = np.shape(R2)[2] // 2
    plot_R2 = np.zeros((np.shape(R2)[0], np.shape(R2)[1] * 5))
    plot_R2[:, 0:np.shape(R2)[1]] = R2[..., max(sl - 10, 0)]
    plot_R2[:, np.shape(R2)[1]:(np.shape(R2)[1] * 2)] = R2[..., max(sl - 5, 0)]
    plot_R2[:, (np.shape(R2)[1] * 2):(np.shape(R2)[1] * 3)] = R2[..., sl]
    plot_R2[:, (np.shape(R2)[1] * 3):(np.shape(R2)[1] * 4)] = R2[..., min(sl + 5, 2*sl-1)]
    plot_R2[:, (np.shape(R2)[1] * 4):(np.shape(R2)[1] * 5)] = R2[..., min(sl + 10, 2*sl-1)]
    im1 = axs[1].imshow(plot_R2, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title('R2')
    axs[1].set_axis_off()
    fig.colorbar(im1, ax=axs[1], orientation='horizontal');
    plt.tight_layout()
    plt.savefig(qc_path + "/error.jpg", dpi=300, bbox_inches='tight');


    """Save as a pdf"""

    elem = [qc_path + "/title.jpg", qc_path + "/error.jpg"]

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
            self.cell(60, 1, 'Quality control report - verdict', 0, 0, 'R')
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

    if not os.path.exists(folder_path + '/subjects/' + patient_path + '/quality_control.pdf'):
        shutil.copyfile(qc_path + '/qc_report.pdf', folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
    else:
        """Merge with QC of preproc""";
        from pypdf import PdfMerger
        pdfs = [folder_path + '/subjects/' + patient_path + '/quality_control.pdf', qc_path + '/qc_report.pdf']
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(folder_path + '/subjects/' + patient_path + '/quality_control_verdict.pdf')
        merger.close()
        os.remove(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        os.rename(folder_path + '/subjects/' + patient_path + '/quality_control_verdict.pdf',folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f = open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/verdict/verdict_logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
        f.close()

    update_status(folder_path, patient_path, "verdict")

def report_solo(folder_path,patient_path, slices=None, short=False):
    """ Legacy report function.

    :param folder_path: path to the root directory.
    :param patient_path: Name of the subject.
    :param slices: Add additional slices cut to specific volumes
    :param short: Only output raw data, preprocessed data and FA data.
    """

    report_path = folder_path + '/subjects/' + patient_path + "/report/raw/"
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
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/raw/"+patient_path+"_raw_dmri" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/raw/"+patient_path+"_raw_dmri","raw_drmi","Raw dMRI ("+patient_path+"_raw_drmi.nii.gz)"))
    if os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc","drmi_preproc","dMRI preprocessed ("+patient_path+"_drmi_preproc.nii.gz)"))

        if slices:
            i = slices
            fslroi = "fslroi " + folder_path + '/subjects/' + patient_path + "/dMRI/preproc/"+patient_path+"_dmri_preproc" + ".nii.gz" + " " + report_path + "/preproc_" + str(i) + ".nii.gz " + str(i - 1) + " 1"
            process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=report_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()
            image.append((report_path + "/preproc_" + str(i),
                          "drmi_preproc_" + str(i), "dMRI preprocessed slice "+ str(i) + " (" + patient_path + "_drmi_preproc.nii.gz)"))

    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/bet/"+patient_path+"_mask" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/preproc/bet/"+patient_path+"_mask","drmi_preproc_bet","dMRI BET preprocessing ("+patient_path+"_mask.nii.gz)"))
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/mppca/"+patient_path+"_mppca" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/preproc/mppca/"+patient_path+"_mppca","drmi_preproc_mppca","dMRI Denoised preprocessing ("+patient_path+"_mppca.nii.gz)"))
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected","drmi_preproc_gibbs","dMRI Gibbs preprocessing ("+patient_path+"_gibbscorrected.nii.gz)"))

        if slices:
            i = slices
            fslroi = "fslroi " + folder_path + '/subjects/' + patient_path + "/dMRI/preproc/gibbs/"+patient_path+"_gibbscorrected" + ".nii.gz" + " " + report_path + "/preproc_gibbs_" + str(
                i) + ".nii.gz " + str(i - 1) + " 1"
            process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=report_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()
            image.append((report_path + "/preproc_gibbs_" + str(i),
                          "drmi_preproc_gibbs_" + str(i),
                          "dMRI Gibbs preprocessing " + str(i) + " ("+patient_path+"_gibbscorrected.nii.gz)"))

    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/topup/"+patient_path+"_topup" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/preproc/topup/"+patient_path+"_topup","drmi_preproc_topup","dMRI Topup preprocessing ("+patient_path+"_topup.nii.gz)"))
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/eddy/" + patient_path + "_eddy_corr" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/preproc/eddy/" + patient_path + "_eddy_corr","drmi_preproc_eddy_corr", "dMRI Eddy preprocessing (" + patient_path + "_eddy_corr.nii.gz)"))

    if os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_FA" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_FA","dti_FA", "Microstructure: FA of dti (" + patient_path + "_FA.nii.gz)"))
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_AD" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_AD","dti_AD", "Microstructure: AD of dti (" + patient_path + "_AD.nii.gz)"))
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_MD" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_MD","dti_MD", "Microstructure: MD of dti (" + patient_path + "_MD.nii.gz)"))
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_RD" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_RD","dti_RD", "Microstructure: RD of dti (" + patient_path + "_RD.nii.gz)"))
    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_dtensor" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/" + patient_path + "_dtensor","dti_dtensor", "Microstructure: Dtensor of dti (" + patient_path + "_dtensor.nii.gz)"))

    if not short and os.path.exists(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/" + patient_path + "_dtensor" + ".nii.gz"):
        image.append((folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/" + patient_path + "_dtensor","noddi_dtensor", "Microstructure: ICVF of Noddi (" + patient_path + "_dtensor.nii.gz)"))

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


    pdf.output(folder_path + '/subjects/' + patient_path + "/report/report_"+patient_path+".pdf", 'F')



def clean_study_solo(folder_path, p):
    """Clean a study folder for a specific patient p.
    """
    import glob
    log_prefix = "CLEAN-STUDY SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual CLEAN-STUDY for patient %s \n" % p, flush = True)
    patient_path = p

    subject_main_path = os.path.join(folder_path, "subjects", p)

    # Delete all the space consuming intermediate files and folders of the preproc folder
    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "bet")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "bet"))

    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "eddy")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "eddy"))

    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "mppca")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "mppca"))

    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "topup")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "topup"))

    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "patch2self")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "patch2self"))

    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "gibbs")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "gibbs"))

    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "reslice")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "reslice"))

    if os.path.exists(os.path.join(subject_main_path, "dMRI", "preproc", "biasfield")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "preproc", "biasfield"))

    # delete slurm files in preproc folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "preproc") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")


    # Delete all the intermediate files of the T1 folder
    if os.path.exists(os.path.join(subject_main_path, "T1")):
        shutil.rmtree(os.path.join(subject_main_path, "T1", patient_path + "_T1_gibbscorrected.nii.gz"))
        shutil.rmtree(os.path.join(subject_main_path, "T1", patient_path + "_T1_corr_projected.nii.gz"))
        shutil.rmtree(os.path.join(subject_main_path, "T1", patient_path + "_T1_brain.nii.gz"))


    # delete slurm files in masks folder
    slurmList = glob.glob(os.path.join(subject_main_path, "masks") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")


    # delete slurm files in CSD folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "ODF", "CSD") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")


    # delete slurm files in MSMT-CSD folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "ODF", "MSMT-CSD") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")

    # delete slurm files in MSMT-CSD folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "ODF", "MSMT-CSD") + '/' + patient_path + '_CSD_peak_-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")


    # Delete all the intermediate files of the diamond folder
    if os.path.exists(os.path.join(subject_main_path, "dMRI", "microstructure", "diamond")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "microstructure", "diamond", patient_path + "_diamond_residuals.nii.gz"))


    # Delete all the intermediate files of the dti folder
    if os.path.exists(os.path.join(subject_main_path, "dMRI", "microstructure", "dti")):
        shutil.rmtree(os.path.join(subject_main_path, "dMRI", "microstructure", "dti", patient_path + "_residual.nii.gz"))



    # delete slurm files in dti folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "microstructure", "dti") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")

    # delete slurm files in noddi folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "microstructure", "noddi") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")

    # delete slurm files in mf folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "microstructure", "mf") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")

    # delete slurm files in diamond folder
    slurmList = glob.glob(os.path.join(subject_main_path, "dMRI", "microstructure", "diamond") + '/slurm-*', recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for slurmfilePath in slurmList:
        try:
            os.remove(slurmfilePath)
        except OSError:
            print("Error while deleting file")
