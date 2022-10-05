import numpy as np
import nibabel as nib
import datetime
import os
import shutil
import json
import math

import re
from bruker2nifti.converter import Bruker2Nifti

from threading import Thread

import subprocess
from elikopy.utils import makedir

from dipy.denoise.localpca import mppca
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table, reorient_bvecs
from dipy.viz import regtools
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D,
                                   TranslationTransform2D,
                                   RigidTransform2D,
                                   AffineTransform2D,
                                   RotationTransform2D,
                                   RotationTransform3D)

from skimage.morphology import binary_dilation, binary_erosion
denoised = None


def threaded_mppca(a, b, chunk):
    global denoised
    pr = math.ceil((np.shape(chunk)[3] ** (1 / 3) - 1) / 2)
    denoised_chunk, sigma = mppca(chunk, patch_radius=pr, return_sigma=True)
    denoised[:, :, :, a:b] = denoised_chunk


def convertAndMerge(folder_path, raw_bruker_dicom_subfolder, nifti_bruker_folder, subject):
    # instantiate a converter
    log_prefix= "ConvertAndMerge"
    f = open(folder_path + "/mouse_logs.txt", "a+")
    msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of convert and merge for mouse %s \n" % subject
    print(msg)
    f.write(msg)

    bru = Bruker2Nifti(raw_bruker_dicom_subfolder, nifti_bruker_folder, study_name=subject)

    # select the options (attributes) you may want to change - the one shown below are the default one:
    bru.verbose = 0
    # converter settings for the nifti values
    bru.nifti_version = 1
    bru.qform_code = 1
    bru.sform_code = 2
    bru.save_human_readable = True
    bru.save_b0_if_dwi = (True)  # if DWI, it saves the first layer as a single nfti image.
    bru.correct_slope = True
    bru.correct_offset = True
    # advanced sample positioning
    bru.sample_upside_down = False
    bru.frame_body_as_frame_head = False
    # chose to convert extra files:
    bru.get_acqp = False
    bru.get_method = False
    bru.get_reco = False

    # Check that the list of scans and the scans names automatically selected makes some sense:
    print(bru.scans_list)
    print(bru.list_new_name_each_scan)

    # call the function convert, to convert the study from DICOM to NifTi:
    bru.convert()
    # Merge all DWI volumes:
    bvecs = None
    bvals = None
    mergedDWI = None
    shell_index = []

    msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Merging images \n"
    print(msg)
    f.write(msg)
    for dirr in os.listdir(os.path.join(nifti_bruker_folder, subject)):
        basedir = os.path.join(nifti_bruker_folder, subject, dirr)
        if not os.path.isfile(os.path.join(basedir, "acquisition_method.txt")):
            continue
        with open(os.path.join(basedir, "acquisition_method.txt")) as facq_method:
            method = facq_method.readlines()[0]

        if (method == "DtiEpi"):
            print("DtiEpi", dirr)
            bvec = np.load(os.path.join(basedir, dirr + "_DwGradVec.npy"))
            bval = np.load(os.path.join(basedir, dirr + "_DwEffBval.npy"))
            dwi = nib.load(os.path.join(basedir, dirr + ".nii.gz"))
            zerobvec = np.where((bvec[:, 0] == 0) & (bvec[:, 1] == 0) & (bvec[:, 2] == 0))
            bvec[zerobvec] = [1, 0, 0]
            if any(x is None for x in [bvecs, bvals, mergedDWI]):
                bvecs = bvec
                bvals = bval
                mergedDWI = dwi
                shell_index.append(0)
            else:
                bvecs = np.concatenate((bvecs, bvec), axis=0)
                bvals = np.concatenate((bvals, bval), axis=0)
                shell_index.append(mergedDWI.shape[-1])
                mergedDWI = nib.concat_images([mergedDWI, dwi], axis=3)
        elif (method == "FLASH"):
            print("FLASH", dirr)
        elif (method == "RARE"):
            print("RARE", dirr)
        elif (method == "MSME"):
            print("MSME", dirr)
        elif (method == "FieldMap"):
            print("FieldMap", dirr)
        elif (method == "nmrsuDtiEpi"):
            print("nmrsuDtiEpi", dirr)
            makedir(os.path.join(nifti_bruker_folder, subject, "reverse_encoding"), folder_path + "/logs.txt", log_prefix)
            dwi = nib.load(os.path.join(basedir, dirr + ".nii.gz"))
            bval = np.load(os.path.join(basedir, dirr + "_DwEffBval.npy"))
            bvec = np.load(os.path.join(basedir, dirr + "_DwGradVec.npy"))
            zerobvec = np.where((bvec[:, 0] == 0) & (bvec[:, 1] == 0) & (bvec[:, 2] == 0))
            bvec[zerobvec] = [1, 0, 0]
            np.savetxt(os.path.join(nifti_bruker_folder, subject, "reverse_encoding", subject + ".bvec"), bvec, fmt="%.42f")
            np.savetxt(os.path.join(nifti_bruker_folder, subject, "reverse_encoding", subject + ".bval"), bval, newline=' ',
                       fmt="%.42f")
            dwi.to_filename(os.path.join(nifti_bruker_folder, subject, "reverse_encoding", subject + ".nii.gz"))
            fnVol = open(os.path.join(nifti_bruker_folder, subject, "reverse_encoding", "nVol.txt"), "w")
            fnVol.write(str(bval.shape[0]))
            fnVol.close()
        else:
            print("Unknow acquisition method:", method)

    np.savetxt(os.path.join(nifti_bruker_folder, subject, subject + ".bvec"), bvecs.T, fmt="%.42f")
    np.savetxt(os.path.join(nifti_bruker_folder, subject, subject + ".bval"), bvals.T, newline=' ', fmt="%.42f")
    mergedDWI.to_filename(os.path.join(nifti_bruker_folder, subject, subject + ".nii.gz"))
    print("Number of total volumes: ", bvals.shape)

    fnVol = open(os.path.join(nifti_bruker_folder, subject, "nVol.txt"), "w")
    fnVol.write(str(int(bvals.shape[0])))
    fnVol.close()

    fshellIndex = open(os.path.join(nifti_bruker_folder, subject, "shell_index.txt"), "w")
    for element in shell_index:
        fshellIndex.write(str(element) + "\n")
    fshellIndex.close()
    msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": End of convert and merge for mouse %s \n" % subject
    print(msg)
    f.write(msg)
    f.close()
    return int(bvals.shape[0])


def link(folder_path, nifti_bruker_folder, nVol, subjectName, acqparams_path):
    pattern = re.compile("data_\\d")
    patternNum = re.compile("\\d")

    log_prefix = "link"

    match = False
    dataFolderIndex = []
    for typeFolder in os.listdir(folder_path):
        if pattern.match(typeFolder):
            nVolDataFolder = int(np.loadtxt(os.path.join(folder_path, typeFolder, "nVol.txt")))
            dataFolderIndex.append(patternNum.search(typeFolder).group(0))
            if nVol == nVolDataFolder:
                print("match")
                match = True
                break

    if not match:
        dataFolderIndex.sort()
        if len(dataFolderIndex) > 0:
            typeFolder = "data_" + str(dataFolderIndex[-1] + 1)
        else:
            typeFolder = "data_1"
        makedir(os.path.join(folder_path, typeFolder), folder_path + "/logs.txt", log_prefix)
        makedir(os.path.join(folder_path, typeFolder, "reverse_encoding"), folder_path + "/logs.txt", log_prefix)
        index = open(os.path.join(folder_path, typeFolder, "index.txt"), "w")
        for i in range(nVol):
            index.write('1 ')
        index.close()
        index = open(os.path.join(folder_path, typeFolder, "reverse_encoding", "index.txt"), "w")
        for i in range(4):
            index.write('1 ')
        index.close()
        fnVol = open(os.path.join(folder_path, typeFolder, "nVol.txt"), "w")
        fnVol.write(str(nVol))
        fnVol.close()
        shutil.copyfile(os.path.join(nifti_bruker_folder, "shell_index.txt"), os.path.join(folder_path, typeFolder, "shell_index.txt"))

        fraw = open(acqparams_path, 'r')

        acq = open(os.path.join(folder_path, typeFolder, "acqparams.txt"), "w")
        acq.write(fraw.readline())
        acq.close()
        acq = open(os.path.join(folder_path, typeFolder, "reverse_encoding", "acqparams.txt"), "w")
        acq.write(fraw.readline())
        acq.close()

        fraw.close()

    subjectPath = os.path.join(nifti_bruker_folder, subjectName)
    shutil.copyfile(subjectPath + ".nii.gz", os.path.join(folder_path, typeFolder, subjectName + ".nii.gz"))
    shutil.copyfile(subjectPath + ".bvec", os.path.join(folder_path, typeFolder, subjectName + ".bvec"))
    shutil.copyfile(subjectPath + ".bval", os.path.join(folder_path, typeFolder, subjectName + ".bval"))

    subjectPath_reverse = os.path.join(nifti_bruker_folder, "reverse_encoding", subjectName)
    shutil.copyfile(subjectPath_reverse + ".nii.gz",
                    os.path.join(folder_path, typeFolder, "reverse_encoding", subjectName + ".nii.gz"))
    shutil.copyfile(subjectPath_reverse + ".bvec",
                    os.path.join(folder_path, typeFolder, "reverse_encoding", subjectName + ".bvec"))
    shutil.copyfile(subjectPath_reverse + ".bval",
                    os.path.join(folder_path, typeFolder, "reverse_encoding", subjectName + ".bval"))


def gen_Nifti(folder_path, raw_bruker_dicom_folder, nifti_bruker_folder):
    """
    :param BaseIN: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/raw"
    :param BaseOUT: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/bruker_elikopy"
    :param ProcIN: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/test_b2e_study"
    """
    log_prefix= "gen_Nifti"
    f = open(folder_path + "/mouse_logs.txt", "a+")
    msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of gen Nifti \n"
    print(msg)
    f.write(msg)
    f.close()
    for file in os.listdir(raw_bruker_dicom_folder):
        if (os.path.isdir(os.path.join(raw_bruker_dicom_folder, file)) and file != 'cleaning_data'):
            raw_bruker_dicom_subfolder = os.path.join(raw_bruker_dicom_folder, file)
            nVol = convertAndMerge(folder_path,raw_bruker_dicom_subfolder, nifti_bruker_folder, file)
            link(folder_path, os.path.join(nifti_bruker_folder, file), nVol, file, os.path.join(raw_bruker_dicom_subfolder, "acqparams.txt"))



def preprocessing_solo(folder_path, patient_path, denoising=True, denoising_algorithm="mppca_dipy", motion_correction=True, brain_extraction=True, topup=True, core_count = 1):
    p = patient_path
    log_prefix = "MOUSE PREPROC SOLO"
    f = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
    msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for mouse %s \n" % p
    print(msg)
    f.write(msg)

    global denoised

    fdwi = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.nii.gz'
    fbval = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval"
    fbvec = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec"

    fcorr_dwi = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz'
    fcorr_bval = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval"
    fcorr_bvec = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec"

    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=65)

    denoising_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/denoising/'
    motionCorr_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/motionCorrection/'
    topup_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/topup"
    brainExtraction_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/brainExtraction/'
    mask_path = folder_path + '/subjects/' + patient_path + '/masks/'

    imain_tot = fdwi

    import json
    with open(os.path.join(folder_path + '/subjects/', "subj_type.json")) as json_file:
        subj_type = json.load(json_file)



    ############################
    ### MPPCA Denoising step ###
    ############################
    if denoising:
        msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Start of denoising step \n"
        print(msg)
        f.write(msg)

        makedir(denoising_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        shell_index = []
        with open(os.path.join(folder_path, "data_" + str(subj_type[patient_path]), "shell_index.txt"), "r") as f:
            for line in f:
                shell_index.append(int(line.strip()))

        denoised = data.copy()

        print(shell_index, data.shape)

        if denoising_algorithm == "mppca_mrtrix":
            makedir(denoising_path + "/mrtrix",folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        threads = []
        for i in range(len(shell_index) - 1):
            msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
                  ": " + "Start of mppca for shell" + str(i) + " (index:" + str(shell_index[i]) + "," + str(shell_index[i + 1]) + ")\n"
            print(msg)
            f.write(msg)
            a = shell_index[i]
            b = shell_index[i + 1]
            chunk = data[:, :, :, a:b].copy()

            if denoising_algorithm == "mppca_dipy":
                threads.append(Thread(target=threaded_mppca, args=(a, b, chunk)))
                threads[-1].start()
            elif denoising_algorithm == "mppca_mrtrix":
                makedir(denoising_path + "/mrtrix",folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
                import subprocess
                bashCommand = "dwidenoise -nthreads " + str(core_count) + " " + imain_tot + \
                              " " + denoising_path + '/mrtrix/' + 'mppca_shell' + str(i) + '_.nii.gz' + \
                              " -noise " + denoising_path + '/' + patient_path + '_sigmaNoise_shell' + str(i) + '.nii.gz -force'

                process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=f,
                                           stderr=subprocess.STDOUT)

                output, error = process.communicate()




        if denoising_algorithm == "mppca_dipy":
            msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
                  ": " + "All threads have been launched.\n"
            print(msg)
            f.write(msg)
            for i in range(len(threads)):
                threads[i].join()
        elif denoising_algorithm == 'mppca_mrtrix':
            print("Merging denoising results")
            #TODO: Merge denoising results

        msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
              ": " + "All threads have finished.\n"
        print(msg)
        f.write(msg)

        save_nifti(denoising_path + '/' + patient_path + '_mppca.nii.gz', denoised.astype(np.float32), affine)
        data = denoised
        imain_tot = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/denoisingMPPCA/' + patient_path + '_mppca.nii.gz'

    ##############################
    ### Motion correction step ###
    ##############################

    if motion_correction:
        makedir(motionCorr_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)


        msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
              ": " + " Motion correction step.\n"
        print(msg)
        f.write(msg)

        reg_affines_precorrection = []
        static_precorrection = data[..., 0]
        static_grid2world_precorrection = affine
        moving_grid2world_precorrection = affine
        for i in range(data.shape[-1]):
            if gtab.b0s_mask[i]:
                msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
                      ": " + " Motion correction step: Premoving b0 number " + str(i) + "\n"
                print(msg)
                f.write(msg)
                moving = data[..., i]
                moved, trans_affine = affine_reg(static_precorrection, static_grid2world_precorrection,
                                                 moving, moving_grid2world_precorrection)
                data[..., i] = moved
            else:
                moving = data[..., i]
                data[..., i] = trans_affine.transform(moving)
                reg_affines_precorrection.append(trans_affine.affine)

        gtab_precorrection = reorient_bvecs(gtab, reg_affines_precorrection)

        bvec = gtab.bvecs
        zerobvec = np.where((bvec[:, 0] == 0) & (bvec[:, 1] == 0) & (bvec[:, 2] == 0))
        bvec[zerobvec] = [1, 0, 0]
        save_nifti(motionCorr_path + patient_path + '_motionCorrected.nii.gz', data, affine)
        np.savetxt(motionCorr_path + patient_path + '_motionCorrected.bval', bvals)
        np.savetxt(motionCorr_path + patient_path + '_motionCorrected.bvec', bvec)

        gtab = gtab_precorrection
        msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
              ": " + " End of Motion correction step\n"
        print(msg)
        f.write(msg)

        imain_tot = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/motionCorrection/' + patient_path + 'motionCorrected.nii.gz'

    #############################
    ###      Topup step       ###
    #############################

    if topup:
        msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
              ": " + " Start of topup step\n"
        print(msg)
        f.write(msg)

        multiple_encoding = False
        topup_log = open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/topup/topup_logs.txt", "a+")

        with open(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'index.txt') as f:
            line = f.read()
            line = " ".join(line.split())
            topup_index = [int(s) for s in line.split(' ')]

        with open(folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt') as f:
            topup_acq = [[float(x) for x in line2.split()] for line2 in f]

        # Find all the bo to extract.
        current_index = 0
        all_index = {}
        i = 1
        roi = []
        for ind in topup_index:
            if ind != current_index and ind not in all_index:
                roi.append(i)
                fslroi = "fslroi " + imain_tot + " " + topup_path + "/b0_" + str(i) + ".nii.gz " + str(i - 1) + " 1"
                process = subprocess.Popen(fslroi, universal_newlines=True, shell=True, stdout=topup_log,
                                           stderr=subprocess.STDOUT)

                output, error = process.communicate()
                print("B0 of index" + str(i) + " extracted!")
            current_index = ind
            all_index[ind] = all_index.get(ind, 0) + 1
            i = i + 1

        # Merge b0
        if len(roi) == 1:
            shutil.copyfile(topup_path + "/b0_" + str(roi[0]) + ".nii.gz", topup_path + "/b0.nii.gz")
        else:
            roi_to_merge = ""
            for r in roi:
                roi_to_merge = roi_to_merge + " " + topup_path + "/b0_" + str(r) + ".nii.gz"
            cmd = "fslmerge -t " + topup_path + "/b0.nii.gz" + roi_to_merge
            process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=topup_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()

        # Check if multiple or single encoding direction
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


        if multiple_encoding:
            makedir(topup_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
            topupConfig = 'b02b0.cnf'  # if topupConfig is None else topupConfig
            bashCommand = 'export OMP_NUM_THREADS=' + str(core_count) + ' ; export FSLPARALLEL=' + str(
                core_count) + ' ; topup --imain="' + topup_path + '/b0.nii.gz" --config="' + topupConfig + '" --datain="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --fout="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_fout_estimate" --iout="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_iout_estimate" --verbose'
            bashcmd = bashCommand.split()
            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=topup_log,
                                       stderr=subprocess.STDOUT)
            # wait until topup finish
            output, error = process.communicate()

        inindex = ""
        first = True
        for r in roi:
            if first:
                inindex = str(topup_index[r - 1])
                first = False
            else:
                inindex = inindex + "," + str(topup_index[r - 1])

        bashCommand2 = 'applytopup --imain="' + imain_tot + '" --inindex=' + inindex + ' --datain="' + folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + 'acqparams.txt" --topup="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_estimate" --out="' + folder_path + '/subjects/' + patient_path + '/dMRI/preproc/topup/' + patient_path + '_topup_corr"'

        process2 = subprocess.Popen(bashCommand2, universal_newlines=True, shell=True, stdout=topup_log,
                                    stderr=subprocess.STDOUT)
        # wait until apply topup finish
        output, error = process2.communicate()

        topup_log.close()

    #############################
    ### Brain extraction step ###
    #############################
    if brain_extraction:
        msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
              ": " + " Start of brain extraction step\n"
        print(msg)
        f.write(msg)

        makedir(brainExtraction_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
        makedir(mask_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        # created a brain for designing a mask (sum of all shells)
        b0final = np.zeros(data.shape[:-1])
        for i in range(data.shape[-1]):
            b0final += data[:, :, :, i]

        save_nifti(brainExtraction_path + patient_path + '_Brain_extraction_ref.nii.gz', b0final, affine)

        # function to find a mask
        final_mask = mask_Wizard(b0final, 4, 7, work='2D')

        # Saving
        out = nib.Nifti1Image(final_mask, affine)
        out.to_filename(mask_path + patient_path + '_brain_mask.nii.gz')

        data[final_mask == 0] = 0

        save_nifti(brainExtraction_path + patient_path + '_Extracted_brain.nii.gz', data, affine)
        msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
              ": " + " End of brain extraction step\n"
        print(msg)
        f.write(msg)

    ################################
    ### Final preprocessing step ###
    ################################

    final_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/'
    save_nifti(final_path + '/' + patient_path + '_dmri_preproc.nii.gz', data, affine)
    np.savetxt(final_path + '/' + patient_path + '_dmri_preproc.bval', bvals)
    np.savetxt(final_path + '/' + patient_path + '_dmri_preproc.bvec', bvecs)

    msg = "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + \
          ": " + " End of preprocessing\n"
    print(msg)
    f.write(msg)




"""
=========================================================
Motion correction of DWI data
@author: DELINTE Nicolas, BIOUL Nicolas, HENAUT Eliott
=========================================================
"""

def affine_reg(static, static_grid2world, moving, moving_grid2world, work='3D'):
    """


    Parameters
    ----------
    static : TYPE
        DESCRIPTION.
    static_grid2world : TYPE
        DESCRIPTION.
    moving : TYPE
        DESCRIPTION.
    moving_grid2world : TYPE
        DESCRIPTION.
    work : TYPE, optional
        DESCRIPTION. The default is '3D'.

    Returns
    -------
    transformed : TYPE
        DESCRIPTION.
    affine : TYPE
        DESCRIPTION.

    """

    if work == '3D':
        c_of_mass = transform_centers_of_mass(static, static_grid2world, moving, moving_grid2world)
        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)

        level_iters = [500, 100, 10]

        sigmas = [3.0, 1.0, 0.0]

        factors = [4, 2, 1]

        affreg = AffineRegistration(metric=metric,
                                    level_iters=level_iters,
                                    sigmas=sigmas,
                                    factors=factors)

        transform = TranslationTransform3D()
        params0 = None
        starting_affine = c_of_mass.affine
        translation = affreg.optimize(static, moving, transform, params0,
                                      static_grid2world, moving_grid2world,
                                      starting_affine=starting_affine)

        transformed = translation.transform(moving)
        # save_nifti('/Users/nicolasbioul/Desktop/translation.nii.gz', transformed, moving_grid2world)

        transform = RotationTransform3D()
        params0 = None
        starting_affine = translation.affine
        rotation = affreg.optimize(static, moving, transform, params0,
                                   static_grid2world, moving_grid2world,
                                   starting_affine=starting_affine)

        transformed = rotation.transform(moving)
        # save_nifti('/Users/nicolasbioul/Desktop/rotation.nii.gz', transformed, moving_grid2world)

        transform = RigidTransform3D()
        params0 = None
        starting_affine = rotation.affine
        rigid = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=starting_affine)
        transformed = rigid.transform(moving)
        # save_nifti('/Users/nicolasbioul/Desktop/rigid.nii.gz', transformed, moving_grid2world)

        transform = AffineTransform3D()
        params0 = None
        starting_affine = rigid.affine
        affine = affreg.optimize(static, moving, transform, params0,
                                 static_grid2world, moving_grid2world,
                                 starting_affine=starting_affine)

        transformed = affine.transform(moving)
        # save_nifti('/Users/nicolasbioul/Desktop/affine.nii.gz', transformed, moving_grid2world)

        return transformed, affine

    elif work == '2D':

        c_of_mass = transform_centers_of_mass(static, static_grid2world, moving, moving_grid2world)
        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)

        level_iters = [500, 100, 10]

        sigmas = [3.0, 1.0, 0.0]

        factors = [4, 2, 1]

        affreg = AffineRegistration(metric=metric,
                                    level_iters=level_iters,
                                    sigmas=sigmas,
                                    factors=factors)

        transform = TranslationTransform2D()
        params0 = None
        starting_affine = c_of_mass.affine
        translation = affreg.optimize(static, moving, transform, params0,
                                      static_grid2world, moving_grid2world,
                                      starting_affine=starting_affine)

        transformed = translation.transform(moving)

        transform = RigidTransform2D()
        params0 = None
        starting_affine = translation.affine
        rigid = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=starting_affine)
        transformed = rigid.transform(moving)

        transform = RotationTransform2D()
        params0 = None
        starting_affine = rigid.affine
        rotation = affreg.optimize(static, moving, transform, params0,
                                   static_grid2world, moving_grid2world,
                                   starting_affine=starting_affine)

        transformed = rotation.transform(moving)

        transform = AffineTransform2D()
        params0 = None
        starting_affine = rotation.affine
        affine = affreg.optimize(static, moving, transform, params0,
                                 static_grid2world, moving_grid2world,
                                 starting_affine=starting_affine)

        transformed = affine.transform(moving)

        return transformed, affine

"""
=======================================================
Brain extraction of DWI data
@author: HENAUT Eliott BIOUL Nicolas
=======================================================
"""

def mask_Wizard(data, r_fill, r_shape, scal=1, geo_shape='ball', height=5, work='2D'):
    """

    Parameters
    ----------
    data : TYPE : ArrayList
        DESCRIPTION: The brain data that we want to find a mask.
    r_fill : TYPE : Interger
        DESCRIPTION: The radius of the circle,cylinder,or ball that we use for growing the surface. Exclusively used in the fill function.
    r_shape : TYPE : Interger
        DESCRIPTION: The radius of the circle, cylinder, or ball that we use for the opening/closing.  Exclusively used in the shape_matrix function.
    scal : TYPE : Interger, optional
        DESCRIPTION: The param λ in the formula y=µ+λ*σ. y is the threshold defining the inclusion of a voxel or not. µ is the mean of the array data. σ is it's standard deviation'
        The default is 1.
    geo_shape : TYPE : String, optional
        DESCRIPTION: The shape of the convolution in the opening/closing. ('ball','cylinder')
        The default is 'ball'.
    height : TYPE : int, optional
        DESCRIPTION: The height of the cylinder of the convolution in the opening/closing. ('ball','cylinder')
        The default is 'ball'.
    work : TYPE : String, optional
        DESCRIPTION: The dimension. ('2D','3D')
        The default is '2D'. Allows working in 2 or 3 dimensions. Switch to '3D'

    Returns
    -------
    mask : TYPE : Arraylist
        The matrice of the mask, 1 if we take it, O otherwise.

    """
    if work == '3D':

        (x, y, z) = data.shape
        seed = binsearch(data, x, y, z)
        init_val = int(np.mean(data) + scal * np.sqrt(np.var(data)))
        brainish = fill(seed, data, r_fill, init_val)
        geo_shape = shape_matrix(r_shape, geo_shape, height, work)
        closing = binary_erosion(binary_dilation(brainish, selem=geo_shape))
        opening = binary_dilation(binary_erosion(closing, selem=geo_shape))
        mask = np.zeros(opening.shape)
        mask[opening] = 1
    else:
        (x, y, z) = data.shape
        b0final = data
        for i in range(z):
            seed = ((b0final.shape[0]) // 2, (b0final.shape[1]) // 2, i)
            if np.std(b0final[:, :, i] / np.mean(b0final[:, :, i])) > 0.5:
                seed = binsearch(b0final[:, :, i], x, y, work='2D')
                init_val = int(np.mean(b0final[:, :, i]) + scal * np.sqrt(np.var(b0final[:, :, i])))
                brainish = fill([seed], b0final[:, :, i], r_fill, init_val)
                mat = shape_matrix(r_shape)
                closing = binary_erosion(binary_dilation(brainish, selem=mat), selem=mat)
                opening = binary_dilation(binary_erosion(closing, selem=mat), selem=mat)

                # Correction element
                final = binary_dilation(opening, selem=shape_matrix(2, work='2D'))
                #
                inter = np.zeros(final.shape)
                inter[final] = 1
                b0final[:, :, i] = inter
            else:
                b0final[:, :, i] = np.zeros(b0final[:, :, i].shape)
                b0final[seed] = 1
        mask = b0final

    return mask

def fill(position, data, rad, init_val, work='2D'):
    """

    Parameters
    ----------
    position : TYPE: List
        DESCRIPTION: List containing the seed point or set of mask points on which to start the filling.
    data : TYPE: Array
        DESCRIPTION: Image array on which to fill.
    rad : TYPE: Integer
        DESCRIPTION: Radius of the neighbourhood to be taken into account when filling. Exclusively used when calling getVoxel
    init_val : TYPE: Float
        DESCRIPTION: Inclusing value. If above or equal to this value, the fill function will select the voxel/pixel otherwise it will be excluded.
    work : TYPE : String, optional
        DESCRIPTION: The dimension. ('2D','3D')
        The default is '2D'. Allows working in 2 or 3 dimensions. Switch to '3D'

    Returns
    -------
    data_new : TYPE: Array
        DESCRIPTION: returns the filled out mask with all selected pixels from 2D or 3D image.

    """
    data_new = np.zeros(data.shape)
    voxelList = set()
    for i in position:
        voxelList.add(i)

    while len(voxelList) > 0:
        voxel = voxelList.pop()
        voxelList = getVoxels(voxel, data, init_val, voxelList, rad, data_new, work)
        data_new[voxel] = 1
    return data_new

def getVoxels(voxel, data, init_val, voxelList, rad, data_new, work='2D'):
    """


    Parameters
    ----------
    voxel : TYPE: Tuple
        DESCRIPTION: Contains the coordinates (x,y) in work='2D' or (x,y,z) in work='3D' of the voxel to be analysed
    data : TYPE: Array
        DESCRIPTION: image array
    init_val : TYPE: Float
        DESCRIPTION: Inclusing value. If above or equal to this value, the fill function will select the voxel/pixel otherwise it will be excluded.
    voxelList : TYPE: set()
        DESCRIPTION: Set of all qualified voxels. Each qualifying neighbour of an analysed voxel will be placed in the set adjacentVoxelList,
        if they qualify when their analysis comes, they will be placed in voxelList.
    rad : TYPE: Integer
        DESCRIPTION: Radius of the neighbourhood to be taken into account when filling. Exclusively used when calling getVoxel
    data_new : TYPE: Array
        DESCRIPTION: Mask in the making. See fill function
    work : TYPE : String, optional
        DESCRIPTION: The dimension. ('2D','3D')
        The default is '2D'. Allows working in 2 or 3 dimensions. Switch to '3D'

    Returns
    -------
    voxelList : TYPE: set
        DESCRIPTION: The original voxelList (see arguments) in which we have added all qualifying voxels of the image.

    """

    if work == '3D':
        (x, y, z) = voxel
        adjacentVoxelList = set()
        for i in range(rad):
            h = int(np.sqrt(rad ** 2 - i ** 2))
            for j in range(-h, h + 1):
                for k in range(-1, 2):
                    adjacentVoxelList.add((x + i, y + j, z + k))
                    adjacentVoxelList.add((x - i, y + j, z + k))
        for adja in adjacentVoxelList:
            if isInbound(adja, data, work):
                if data[adja] >= init_val and data_new[adja] != 1:
                    voxelList.add(adja)
    else:
        pixel = voxel
        (x, y) = pixel
        adjacentPixelList = set()
        for i in range(rad):
            h = int(np.sqrt(rad ** 2 - i ** 2))
            for j in range(-h, h + 1):
                adjacentPixelList.add((x + i, y + j))
                adjacentPixelList.add((x - i, y + j))
        for adja in adjacentPixelList:
            if isInbound(adja, data, work):
                if data[adja] >= init_val and data_new[adja] != 1:
                    voxelList.add(adja)
    return voxelList

def isInbound(voxel, data, work='2D'):
    """
    Boolean function to know if still in bound. To assess the borders of an array

    Parameters
    ----------
    voxel : TYPE: Tuple
        DESCRIPTION: Contains the coordinates (x,y) in work='2D' or (x,y,z) in work='3D' of the voxel to be analysed
    data : TYPE: Array
        DESCRIPTION: image array
    work : TYPE : String, optional
        DESCRIPTION: The dimension. ('2D','3D')
        The default is '2D'. Allows working in 2 or 3 dimensions. Switch to '3D'

    Returns
    -------
    TYPE: Boolean
        DESCRIPTION: True if the voxel lies within the image(array). False otherwise

    """
    if work == '3D':
        return voxel[0] < data.shape[0] and voxel[0] >= 0 and voxel[1] < data.shape[1] and voxel[1] >= 0 and voxel[
            2] < \
               data.shape[2] and voxel[2] >= 0
    else:
        return voxel[0] < data.shape[0] and voxel[0] >= 0 and voxel[1] < data.shape[1] and voxel[1] >= 0

def binsearch(img, x2, y2, z=0, x1=0, y1=0, work='2D'):
    """
    This function is a binary search tree that works using a recursive call.

    Parameters
    ----------
    img : TYPE: Array
        DESCRIPTION: 2 or 3D array of an image
    x2 : TYPE: Integer
        DESCRIPTION: Lower cut of the image on x-axis
    y2 : TYPE: Integer
        DESCRIPTION: Lower cut of the image on y-axis
    z : TYPE, optional
        DESCRIPTION. The default is 0.
    x1 : TYPE, optional: Integer
        DESCRIPTION. Upper cut of the image on x-axis. The default is 0.
    y1 : TYPE, optional: Integer
        DESCRIPTION. Upper cut of the image on y-axis. The default is 0.
    work : TYPE : String, optional
        DESCRIPTION: The dimension. ('2D','3D')
        The default is '2D'. Allows working in 2 or 3 dimensions. Switch to '3D'

    Returns
    -------
    TYPE Integer
        DESCRIPTION: Returns the best

    """
    if work == '3D':
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return (x1, y1, z // 2)
        cand = [[[x1, (x1 + x2) // 2], [y1, (y1 + y2) // 2]], [[(x1 + x2) // 2, x2], [y1, (y1 + y2) // 2]],
                [[x1, (x1 + x2) // 2], [(y1 + y2) // 2, y2]], [[(x1 + x2) // 2, x2], [(y1 + y2) // 2, y2]]]

        Im1 = img[cand[0][0][0]:cand[0][0][1], cand[0][1][0]:cand[0][1][1], z // 2]
        Im2 = img[cand[1][0][0]:cand[1][0][1], cand[1][1][0]:cand[1][1][1], z // 2]
        Im3 = img[cand[2][0][0]:cand[2][0][1], cand[2][1][0]:cand[2][1][1], z // 2]
        Im4 = img[cand[3][0][0]:cand[3][0][1], cand[3][1][0]:cand[3][1][1], z // 2]

        idx = search([Im1, Im2, Im3, Im4], [0, 1, 2, 3])

    else:
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return (x1, y1)
        cand = [[[x1, (x1 + x2) // 2], [y1, (y2 + y1) // 2]], [[(x1 + x2) // 2, x2], [y1, (y2 + y1) // 2]],
                [[x1, (x1 + x2) // 2], [(y1 + y2) // 2, y2]], [[(x1 + x2) // 2, x2], [(y1 + y2) // 2, y2]]]
        Im1 = img[cand[0][0][0]:cand[0][0][1], cand[0][1][0]:cand[0][1][1]]
        Im2 = img[cand[1][0][0]:cand[1][0][1], cand[1][1][0]:cand[1][1][1]]
        Im3 = img[cand[2][0][0]:cand[2][0][1], cand[2][1][0]:cand[2][1][1]]
        Im4 = img[cand[3][0][0]:cand[3][0][1], cand[3][1][0]:cand[3][1][1]]

        idx = search([Im1, Im2, Im3, Im4], [0, 1, 2, 3])
    return binsearch(img, cand[idx][0][1], cand[idx][1][1], z=z, x1=cand[idx][0][0], y1=cand[idx][1][0], work=work)

def search(l, idx):
    """


    Parameters
    ----------
    l : TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if len(idx) == 1: return idx[0]
    newidx = []
    for i in range(0, len(idx), 2):
        if np.mean(l[idx[i]]) < np.mean(l[idx[i + 1]]):
            newidx.append(idx[i + 1])
        else:
            newidx.append(idx[i])
    return search(l, newidx)

def shape_matrix(radius, shape='ball', height=5, work='2D'):
    """


    Parameters
    ----------
    radius : TYPE: float
        DESCRIPTION: the radius of the sphere (3D) or circle (2D)
    shape : TYPE, optional: String
        DESCRIPTION. The default is 'ball'. Only in work='3D' is this parameter important. Switch between 'ball' or 'cylinder' in order to choose the form of the geometric shape that the matrix will depict
    height : TYPE, optional: Integer
        DESCRIPTION. The default is 5. If the shape is 'cylinder' then a height is to be input. No use if work='2D' or shape='ball'
    work : TYPE : String, optional
        DESCRIPTION: The dimension. ('2D','3D')
        The default is '2D'. Allows working in 2 or 3 dimensions. Switch to '3D'

    Returns
    -------
    mat : TYPE
        DESCRIPTION.

    """
    if work == '3D':
        mat = 0
        if shape == 'cylinder':
            mat = np.zeros((2 * radius + 1, 2 * radius + 1, height))
            for i in range(radius):
                h = int(np.sqrt(radius ** 2 - i ** 2))
                for j in range(-h, h + 1):
                    for k in range(height):
                        mat[radius + i, radius + j, k] = 1
                        mat[radius - i, radius + j, k] = 1

        else:
            mat = np.zeros((2 * radius + 1, 2 * radius + 1, 2 * radius + 1))
            for i in range(radius):
                h = int(np.sqrt(radius ** 2 - i ** 2))
                for j in range(-h, h + 1):
                    h1 = int(np.sqrt(radius ** 2 - i ** 2 - j ** 2))
                    for k in range(-h1, h1 + 1):
                        mat[radius + i, radius + j, radius + k] = 1
                        mat[radius - i, radius + j, radius + k] = 1
    else:
        mat = 0
        mat = np.zeros((2 * radius + 1, 2 * radius + 1))
        for i in range(radius):
            h = int(np.sqrt(radius ** 2 - i ** 2))
            for j in range(-h, h + 1):
                mat[radius + i, radius + j] = 1
                mat[radius - i, radius + j] = 1
    return mat


def pos_reader(data, work='2D'):
    """

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    work : TYPE : String, optional
        DESCRIPTION: The dimension. ('2D','3D')
        The default is '2D'. Allows working in 2 or 3 dimensions. Switch to '3D'

    Returns
    -------
    pos_list : TYPE
        DESCRIPTION.

    """
    pos_list = []
    if work == '2D':
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                if data[i, j] != 0:
                    pos_list.append((i, j))

    elif work == '3D':
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                for k in range(np.shape(data)[2]):
                    if data[i, j, k] != 0:
                        pos_list.append((i, j, k))
    return pos_list

def AffineSwitch(affine, to='2D'):
    if to == '2D' and affine.shape == (4, 4):
        affine = np.delete(affine, 2, 0)
        affine = np.delete(affine, 2, 1)
    elif to == '3D' and affine.shape == (3, 3):
        temp = np.zeros((4, 4))
        temp[:2, :2] = affine[:2, :2]
        temp[:2, 3] = affine[:2, 2]
        temp[2:, 2:] = np.eye(2)
        affine = temp
    if np.linalg.det(affine) == 0.0:
        for i in range(affine.shape[0]):
            if affine[i, i] == 0.0:
                affine[i, i] = 1
    return affine

