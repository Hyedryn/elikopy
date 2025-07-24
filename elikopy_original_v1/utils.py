import elikopy.utilsSynb0Disco as util
import nibabel as nib
import numpy as np
from random import shuffle
import math
import glob
import sys
import datetime
import time
import os
import json
import shutil
import matplotlib.pyplot

from future.utils import iteritems
import subprocess

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sf_to_sh


def submit_job(job_info):
    """
    Submit a job to the Slurm Workload Manager using a crafted sbatch.

    :param job_info: The parameters to use in the sbatch.
    :return job_id: The id of the submited job.
    """
    # Construct sbatch command
    slurm_cmd = ["sbatch"]
    script = False
    for key, value in iteritems(job_info):
        # Check for special case keys
        if key == "cpus_per_task":
            key = "cpus-per-task"
        if key == "mem_per_cpu":
            key = "mem-per-cpu"
        if key == "mail_user":
            key = "mail-user"
        if key == "mail_type":
            key = "mail-type"
        if key == "job_name":
            key = "job-name"
        elif key == "script":
            script = True
            continue
        slurm_cmd.append("--%s=%s" % (key, value))
    if script:
        slurm_cmd.append(job_info["script"])
    slurm_cmd.append("--hint=multithread")
    print("[INFO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") +
          ": Generated slurm batch command: '%s'" % slurm_cmd)

    # Run sbatch command as subprocess.
    try:
        sbatch_output = subprocess.check_output(slurm_cmd)
    except subprocess.CalledProcessError as e:
        # Print error message from sbatch for easier debugging, then pass on exception
        if sbatch_output is not None:
            print("ERROR: Subprocess call output: %s" % sbatch_output)
        raise e

    # Parse job id from sbatch output.
    sbatch_output = sbatch_output.decode().strip("\n ").split()
    for s in sbatch_output:
        if s.isdigit():
            job_id = int(s)
            return job_id
            # break


def anonymise_nifti(rootdir, anonymize_json, rename):
    """
    Anonymise all nifti present in rootdir by removing the PatientName and PatientBirthDate (only month and day) in the json and
    renaming the nifti file name to the PatientID.

    :param rootdir: Folder containing all the nifti to anonimyse.
    :param anonymize_json: If true, edit the json to remove the PatientName and replace the PatientBirthDate by the year of birth.
    :param rename: If true, rename the nifti to the PatientID.
    """
    import json
    import os

    extensions1 = ('.json')
    extensions2 = ('.gz', '.bval', '.bvec', '.json')

    name_key = {}
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in extensions1:
                print('json: ', os.path.join(subdir, file))
                f = open(os.path.join(subdir, file), 'r+')
                data = json.load(f)

                print(data.get('PatientID'))
                print()
                name_key.update(
                    {os.path.splitext(file)[0]: data.get('PatientID')})

                if anonymize_json:
                    data["PatientName"] = data.get('PatientID')
                    data["PatientBirthDate"] = data.get('PatientBirthDate')[:-3]
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
                f.close()
    print()
    print('Dict: ' + str(name_key))
    print()
    print()
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions2:
                print("Processing " + file)
                if ext in '.gz':
                    ext = r'.nii.gz'
                    ID = name_key.get(os.path.splitext(
                        os.path.splitext(file)[0])[0])
                else:
                    ID = name_key.get(os.path.splitext(file)[0])

                if ID:
                    new_file = ID + "_DTI" + ext
                    new_path = os.path.join(subdir, new_file)
                    old_path = os.path.join(subdir, file)
                    print("New path: " + new_path)
                    print("Old path: " + old_path)
                    if rename:
                        os.rename(old_path, new_path)
                else:
                    print("ID is none " + file)
                    print(name_key)

                print()


def getJobsState(folder_path, job_list, step_name):
    """
    Periodically checks the status of all jobs in the job_list. When a job status change to complete or a failing state.
    Write the status in the log and remove the job from the job_list. This function end when all jobs are completed or failed.

    :param folder_path: The path to the root dir of the study (used to write the logs.txt file)
    :param job_list: The list of job to check for state update
    :param step_name: The string value of the prefix to put in the log file
    """
    job_info = {}
    job_failed = []
    job_successed = []
    while job_list:
        for job_data in job_list[:]:
            job_info["job_state"] = get_job_state(job_data["id"])
            if job_info["job_state"] == 'COMPLETED':
                job_list.remove(job_data)
                f = open(folder_path + "/logs.txt", "a+")
                f.write("["+step_name+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(
                    job_data) + " COMPLETED\n")
                f.close()
                job_successed.append(job_data["name"])
            if job_info["job_state"] == 'FAILED':
                job_list.remove(job_data)
                f = open(folder_path + "/logs.txt", "a+")
                f.write("["+step_name+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(
                    job_data) + " FAILED\n")
                f.close()
                job_failed.append(job_data["name"])
            if job_info["job_state"] == 'OUT_OF_MEMORY':
                job_list.remove(job_data)
                f = open(folder_path + "/logs.txt", "a+")
                f.write("["+step_name+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(
                    job_data) + " OUT_OF_MEMORY\n")
                f.close()
                job_failed.append(job_data["name"])
            if job_info["job_state"] == 'TIMEOUT':
                job_list.remove(job_data)
                f = open(folder_path + "/logs.txt", "a+")
                f.write("["+step_name+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(
                    job_data) + " TIMEOUT\n")
                f.close()
                job_failed.append(job_data["name"])
            if job_info["job_state"] == 'CANCELLED':
                job_list.remove(job_data)
                f = open(folder_path + "/logs.txt", "a+")
                f.write("["+step_name+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(
                    job_data) + " CANCELLED\n")
                f.close()
                job_failed.append(job_data["name"])
        time.sleep(30)

    f = open(folder_path + "/logs.txt", "a+")
    f.write("[" + step_name + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": List of successful jobs:\n " + str(
        job_successed) + "\n")
    f.write("[" + step_name + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": List of failed jobs:\n " + str(
        job_failed) + "\n")
    f.close()


def export_files(folder_path, step, patient_list_m=None):
    """
    Creates an export folder in the root folder containing the results of 'step' for each patient in a single folder

    example : export_files('user/my_rootfolder', 'dMRI/microstructure/dti')

    :param folder_path: root folder
    :param step: step to export
    :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
    """

    export_path = folder_path + "/export_" + step.rsplit('/', 1)[1]
    if not (os.path.exists(export_path)):
        try:
            os.makedirs(export_path)
        except OSError:
            print("Creation of the directory %s failed" % export_path)
        else:
            print("Successfully created the directory %s " % export_path)

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    if patient_list_m:
        patient_list = patient_list_m

    for p in patient_list:
        copy_path = folder_path + '/subjects/' + \
            os.path.splitext(p)[0] + '/' + step
        shutil.copytree(copy_path, export_path, dirs_exist_ok=True)

    shutil.copyfile(folder_path + "/subjects/subj_list.json",
                    export_path + "/subj_list.json")
    shutil.copyfile(folder_path + "/subjects/subj_error.json",
                    export_path + "/subj_error.json")
    shutil.copyfile(folder_path + "/subjects/subj_type.json",
                    export_path + "/subj_type.json")


def get_job_state(job_id):
    """
    Retrieve the state of a job through the sacct bash command offered by the lurm Workload Manager.
    :param job_id: The id of the job to retrieve the state of.
    :return state: The string value representing the state of the job.
    """
    cmd = "sacct --jobs=" + str(job_id) + " -n -o state"

    proc = subprocess.Popen(cmd, universal_newlines=True,
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    try:
        state = (out.partition('\n')[0]).rstrip().strip()
    except Exception:
        try:
            state = out.rstrip().strip()
        except Exception:
            print("Double error" + out)
            state = "NOSTATE"
    return state


def makedir(dir_path, log_path, log_prefix):
    """
    Create a directory in the location specified by the dir_path and write the log in the log_path.

    :param dir_path: The path to the directory to create.
    :param log_path: The path to the log file to write verbose data.
    :param log_prefix: The prefix to use in the log file.
    """
    if not(os.path.exists(dir_path)):
        try:
            os.makedirs(dir_path)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)
            f = open(log_path, "a+")
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dir_path)
            f.close()
        else:
            print("Successfully created the directory %s " % dir_path)
            f = open(log_path, "a+")
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dir_path)
            f.close()


def synb0DisCo(folder_path, topuppath, patient_path, static_files_path=None, starting_step=None, topup=True, gpu=True):
    """
    synb0DISCO adapted from https://github.com/MASILab/Synb0-DISCO

    :param folder_path: path to the root directory.
    :param topuppath: Path to the subject's topup folder.
    :param patient_path: Name of the subject.
    :param starting_step: Define the starting step, usefull if previous step had already been run.
    :param topup: If true, topup will be perfomed after synb0Disco.
    :param gpu: If true, torch will use the gpu.
    :rtype: object
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    assert starting_step in (None, "Registration",
                             "Inference", "Apply", "topup")

    static_files_path = folder_path + \
        "/static_files" if static_files_path is None else static_files_path

    synb0path = topuppath + "/synb0-DisCo"

    if starting_step == None:
        """
        Step 1 - Normalize T1
        """

        mri_convert_T1 = "mri_convert " + synb0path + \
            "/T1.nii.gz " + synb0path + "/T1.mgz"

        n3_correction = "mri_nu_correct.mni --i " + synb0path + \
            "/T1.mgz --o " + synb0path + "/T1_N3.mgz --n 2"

        mri_convert_N3 = "mri_convert " + synb0path + \
            "/T1_N3.mgz " + synb0path + "/T1_N3.nii.gz"

        mri_normalize = "mri_normalize -g 1 -mprage " + \
            synb0path + "/T1_N3.mgz " + synb0path + "/T1_norm.mgz"

        mri_convert_norm = "mri_convert " + synb0path + \
            "/T1_norm.mgz " + synb0path + "/T1_norm.nii.gz"

        bashCommand_step1 = mri_convert_T1 + "; " + n3_correction + "; " + \
            mri_convert_N3 + "; " + mri_normalize + "; " + mri_convert_norm
        step1_log = open(synb0path + "/step1_logs.txt", "a+")
        step1_log.write("[SynB0DISCO] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of step 1 \n\n")
        step1_log.flush()
        process = subprocess.Popen(bashCommand_step1, universal_newlines=True, shell=True, stdout=step1_log,
                                   stderr=subprocess.STDOUT)

        output, error = process.communicate()
        step1_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of step 1 \n\n")
        step1_log.close()

    if starting_step in (None, "Registration"):
        """
        Step 2 - Registration
        """

        # Skull strip T1

        bet = shutil.copyfile(os.path.join(folder_path,"subjects",patient_path,"T1",f"{patient_path}_T1_brain.nii.gz"), synb0path + "/T1_mask.nii.gz")

        # epi_reg distorted b0 to T1; wont be perfect since B0 is distorted

        epi_reg_b0_dist = "epi_reg --epi=" + synb0path + "/b0.nii.gz  --t1=" + synb0path + \
            "/T1.nii.gz --t1brain=" + synb0path + \
            "/T1_mask.nii.gz --out=" + synb0path + "/epi_reg_d"

        # Convert FSL transform to ANTS transform
        c3d_affine_tool = "c3d_affine_tool -ref " + synb0path + "/T1.nii.gz -src " + synb0path + \
            "/b0.nii.gz " + synb0path + "/epi_reg_d.mat -fsl2ras -oitk " + \
            synb0path + "/epi_reg_d_ANTS.txt"

        # ANTs register T1 to atlas
        antsRegistrationSyNQuick = "antsRegistrationSyNQuick.sh -d 3 -f " + static_files_path + \
            "/atlases/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz -m " + \
            synb0path + "/T1.nii.gz -o " + synb0path + "/ANTS"

        # Apply linear transform to normalized T1 to get it into atlas space
        antsApplyTransforms_lin_T1 = "antsApplyTransforms -d 3 -i " + synb0path + "/T1_norm.nii.gz -r " + static_files_path + \
            "/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + synb0path + \
            "/ANTS0GenericAffine.mat -o " + synb0path + "/T1_norm_lin_atlas_2_5.nii.gz"

        # Apply linear transform to distorted b0 to get it into atlas space
        antsApplyTransforms_lin_b0 = "antsApplyTransforms -d 3 -i " + synb0path + "/b0.nii.gz -r " + static_files_path + "/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + \
            synb0path + "/ANTS0GenericAffine.mat -t " + synb0path + \
            "/epi_reg_d_ANTS.txt -o " + synb0path + "/b0_d_lin_atlas_2_5.nii.gz"

        # Apply nonlinear transform to normalized T1 to get it into atlas space
        antsApplyTransforms_nonlin_T1 = "antsApplyTransforms -d 3 -i " + synb0path + "/T1_norm.nii.gz -r " + static_files_path + "/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + \
            synb0path + "/ANTS1Warp.nii.gz -t " + synb0path + \
            "/ANTS0GenericAffine.mat -o " + synb0path + "/T1_norm_nonlin_atlas_2_5.nii.gz"

        # Apply nonlinear transform to distorted b0 to get it into atlas space
        antsApplyTransforms_nonlin_b0 = "antsApplyTransforms -d 3 -i " + synb0path + "/b0.nii.gz -r " + static_files_path + "/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + \
            synb0path + "/ANTS1Warp.nii.gz -t " + synb0path + "/ANTS0GenericAffine.mat -t " + \
            synb0path + "/epi_reg_d_ANTS.txt -o " + \
            synb0path + "/b0_d_nonlin_atlas_2_5.nii.gz"

        bashCommand_step2 = bet + "; " + epi_reg_b0_dist + "; " + c3d_affine_tool + "; " + antsRegistrationSyNQuick + "; " + \
            antsApplyTransforms_lin_T1 + "; " + antsApplyTransforms_lin_b0 + "; " + \
            antsApplyTransforms_nonlin_T1 + "; " + antsApplyTransforms_nonlin_b0
        step2_log = open(synb0path + "/step2_logs.txt", "a+")
        step2_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of step 2 \n\n")
        step2_log.flush()
        process = subprocess.Popen(bashCommand_step2, universal_newlines=True, shell=True, stdout=step2_log,
                                   stderr=subprocess.STDOUT)

        output, error = process.communicate()
        step2_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of step 2 \n\n")
        step2_log.close()

    if starting_step in (None, "Registration", "Inference"):
        """
        Step 3 -  Run inference
        """
        step3_log = open(synb0path + "/step3_logs.txt", "a+")
        step3_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of step 3 \n\n")
        numfold = 5
        # Get device
        if gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        T1_input_path = synb0path + "/T1_norm_lin_atlas_2_5.nii.gz"
        b0_input_path = synb0path + "/b0_d_lin_atlas_2_5.nii.gz"

        from elikopy.modelSynb0Disco import UNet3D

        import glob

        for i in range(1, numfold+1):
            torch.cuda.empty_cache()
            b0_output_path = synb0path + \
                "/b0_u_lin_atlas_2_5_FOLD_" + str(i) + ".nii.gz"
            model_path = static_files_path + "/dual_channel_unet/num_fold_" + str(i) + "_total_folds_" + str(
                numfold) + "_seed_1_num_epochs_100_lr_0.0001_betas_(0.9, 0.999)_weight_decay_1e-05_num_epoch_*.pth"
            model_path = glob.glob(model_path)[0]
            # Get model
            model = UNet3D(2, 1).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            # Inference
            step3_log.write("[SynB0DISCO] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Beginning of inference " + str(i) + "\n\n")
            step3_log.flush()
            img_model = inference(T1_input_path, b0_input_path, model, device)

            # Save
            nii_template = nib.load(b0_input_path)
            nii = nib.Nifti1Image(util.torch2nii(
                img_model.detach().cpu()), nii_template.affine, nii_template.header)
            nib.save(nii, b0_output_path)

        step3_log.write("[SynB0DISCO] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of step 3 \n\n")
        step3_log.close()

    if starting_step in (None, "Registration", "Inference", "Apply"):
        """
        Step 4 -  Apply
        """

        # Take mean
        mean_merge = "fslmerge -t " + synb0path + "/b0_u_lin_atlas_2_5_merged.nii.gz " + \
            synb0path + "/b0_u_lin_atlas_2_5_FOLD_*.nii.gz"
        mean_math = "fslmaths " + synb0path + "/b0_u_lin_atlas_2_5_merged.nii.gz -Tmean " + \
            synb0path + "/b0_u_lin_atlas_2_5.nii.gz"

        # Apply inverse xform to undistorted b0
        antsApplyTransforms_inv_xform = "antsApplyTransforms -d 3 -i " + synb0path + "/b0_u_lin_atlas_2_5.nii.gz -r " + synb0path + \
            "/b0.nii.gz -n BSpline -t [" + synb0path + "/epi_reg_d_ANTS.txt,1] -t [" + \
            synb0path + "/ANTS0GenericAffine.mat,1] -o " + synb0path + "/b0_u.nii.gz"

        # Smooth image
        smooth_math = "fslmaths " + synb0path + \
            "/b0.nii.gz -s 1.15 " + synb0path + "/b0_d_smooth.nii.gz"

        # Merge for topup
        merge_image = "fslmerge -t " + synb0path + "/b0_all.nii.gz " + \
            synb0path + "/b0_d_smooth.nii.gz " + synb0path + "/b0_u.nii.gz"

        bashCommand_step4 = mean_merge + "; " + mean_math + "; " + \
            antsApplyTransforms_inv_xform + "; " + smooth_math + "; " + merge_image
        step4_log = open(synb0path + "/step4_logs.txt", "a+")
        step4_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of step 4 \n\n")
        process = subprocess.Popen(bashCommand_step4, universal_newlines=True, shell=True, stdout=step4_log,
                                   stderr=subprocess.STDOUT)

        output, error = process.communicate()

        with open(synb0path + '/' + 'acqparams_topup.txt') as f:
            topup_acq = [[float(x) for x in line2.split()] for line2 in f]

        topup_acq.append(
            [topup_acq[0][0], - topup_acq[0][1], topup_acq[0][2], 0])

        print(topup_acq)

        with open(synb0path + '/' + "acqparams_topup.txt", 'w') as file:
            file.writelines(' '.join(str(j)
                            for j in i) + '\n' for i in topup_acq)

        step4_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of step 4 \n\n")
        step4_log.close()

    if starting_step in (None, "Registration", "Inference", "Apply", "topup") and topup:
        run_topup = "topup -v --imain=" + synb0path + "/b0_all.nii.gz --datain=" + synb0path + "/acqparams_topup.txt --config=b02b0.cnf --iout=" + topuppath + "/" + patient_path + "_topup_iout_estimate --out=" + topuppath + "/" + patient_path + \
            "_topup_estimate --subsamp=1,1,1,1,1,1,1,1,1 --miter=10,10,10,10,10,20,20,30,30 --lambda=0.00033,0.000067,0.0000067,0.000001,0.00000033,0.000000033,0.0000000033,0.000000000033,0.00000000000067 --scale=0 " + \
            '--fout="' + topuppath + '/' + patient_path + '_topup_fout_estimate" '
        bashCommand_topup = run_topup
        topup_log = open(topuppath + "/topup_logs.txt", "a+")
        topup_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of topup \n\n")
        process = subprocess.Popen(bashCommand_topup, universal_newlines=True, shell=True, stdout=topup_log,
                                   stderr=subprocess.STDOUT)

        output, error = process.communicate()
        topup_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of topup \n\n")
        topup_log.close()


#from torchvision import datasets, transforms


def inference(T1_path, b0_d_path, model, device):
    """ synb0DISCO adapted from https://github.com/MASILab/Synb0-DISCO

    :param T1_path: Path to the normalized projected T1.
    :param b0_d_path: Path to the b0 atlases.
    :param model: DL Model
    :param device: Define if cuda or cpu is used.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    # Eval mode
    model.eval()

    # Get image
    img_T1 = np.expand_dims(util.get_nii_img(T1_path), axis=3)
    img_b0_d = np.expand_dims(util.get_nii_img(b0_d_path), axis=3)

    # Pad array since I stupidly used template with dimensions not factorable by 8
    # Assumes input is (77, 91, 77) and pad to (80, 96, 80) with zeros
    img_T1 = np.pad(img_T1, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_b0_d = np.pad(img_b0_d, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')

    # Convert to torch img format
    img_T1 = util.nii2torch(img_T1)
    img_b0_d = util.nii2torch(img_b0_d)

    # Normalize data
    img_T1 = util.normalize_img(img_T1, 150, 0, 1, -1)
    max_img_b0_d = np.percentile(img_b0_d, 99)
    min_img_b0_d = 0
    img_b0_d = util.normalize_img(img_b0_d, max_img_b0_d, min_img_b0_d, 1, -1)

    # Set "data"
    img_data = np.concatenate((img_b0_d, img_T1), axis=1)

    # Send data to device
    img_data = torch.from_numpy(img_data).float().to(device)

    # Pass through model
    img_model = model(img_data)

    # Unnormalize model
    img_model = util.unnormalize_img(
        img_model, max_img_b0_d, min_img_b0_d, 1, -1)

    # Remove padding
    img_model = img_model[:, :, 2:-1, 2:-1, 3:-2]

    # Return model
    return img_model


def get_patient_list_by_types(folder_path, type=None):
    """Print the list of patient corresponding to a specfic type of patient.

    :param folder_path: Path to the root folder of the study.
    :param type: The selected type
    """

    import json
    import os

    # open the subject list and subj_type dic
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)
    dest_subj_type = folder_path + "/subjects/subj_type.json"
    with open(dest_subj_type, 'r') as f:
        subj_type = json.load(f)

    patients_by_id = {}

    for p in patient_list:
        patient_path = os.path.splitext(p)[0]
        control_info = subj_type[patient_path]
        if control_info not in patients_by_id:
            patients_by_id[control_info] = []
        patients_by_id[control_info].append(patient_path)

    if type:
        print(patients_by_id.get(type, "Type not found!"))
    else:
        for key, value in patients_by_id.items():
            print("Patient list of cat " + str(key) + ": \n")
            print(value)
            print("\n")


def merge_all_reports(folder_path):
    """ Merge all subjects quality control reports into a single report.

    :param folder_path: Path to the root folder of the study.
    """
    from pypdf import PdfWriter, PdfReader
    import json
    import os

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    writer = PdfWriter()

    for p in patient_list:
        patient_path = os.path.splitext(p)[0]
        pdf_path = folder_path + '/subjects/' + patient_path + '/quality_control.pdf'
        if (os.path.exists(pdf_path)):
            reader = PdfReader(pdf_path)
            for i in range(len(reader.pages)):
                page = reader.pages[i]
                page.compress_content_streams()
                writer.add_page(page)

    with open(folder_path + '/quality_control_all_tmp.pdf', 'wb') as f:
        writer.write(f)

    # try to compress pdf with ghostscript
    bashCommand = "command -v gs && gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=" + folder_path + \
        '/quality_control_all.pdf ' + folder_path + '/quality_control_all_tmp.pdf || mv ' + \
        folder_path + '/quality_control_all_tmp.pdf ' + \
        folder_path + '/quality_control_all.pdf'
    bashcmd = bashCommand.split()
    print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True,
                               stderr=subprocess.STDOUT)
    output, error = process.communicate()


def merge_all_specific_reports(folder_path, merge_wm_report=False, merge_legacy_report=False):
    """ Merge all selected specific subject's report into a single big report.

    :param folder_path: Path to the root folder of the study.
    :param merge_wm_report: Select wm report.
    :param merge_legacy_report: Select legacy report.
    """
    from pypdf import PdfWriter, PdfReader
    import json
    import os

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    if merge_wm_report:
        wm_writer = PdfWriter()

    if merge_legacy_report:
        legacy_writer = PdfWriter()

    for p in patient_list:
        patient_path = os.path.splitext(p)[0]

        if merge_wm_report:
            pdf_path = folder_path + '/subjects/' + patient_path + \
                '/masks/quality_control/qc_report.pdf'
            if (os.path.exists(pdf_path)):
                reader = PdfReader(pdf_path)
                for i in range(len(reader.pages)):
                    page = reader.pages[i]
                    page.compress_content_streams()
                    wm_writer.add_page(page)

        if merge_legacy_report:
            pdf_path = folder_path + '/subjects/' + patient_path + \
                '/report/report_' + patient_path + '.pdf'
            if (os.path.exists(pdf_path)):
                reader = PdfReader(pdf_path)
                for i in range(len(reader.pages)):
                    page = reader.pages[i]
                    page.compress_content_streams()
                    legacy_writer.add_page(page)

    if merge_wm_report:
        with open(folder_path + '/wm_mask_qc_report_all.pdf', 'wb') as f:
            wm_writer.write(f)

    if merge_legacy_report:
        with open(folder_path + '/legacy_report_all.pdf', 'wb') as f:
            legacy_writer.write(f)


def deltas_to_D(dx: float, dy: float, dz: float, lamb=np.diag([1, 0, 0]),
                vec_len: float = 500):
    """     Function creating a diffusion tensor from three orthogonal components.
    Can raises np.linalg.LinAlgError
    @author: DELINTE  Nicolas

    :param dx: float 'x' component.
    :param dy: float 'y' component.
    :param dz: float 'z' component.
    :param lamb: 3x3 array, optional. Diagonal matrix containing the diffusion eigenvalues. The default is np.diag([1, 0, 0]).
    :param vec_len: float, optional. Value decreasing the diffusion. The default is 500.
    :return: D : 3x3 array. Matrix containing the diffusion tensor.
    """

    e = np.array([[dx, -dz-dy, dy*dx-dx*dz],
                  [dy, dx, -dx**2-(dz+dy)*dz],
                  [dz, dx, dx**2+(dy+dz)*dy]])

    try:
        e_1 = np.linalg.inv(e)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError

    D = (e.dot(lamb)).dot(e_1)/vec_len

    return D


def peak_to_tensor(peaks, norm=None, pixdim=[2, 2, 2]):
    """ Takes peaks, such as the ones obtained with Microstructure Fingerprinting,
    and return the corresponding tensor, in the format used in DIAMOND.
    @author: DELINTE  Nicolas

    :param peaks: 4-D array containing the peaks of shape (x,y,z,3)
    :return: t, a 5-D array Tensor array of shape (x,y,z,1,6).
    """

    t = np.zeros(peaks.shape[:3]+(1, 6))

    scaleFactor = 1000 / min(pixdim)

    for xyz in np.ndindex(peaks.shape[:3]):

        if peaks[xyz].all() == 0:
            continue

        dx, dy, dz = peaks[xyz]

        try:
            if norm is not None:
                D = deltas_to_D(dx, dy, dz, vec_len=scaleFactor/norm[xyz])
            else:
                D = deltas_to_D(dx, dy, dz, vec_len=scaleFactor)
        except np.linalg.LinAlgError:
            continue

        t[xyz+(0, 0)] = D[0, 0]
        t[xyz+(0, 1)] = D[0, 1]
        t[xyz+(0, 2)] = D[1, 1]
        t[xyz+(0, 3)] = D[0, 2]
        t[xyz+(0, 4)] = D[1, 2]
        t[xyz+(0, 5)] = D[2, 2]

    return t


def tensor_to_peak(t):
    """ Takes peaks, such as the ones obtained with DIAMOND, and return the
    corresponding tensor, in the format used in Microstructure Fingerprinting.
    @author: DELINTE  Nicolas

    :param t: 5-D array. Tensor array of shape (x,y,z,1,6).
    :return: peaks, a 4-D Array containing the peaks of shape (x,y,z,3)
    """

    if len(t.shape) == 4:
        t = t[..., np.newaxis]
        t = np.transpose(t, (1, 2, 3, 4, 0))

        D_t = np.array([[t[:, :, :, 0, 0], t[:, :, :, 0, 1], t[:, :, :, 0, 2]],
                        [t[:, :, :, 0, 1], t[:, :, :, 0, 3], t[:, :, :, 0, 4]],
                        [t[:, :, :, 0, 2], t[:, :, :, 0, 4], t[:, :, :, 0, 5]]]
                       )

    else:

        D_t = np.array([[t[:, :, :, 0, 0], t[:, :, :, 0, 1], t[:, :, :, 0, 3]],
                        [t[:, :, :, 0, 1], t[:, :, :, 0, 2], t[:, :, :, 0, 4]],
                        [t[:, :, :, 0, 3], t[:, :, :, 0, 4], t[:, :, :, 0, 5]]]
                       )

    D_t = np.transpose(D_t, (2, 3, 4, 0, 1))

    val_t, vec_t = np.linalg.eig(D_t)

    vol_shape = t.shape[0]*t.shape[1]*t.shape[2]

    vec_t = vec_t.reshape((vol_shape, 3, 3))
    vec_t = np.transpose(vec_t, (0, 2, 1))
    idx = np.argmax(val_t.reshape((vol_shape, 3)), axis=1)

    peaks = vec_t[range(vol_shape), idx].reshape(t.shape[:3]+(3,)).real

    return peaks


def _flip_m_neg(sh, sh_order: int, full_basis: bool = False):
    """
    :param sh: 4-D array. Spherical harmonics coefficient array of shape (x,y,z,coeff).
    :param sh_order: int. Order of the spherical harmonics.
    :param full_basis: bool, optional.If True, takes a full basis as input.
    The default is False.
    :return: 4-D array. Spherical harmonics coefficient array of shape (x,y,z,coeff).

    """

    counter = 0
    for l in range(sh_order.astype(int)):
        n = 1+2*l
        m_list = np.linspace((n-1)/2, -(n-1)/2, n)
        for m in m_list:
            if full_basis:
                if m % 2 == 0 and m < 0:
                    sh[:, :, :, counter] *= -1
                counter += 1
            else:
                if l % 2 == 0:
                    if m % 2 == 0 and m < 0:
                        sh[:, :, :, counter] *= -1
                    counter += 1

    return sh


def dipy_fod_to_mrtrix(sh):
    """
    Converts spherical harmonics (sh) file from dipy format to mrtrix format.
    Does not work with full basis, only symmetrical SH.

    :param sh: 4-D array. Spherical harmonics coefficient array of shape (x,y,z,coeff).
    :return: 4-D array. Spherical harmonics coefficient array of shape (x,y,z,coeff).

    """

    sh_order = (np.sqrt(sh.shape[3]*8+1)-3)/2

    sh = _flip_m_neg(sh, sh_order)

    default_sphere = get_sphere('repulsion724')

    temp = sh_to_sf(sh, sphere=default_sphere, sh_order=sh_order,
                    basis_type='descoteaux07', legacy=False)
    sh = sf_to_sh(temp, sphere=default_sphere, sh_order=sh_order,
                  basis_type='tournier07', legacy=False)

    return sh


def mrtrix_fod_to_dipy(sh):
    """
    Converts spherical harmonics (sh) file from mrtrix format to dipy format.
    Does not work with full basis, only symmetrical SH.

    :param sh: 4-D array. Spherical harmonics coefficient array of shape (x,y,z,coeff).
    :return: 4-D array. Spherical harmonics coefficient array of shape (x,y,z,coeff).

    """

    sh_order = (np.sqrt(sh.shape[3]*8+1)-3)/2

    default_sphere = get_sphere('repulsion724')

    temp = sh_to_sf(sh, sphere=default_sphere, sh_order=sh_order,
                    basis_type='tournier07', legacy=False)
    sh = sf_to_sh(temp, sphere=default_sphere, sh_order=sh_order,
                  basis_type='descoteaux07', legacy=False)

    sh = _flip_m_neg(sh, sh_order)

    return sh


def clean_mask(mask):
    from skimage.morphology import flood

    mask = mask.copy()
    mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    mask_filled = mask.copy()
    for x in range(mask.shape[0]):
        mask_filled[x, :, :] = flood(mask[x, :, :], (0, 0))
    mask = np.where(mask_filled == 0, 1, mask)

    mask_filled = mask.copy()
    for y in range(mask.shape[1]):
        mask_filled[:, y, :] = flood(mask[:, y, :], (0, 0))
    mask = np.where(mask_filled == 0, 1, mask)

    mask_filled = mask.copy()
    for z in range(mask.shape[2]):
        mask_filled[:, :, z] = flood(mask[:, :, z], (0, 0))
    mask = np.where(mask_filled == 0, 1, mask)

    center = tuple([np.average(indices) for indices in np.where(mask == 1)])
    center = tuple([int(point) for point in center])

    mask = flood(mask, center, connectivity=1)
    mask_cleaned = np.zeros((mask.shape))
    mask_cleaned[mask] = 1

    mask_cleaned = mask_cleaned[tuple(slice(1, dim - 1) for dim in mask_cleaned.shape)]

    return mask_cleaned


def get_acquisition_view(affine) -> str:
    '''
    Returns the acquisition view corresponding to the affine.

    Parameters
    ----------
    :param affine: 2-D array of shape (4,4).Array containing the affine information.
    :return str: String of the acquisition view, either 'axial', 'sagittal', 'coronal' or 'oblique'.
    '''

    affine = affine[:3, :3].copy()

    sum_whole_aff = np.sum(abs(affine))
    sum_diag_aff = np.sum(np.diag(abs(affine)))
    sum_extra_diag_aff = sum_whole_aff - sum_diag_aff

    a = affine[0, 0]
    b = affine[1, 1]
    c = affine[2, 2]
    d = affine[0, 2]
    e = affine[1, 0]
    f = affine[2, 1]
    g = affine[0, 1]
    h = affine[1, 2]
    i = affine[2, 0]

    if (a != 0 and b != 0 and c != 0 and sum_extra_diag_aff == 0):
        return "axial"
    elif (d != 0 and e != 0 and f != 0 and sum_diag_aff == 0):
        return "sagittal"
    elif (g != 0 and h != 0 and i != 0 and sum_diag_aff == 0):
        return "coronal"
    else:
        return "oblique"


def get_patient_ref(root: str, patient: str, suffix_length: int = 2):
    """
    Retrieve the reference patient from the patient list based on input patient ID.

    Args:
        root (str): Root directory containing the patient list JSON file.
        patient (str): Patient ID.
        suffix_length (int): Number of characters to trim from the patient ID for non-BIDS matching.

    Returns:
        str: Reference patient ID.
    """
    # Load patient list
    patient_list_path = os.path.join(root, 'subjects', 'subj_list.json')
    with open(patient_list_path, 'r') as f:
        patient_list = json.load(f)

    # Determine if the input patient uses BIDS format
    is_bids = "sub-" in patient and "ses-" in patient

    # Extract ses attributes for sorting
    def extract_ses(patient_id):
        ses_attr = next((attr for attr in patient_id.split("_") if "ses-" in attr), "")
        return ses_attr

    if is_bids:
        # Sort patient list by ses attribute
        patient_list.sort(key=extract_ses)
        patient_sub = next((patient_attr for patient_attr in patient.split("_") if "sub-" in patient_attr), None)
        # Process BIDS patient
        for p in patient_list:
            curr_patient_sub = next((patient_attr for patient_attr in p.split("_") if "sub-" in patient_attr), None)
            if patient_sub == curr_patient_sub:  # Match exact BIDS ID
                return p
        raise ValueError(f"No reference found for BIDS patient {patient}.")
    else:
        # Sort the patient list
        patient_list.sort()

        # Find the index of the patient in the list
        patient_index = patient_list.index(patient)
        # Process non-BIDS patient
        trimmed_names = [p[:-suffix_length] if len(p) > suffix_length else p for p in patient_list]

        unique_names, unique_indices = np.unique(trimmed_names, return_index=True)

        relative_positions = unique_indices - patient_index
        relative_positions[relative_positions > 0] -= len(patient_list)
        ref_idx = np.max(relative_positions)
        ref_index = (patient_index + ref_idx) % len(patient_list)
        return patient_list[ref_index]

def update_status(folder_path, p, step):
    json_status_file = os.path.join(folder_path, "subjects", p, f"{p}_status.json")
    if os.path.exists(json_status_file):
        with open(json_status_file, "r") as f:
            patient_status = json.load(f)
    else:
        patient_status = {}
    patient_status[step] = True

    with open(json_status_file, "w") as f:
        json.dump(patient_status, f, indent=6)
