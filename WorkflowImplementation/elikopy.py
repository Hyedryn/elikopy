"""
 
"""
import datetime
import os
import json
import math
import time
try:
    from WorkflowImplementation.utils import preproc_solo, dti_solo, submit_job
    print("Importation of WorkflowImplementation.utils is a success")
except ImportError:
    print("Warning: Importation of WorkflowImplementation.utils failed")
    ## check whether in the source directory...
try:
    from utils import preproc_solo, dti_solo, submit_job
    print("Importation of utils is a success")
except ImportError:
    ## check whether in the source directory...
    print("Warning: Importation of utils failed")


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
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[DICOM TO NIFTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning sequential dicom convertion\n")
    f.close()

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

    f=open(folder_path + "/out/logs.txt", "a+")
    for f in files:
        if "mrdc" in f or "MRDC" in f:
            shutil.move(folder_path + '/' + f, dest)

            f.write("[DICOM TO NIFTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Moved " + f + " to " + dest + "\n")
    f.close()


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
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dest)
            f.close()
        else:
            print("Successfully created the directory %s " % dest)
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dest)
            f.close()


    import json
    dest_error = folder_path + "/out/patient_error.json"
    with open(dest_error, 'w') as f:
        json.dump(error, f)

    dest_success = folder_path + "/out/patient_list.json"
    with open(dest_success, 'w') as f:
        json.dump(success, f)

    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient list generated\n")
    f.close()


def preproc(folder_path, eddy=False, denoising=False, slurm=False, reslice=False):
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

    if slurm:
        import pyslurm

    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ":  Beginning preprocessing with eddy:" + str(eddy) + ", denoising:" + str(denoising) + ", slurm:" + str(slurm) + "\n")
    f.close()

    dest_success = folder_path + "/out/patient_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    preproc_path = folder_path + "/out/preproc"
    if not (os.path.exists(preproc_path)):
        try:
            os.mkdir(preproc_path)
        except OSError:
            print("Creation of the directory %s failed" % preproc_path)
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ":  Creation of the directory %s failed\n" % preproc_path)
            f.close()
        else:
            print("Successfully created the directory %s " % preproc_path)
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s\n" % preproc_path)
            f.close()

    bet_path = folder_path + "/out/preproc/bet"
    if not (os.path.exists(bet_path)):
        try:
            os.mkdir(bet_path)
        except OSError:
            print("Creation of the directory %s failed" % bet_path)
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % bet_path)
            f.close()
        else:
            print("Successfully created the directory %s " % bet_path)
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s\n" % bet_path)
            f.close()

    final_path = folder_path + "/out/preproc/final"
    if not (os.path.exists(final_path)):
        try:
            os.mkdir(final_path)
        except OSError:
            print("Creation of the directory %s failed" % final_path)
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % final_path)
            f.close()
        else:
            print("Successfully created the directory %s " % final_path)
            f=open(folder_path + "/out/logs.txt", "a+")
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s\n" % final_path)
            f.close()


    job_list = []

    f=open(folder_path + "/out/logs.txt", "a+")
    for p in patient_list:
        if slurm:
            if not denoising and not eddy:
                p_job = {
                    "wrap": "python -c 'from utils import preproc_solo; preproc_solo(\"" + folder_path + "\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "01:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                }
            elif denoising and eddy:
                p_job = {
                    "wrap": "python -c 'from utils import preproc_solo; preproc_solo(\"" + folder_path + "\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 8,
                    "mem_per_cpu": 6096,
                    "time": "12:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                }
            elif denoising and not eddy:
                p_job = {
                    "wrap": "python -c 'from utils import preproc_solo; preproc_solo(\"" + folder_path + "\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 9096,
                    "time": "4:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                }
            elif not denoising and eddy:
                p_job = {
                    "wrap": "python -c 'from utils import preproc_solo; preproc_solo(\"" + folder_path + "\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 4,
                    "mem_per_cpu": 6096,
                    "time": "12:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                }
            else:
                p_job = {
                    "wrap": "python -c 'from utils import preproc_solo; preproc_solo(\"" + folder_path + "\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "1:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            preproc_solo(folder_path,p,eddy,denoising,reslice)
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully preproceced patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": All the preprocessing operation are finished!\n")
    f.close()


def dti(folder_path, slurm=False):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of DTI with slurm:" + str(slurm) + "\n")
    f.close()

    if slurm:
        import pyslurm

    dti_path = folder_path + "/out/dti"
    try:
        os.mkdir(dti_path)
    except OSError:
        print("Creation of the directory %s failed" % dti_path)
        f=open(folder_path + "/out/logs.txt", "a+")
        f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed \n" % dti_path)
        f.close()
    else:
        print("Successfully created the directory %s " % dti_path)
        f=open(folder_path + "/out/logs.txt", "a+")
        f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dti_path)
        f.close()


    dest_success = folder_path + "/out/patient_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []
    f=open(folder_path + "/out/logs.txt", "a+")
    for p in patient_list:
        if slurm:
            p_job = {
                    "wrap": "python -c 'from utils import dti_solo; dti_solo(\"" + folder_path + "\",\"" + p + "\")'",
                    "job_name": "dti_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "1:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            dti_solo(folder_path,p)
            f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied DTI on patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of DTI\n")
    f.close()


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



def white_mask(folder_path, slurm=False):
    """ Compute a white matter mask of the diffusion data for each patient based on T1 volumes
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    Remark
    ----------
    The T1 images must have the same name as the patient it corresponds to with _T1 at the end and must be in
    a folder named anat in the root folder
    """
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of white with slurm:" + str(slurm) + "\n")
    f.close()

    if slurm:
        import pyslurm

    whitemask_path = folder_path + "/out/whitemask"
    try:
        os.mkdir(whitemask_path)
    except OSError:
        print("Creation of the directory %s failed" % whitemask_path)
        f=open(folder_path + "/out/logs.txt", "a+")
        f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed \n" % whitemask_path)
        f.close()
    else:
        print("Successfully created the directory %s " % whitemask_path)
        f=open(folder_path + "/out/logs.txt", "a+")
        f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % whitemask_path)
        f.close()


    dest_success = folder_path + "/out/patient_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []
    f=open(folder_path + "/out/logs.txt", "a+")
    for p in patient_list:
        if slurm:
            p_job = {
                    "wrap": "python -c 'from utils import dti_solo; white_mask_solo(\"" + folder_path + "\",\"" + p + "\")'",
                    "job_name": "whitemask_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "3:00:00",
                    "mail_user": "mathieu.simon@student.uclouvain.be",
                    "mail_type": "FAIL",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            white_mask(folder_path,p)
            f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied white mask on patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/out/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of White mask\n")
    f.close()
