"""
 Elikopy
 @author: qdessain, msimon
"""
import datetime
import os
import json
import math
import shutil
import time
import subprocess

from elikopy.individual_subject_processing import preproc_solo, dti_solo, white_mask_solo, noddi_solo, diamond_solo, mf_solo
from elikopy.utils import submit_job


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
    f=open(folder_path + "/logs.txt", "a+")
    f.write("[DICOM TO NIFTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning sequential dicom convertion\n")
    f.close()

    bashCommand = 'dcm2niix -f "%i_%p_%z" -p y -z y -o ' + folder_path + ' ' + folder_path + ''
    bashcmd = bashCommand.split()
    #print("Bash command is:\n{}\n".format(bashcmd))
    process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)

    #wait until mricron finish
    output, error = process.communicate()


    #Move all old dicom to dicom folder

    dest = folder_path + "/dicom"
    files = os.listdir(folder_path)
    if not(os.path.exists(dest)):
        try:
            os.mkdir(dest)
        except OSError:
            print ("Creation of the directory %s failed" % dest)
        else:
            print ("Successfully created the directory %s " % dest)

    f=open(folder_path + "/logs.txt", "a+")
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
    iscontrol = {}

    for file in os.listdir(folder_path + "/CONTROL"):

        if file.endswith(".nii"):
            name = os.path.splitext(file)[0]
            bvec = os.path.splitext(file)[0] + ".bvec"
            bval = os.path.splitext(file)[0] + ".bval"
            if bvec not in os.listdir(folder_path) or bval not in os.listdir(folder_path):
                error.append(name)
            else:
                success.append(name)
                iscontrol[name]=True

        if file.endswith(".nii.gz"):
            name = os.path.splitext(os.path.splitext(file)[0])[0]
            bvec = os.path.splitext(os.path.splitext(file)[0])[0] + ".bvec"
            bval = os.path.splitext(os.path.splitext(file)[0])[0] + ".bval"
            if bvec not in os.listdir(folder_path + "/CONTROL") or bval not in os.listdir(folder_path + "/CONTROL"):
                error.append(name)
            else:
                success.append(name)
                iscontrol[name]=True
                dest = folder_path + "/subjects/" + name + "/dMRI/raw/"
                if not (os.path.exists(dest)):
                    try:
                        os.makedirs(dest)
                    except OSError:
                        print("Creation of the directory %s failed" % dest)
                        f=open(folder_path + "/logs.txt", "a+")
                        f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dest)
                        f.close()
                    else:
                        print("Successfully created the directory %s " % dest)
                        f=open(folder_path + "/logs.txt", "a+")
                        f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dest)
                        f.close()

                shutil.copyfile(folder_path + "/CONTROL/" + name + ".bvec",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec")
                shutil.copyfile(folder_path + "/CONTROL/" + name + ".bval",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval")
                shutil.copyfile(folder_path + "/CONTROL/" + name + ".nii.gz",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.nii.gz")
                shutil.copyfile(folder_path + "/CONTROL/" + name + ".json",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.json")
                shutil.copyfile(folder_path + "/CONTROL/" + "index.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "index.txt")
                shutil.copyfile(folder_path + "/CONTROL/" + "acqparams.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "acqparams.txt")

                anat_path = folder_path + '/T1/' + name + '_T1.nii.gz'
                if os.path.isfile(anat_path):
                    dest = folder_path + "/subjects/" + name + "/T1/"
                    if not (os.path.exists(dest)):
                        try:
                            os.makedirs(dest)
                        except OSError:
                            print("Creation of the directory %s failed" % dest)
                            f = open(folder_path + "/logs.txt", "a+")
                            f.write("[PATIENT LIST] " + datetime.datetime.now().strftime(
                                "%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dest)
                            f.close()
                        else:
                            print("Successfully created the directory %s " % dest)
                            f = open(folder_path + "/logs.txt", "a+")
                            f.write("[PATIENT LIST] " + datetime.datetime.now().strftime(
                                "%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dest)
                            f.close()
                    shutil.copyfile(folder_path + "/T1/" + name + "_T1.nii.gz", folder_path + "/subjects/" + name + "/T1/" + name + "_T1.nii.gz")




    for file in os.listdir(folder_path + "/CASE"):

        if file.endswith(".nii"):
            name = os.path.splitext(file)[0]
            bvec = os.path.splitext(file)[0] + ".bvec"
            bval = os.path.splitext(file)[0] + ".bval"
            if bvec not in os.listdir(folder_path) or bval not in os.listdir(folder_path):
                error.append(name)
            else:
                success.append(name)
                iscontrol[name]=False

        if file.endswith(".nii.gz"):
            name = os.path.splitext(os.path.splitext(file)[0])[0]
            bvec = os.path.splitext(os.path.splitext(file)[0])[0] + ".bvec"
            bval = os.path.splitext(os.path.splitext(file)[0])[0] + ".bval"
            if bvec not in os.listdir(folder_path+ "/CASE") or bval not in os.listdir(folder_path+ "/CASE"):
                error.append(name)
            else:
                success.append(name)
                iscontrol[name]=False
                dest = folder_path + "/subjects/" + name + "/dMRI/raw/"
                if not (os.path.exists(dest)):
                    try:
                        os.makedirs(dest)
                    except OSError:
                        print("Creation of the directory %s failed" % dest)
                        f=open(folder_path + "/logs.txt", "a+")
                        f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dest)
                        f.close()
                    else:
                        print("Successfully created the directory %s " % dest)
                        f=open(folder_path + "/logs.txt", "a+")
                        f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dest)
                        f.close()

                shutil.copyfile(folder_path + "/CASE/" + name + ".bvec",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec")
                shutil.copyfile(folder_path + "/CASE/" + name + ".bval",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval")
                shutil.copyfile(folder_path + "/CASE/" + name + ".nii.gz",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.nii.gz")
                shutil.copyfile(folder_path + "/CASE/" + name + ".json",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.json")
                shutil.copyfile(folder_path + "/CASE/" + "index.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "index.txt")
                shutil.copyfile(folder_path + "/CASE/" + "acqparams.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "acqparams.txt")

                anat_path = folder_path + '/T1/' + name + '_T1.nii.gz'
                if os.path.isfile(anat_path):
                    dest = folder_path + "/subjects/" + name + "/T1/"
                    if not (os.path.exists(dest)):
                        try:
                            os.makedirs(dest)
                        except OSError:
                            print("Creation of the directory %s failed" % dest)
                            f = open(folder_path + "/logs.txt", "a+")
                            f.write("[PATIENT LIST] " + datetime.datetime.now().strftime(
                                "%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dest)
                            f.close()
                        else:
                            print("Successfully created the directory %s " % dest)
                            f = open(folder_path + "/logs.txt", "a+")
                            f.write("[PATIENT LIST] " + datetime.datetime.now().strftime(
                                "%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dest)
                            f.close()
                    shutil.copyfile(folder_path + "/T1/" + name + "_T1.nii.gz",folder_path + "/subjects/" + name + "/T1/" + name + "_T1.nii.gz")

    error = list(dict.fromkeys(error))
    success = list(dict.fromkeys(success))

    dest_error = folder_path + "/subjects/subj_error.json"
    with open(dest_error, 'w') as f:
        json.dump(error, f)

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'w') as f:
        json.dump(success, f)

    dest_iscontrol = folder_path + "/subjects/is_control.json"
    with open(dest_iscontrol, 'w') as f:
        json.dump(iscontrol, f)

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[PATIENT LIST] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient list generated\n")
    f.close()


def preproc(folder_path, eddy=False, denoising=False, slurm=False, reslice=False, gibbs=False):
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

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ":  Beginning preprocessing with eddy:" + str(eddy) + ", denoising:" + str(denoising) + ", slurm:" + str(slurm) + ", reslice:" + str(reslice) + ", gibbs:" + str(gibbs) +"\n")
    f.close()

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []

    f=open(folder_path + "/logs.txt", "a+")
    for p in patient_list:
        patient_path = os.path.splitext(p)[0]
        preproc_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/bet"
        if not(os.path.exists(preproc_path)):
            try:
                os.makedirs(preproc_path)
            except OSError:
                print ("Creation of the directory %s failed" % preproc_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f2.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % preproc_path)
                f2.close()
            else:
                print ("Successfully created the directory %s " % preproc_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", "a+")
                f2.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % preproc_path)
                f2.close()
        if slurm:
            if not denoising and not eddy:
                p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/subjects\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ",gibbs=" + str(gibbs) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "00:30:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.err",
                }
            elif denoising and eddy:
                p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/subjects\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ",gibbs=" + str(gibbs) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 8,
                    "mem_per_cpu": 6096,
                    "time": "14:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.err",
                }
            elif denoising and not eddy:
                p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/subjects\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ",gibbs=" + str(gibbs) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 9096,
                    "time": "3:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.err",
                }
            elif not denoising and eddy:
                p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/subjects\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ",gibbs=" + str(gibbs) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 4,
                    "mem_per_cpu": 6096,
                    "time": "12:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.err",
                }
            else:
                p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/subjects\",\"" + p + "\",eddy=" + str(eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "1:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.err",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            preproc_solo(folder_path + "/subjects",p,eddy=eddy,denoising=denoising,reslice=reslice,gibbs=gibbs)
            f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully preproceced patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        import pyslurm
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[PREPROC] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": All the preprocessing operation are finished!\n")
    f.close()


def dti(folder_path, slurm=False):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """
    f=open(folder_path + "/logs.txt", "a+")
    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of DTI with slurm:" + str(slurm) + "\n")
    f.close()

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []
    f=open(folder_path + "/logs.txt", "a+")
    for p in patient_list:
        patient_path = os.path.splitext(p)[0]

        dti_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti"
        if not(os.path.exists(dti_path)):
            try:
                os.makedirs(dti_path)
            except OSError:
                print ("Creation of the directory %s failed" % dti_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
                f2.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dti_path)
                f2.close()
            else:
                print ("Successfully created the directory %s " % dti_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt", "a+")
                f2.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dti_path)
                f2.close()

        if slurm:
            p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import dti_solo; dti_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                    "job_name": "dti_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "1:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + "slurm-%j.err",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            dti_solo(folder_path + "/subjects",p)
            f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied DTI on patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        import pyslurm
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[DTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of DTI\n")
    f.close()


def fingerprinting(folder_path, slurm=False):
    """Perform microstructure fingerprinting and store the data in the subjID/dMRI/microstructure/mf folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the nifti
    Remark
    ----------
    need the dictionary in the CASE and CONTROL directory
    """
    f=open(folder_path + "/logs.txt", "a+")
    f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of Microstructure Fingerprinting with slurm:" + str(slurm) + "\n")
    f.close()

    dest_success = folder_path + "/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []
    f=open(folder_path + "/logs.txt", "a+")
    for p in patient_list:
        patient_path = os.path.splitext(p)[0]

        mf_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/mf"
        if not(os.path.exists(mf_path)):
            try:
                os.makedirs(mf_path)
            except OSError:
                print ("Creation of the directory %s failed" % mf_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")
                f2.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % mf_path)
                f2.close()
            else:
                print ("Successfully created the directory %s " % mf_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/mf/mf_logs.txt", "a+")
                f2.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % mf_path)
                f2.close()
        shutil.copyfile(folder_path + '/CASE/fixed_rad_dist.mat', mf_path + '/fixed_rad_dist.mat')

        if slurm:
            p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import mf_solo; mf_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                    "job_name": "mf_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 4,
                    "mem_per_cpu": 2096,
                    "time": "2:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/mf/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/mf/' + "slurm-%j.err",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            mf_solo(folder_path + "/subjects",p)
            f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied microstructure fingerprinting on patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        import pyslurm
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[MF] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of microstructure fingerprinting\n")
    f.close()


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
    f=open(folder_path + "/logs.txt", "a+")
    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of white with slurm:" + str(slurm) + "\n")
    f.close()

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []
    f=open(folder_path + "/logs.txt", "a+")
    for p in patient_list:
        patient_path = os.path.splitext(p)[0]
        if slurm:
            p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import white_mask_solo; white_mask_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                    "job_name": "whitemask_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "3:00:00",
                    "mail_user": "mathieu.simon@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/T1/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/T1/' + "slurm-%j.err",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            white_mask_solo(folder_path + "/subjects", p)
            f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied white mask on patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        import pyslurm
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of White mask\n")
    f.close()


def noddi(folder_path, slurm=False):
    """Perform noddi and store the data in the subjID/dMRI/microstructure/noddi folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """
    f=open(folder_path + "/logs.txt", "a+")
    f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of Noddi with slurm:" + str(slurm) + "\n")
    f.close()

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []
    f=open(folder_path + "/logs.txt", "a+")
    for p in patient_list:
        patient_path = os.path.splitext(p)[0]

        noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi"
        if not(os.path.exists(noddi_path)):
            try:
                os.makedirs(noddi_path)
            except OSError:
                print ("Creation of the directory %s failed" % noddi_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
                f2.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % noddi_path)
                f2.close()
            else:
                print ("Successfully created the directory %s " % noddi_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt", "a+")
                f2.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % noddi_path)
                f2.close()

        if slurm:
            p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import noddi_solo; noddi_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                    "job_name": "noddi_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 1,
                    "mem_per_cpu": 8096,
                    "time": "3:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi/' + "slurm-%j.err",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            noddi_solo(folder_path + "/subjects",p)
            f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied NODDI on patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        import pyslurm
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[NODDI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of NODDI\n")
    f.close()


def diamond(folder_path, slurm=False):
    """Perform diamond and store the data in the subjID/dMRI/microstructure/diamond folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the nifti
    """
    f=open(folder_path + "/logs.txt", "a+")
    f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of DIAMOND with slurm:" + str(slurm) + "\n")
    f.close()

    dest_success = folder_path + "/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    job_list = []
    f=open(folder_path + "/logs.txt", "a+")
    for p in patient_list:
        patient_path = os.path.splitext(p)[0]

        diamond_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond"
        if not(os.path.exists(diamond_path)):
            try:
                os.makedirs(diamond_path)
            except OSError:
                print ("Creation of the directory %s failed" % diamond_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
                f2.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % diamond_path)
                f2.close()
            else:
                print ("Successfully created the directory %s " % diamond_path)
                f2=open(folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond/diamond_logs.txt", "a+")
                f2.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % diamond_path)
                f2.close()

        if slurm:
            p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import diamond_solo; diamond_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                    "job_name": "diamond_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 4,
                    "mem_per_cpu": 2096,
                    "time": "2:00:00",
                    "mail_user": "quentin.dessain@student.uclouvain.be",
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/diamond/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/diamond/' + "slurm-%j.err",
                }
            #p_job_id = pyslurm.job().submit_batch_job(p_job)
            p_job_id = submit_job(p_job)
            job_list.append(p_job_id)
            f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
            f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            diamond_solo(folder_path + "/subjects",p)
            f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied diamond on patient %s\n" % p)
            f.flush()
    f.close()

    #Wait for all jobs to finish
    if slurm:
        import pyslurm
        while job_list:
            for job_id in job_list[:]:
                job_info = pyslurm.job().find_id(job_id)[0]
                if job_info["job_state"] == 'COMPLETED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " COMPLETED\n")
                    f.close()
                if job_info["job_state"] == 'FAILED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " FAILED\n")
                    f.close()
                if job_info["job_state"] == 'OUT_OF_MEMORY':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " OUT_OF_MEMORY\n")
                    f.close()
                if job_info["job_state"] == 'TIMEOUT':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " TIMEOUT\n")
                    f.close()
                if job_info["job_state"] == 'CANCELLED':
                    job_list.remove(job_id)
                    f=open(folder_path + "/logs.txt", "a+")
                    f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Job " + str(job_id) + " CANCELLED\n")
                    f.close()
            time.sleep(30)

    f=open(folder_path + "/logs.txt", "a+")
    f.write("[DIAMOND] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of DIAMOND\n")
    f.close()
