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

import elikopy.utils
from elikopy.individual_subject_processing import preproc_solo, dti_solo, white_mask_solo, noddi_solo, diamond_solo, \
    mf_solo, noddi_amico_solo
from elikopy.utils import submit_job, get_job_state, makedir, tbss_utils


def dicom_to_nifti(folder_path):
    """ Convert dicom data into nifti. Converted dicom are then moved to a sub-folder named original_data
    Parameters

    :param folder_path: Path to root folder containing all the dicom
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

class Elikopy:
    r'''
    The MultiCompartmentModel class allows to combine any number of
    CompartmentModels and DistributedModels into one combined model that can
    be used to fit and simulate dMRI data.
    Parameters
    ----------
    models : list of N CompartmentModel instances,
        the models to combine into the MultiCompartmentModel.
    parameter_links : list of iterables (model, parameter name, link function,
        argument list),
        deprecated, for testing only.
    '''

    def __init__(self, folder_path, cuda=False, slurm=False, slurm_email='example@example.com'):
        self._folder_path = folder_path
        self._slurm = slurm
        self._slurm_email = slurm_email
        self._cuda = cuda


    def patient_list(self, folder_path=None):
        """ Verify the validity of all the nifti present in the root folder. If some nifti does not posses an associated
        bval and bvec file, they are discarded and the user is notified by the mean of a summary file named
        patient_error.json generated in the out sub-directory. All the valid patient are stored in a file named patient_list.json

        :param folder_path: Path to root folder of the study.
        """
        log_prefix = "PATIENT LIST"
        folder_path = self._folder_path if folder_path is None else folder_path

        import os
        import re

        error = []
        success = []
        type = {}
        pattern = re.compile("data_\\d")

        for typeFolder in os.listdir(folder_path):
            if pattern.match(typeFolder):
                subjectType = int(re.findall(r'\d+', typeFolder)[0])
                typeFolderName = "/" + typeFolder + "/"

                for file in os.listdir(folder_path + typeFolderName):

                    if file.endswith(".nii"):
                        name = os.path.splitext(file)[0]
                        bvec = os.path.splitext(file)[0] + ".bvec"
                        bval = os.path.splitext(file)[0] + ".bval"
                        if bvec not in os.listdir(folder_path) or bval not in os.listdir(folder_path):
                            error.append(name)
                        else:
                            success.append(name)
                            type[name]=subjectType

                    if file.endswith(".nii.gz"):
                        name = os.path.splitext(os.path.splitext(file)[0])[0]
                        bvec = os.path.splitext(os.path.splitext(file)[0])[0] + ".bvec"
                        bval = os.path.splitext(os.path.splitext(file)[0])[0] + ".bval"
                        if bvec not in os.listdir(folder_path + typeFolderName) or bval not in os.listdir(folder_path + typeFolderName):
                            error.append(name)
                        else:
                            success.append(name)
                            type[name]=subjectType
                            dest = folder_path + "/subjects/" + name + "/dMRI/raw/"
                            makedir(dest, folder_path + "/logs.txt", log_prefix)

                            shutil.copyfile(folder_path + typeFolderName + name + ".bvec",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec")
                            shutil.copyfile(folder_path + typeFolderName + name + ".bval",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval")
                            shutil.copyfile(folder_path + typeFolderName + name + ".nii.gz",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.nii.gz")
                            try:
                                shutil.copyfile(folder_path + typeFolderName + name + ".json",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.json")
                            except:
                                print('WARNING: JSON missing for patient', name)

                            try:
                                shutil.copyfile(folder_path + typeFolderName + "index.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "index.txt")
                                shutil.copyfile(folder_path + typeFolderName + "acqparams.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "acqparams.txt")
                            except:
                                print('WARNING: acqparam or index missing, you will get error trying to run EDDY correction')

                            anat_path = folder_path + '/T1/' + name + '_T1.nii.gz'
                            if os.path.isfile(anat_path):
                                dest = folder_path + "/subjects/" + name + "/T1/"
                                makedir(dest, folder_path + "/logs.txt", log_prefix)
                                shutil.copyfile(folder_path + "/T1/" + name + "_T1.nii.gz", folder_path + "/subjects/" + name + "/T1/" + name + "_T1.nii.gz")

        error = list(dict.fromkeys(error))
        success = list(dict.fromkeys(success))

        dest_error = folder_path + "/subjects/subj_error.json"
        with open(dest_error, 'w') as f:
            json.dump(error, f)

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'w') as f:
            json.dump(success, f)

        dest_type = folder_path + "/subjects/subj_type.json"
        with open(dest_type, 'w') as f:
            json.dump(type, f)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient list generated\n")
        f.close()


    def preproc(self, folder_path=None, reslice=False, denoising=False, gibbs=False, topup=False, eddy=False, patient_list_m=None, starting_state=None, bet_median_radius=2, bet_numpass=1, bet_dilate=2, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Wrapper function for the preprocessing. Perform bet and optionnaly denoising, gibbs, topup and eddy. Generated data are stored in bet, eddy, denoising and final directory
        located in the folder out/preproc. All the function executed after this function MUST take input data from folder_path/out/preproc/final

        :param folder_path: the path to the root directory.
        :param reslice: If true, data will be resliced with a new voxel resolution of 2*2*2.
        :param denoising: If true, PCA-based denoising using the Marcenko-Pastur distribution will be performed.
        :param gibbs: If true, Gibbs ringing artefacts of images volumes will be suppressed.
        :param topup: If true, topup will estimate and correct susceptibility induced distortions.
        :param eddy: If true, eddy will correct eddy currents and movements in diffusion data.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param starting_state: Manually set which step of the preprocessing to execute first. Could either be None, denoising, gibbs, topup or eddy.
        :param bet_median_radius: Radius (in voxels) of the applied median filter during bet.
        :param bet_num_pass: Number of pass of the median filter during bet.
        :param bet_dilate: Number of iterations for binary dilation during bet.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task by a custom amount of ram.
        """

        assert starting_state in (None, "denoising", "gibbs", "topup", "eddy"), 'invalid starting state!'
        if starting_state=="denoising":
            assert denoising == True, 'if starting_state is denoising, denoising must be True!'
        if starting_state=="gibbs":
            assert gibbs == True, 'if starting_state is gibbs, gibbs must be True!'
        if starting_state=="topup":
            assert topup == True, 'if starting_state is topup, topup must be True!'
        if starting_state=="eddy":
            assert eddy == True, 'if starting_state is eddy, eddy must be True!'

        log_prefix = "PREPROC"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ":  Beginning preprocessing with eddy:" + str(eddy) + ", denoising:" + str(denoising) + ", slurm:" + str(slurm) + ", reslice:" + str(reslice) + ", gibbs:" + str(gibbs) + ", starting_state:" + str(starting_state) +"\n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []

        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]
            preproc_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/bet"
            makedir(preproc_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

            if slurm:
                p_job = {
                    "wrap": "python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/subjects\",\"" + p + "\",eddy=" + str(
                        eddy) + ",denoising=" + str(denoising) + ",reslice=" + str(reslice) + ",gibbs=" + str(
                        gibbs) + ",topup=" + str(topup) + ",starting_state=\"" + str(starting_state) + ",bet_median_radius=\"" + str(
                        bet_median_radius) + ",bet_dilate=\"" + str(bet_dilate) + ",bet_numpass=\"" + str(bet_numpass) + "\")'",
                    "job_name": "preproc_" + p,
                    "ntasks": 1,
                    "cpus_per_task": 8,
                    "mem_per_cpu": 6096,
                    "time": "03:30:00",
                    "mail_user": slurm_email,
                    "mail_type": "FAIL",
                    "output": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.out",
                    "error": folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + "slurm-%j.err",
                }
                if not denoising and not eddy:
                    p_job["time"] = "00:30:00"
                    p_job["cpus_per_task"] = 1
                    p_job["mem_per_cpu"] = 8096
                elif denoising and eddy:
                    p_job["time"] = "14:00:00"
                    p_job["cpus_per_task"] = 8
                    p_job["mem_per_cpu"] = 6096
                elif denoising and not eddy:
                    p_job["time"] = "3:00:00"
                    p_job["cpus_per_task"] = 1
                    p_job["mem_per_cpu"] = 9096
                elif not denoising and eddy:
                    p_job["time"] = "12:00:00"
                    p_job["cpus_per_task"] = 4
                    p_job["mem_per_cpu"] = 6096
                else:
                    p_job["time"] = "1:00:00"
                    p_job["cpus_per_task"] = 1
                    p_job["mem_per_cpu"] = 8096
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["cpus_per_task"] = p_job["cpus_per_task"] if slurm_cpus is None else slurm_cpus
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem

                p_job_id={}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p

                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                preproc_solo(folder_path + "/subjects",p,reslice=reslice,denoising=denoising,gibbs=gibbs,topup=topup,eddy=eddy,starting_state=starting_state,bet_median_radius=bet_median_radius,bet_dilate=bet_dilate,bet_numpass=bet_numpass,cuda=self._cuda)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully preproceced patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": All the preprocessing operation are finished!\n")
        f.close()


    def dti(self,folder_path=None, patient_list_m=None, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Wrapper function for tensor reconstruction and computation of DTI metrics using Weighted Least-Squares.
        Performs a tensor reconstruction and saves the DTI metrics.

        :param folder_path: the path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 1h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """
        log_prefix = "DTI"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of DTI with slurm:" + str(slurm) + "\n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            dti_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti"
            makedir(dti_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt",
                    log_prefix)

            if slurm:
                p_job = {
                        "wrap": "python -c 'from elikopy.individual_subject_processing import dti_solo; dti_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                        "job_name": "dti_" + p,
                        "ntasks": 1,
                        "cpus_per_task": 1,
                        "mem_per_cpu": 8096,
                        "time": "1:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + "slurm-%j.err",
                    }
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["cpus_per_task"] = p_job["cpus_per_task"] if slurm_cpus is None else slurm_cpus
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                dti_solo(folder_path + "/subjects",p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied DTI on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "DTI")

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of DTI\n")
        f.close()


    def fingerprinting(self, dictionary_path, folder_path=None, CSD_bvalue = None, slurm=None, patient_list_m=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Wrapper function for microstructure estimation. Perform microstructure fingerprinting and store the data in the subjID/dMRI/microstructure/mf folder.

        :param folder_path: the path to the root directory.
        :param dictionary_path: Path to the dictionary to use
        :param CSD_bvalue:
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """
        log_prefix="MF"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of Microstructure Fingerprinting with slurm:" + str(slurm) + "\n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            mf_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/mf"
            makedir(mf_path,folder_path + '/subjects/' + patient_path+"/dMRI/microstructure/mf/mf_logs.txt",log_prefix)

            if slurm:
                p_job = {
                        "wrap": "python -c 'from elikopy.individual_subject_processing import mf_solo; mf_solo(\"" + folder_path + "/subjects\",\"" + p + "\", \"" + dictionary_path + "\", CSD_bvalue =" + str(CSD_bvalue) + ")'",
                        "job_name": "mf_" + p,
                        "ntasks": 1,
                        "cpus_per_task": 1,
                        "mem_per_cpu": 8096,
                        "time": "20:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/mf/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/mf/' + "slurm-%j.err",
                    }
                #p_job_id = pyslurm.job().submit_batch_job(p_job)
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["cpus_per_task"] = p_job["cpus_per_task"] if slurm_cpus is None else slurm_cpus
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                mf_solo(folder_path + "/subjects", p, dictionary_path, CSD_bvalue = CSD_bvalue)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied microstructure fingerprinting on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of microstructure fingerprinting\n")
        f.close()


    def white_mask(self, folder_path=None, patient_list_m=None, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """ Wrapper function for whitematter mask computation. Compute a white matter mask of the diffusion data for each patient based on T1 volumes or on diffusion data if
        T1 is not available. The T1 images must have the same name as the patient it corresponds to with _T1 at the end and must be in
        a folder named anat in the root folder.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 3h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """

        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of white with slurm:" + str(slurm) + "\n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

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
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/masks/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/masks/' + "slurm-%j.err",
                    }
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["cpus_per_task"] = p_job["cpus_per_task"] if slurm_cpus is None else slurm_cpus
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem
                #p_job_id = pyslurm.job().submit_batch_job(p_job)

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
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
            elikopy.utils.getJobsState(folder_path, job_list, "White mask")

        f=open(folder_path + "/logs.txt", "a+")
        f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of White mask\n")
        f.close()


    def noddi(self, folder_path=None, patient_list_m=None, force_brain_mask=False, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Wrapper function for noddi. Perform noddi and store the data in the subjID/dMRI/microstructure/noddi folder.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param force_brain_mask:
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 10h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """
        log_prefix="NODDI"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of Noddi with slurm:" + str(slurm) + "\n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi"
            makedir(noddi_path,folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt",
                    log_prefix)

            if slurm:
                p_job = {
                        "wrap": "python -c 'from elikopy.individual_subject_processing import noddi_solo; noddi_solo(\"" + folder_path + "/subjects\",\"" + p + "\"," + str(force_brain_mask) + ")'",
                        "job_name": "noddi_" + p,
                        "ntasks": 1,
                        "cpus_per_task": 1,
                        "mem_per_cpu": 8096,
                        "time": "10:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi/' + "slurm-%j.err",
                    }
                #p_job_id = pyslurm.job().submit_batch_job(p_job)
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["cpus_per_task"] = p_job["cpus_per_task"] if slurm_cpus is None else slurm_cpus
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                noddi_solo(folder_path + "/subjects",p,force_brain_mask)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied NODDI on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of NODDI\n")
        f.close()


    def noddi_amico(self, folder_path=None, patient_list_m=None, force_brain_mask=False, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Wrapper function for noddi amico. Perform noddi and store the data in the subjID/dMRI/microstructure/noddi_amico folder.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param force_brain_mask:
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 10h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """
        log_prefix = "NODDI AMICO"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of Noddi AMICO with slurm:" + str(slurm) + "\n")
        f.close()

        kernel_path = folder_path + '/noddi_AMICO/'
        makedir(kernel_path,folder_path + "/logs.txt",log_prefix)

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi_amico"
            makedir(noddi_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi_amico/noddi_amico_logs.txt", log_prefix)

            if slurm:
                p_job = {
                        "wrap": "python -c 'from elikopy.individual_subject_processing import noddi_amico_solo; noddi_amico_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                        "job_name": "noddi_amico_" + p,
                        "ntasks": 1,
                        "cpus_per_task": 1,
                        "mem_per_cpu": 8096,
                        "time": "10:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi_amico/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi_amico/' + "slurm-%j.err",
                    }
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["cpus_per_task"] = p_job["cpus_per_task"] if slurm_cpus is None else slurm_cpus
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem
                #p_job_id = pyslurm.job().submit_batch_job(p_job)

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                noddi_amico_solo(folder_path + "/subjects",p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied NODDI AMICO on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of NODDI AMICO\n")
        f.close()


    def diamond(self, folder_path=None, patient_list_m=None, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Wrapper function for DIAMOND. Perform diamond and store the data in the subjID/dMRI/microstructure/diamond folder.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 14h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 4 by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (6096MO by cpu) by a custom amount of ram.
        """
        log_prefix = "DIAMOND"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of DIAMOND with slurm:" + str(slurm) + "\n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            diamond_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond"
            makedir(diamond_path,folder_path+'/subjects/'+patient_path+"/dMRI/microstructure/diamond/diamond_logs.txt",
                    log_prefix)

            if slurm:
                p_job = {
                        "wrap": "python -c 'from elikopy.individual_subject_processing import diamond_solo; diamond_solo(\"" + folder_path + "/subjects\",\"" + p + "\")'",
                        "job_name": "diamond_" + p,
                        "ntasks": 1,
                        "cpus_per_task": 4,
                        "mem_per_cpu": 6096,
                        "time": "14:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/diamond/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/diamond/' + "slurm-%j.err",
                    }
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["cpus_per_task"] = p_job["cpus_per_task"] if slurm_cpus is None else slurm_cpus
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem
                #p_job_id = pyslurm.job().submit_batch_job(p_job)

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                diamond_solo(folder_path + "/subjects",p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied diamond on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of DIAMOND\n")
        f.close()


    def tbss(self, folder_path=None, grp1=None, grp2=None, starting_state=None, last_state=None, registration_type="-T", postreg_type="-S", prestats_treshold=0.2, slurm=None, slurm_email=None, slurm_timeout=None, slurm_tasks=None, slurm_mem=None):
        """ Wrapper function for TBSS. Perform tract base spatial statistics between the control data and case data.
        DTI needs to have been performed on the data first !!

        :param folder_path: path to the root directory.
        :param grp1: List of number corresponding to the type of the patients to put in the first group.
        :param grp2: List of number corresponding to the type of the patients to put in the second group.
        :param starting_state: Manually set which step of TBSS to execute first. Could either be None, reg, post_reg, prestats, design or randomise.
        :param last_state: Manually set which step of TBSS to execute last. Could either be None, preproc, reg, post_reg, prestats, design or randomise.
        :param registration_type: Define the argument used by the tbss command tbss_2_reg. Could either by '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is align to every other one, identify the "most representative" one, and use this as the target image.
        :param postreg_type: Define the argument used by the tbss command tbss_3_postreg. Could either by '-S' or '-T'. If you wish to use the FMRIB58_FA mean FA image and its derived skeleton, instead of the mean of your subjects in the study, use the '-T' option. Otherwise, use the '-S' option.
        :param prestats_treshold: Thresholds the mean FA skeleton image at the chosen threshold during prestats.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param slurm_tasks: Replace the default number of slurm task of 8 by a custom number of tasks.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """

        assert starting_state in (None, "reg", "postreg", "prestats", "design", "randomise"), 'invalid starting state!'
        assert last_state in (None, "preproc", "reg", "postreg", "prestats", "design", "randomise"), 'invalid last state!'
        assert registration_type in ("-T", "-t", "-n"), 'invalid registration type!'
        assert postreg_type in ("-S", "-T"), 'invalid postreg type!'

        if grp1 is None:
            grp1 = [1]
        if grp2 is None:
            grp2 = [2]
        log_prefix = "TBSS"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f = open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of TBSS with slurm:" + str(slurm) + "\n")
        f.close()

        tbss_path = folder_path + "/TBSS"
        makedir(tbss_path,folder_path + "/logs.txt",log_prefix)

        job_list = []
        f = open(folder_path + "/logs.txt", "a+")
        if slurm:
            job = {
                "wrap": "python -c 'from elikopy.utils import tbss_utils; tbss_utils(\"" + str(folder_path) + "\",grp1=" + str(grp1) + ",grp2=" + str(grp2) + ",starting_state=\"" + str(starting_state) + "\",last_state=\"" + str(last_state) + "\",registration_type=\"" + str(registration_type) + "\",postreg_type=\"" + str(postreg_type) + "\",prestats_treshold=" + str(prestats_treshold) + ")'",
                "job_name": "tbss",
                "ntasks": 8,
                "cpus_per_task": 1,
                "mem_per_cpu": 8096,
                "time": "20:00:00",
                "mail_user": slurm_email,
                "mail_type": "FAIL",
                "output": tbss_path + '/' + "slurm-%j.out",
                "error": tbss_path + '/' + "slurm-%j.err",
            }
            job["time"] = job["time"] if slurm_timeout is None else slurm_timeout
            job["ntasks"] = job["ntasks"] if slurm_tasks is None else slurm_tasks
            job["mem_per_cpu"] = job["mem_per_cpu"] if slurm_mem is None else slurm_mem
            p_job_id = {}
            p_job_id["id"] = submit_job(job)
            p_job_id["name"] = "tbss"
            job_list.append(p_job_id)
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            tbss_utils(folder_path=folder_path, grp1=grp1, grp2=grp2, starting_state=starting_state, last_state=last_state, registration_type=registration_type, postreg_type=postreg_type, prestats_treshold=prestats_treshold)
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied TBSS \n")
            f.flush()
        f.close()

        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "TBSS")

        f = open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of TBSS\n")
        f.close()
