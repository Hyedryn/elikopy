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

import matplotlib.pyplot
import matplotlib

import elikopy.utils
from elikopy.individual_subject_processing import preproc_solo, dti_solo, white_mask_solo, noddi_solo, diamond_solo, \
    mf_solo, noddi_amico_solo
from elikopy.utils import submit_job, get_job_state, makedir, tbss_utils, regall_FA, regall, randomise_all


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
    Main class containing all the necessary function to process and preprocess a specific study.
    '''

    def __init__(self, folder_path, cuda=False, slurm=False, slurm_email='example@example.com'):
        self._folder_path = folder_path
        self._slurm = slurm
        self._slurm_email = slurm_email
        self._cuda = cuda

    def patient_list(self, folder_path=None, bids_path=None, reverseEncoding=True):
        """ Verify the validity of all the nifti present in the root folder. If some nifti does not possess an associated
        bval and bvec file, they are discarded and the user is notified by a summary file named
        patient_error.json generated in the out sub-directory. All valid patients are stored in a file named patient_list.json

        :param folder_path: Path to root folder of the study.
        :param bids_path: Path to the optional folder containing subjects' data in the BIDS format.
        :param reverseEncoding: Append reverse encoding direction to the DW-MRI data if available
        """
        log_prefix = "PATIENT LIST"
        folder_path = self._folder_path if folder_path is None else folder_path

        makedir(folder_path + '/subjects/', folder_path + "/logs.txt", log_prefix)

        import os
        import re

        error = []
        success = []
        type = {}
        pattern = re.compile("data_\\d")


        if bids_path is not None:
            for subj in os.listdir(bids_path):
                subjectType = 99
                dwi_path = bids_path + "/" + subj + "/dwi/"

                if os.path.exists(dwi_path):
                    for file in os.listdir(dwi_path):
                        DWI_present=False
                        if file.endswith(".nii"):
                            nii = True
                            DWI_present=True

                            name = os.path.splitext(file)[0]
                            bvec = os.path.splitext(file)[0] + ".bvec"
                            bval = os.path.splitext(file)[0] + ".bval"
                            entities_0 = name.split('_')[0]

                        if file.endswith(".nii.gz"):
                            nii = False
                            DWI_present = True
                            name = os.path.splitext(os.path.splitext(file)[0])[0]
                            bvec = os.path.splitext(os.path.splitext(file)[0])[0] + ".bvec"
                            bval = os.path.splitext(os.path.splitext(file)[0])[0] + ".bval"
                            jsonpath = os.path.splitext(os.path.splitext(file)[0])[0] + ".json"

                        anat_path = bids_path + "/" + subj + "/anat/" + name + "_T1w.nii.gz"

                        if os.path.exists(bids_path + "/" + subj + "/anat/" + name + "_T1w.nii"):
                            import gzip
                            f_in = open(bids_path + "/" + subj + "/anat/" + name + "_T1w.nii")
                            f_out = gzip.open(
                                folder_path + "/subjects/" + entities_0 + "/T1/" + entities_0 + "_T1.nii.gz",
                                'wb')
                            f_out.writelines(f_in)
                            f_out.close()
                            f_in.close()
                            anat_path_json = bids_path + "/" + subj + "/anat/" + entities_0 + '_T1w.json'
                            if os.path.isfile(anat_path_json):
                                shutil.copyfile(anat_path_json,
                                                folder_path + "/subjects/" + entities_0 + "/T1/" + entities_0 + "_T1.json")

                        if DWI_present==False or (bvec not in os.listdir(dwi_path) or bval not in os.listdir(dwi_path)):
                            if file.endswith(".nii.gz") or file.endswith(".nii"):
                                error.append(entities_0)
                        else:
                            success.append(entities_0)
                            type[entities_0] = subjectType
                            dest = folder_path + "/subjects/" + name + "/dMRI/raw/"
                            makedir(dest, folder_path + "/logs.txt", log_prefix)

                            shutil.copyfile(dwi_path + name + ".bvec",
                                            folder_path + "/subjects/" + entities_0 + "/dMRI/raw/" + entities_0 + "_raw_dmri.bvec")
                            shutil.copyfile(dwi_path + name + ".bval",
                                            folder_path + "/subjects/" + entities_0 + "/dMRI/raw/" + entities_0 + "_raw_dmri.bval")
                            if nii:
                                import gzip
                                f_in = open(dwi_path + name + ".nii")
                                f_out = gzip.open(folder_path + "/subjects/" + entities_0 + "/dMRI/raw/" + entities_0 + "_raw_dmri.nii.gz", 'wb')
                                f_out.writelines(f_in)
                                f_out.close()
                                f_in.close()
                            else:
                                shutil.copyfile(dwi_path + name + ".nii.gz",
                                                folder_path + "/subjects/" + entities_0 + "/dMRI/raw/" + entities_0 + "_raw_dmri.nii.gz")
                            try:
                                shutil.copyfile(dwi_path + name + ".json",
                                                folder_path + "/subjects/" + entities_0 + "/dMRI/raw/" + entities_0 + "_raw_dmri.json")
                            except:
                                print('WARNING: JSON missing for patient', entities_0)

                            try:
                                shutil.copyfile(dwi_path + "index.txt",
                                                folder_path + "/subjects/" + name + "/dMRI/raw/" + "index.txt")
                                shutil.copyfile(dwi_path + "acqparams.txt",
                                                folder_path + "/subjects/" + entities_0 + "/dMRI/raw/" + "acqparams.txt")
                            except:
                                print(
                                    'WARNING: acqparam or index missing, you will get error trying to run EDDY correction')

                            try:
                                shutil.copyfile(dwi_path + "slspec.txt",
                                                folder_path + "/subjects/" + entities_0 + "/dMRI/raw/" + "slspec.txt")
                            except:
                                print(
                                    'WARNING: slspec missing, EDDY outlier replacement and slice-to-volume motion correction will not correct properly')

                            if os.path.isfile(anat_path):
                                dest = folder_path + "/subjects/" + entities_0 + "/T1/"
                                makedir(dest, folder_path + "/logs.txt", log_prefix)
                                shutil.copyfile(anat_path,
                                                folder_path + "/subjects/" + entities_0 + "/T1/" + entities_0 + "_T1.nii.gz")
                                anat_path_json = bids_path + "/" + subj + "/anat/" + entities_0 + '_T1w.json'
                                if os.path.isfile(anat_path_json):
                                    shutil.copyfile(anat_path_json,
                                                    folder_path + "/subjects/" + entities_0 + "/T1/" + entities_0 + "_T1.json")

        for typeFolder in os.listdir(folder_path):
            if pattern.match(typeFolder):
                subjectType = int(re.findall(r'\d+', typeFolder)[0])
                typeFolderName = "/" + typeFolder + "/"

                for file in os.listdir(folder_path + typeFolderName):

                    if os.path.isdir(file):
                        continue

                    DWI_present = False

                    if file.endswith(".nii"):
                        DWI_present = True
                        nii = True
                        name = os.path.splitext(file)[0]
                        bvec = os.path.splitext(file)[0] + ".bvec"
                        bval = os.path.splitext(file)[0] + ".bval"

                    if file.endswith(".nii.gz"):
                        DWI_present = True
                        nii = False
                        name = os.path.splitext(os.path.splitext(file)[0])[0]
                        bvec = os.path.splitext(os.path.splitext(file)[0])[0] + ".bvec"
                        bval = os.path.splitext(os.path.splitext(file)[0])[0] + ".bval"

                    if DWI_present==False or (bvec not in os.listdir(folder_path + typeFolderName) or bval not in os.listdir(folder_path + typeFolderName)):
                        if file.endswith(".nii.gz") or file.endswith(".nii"):
                            error.append(file)
                    else:
                        success.append(name)
                        type[name]=subjectType
                        dest = folder_path + "/subjects/" + name + "/dMRI/raw/"
                        makedir(dest, folder_path + "/logs.txt", log_prefix)

                        shutil.copyfile(folder_path + typeFolderName + name + ".bvec",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec")
                        shutil.copyfile(folder_path + typeFolderName + name + ".bval",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval")
                        if nii:
                            import gzip
                            f_in = open(folder_path + typeFolderName + name + ".nii")
                            f_out = gzip.open(
                                folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.nii.gz", 'wb')
                            f_out.writelines(f_in)
                            f_out.close()
                            f_in.close()
                        else:
                            shutil.copyfile(folder_path + typeFolderName + name + ".nii.gz",
                                            folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.nii.gz")
                        try:
                            shutil.copyfile(folder_path + typeFolderName + name + ".json",folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.json")
                        except:
                            print('WARNING: JSON missing for patient', name)

                        try:
                            shutil.copyfile(folder_path + typeFolderName + "index.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "index.txt")
                            shutil.copyfile(folder_path + typeFolderName + "acqparams.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "acqparams.txt")
                        except:
                            print('WARNING: acqparam or index missing, you will get error trying to run EDDY correction')

                        try:
                            shutil.copyfile(folder_path + typeFolderName + "slspec.txt",folder_path + "/subjects/" + name + "/dMRI/raw/" + "slspec.txt")
                        except:
                            print('WARNING: slspec missing, EDDY outlier replacement and slice-to-volume motion correction will not correct properly')

                        anat_path = folder_path + '/T1/' + name + '_T1.nii.gz'
                        anat_path_json = folder_path + '/T1/' + name + '_T1.json'
                        if os.path.isfile(anat_path):
                            dest = folder_path + "/subjects/" + name + "/T1/"
                            makedir(dest, folder_path + "/logs.txt", log_prefix)
                            shutil.copyfile(anat_path, folder_path + "/subjects/" + name + "/T1/" + name + "_T1.nii.gz")
                            if os.path.isfile(anat_path_json):
                                shutil.copyfile(anat_path_json,
                                            folder_path + "/subjects/" + name + "/T1/" + name + "_T1.json")

                        anat_path = folder_path + '/T1/' + name + '.nii.gz'
                        anat_path_json = folder_path + '/T1/' + name + '.json'
                        if os.path.isfile(anat_path):
                            dest = folder_path + "/subjects/" + name + "/T1/"
                            makedir(dest, folder_path + "/logs.txt", log_prefix)
                            shutil.copyfile(anat_path, folder_path + "/subjects/" + name + "/T1/" + name + "_T1.nii.gz")
                            if os.path.isfile(anat_path_json):
                                shutil.copyfile(anat_path_json,
                                                folder_path + "/subjects/" + name + "/T1/" + name + "_T1.json")

                        if os.path.exists(folder_path + '/T1/' + name + '.nii'):
                            import gzip
                            f_in = open(folder_path + '/T1/' + name + '.nii')
                            f_out = gzip.open(
                                folder_path + "/subjects/" + name + "/T1/" + name + "_T1.nii.gz",
                                'wb')
                            f_out.writelines(f_in)
                            f_out.close()
                            f_in.close()
                            anat_path_json = folder_path + '/T1/' + name + '.json'
                            if os.path.isfile(anat_path_json):
                                shutil.copyfile(anat_path_json,
                                                folder_path + "/subjects/" + name + "/T1/" + name + "_T1.json")

                        if os.path.exists(folder_path + '/T1/' + name + '_T1.nii'):
                            import gzip
                            f_in = open(folder_path + '/T1/' + name + '_T1.nii')
                            f_out = gzip.open(
                                folder_path + "/subjects/" + name + "/T1/" + name + "_T1.nii.gz",
                                'wb')
                            f_out.writelines(f_in)
                            f_out.close()
                            f_in.close()
                            anat_path_json = folder_path + '/T1/' + name + '_T1.json'
                            if os.path.isfile(anat_path_json):
                                shutil.copyfile(anat_path_json,
                                                folder_path + "/subjects/" + name + "/T1/" + name + "_T1.json")

                        reverse_path = folder_path + typeFolderName + '/reverse_encoding/' + name + '.nii.gz'
                        reverse_path_bvec = folder_path + typeFolderName + '/reverse_encoding/' + name + '.bvec'
                        reverse_path_bval = folder_path + typeFolderName + '/reverse_encoding/' + name + '.bval'
                        reverse_path_acqparameters = folder_path + typeFolderName + '/reverse_encoding/' + "acqparams.txt"
                        if reverseEncoding and os.path.isfile(reverse_path) and os.path.isfile(reverse_path_bvec) and os.path.isfile(reverse_path_bval) and os.path.isfile(reverse_path_acqparameters):
                            print('Topup will use a reverse encoding direction for patient ', name)
                            dw_mri_path = folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.nii.gz"
                            b0_path = folder_path + "/subjects/" + name + "/dMRI/raw/" + name +"_b0_reverse.nii.gz"

                            #Copy b0 to patient path:
                            fslroi = "fslroi " + reverse_path + " " + b0_path + " 0 1"
                            reverse_log = open(folder_path + "/logs.txt","a+")
                            try:
                                output = ""
                                output = subprocess.check_output(fslroi, universal_newlines=True, shell=True, stderr=subprocess.STDOUT)
                            except subprocess.CalledProcessError as e:
                                print("Error when calling fslroi, no reverse direction will be available")
                                reverse_log.write("Error when calling fslroi, no reverse direction will be available\n")
                                print(e.returncode)
                                print(e.cmd)
                                print(e.output)
                                reverse_log.write(e.output + "\n")
                            finally:
                                print(output)
                                reverse_log.write(output)
                            #print(error_log)


                            #Merge b0 with original DW-MRI:
                            merge_b0 = "fslmerge -t " + dw_mri_path + " " + dw_mri_path + " " + b0_path + " "
                            try:
                                output = ""
                                output = subprocess.check_output(merge_b0, universal_newlines=True, shell=True, stderr=subprocess.STDOUT)
                            except subprocess.CalledProcessError as e:
                                print("Error when calling fslmerge, no reverse direction will be available")
                                reverse_log.write("Error when calling fslmerge, no reverse direction will be available\n")
                                print(e.returncode)
                                print(e.cmd)
                                print(e.output)
                                reverse_log.write(e.output + "\n")
                            finally:
                                print(output)
                                reverse_log.write(output)

                            #Edit bvec:
                            with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec", "r") as file_object:
                                lines = file_object.readlines()
                                for index, line in enumerate(lines):
                                    if index < 3:
                                        lines[index] = line.strip() + " 0" + "\n"

                            with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec", "w") as f:
                                for line in lines:
                                    f.write(line)

                            #Edit bval
                            with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval", "r") as file_object:
                                file_object=file_object.read().rstrip()

                            with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval", "w") as myfile:
                                myfile.write(file_object + " 0"+ "\n")

                            #Edit index:
                            with open(folder_path + "/subjects/" + name + '/dMRI/raw/' + 'index.txt', "r") as f0:
                                line = f0.read()
                                line = " ".join(line.split())
                                original_index = [int(s) for s in line.split(' ')]
                            original_index.append(original_index[-1]+1)

                            with open(folder_path + "/subjects/" + name + '/dMRI/raw/' + 'index.txt', "w") as myfile:
                                new_index = ''.join(str(j) + " " for j in original_index).rstrip() + "\n"
                                myfile.write(new_index)

                            #Edit acqparameters:
                            with open(folder_path + "/subjects/" + name + '/dMRI/raw/' + 'acqparams.txt') as f:
                                original_acq = [[float(x) for x in line2.split()] for line2 in f]

                            with open(reverse_path_acqparameters) as f2:
                                reverse_acq = [[float(x) for x in line2.split()] for line2 in f2]

                            original_acq.append([reverse_acq[0][0], reverse_acq[0][1], reverse_acq[0][2], reverse_acq[0][3]])

                            with open(folder_path + "/subjects/" + name + '/dMRI/raw/' + 'acqparams.txt', 'w') as file:
                                file.writelines(' '.join(str(j) for j in i) + '\n' for i in original_acq)
                            print(original_acq)


                            reverse_log.close()

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

    def preproc(self, folder_path=None, reslice=False, reslice_addSlice=False, denoising=False, gibbs=False, topup=False, topupConfig=None, forceSynb0DisCo=False, useGPUsynb0DisCo=False, eddy=False, biasfield=False, biasfield_bsplineFitting=[100,3], biasfield_convergence=[1000,0.001], patient_list_m=None, starting_state=None, bet_median_radius=2, bet_numpass=1, bet_dilate=2, cuda=None, cuda_name="eddy_cuda10.1", s2v=[0,5,1,'trilinear'], olrep=[False, 4, 250, 'sw'], slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None, qc_reg=True, niter=5, slspec_gc_path=None, report=True):
        """Wrapper function for the preprocessing. Perform brain extraction and optionally reslicing, denoising, gibbs correction, susceptibility field estimation using topup, movement correction using eddy and biasfield correction. Generated data are stored in bet, reslice, mppca, gibbs, topup, eddy, biasfield and final directory
        located in the folder <folder_path>/subjects/<subjects_ID>/dMRI/preproc.

        :param folder_path: the path to the root directory.
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

        assert starting_state in (None, "denoising", "gibbs", "topup", "eddy", "biasfield", "report", "post_report"), 'invalid starting state!'
        if starting_state=="denoising":
            assert denoising == True, 'if starting_state is denoising, denoising must be True!'
        if starting_state=="gibbs":
            assert gibbs == True, 'if starting_state is gibbs, gibbs must be True!'
        if starting_state=="topup":
            assert topup == True, 'if starting_state is topup, topup must be True!'
        if starting_state=="eddy":
            assert eddy == True, 'if starting_state is eddy, eddy must be True!'
        if starting_state == "biasfield":
            assert biasfield == True, 'if starting_state is biasfield, biasfield must be True!'

        log_prefix = "PREPROC"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email
        cuda = self._cuda if cuda not in (True,False) else cuda

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

        core_count = 1 if cpus is None else cpus

        if starting_state!="post_report":
            for p in patient_list:
                patient_path = os.path.splitext(p)[0]
                preproc_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/bet"
                makedir(preproc_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

                if slurm:
                    tot_cpu = 8 if cpus is None else cpus
                    core_count = tot_cpu
                    p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(tot_cpu)+" ; export FSLPARALLEL="+str(tot_cpu)+" ; python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/subjects\",\"" + p + "\",eddy=" + str(
                            eddy) + ",biasfield=" + str(biasfield)  + ",biasfield_convergence=[" + str(biasfield_convergence[0]) + "," + str(biasfield_convergence[1]) + "],biasfield_bsplineFitting=[" + str(biasfield_bsplineFitting[0]) + "," + str(biasfield_bsplineFitting[1]) + "],denoising=" + str(
                            denoising) + ",reslice=" + str(reslice) + ",reslice_addSlice=" + str(reslice_addSlice) + ",gibbs=" + str(
                            gibbs) + ",topup=" + str(topup) + ",forceSynb0DisCo=" + str(forceSynb0DisCo) + ",useGPUsynb0DisCo=" + str(useGPUsynb0DisCo) + ",topupConfig=\"" + str(topupConfig) + "\",starting_state=\"" + str(starting_state) + "\",bet_median_radius=" + str(
                            bet_median_radius) + ",bet_dilate=" + str(bet_dilate) + ", qc_reg=" + str(qc_reg) + ", report=" + str(report) + ", slspec_gc_path=" + str(slspec_gc_path) + ", core_count=" + str(core_count)+ ", niter=" + str(niter)+",bet_numpass=" + str(bet_numpass) + ",cuda=" + str(cuda) + ",cuda_name=\"" + str(cuda_name) + "\",s2v=[" + str(s2v[0]) + "," + str(s2v[1]) + "," + str(s2v[2]) + ",\"" + str(s2v[3]) + "\"],olrep=[" + str(olrep[0]) + "," + str(olrep[1]) + "," + str(olrep[2]) + ",\"" + str(olrep[3]) + "\"])'",
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
                    p_job["cpus_per_task"] = p_job["cpus_per_task"] if cpus is None else cpus
                    p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem

                    p_job_id={}
                    p_job_id["id"] = submit_job(p_job)
                    p_job_id["name"] = p

                    job_list.append(p_job_id)
                    f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be preprocessed\n" % p)
                    f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
                else:
                    core_count = 1 if cpus is None else cpus
                    preproc_solo(folder_path + "/subjects",p,reslice=reslice,reslice_addSlice=reslice_addSlice,denoising=denoising,gibbs=gibbs,
                                 topup=topup, topupConfig=topupConfig, forceSynb0DisCo=forceSynb0DisCo, useGPUsynb0DisCo=useGPUsynb0DisCo,
                                 eddy=eddy,biasfield=biasfield, biasfield_bsplineFitting=biasfield_bsplineFitting, biasfield_convergence=biasfield_convergence,
                                 starting_state=starting_state,
                                 bet_median_radius=bet_median_radius,bet_dilate=bet_dilate,bet_numpass=bet_numpass,cuda=self._cuda, qc_reg=qc_reg, core_count=core_count,
                                 cuda_name=cuda_name, s2v=s2v, olrep=olrep, niter=niter, slspec_gc_path=slspec_gc_path, report=report)
                    matplotlib.pyplot.close(fig='all')
                    f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully preprocessed patient %s\n" % p)
                    f.flush()
            f.close()

            #Wait for all jobs to finish
            if slurm:
                elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        """Outside of preprocsolo : Eddy squad + Merge both individual QC pdf""";
        if eddy and report:
            # 1) update the QC with squad
            squadlist = [folder_path + '/subjects/' + x + "/dMRI/preproc/eddy/" + x + '_eddy_corr.qc\n' for x in
                         patient_list]
            dest_list = folder_path + "/subjects/squad_list.txt"
            f = open(dest_list, "w")
            for elem in squadlist:
                f.write(elem)
            f.close()

            shutil.rmtree(folder_path + '/eddy_squad', ignore_errors=True)
            bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; eddy_squad "' + dest_list + '" --update -o ' + folder_path + '/eddy_squad'
            bashcmd = bashCommand.split()
            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)
            output, error = process.communicate()

            # 2) merge both pdfs
            from PyPDF2 import PdfFileMerger
            for p in patient_list:
                patient_path = os.path.splitext(p)[0]
                if os.path.exists(folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr.qc/qc_updated.pdf'):
                    pdfs = [folder_path + '/subjects/' + patient_path + '/dMRI/preproc/quality_control/qc_report.pdf',
                            folder_path + '/subjects/' + patient_path + '/dMRI/preproc/eddy/' + patient_path + '_eddy_corr.qc/qc_updated.pdf']
                    merger = PdfFileMerger()
                    for pdf in pdfs:
                        merger.append(pdf)
                    merger.write(folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
                    merger.close()
                else:
                    shutil.copyfile(
                        folder_path + '/subjects/' + patient_path + '/dMRI/preproc/quality_control/qc_report.pdf',
                        folder_path + '/subjects/' + patient_path + '/quality_control.pdf')
        else:
            for p in patient_list:
                patient_path = os.path.splitext(p)[0]
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/preproc/quality_control/qc_report.pdf',
                    folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

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
                matplotlib.pyplot.close(fig='all')
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied DTI on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "DTI")

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of DTI\n")
        f.close()

    def fingerprinting(self, dictionary_path, folder_path=None, CSD_bvalue = None, slurm=None, patient_list_m=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """Wrapper function for microstructure estimation. Perform microstructure fingerprinting and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/mf/.

        :param folder_path: the path to the root directory.
        :param dictionary_path: Path to the dictionary to use
        :param CSD_bvalue: Define a csd value.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
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

        core_count = 4 if cpus is None else cpus

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            mf_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/mf"
            makedir(mf_path,folder_path + '/subjects/' + patient_path+"/dMRI/microstructure/mf/mf_logs.txt",log_prefix)

            if slurm:
                p_job = {
                        "wrap": "export MKL_NUM_THREADS="+ str(core_count)+" ; export OMP_NUM_THREADS="+ str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import mf_solo; mf_solo(\"" + folder_path + "/subjects\",\"" + p + "\", \"" + dictionary_path + "\", CSD_bvalue =" + str(CSD_bvalue) + ", core_count=" + str(core_count) + ")'",
                        "job_name": "mf_" + p,
                        "ntasks": 1,
                        "cpus_per_task": core_count,
                        "mem_per_cpu": 8096,
                        "time": "20:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/mf/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/mf/' + "slurm-%j.err",
                    }
                #p_job_id = pyslurm.job().submit_batch_job(p_job)
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem
                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                mf_solo(folder_path + "/subjects", p, dictionary_path, CSD_bvalue = CSD_bvalue, core_count=core_count)
                matplotlib.pyplot.close(fig='all')
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied microstructure fingerprinting on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of microstructure fingerprinting\n")
        f.close()

    def white_mask(self, folder_path=None, patient_list_m=None, corr_gibbs=True, forceUsePowerMap=False, debug=False, slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """ Wrapper function for whitematter mask computation. Compute a white matter mask of the diffusion data for each patient based on T1 volumes or on diffusion data if
        T1 is not available. Otherwise, compute the whitematter mask based on an anisotropic power map.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param corr_gibbs: Correct for gibbs oscillation.
        :param forceUsePowerMap: Force the use of an AnisotropicPower map for the white matter mask generation.
        :param debug: If true, additional intermediate output will be saved.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 3h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
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

        core_count = 1 if cpus is None else cpus

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]
            if slurm:
                core_count = 1 if cpus is None else cpus
                p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import white_mask_solo; white_mask_solo(\"" + folder_path + "/subjects\",\"" + p + "\"" + ",corr_gibbs=" + str(corr_gibbs) + ",forceUsePowerMap=" + str(forceUsePowerMap) + ",debug=" + str(debug) + ",core_count=" + str(core_count) + " )'",
                        "job_name": "whitemask_" + p,
                        "ntasks": 1,
                        "cpus_per_task": core_count,
                        "mem_per_cpu": 8096,
                        "time": "3:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/masks/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/masks/' + "slurm-%j.err",
                    }
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem
                #p_job_id = pyslurm.job().submit_batch_job(p_job)

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                white_mask_solo(folder_path + "/subjects", p, corr_gibbs=corr_gibbs, core_count=core_count, forceUsePowerMap=forceUsePowerMap, debug=debug)
                matplotlib.pyplot.close(fig='all')
                f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied white mask on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "White mask")

        f=open(folder_path + "/logs.txt", "a+")
        f.write("[White mask] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of White mask\n")
        f.close()

    def noddi(self, folder_path=None, patient_list_m=None, force_brain_mask=False, slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """Wrapper function for noddi. Perform noddi and store the data in <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi/.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param force_brain_mask: Force the use of a brain mask even if a whitematter mask exist.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 10h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus.
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

        core_count = 1 if cpus is None else cpus

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi"
            makedir(noddi_path,folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt",
                    log_prefix)

            if slurm:
                core_count = 1 if cpus is None else cpus
                p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import noddi_solo; noddi_solo(\"" + folder_path + "/subjects\",\"" + p + "\"," + str(force_brain_mask) + ",core_count="+str(core_count)+ ")'",
                        "job_name": "noddi_" + p,
                        "ntasks": 1,
                        "cpus_per_task": core_count,
                        "mem_per_cpu": 8096,
                        "time": "10:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/noddi/' + "slurm-%j.err",
                    }
                #p_job_id = pyslurm.job().submit_batch_job(p_job)
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                noddi_solo(folder_path + "/subjects",p,force_brain_mask,core_count=cpus)
                matplotlib.pyplot.close(fig='all')
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
        """Wrapper function for noddi amico. Perform noddi and store the data in <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi_amico/.

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
                matplotlib.pyplot.close(fig='all')
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied NODDI AMICO on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of NODDI AMICO\n")
        f.close()

    def diamond(self, folder_path=None, patient_list_m=None, reportOnly=False, slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """Wrapper function for DIAMOND. Perform diamond and store the data in the <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/diamond/.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 14h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 4 by a custom number of cpus.
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

        core_count = 4 if cpus is None else cpus

        job_list = []
        f=open(folder_path + "/logs.txt", "a+")

        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            diamond_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond"
            makedir(diamond_path,folder_path+'/subjects/'+patient_path+"/dMRI/microstructure/diamond/diamond_logs.txt",
                    log_prefix)

            if slurm:
                p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import diamond_solo; diamond_solo(\"" + folder_path + "/subjects\",\"" + p + "\", reportOnly="+str(reportOnly)+", core_count="+str(core_count)+")'",
                        "job_name": "diamond_" + p,
                        "ntasks": 1,
                        "cpus_per_task": core_count,
                        "mem_per_cpu": 6096,
                        "time": "30:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/diamond/' + "slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/diamond/' + "slurm-%j.err",
                    }
                p_job["time"] = p_job["time"] if slurm_timeout is None else slurm_timeout
                p_job["mem_per_cpu"] = p_job["mem_per_cpu"] if slurm_mem is None else slurm_mem
                #p_job_id = pyslurm.job().submit_batch_job(p_job)

                p_job_id = {}
                p_job_id["id"] = submit_job(p_job)
                p_job_id["name"] = p
                job_list.append(p_job_id)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s is ready to be processed\n" % p)
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                diamond_solo(folder_path + "/subjects",p,core_count=core_count,reportOnly=reportOnly)
                matplotlib.pyplot.close(fig='all')
                f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied diamond on patient %s\n" % p)
                f.flush()
        f.close()

        #Wait for all jobs to finish
        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of DIAMOND\n")
        f.close()

    def tbss(self, folder_path=None, grp1=None, grp2=None, starting_state=None, last_state=None, registration_type="-T", postreg_type="-S", prestats_treshold=0.2, randomise_numberofpermutation=5000, slurm=None, slurm_email=None, slurm_timeout=None, slurm_tasks=None, slurm_mem=None):
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
        :param randomise_numberofpermutation: Define the number of permutations.
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
                "wrap": "python -c 'from elikopy.utils import tbss_utils; tbss_utils(\"" + str(folder_path) + "\",grp1=" + str(grp1) + ",grp2=" + str(grp2) + ",starting_state=\"" + str(starting_state) + "\",last_state=\"" + str(last_state) + "\",registration_type=\"" + str(registration_type) + "\",postreg_type=\"" + str(postreg_type) + "\",prestats_treshold=" + str(prestats_treshold) + ",randomise_numberofpermutation=" + str(randomise_numberofpermutation) + ")'",
                "job_name": "tbss",
                "ntasks": 1,
                "cpus_per_task": 1,
                "mem_per_cpu": 8096,
                "time": "20:00:00",
                "mail_user": slurm_email,
                "mail_type": "FAIL",
                "output": tbss_path + '/' + "slurm-%j.out",
                "error": tbss_path + '/' + "slurm-%j.err",
            }
            job["time"] = job["time"] if slurm_timeout is None else slurm_timeout
            job["cpus_per_task"] = job["cpus_per_task"] if slurm_tasks is None else slurm_tasks
            job["mem_per_cpu"] = job["mem_per_cpu"] if slurm_mem is None else slurm_mem
            p_job_id = {}
            p_job_id["id"] = submit_job(job)
            p_job_id["name"] = "tbss"
            job_list.append(p_job_id)
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            tbss_utils(folder_path=folder_path, grp1=grp1, grp2=grp2, starting_state=starting_state, last_state=last_state, registration_type=registration_type, postreg_type=postreg_type, prestats_treshold=prestats_treshold,randomise_numberofpermutation=randomise_numberofpermutation)
            matplotlib.pyplot.close(fig='all')
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied TBSS \n")
            f.flush()
        f.close()

        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "TBSS")

        f = open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of TBSS\n")
        f.close()

    def export(self, folder_path=None, raw=False, preprocessing=False, dti=False, noddi=False, diamond=False, mf=False, wm_mask=False, report=False, preprocessed_first_b0=False, patient_list_m=None, tractography=False):
        """Wrapper function for tensor reconstruction and computation of DTI metrics using Weighted Least-Squares.
        Performs a tensor reconstruction and saves the DTI metrics.

        :param folder_path: the path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param raw: If true, export files corresponding to the parameter's name.
        :param preprocessing: If true, export files corresponding to the parameter's name.
        :param dti: If true, export files corresponding to the parameter's name.
        :param noddi: If true, export files corresponding to the parameter's name.
        :param diamond: If true, export files corresponding to the parameter's name.
        :param mf: If true, export files corresponding to the parameter's name.
        :param wm_mask: If true, export files corresponding to the parameter's name.
        :param report: If true, export files corresponding to the parameter's name.
        :param preprocessed_first_b0: If true, export files corresponding to the parameter's name.
        :param tractography: If true, export files corresponding to the parameter's name.
        """


        def safe_copy(src,dst):
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))

            if os.path.exists(src):
                shutil.copyfile(src,dst)


        log_prefix = "Export"
        folder_path = self._folder_path if folder_path is None else folder_path
        #slurm = self._slurm if slurm is None else slurm
        #slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of Export\n")# with slurm:" + str(slurm) + "\n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        raw_path = "/dMRI/raw/"
        preprocessing_path = "/dMRI/preproc/"
        dti_path = "/dMRI/microstructure/dti/"
        noddi_path = "/dMRI/microstructure/noddi/"
        diamond_path = "/dMRI/microstructure/diamond/"
        mf_path = "/dMRI/microstructure/mf/"
        wm_mask_path = "/masks/"

        tractography_path = "/dMRI/tracking/"

        makedir(folder_path + '/export/', folder_path + '/logs.txt', log_prefix)
        makedir(folder_path + '/export/' + raw_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/' + preprocessing_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/' + dti_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/' + noddi_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/' + diamond_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/' + mf_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/' + wm_mask_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/' + tractography_path, folder_path + '/export/' + '/export_logs.txt', log_prefix)
        makedir(folder_path + '/export/qc/', folder_path + '/export/' + '/export_logs.txt', log_prefix)

        f=open(folder_path + "/logs.txt", "a+")
        for p in patient_list:

            patient_name = os.path.splitext(p)[0]
            patient_path = folder_path + '/subjects/' + patient_name

            if raw:
                safe_copy(patient_path + raw_path + patient_name + "_dmri_preproc.bval",
                                folder_path + '/export/' + raw_path + patient_name + "_raw_dmri.bval")
                safe_copy(patient_path + raw_path + patient_name + "_dmri_preproc.bvec",
                                folder_path + '/export/' + raw_path + patient_name + "_raw_dmri.bvec")
                safe_copy(patient_path + raw_path + patient_name + '_dmri_preproc.nii.gz',
                                folder_path + '/export/' + raw_path + patient_name + '_raw_dmri.nii.gz')

            if preprocessing:
                safe_copy(patient_path + preprocessing_path + patient_name + "_dmri_preproc.bval",
                                folder_path + '/export/' + preprocessing_path + patient_name + "_dmri_preproc.bval")
                safe_copy(patient_path + preprocessing_path + patient_name + "_dmri_preproc.bvec",
                                folder_path + '/export/' + preprocessing_path + patient_name + "_dmri_preproc.bvec")
                safe_copy(patient_path + preprocessing_path + patient_name + '_dmri_preproc.nii.gz',
                                folder_path + '/export/' + preprocessing_path + patient_name + '_dmri_preproc.nii.gz')
                safe_copy(patient_path + preprocessing_path + 'quality_control/qc_report.pdf',
                                folder_path + '/export/' + preprocessing_path + patient_name + '_qc_report.pdf')

            if preprocessed_first_b0:
                fslroi = "fslroi " + patient_path + preprocessing_path + patient_name + '_dmri_preproc.nii.gz' + " " + patient_path + preprocessing_path + patient_name + '_b0_1_preproc.nii.gz ' + str(0) + " 1"
                process = subprocess.Popen(fslroi, universal_newlines=True, shell=True,
                                           stderr=subprocess.STDOUT)
                output, error = process.communicate()
                safe_copy(patient_path + preprocessing_path + patient_name + '_b0_1_preproc.nii.gz',
                                folder_path + '/export/' + preprocessing_path + patient_name + '_b0_1_preproc.nii.gz')

            if dti:
                safe_copy(patient_path + dti_path + patient_name + "_FA.nii.gz",
                                folder_path + '/export/' + dti_path + patient_name + "_FA.nii.gz")
                safe_copy(patient_path + dti_path + patient_name + "_MD.nii.gz",
                                folder_path + '/export/' + dti_path + patient_name + "_MD.nii.gz")
                safe_copy(patient_path + dti_path + patient_name + "_R2.nii.gz",
                          folder_path + '/export/' + dti_path + patient_name + "_R2.nii.gz")
                safe_copy(patient_path + dti_path + patient_name + "_MSE.nii.gz",
                          folder_path + '/export/' + dti_path + patient_name + "_MSE.nii.gz")
                safe_copy(patient_path + dti_path + patient_name + "_residual.nii.gz",
                                folder_path + '/export/' + dti_path + patient_name + "_residual.nii.gz")
                safe_copy(patient_path + dti_path + 'quality_control/qc_report.pdf',
                                folder_path + '/export/' + dti_path + patient_name + '_qc_report.pdf')

            if noddi:
                safe_copy(patient_path + noddi_path + patient_name + "_noddi_icvf.nii.gz",
                                folder_path + '/export/' + noddi_path + patient_name + "_noddi_icvf.nii.gz")
                safe_copy(patient_path + noddi_path + patient_name + "_noddi_mse.nii.gz",
                                folder_path + '/export/' + noddi_path + patient_name + "_noddi_mse.nii.gz")
                safe_copy(patient_path + noddi_path + patient_name + "_noddi_fextra.nii.gz",
                                folder_path + '/export/' + noddi_path + patient_name + "_noddi_fextra.nii.gz")
                safe_copy(patient_path + noddi_path + 'quality_control/qc_report.pdf',
                                folder_path + '/export/' + noddi_path + patient_name + '_qc_report.pdf')

            if diamond:
                safe_copy(patient_path + diamond_path + patient_name + "_diamond_mosemap.nii.gz",
                                folder_path + '/export/' + diamond_path + patient_name + "_diamond_mosemap.nii.gz")
                safe_copy(patient_path + diamond_path + patient_name + "_diamond_fractions.nii.gz",
                                folder_path + '/export/' + diamond_path + patient_name + "_diamond_fractions.nii.gz")
                safe_copy(patient_path + diamond_path + patient_name + "_diamond_residuals.nii.gz",
                                folder_path + '/export/' + diamond_path + patient_name + "_diamond_residuals.nii.gz")
                safe_copy(patient_path + diamond_path + 'quality_control/qc_report.pdf',
                                folder_path + '/export/' + diamond_path + patient_name + '_qc_report.pdf')

            if mf:
                safe_copy(patient_path + mf_path + patient_name + "_mf_MSE.nii.gz",
                                folder_path + '/export/' + mf_path + patient_name + "_mf_MSE.nii.gz")
                safe_copy(patient_path + mf_path + patient_name + '_mf_peaks.nii.gz',
                                folder_path + '/export/' + mf_path + patient_name + '_mf_peaks.nii.gz')
                safe_copy(patient_path + mf_path + patient_name + '_mf_M0.nii.gz',
                                folder_path + '/export/' + mf_path + patient_name + '_mf_M0.nii.gz')
                safe_copy(patient_path + mf_path + 'quality_control/qc_report.pdf',
                                folder_path + '/export/' + mf_path + patient_name + '_qc_report.pdf')

            if wm_mask:
                safe_copy(patient_path + wm_mask_path + patient_name + '_wm_mask.nii.gz',
                                folder_path + '/export/' + wm_mask_path + patient_name + '_wm_mask.nii.gz')
                safe_copy(patient_path + wm_mask_path + patient_name + '_segmentation.nii.gz',
                                folder_path + '/export/' + wm_mask_path + patient_name + '_segmentation.nii.gz')
                safe_copy(patient_path + wm_mask_path + patient_name + '_brain_mask.nii.gz',
                          folder_path + '/export/' + wm_mask_path + patient_name + '_brain_mask.nii.gz')
                safe_copy(patient_path + wm_mask_path + 'quality_control/qc_report.pdf',
                                folder_path + '/export/' + wm_mask_path + patient_name + '_qc_report.pdf')

            if tractography:
                safe_copy(patient_path + tractography_path + patient_name + '_whole_brain.trk',
                          folder_path + '/export/' + tractography_path + patient_name + '_whole_brain.trk')
                safe_copy(patient_path + tractography_path + patient_name + '.trk',
                          folder_path + '/export/' + tractography_path + patient_name + '.trk')

            if report:
                safe_copy(patient_path + '/quality_control.pdf',
                                folder_path + '/export/qc/' + patient_name + '_quality_control.pdf')

            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Patient %s has been exported succesfully\n" % p)

        f.close()

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of Export\n")
        f.close()

    def regall_FA(self, folder_path=None, grp1=None, grp2=None, starting_state=None, registration_type="-T", postreg_type="-S", prestats_treshold=0.2, slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """ Register all the subjects' FA computed using the DTI model into a common space.

        :param folder_path: path to the root directory.
        :param grp1: List of number corresponding to the type of the patients to put in the first group.
        :param grp2: List of number corresponding to the type of the patients to put in the second group.
        :param starting_state: Manually set which step of TBSS to execute first. Could either be None, reg, post_reg, prestats, design or randomise.
        :param registration_type: Define the argument used by the tbss command tbss_2_reg. Could either by '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is align to every other one, identify the "most representative" one, and use this as the target image.
        :param postreg_type: Define the argument used by the tbss command tbss_3_postreg. Could either by '-S' or '-T'. If you wish to use the FMRIB58_FA mean FA image and its derived skeleton, instead of the mean of your subjects in the study, use the '-T' option. Otherwise, use the '-S' option.
        :param prestats_treshold: Thresholds the mean FA skeleton image at the chosen threshold during prestats.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of used cpus of  by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """

        assert registration_type in ("-T", "-t", "-n"), 'invalid registration type!'
        assert postreg_type in ("-S", "-T"), 'invalid postreg type!'
        assert starting_state in (None, "reg", "postreg", "prestats"), 'invalid starting state!'

        if grp1 is None:
            grp1 = [1]
        if grp2 is None:
            grp2 = [2]

        log_prefix = "REGALL_FA"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f = open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of regall_FA with slurm:" + str(slurm) + "\n")
        f.close()

        reg_path = folder_path + "/registration"
        makedir(reg_path,folder_path + "/logs.txt",log_prefix)

        core_count = 1 if cpus is None else cpus

        job_list = []
        f = open(folder_path + "/logs.txt", "a+")
        if slurm:
            job = {
                "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.utils import regall_FA; regall_FA(\"" + str(folder_path) + "\",grp1=" + str(grp1) + ",grp2=" + str(grp2) + ",starting_state=\"" + str(starting_state) + "\",registration_type=\"" + str(registration_type) + "\",postreg_type=\"" + str(postreg_type) + "\",prestats_treshold=" + str(prestats_treshold) + ")'",
                "job_name": "regall_FA",
                "ntasks": 1,
                "cpus_per_task": core_count,
                "mem_per_cpu": 8096,
                "time": "20:00:00",
                "mail_user": slurm_email,
                "mail_type": "FAIL",
                "output": reg_path + '/' + "slurm-%j.out",
                "error": reg_path + '/' + "slurm-%j.err",
            }
            job["time"] = job["time"] if slurm_timeout is None else slurm_timeout
            job["mem_per_cpu"] = job["mem_per_cpu"] if slurm_mem is None else slurm_mem
            p_job_id = {}
            p_job_id["id"] = submit_job(job)
            p_job_id["name"] = "regall_FA"
            job_list.append(p_job_id)
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            regall_FA(folder_path=folder_path, grp1=grp1, grp2=grp2, starting_state=starting_state, registration_type=registration_type, postreg_type=postreg_type, prestats_treshold=prestats_treshold, core_count=core_count)
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully applied REGALL_FA \n")
            f.flush()
        f.close()

        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "REGALL_FA")

        f = open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of REGALL_FA\n")
        f.close()

    def regall(self, folder_path=None, grp1=None, grp2=None, metrics_dic={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'},
               slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """ Register all the subjects' metric passed in parameters into a common space using the transformation computed when registering the FA. The function regall_FA needs to be applied first!

        :param folder_path: path to the root directory.
        :param grp1: List of number corresponding to the type of the patients to put in the first group.
        :param grp2: List of number corresponding to the type of the patients to put in the second group.
        :param metrics_dic: Dictionnary containing multiple metrics. For each metrics, the metric name is the key and the folder metric's is the value.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of used cpus (1) by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """


        if grp1 is None:
            grp1 = [1]
        if grp2 is None:
            grp2 = [2]

        log_prefix = "REGALL"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f = open(folder_path + "/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of regall with slurm:" + str(slurm) + "\n")
        f.close()

        reg_path = folder_path + "/registration"
        makedir(reg_path, folder_path + "/logs.txt", log_prefix)

        core_count = 1 if cpus is None else cpus

        job_list = []
        f = open(folder_path + "/logs.txt", "a+")
        if slurm:
            job = {
                "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.utils import regall; regall(\"" + str(
                    folder_path) + "\",grp1=" + str(grp1) + ",grp2=" + str(grp2) + ",metrics_dic=" + str(
                    json.dumps(metrics_dic)) + ")'",
                "job_name": "regall",
                "ntasks": 1,
                "cpus_per_task": core_count,
                "mem_per_cpu": 8096,
                "time": "20:00:00",
                "mail_user": slurm_email,
                "mail_type": "FAIL",
                "output": reg_path + '/' + "slurm-%j.out",
                "error": reg_path + '/' + "slurm-%j.err",
            }
            job["time"] = job["time"] if slurm_timeout is None else slurm_timeout
            job["mem_per_cpu"] = job["mem_per_cpu"] if slurm_mem is None else slurm_mem
            p_job_id = {}
            p_job_id["id"] = submit_job(job)
            p_job_id["name"] = "regall"
            job_list.append(p_job_id)
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            regall(folder_path=folder_path, grp1=grp1, grp2=grp2, core_count=core_count, metrics_dic=metrics_dic)
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully applied REGALL \n")
            f.flush()
        f.close()

        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "REGALL")

        f = open(folder_path + "/logs.txt", "a+")
        f.write(
            "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of REGALL\n")
        f.close()




    def randomise_all(self, folder_path=None,randomise_numberofpermutation=5000,skeletonised=True,metrics_dic={'FA':'dti','_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}, regionWiseMean=True,
               slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """ Wrapper function for randomise_all. Perform tract base spatial statistics between the control data and case data and region wise stats.
        DTI needs to have been performed on the data first !!

        :param folder_path: path to the root directory.
        :param randomise_numberofpermutation: Define the number of permutation used bu randomize.
        :param skeletonised: If True, randomize will be using the only skeleton instead of the whole brain.
        :param metrics_dic: Dictionnary containing multiple metrics. For each metrics, the metric name is the key and the folder metric's is the value.
        :param regionWiseMean: If true, csv containing atlas-based region wise mean will be generated.
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of used cpus by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """

        log_prefix = "randomise_all"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        f = open(folder_path + "/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of randomise_all with slurm:" + str(slurm) + "\n")
        f.close()

        reg_path = folder_path + "/registration"
        makedir(reg_path, folder_path + "/logs.txt", log_prefix)

        core_count = 1 if cpus is None else cpus

        job_list = []
        f = open(folder_path + "/logs.txt", "a+")
        if slurm:
            job = {
                "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.utils import randomise_all; randomise_all(\"" + str(
                    folder_path) + "\",randomise_numberofpermutation=" + str(randomise_numberofpermutation) + ",skeletonised=" + str(skeletonised) + ",core_count=" + str(core_count) + ",regionWiseMean="  + str(regionWiseMean) + ",metrics_dic=" + str(
                    json.dumps(metrics_dic)) + ")'",
                "job_name": "randomise_all",
                "ntasks": 1,
                "cpus_per_task": core_count,
                "mem_per_cpu": 8096,
                "time": "20:00:00",
                "mail_user": slurm_email,
                "mail_type": "FAIL",
                "output": reg_path + '/' + "slurm-%j.out",
                "error": reg_path + '/' + "slurm-%j.err",
            }
            job["time"] = job["time"] if slurm_timeout is None else slurm_timeout
            job["mem_per_cpu"] = job["mem_per_cpu"] if slurm_mem is None else slurm_mem
            p_job_id = {}
            p_job_id["id"] = submit_job(job)
            p_job_id["name"] = "randomise_all"
            job_list.append(p_job_id)
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
        else:
            randomise_all(folder_path=folder_path, randomise_numberofpermutation=randomise_numberofpermutation, skeletonised=skeletonised, metrics_dic=metrics_dic,core_count=cpus, regionWiseMean=regionWiseMean)
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully applied randomise_all \n")
            f.flush()
        f.close()

        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "randomise_all")

        f = open(folder_path + "/logs.txt", "a+")
        f.write(
            "[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of randomise_all\n")
        f.close()

    def noddi_fix_icvf_thresholding(self, folder_path=None, patient_list_m=None, fintra_threshold=0.99, fbundle_threshold=0.05,
                 use_brain_mask=False, use_wm_mask=False):
        """ A function to quickly change the treshold value applied on the icvf metric of noddi without the needs of executing again the full noddi core function.

        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param fintra_threshold: Threshold applied on the fintra.
        :param fbundle_threshold: Threshold applied on the fbundle.
        :param use_brain_mask: Set to 0 values outside the brain mask.
        :param use_wm_mask: Set to 0 values outside the white matter mask.
        """
        import numpy as np
        from dipy.io.image import load_nifti, save_nifti
        folder_path = self._folder_path if folder_path is None else folder_path

        f = open(folder_path + "/logs.txt", "a+")
        f.write("[Fix icvf] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of fix icvf \n")
        f.close()

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []
        f = open(folder_path + "/logs.txt", "a+")
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            fintra_path = folder_path + "/subjects/" + p + "/dMRI/microstructure/noddi/" + p + "_noddi_fintra.nii.gz"
            fbundle_path = folder_path + "/subjects/" + p + "/dMRI/microstructure/noddi/" + p + "_noddi_fbundle.nii.gz"

            brain_mask_path = folder_path + "/subjects/" + p + "/masks/" + p + "_brain_mask.nii.gz"
            wm_mask_path = folder_path + "/subjects/" + p + "/masks/" + p + "_wm_mask.nii.gz"

            icvf_path = folder_path + "/subjects/" + p + "/dMRI/microstructure/noddi/" + p + "_noddi_icvf.nii.gz"

            if os.path.exists(fintra_path) and os.path.exists(fbundle_path):
                data_fintra, affine_fintra, voxel_size_fintra = load_nifti(fintra_path, return_voxsize=True)
                data_fbundle, affine_fbundle, voxel_size_fbundle = load_nifti(fbundle_path, return_voxsize=True)

                data_icvf_new = data_fintra * (data_fintra < fintra_threshold) * (data_fbundle > fbundle_threshold)

                if use_brain_mask and os.path.exists(brain_mask_path):
                    data_brain_mask, affine_brain_mask, voxel_size_brain_mask = load_nifti(brain_mask_path, return_voxsize=True)
                    data_icvf_new = data_icvf_new * (data_brain_mask > 0.95)

                if use_wm_mask and os.path.exists(wm_mask_path):
                    data_wm_mask, affine_wm_mask, voxel_size_wm_mask = load_nifti(wm_mask_path, return_voxsize=True)
                    data_icvf_new = data_icvf_new * (data_wm_mask > 0.95)

                save_nifti(icvf_path, data_icvf_new.astype(np.float32), affine_fintra)

            f.write("[Fix icvf] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully applied fix icvf on patient %s\n" % p)
            f.flush()

        f.write("[Fix icvf] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of fix icvf\n")
        f.close()


    def patientlist_wrapper(self, function, func_args, folder_path=None, patient_list_m=None, filename=None, function_name=None, slurm=False, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """ A wrapper function that apply a function given as an argument to every patients of the study. The wrapped function must takes two arguments as input, the patient\_name and the path to the root of the study.


        :param folder_path: path to the root directory.
        :param patient_list_m: Define a subset a patient to process instead of all the available subjects.
        :param function: The pointer to the function (only used without slurm)
        :param func_args: Additional arguments to pass to the wrapped function (only used without slurm)
        :param filename: The name of the file containing the wrapped function (only used with slurm)
        :param function_name: The name of the wrapped function (only used with slurm)
        :param slurm: Whether to use the Slurm Workload Manager or not.
        :param slurm_email: Email adress to send notification if a task fails.
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of used cpus by a custom number of cpus.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        log_prefix = "wrapper_elikopy"


        f = open(folder_path + "/logs.txt", "a+")
        f.write("[PatientList Wrapper] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of wrap function on patient list \n")
        print("[PatientList Wrapper] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of wrap function on patient list \n")

        core_count = 1 if cpus is None else cpus

        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f2:
            patient_list = json.load(f2)

        if patient_list_m:
            patient_list = patient_list_m

        job_list = []

        for p in patient_list:
            patient_path = os.path.splitext(p)[0]

            if slurm:
                job = {
                    "wrap": "export OMP_NUM_THREADS=" + str(core_count) + " ; export FSLPARALLEL=" + str(
                        core_count) + " ; python -c 'from " + filename + " import "+function_name+"; "+function_name+"(\"" + str(
                        folder_path) + "\",\"" + str(
                        patient_path) + "\")'",
                    "job_name": "wrapper_elikopy",
                    "ntasks": 1,
                    "cpus_per_task": core_count,
                    "mem_per_cpu": 8096,
                    "time": "20:00:00",
                    "mail_user": slurm_email,
                    "mail_type": "FAIL",
                    "output": folder_path + '/' + "slurm-%j.out",
                    "error": folder_path + '/' + "slurm-%j.err",
                }
                job["time"] = job["time"] if slurm_timeout is None else slurm_timeout
                job["mem_per_cpu"] = job["mem_per_cpu"] if slurm_mem is None else slurm_mem
                p_job_id = {}
                p_job_id["id"] = submit_job(job)
                p_job_id["name"] = "wrapper_elikopy"
                job_list.append(p_job_id)
                f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                    "%d.%b %Y %H:%M:%S") + ": Successfully submited job %s using slurm\n" % p_job_id)
            else:
                function(folder_path, patient_path, *func_args)

            print("[PatientList Wrapper] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully applied wrap function on patient list on patient %s\n" % p)

            f.write("[PatientList Wrapper] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Successfully applied wrap function on patient list on patient %s\n" % p)
            f.flush()

        f.close()

        if slurm:
            elikopy.utils.getJobsState(folder_path, job_list, "randomise_all")

        f = open(folder_path + "/logs.txt", "a+")
        print("[PatientList Wrapper] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of wrap function on patient list\n")
        f.write("[PatientList Wrapper] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of wrap function on patient list\n")
        f.close()
