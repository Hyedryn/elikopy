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
    """ Convert dicom data into compressed nifti. Converted dicoms are then moved to a sub-folder named original_data.
    The niftis are named patientID_ProtocolName_SequenceName.

    :param folder_path: Path to root folder containing all the dicoms
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
    for file in files:
        if "mrdc" in file or "MRDC" in file:
            shutil.move(folder_path + '/' + file, dest)

            f.write("[DICOM TO NIFTI] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Moved " + file + " to " + dest + "\n")
    f.close()


class Elikopy:
    r'''
    Main class containing all the necessary function to process and preprocess a specific study.
    '''

    def __init__(self, folder_path, cuda=False, slurm=False, slurm_email='example@example.com'):
        """ Creates the study class
            example : study = Elikopy(my_floder, slurm=True, slurm_email='my_email_address')

            :param folder_path: Path to root folder of the study.
            :param cuda: wether or not run on cuda when possible. default = FALSE
            :param slurm: wether or not use the slurm job scheduler (e.g. for computer clusters). default = FALSE
            :param slurm_email: the email for the slurm jobs (e.g. for computer clusters)
        """
        self._folder_path = folder_path
        self._slurm = slurm
        self._slurm_email = slurm_email
        self._cuda = cuda

    def patient_list(self, folder_path=None, bids_path=None, reverseEncoding=True):
        """ From the root folder containing data_1, data_2, ... data_n folders with nifti files (and their corresponding
        bvals and bvecs), the Elikopy folder structure is created in a directory named 'subjects' inside folder_path.
        This step is mandatory. The validity of all the nifti present in the root folder is verified. If some nifti
        do not possess an associated bval and bvec file, they are discarded and the user is notified in a summary file named
        subj_error.json generated in the out sub-directory. All valid patients are stored in a file named patient_list.json.
        In addition to the nifti + bval + bvec, the data_n folders can also contain the json files (with the patient informations) as well as
        the acquparam, index and slspec files (used during the preprocessing). If these files are missing a warning is raised.
        In addition to the DW images, T1 structural images can be provided in a directory called 'T1' in the root folder.

        example : study.patient_list()

        :param folder_path: Path to the root folder of the study. default = study_folder
        :param bids_path: Path to the optional folder containing subjects' data in the BIDS format.
        :param reverseEncoding: Append reverse encoding direction to the DW-MRI data if available. default = True
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
                            with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec", "r") as file_object2:
                                nlines = file_object2.read().count('\n')

                            if nlines > 4:
                                lines.append("1 0 0\n")
                                with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec",
                                          "w") as f:
                                    for line in lines:
                                        f.write(line)
                            else:
                                with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bvec",
                                          "w") as f:
                                    i = 0
                                    for line in lines:
                                        if i==0:
                                            f.write(line.rstrip().rstrip("\n") + " 1\n")
                                        elif i==1:
                                            f.write(line.rstrip().rstrip("\n") + " 0\n")
                                        elif i==2:
                                            f.write(line.rstrip().rstrip("\n") + " 0\n")
                                        else:
                                            f.write(line)
                                        i = i + 1

                            #Edit bval
                            with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval", "r") as file_object:
                                file_object=file_object.read().rstrip().rstrip("\n")

                            nlines = file_object.count('\n')
                            if nlines > 4:
                                with open(folder_path + "/subjects/" + name + "/dMRI/raw/" + name + "_raw_dmri.bval", "w") as myfile:
                                    myfile.write(file_object + "\n0"+ "\n")
                            else:
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

    def preproc(self, folder_path=None, reslice=False, reslice_addSlice=False, denoising=False, mppca_legacy_denoising=False, gibbs=False, topup=False, topupConfig=None, forceSynb0DisCo=False, useGPUsynb0DisCo=False, eddy=False, biasfield=False, biasfield_bsplineFitting=[100,3], biasfield_convergence=[1000,0.001], patient_list_m=None, starting_state=None, bet_median_radius=2, bet_numpass=1, bet_dilate=2, cuda=None, cuda_name="eddy_cuda10.1", s2v=[0,5,1,'trilinear'], olrep=[False, 4, 250, 'sw'], slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None, qc_reg=True, niter=5, slspec_gc_path=None, report=True):
        """ Performs data preprocessing. By default only the brain extraction is enabled. Optional preprocessing steps include : reslicing,
        denoising, gibbs ringing correction, susceptibility field estimation, EC-induced distortions and motion correction, bias field correction.
        The results are stored in the preprocessing subfolder of each study subject <folder_path>/subjects/<subjects_ID>/dMRI/preproc.

        example : study.preproc(denoising=True, topup=True, eddy=True, biasfield=True)

        :param folder_path: the path to the root directory. default=study_folder
        :param reslice: If true, data will be resliced with a new voxel resolution of 2*2*2. default=False
        :param reslice_addSlice: If true, an additional empty slice will be added to each volume (might be useful for motion correction if one slice is dropped during the acquisition and the user still wants to perform easily the slice-to-volume motion correction). default=False
        :param denoising: If true, MPPCA-denoising is performed on the data. default=False
        :param gibbs: If true, Gibbs ringing correction is performed. We do not advise to use this correction unless the data suffers from a lot of Gibbs ringing artifacts. default=False
        :param topup: If true, Topup will estimate the susceptibility induced distortions. These distortions are corrected at the same time as EC-induced distortions if eddy=True. In the absence of images acquired with a reverse phase encoding direction, a T1 structural image is required. default=False
        :param topupConfig: If not None, Topup will use additionnal parameters based on the supplied config file located at <topupConfig>. default=None
        :param forceSynb0DisCo: If true, Topup will always estimate the susceptibility field using the T1 structural image. default=False
        :param eddy: If true, Eddy corrects the EC-induced (+ susceptibility, if estimated) distortions and the motion. If these corrections are performed the acquparam and index files are required (see documentation). To perform the slice-to-volume motion correction the slspec file is also needed. default=False
        :param biasfield: If true, low frequency intensity non-uniformity present in MRI image data known as a bias or gain field will be corrected. default=False
        :param biasfield_bsplineFitting: Define the initial mesh resolution in mm and the bspline order of the biasfield correction tool. default=[100,3]
        :param biasfield_convergence: Define the maximum number of iteration and the convergences threshold of the biasfield correction tool. default=[1000,0.001]
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param starting_state: Manually set which step of the preprocessing to execute first. Could either be None, denoising, gibbs, topup, eddy, biasfield, report or post_report. default=None
        :param bet_median_radius: Radius (in voxels) of the applied median filter during brain extraction. default=2
        :param bet_numpass: Number of pass of the median filter during brain extraction. default=1
        :param bet_dilate: Number of iterations for binary dilation during brain extraction. default=2
        :param cuda: If true, eddy will run on cuda with the command name specified in cuda_name. default=False
        :param cuda_name: name of the eddy command to run when cuda==True. default="eddy_cuda10.1"
        :param s2v: list of parameters of Eddy for slice-to-volume motion correction (see Eddy FSL documentation): [mporder,s2v_niter,s2v_lambda,s2v_interp]. The slice-to-volume motion correction is performed if mporder>0, cuda is used and a slspec file is provided during the patient_list command. default=[0,5,1,'trilinear']
        :param olrep: list of parameters of Eddy for outlier replacement (see Eddy FSL documentation): [repol,ol_nstd,ol_nvox,ol_type]. The outlier replacement is performed if repol==True. default=[False, 4, 250, 'sw']
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout by a custom timeout.
        :param cpus: Replace the default number of slurm cpus by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task by a custom amount of ram.
        :param qc_reg: If true, the motion registration step of the quality control will be performed. We do not advise to use this argument as it increases the computation time. default=False
        :param niter: Define the number of iterations for eddy volume-to-volume. default=5
        :param slspec_gc_path: Path to the folder containing volume specific slice-specification for eddy. If not None, eddy motion correction with gradient cycling will be performed.
        :param report: If False, no quality report will be generated. default=True
        """

        assert starting_state in (None, "denoising", "gibbs", "topup", "eddy", "biasfield", "report", "post_report"), 'invalid starting state!'
        if mppca_legacy_denoising==True:
            assert denoising == True, 'if mppca_legacy_denoising is True, denoising must be True!'
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
                patient_path = p
                preproc_path = folder_path + '/subjects/' + patient_path + "/dMRI/preproc/bet"
                makedir(preproc_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

                if slurm:
                    tot_cpu = 8 if cpus is None else cpus
                    core_count = tot_cpu
                    p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(tot_cpu)+" ; export FSLPARALLEL="+str(tot_cpu)+" ; python -c 'from elikopy.individual_subject_processing import preproc_solo; preproc_solo(\"" + folder_path + "/\",\"" + p + "\",eddy=" + str(
                            eddy) + ",biasfield=" + str(biasfield)  + ",biasfield_convergence=[" + str(biasfield_convergence[0]) + "," + str(biasfield_convergence[1]) + "],biasfield_bsplineFitting=[" + str(biasfield_bsplineFitting[0]) + "," + str(biasfield_bsplineFitting[1]) + "],denoising=" + str(
                            denoising) + ",mppca_legacy_denoising=" + str(mppca_legacy_denoising) +",reslice=" + str(reslice) + ",reslice_addSlice=" + str(reslice_addSlice) + ",gibbs=" + str(
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
                    preproc_solo(folder_path + "/",p,reslice=reslice,reslice_addSlice=reslice_addSlice,denoising=denoising, mppca_legacy_denoising=mppca_legacy_denoising, gibbs=gibbs,
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
                patient_path = p
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
                patient_path = p
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/preproc/quality_control/qc_report.pdf',
                    folder_path + '/subjects/' + patient_path + '/quality_control.pdf')

        f=open(folder_path + "/logs.txt", "a+")
        f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": All the preprocessing operation are finished!\n")
        f.close()

    def dti(self,folder_path=None, patient_list_m=None, use_wm_mask=False, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Computes the DTI metrics for each subject using Weighted Least-Squares. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/dti/.

        example : study.dti()

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param use_wm: If true a white matter mask is used. The white_matter() function needs to already be applied. default=False
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 1h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
            patient_path = p

            dti_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti"
            makedir(dti_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/dti/dti_logs.txt",
                    log_prefix)

            if slurm:
                p_job = {
                        "wrap": "python -c 'from elikopy.individual_subject_processing import dti_solo; dti_solo(\"" + folder_path + "/\",\"" + p + "\",use_wm_mask=" + str(use_wm_mask) + ")'",
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
                dti_solo(folder_path + "/",p,use_wm_mask=use_wm_mask)
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

    def fingerprinting(self, dictionary_path=None, folder_path=None, CSD_bvalue = None, use_wm_mask=False, csf_mask=True, ear_mask=False, mfdir=None, slurm=None, patient_list_m=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """Computes the Microstructure Fingerprinting metrics for each subject. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/mf/.

        example : study.fingerprinting(dictionary_path='my_dictionary')

        :param folder_path: the path to the root directory. default=study_folder
        :param dictionary_path: Path to the dictionary of fingerprints (mandatory).
        :param CSD_bvalue: If the DIAMOND outputs are not available, the fascicles directions are estimated using a CSD with the images at the b-values specified in this argument. default=None
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """
        log_prefix="MF"
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        mfdir = "mf" if mfdir is None else mfdir
        slurm_email = self._slurm_email if slurm_email is None else slurm_email

        dictionary_path = folder_path + "/static_files/mf_dic/fixed_rad_dist.mat" if dictionary_path is None else dictionary_path

        import os.path
        assert os.path.isfile(dictionary_path), 'Invalid path to the MF dictionary'


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
            patient_path = p

            mf_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/" + mfdir
            makedir(mf_path,folder_path + '/subjects/' + patient_path+"/dMRI/microstructure/" + mfdir + "/mf_logs.txt",log_prefix)

            if slurm:
                p_job = {
                        "wrap": "export MKL_NUM_THREADS="+ str(core_count)+" ; export OMP_NUM_THREADS="+ str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import mf_solo; mf_solo(\"" + folder_path + "/\",\"" + p + "\", \"" + dictionary_path + "\", CSD_bvalue =" + str(CSD_bvalue) + ", core_count=" + str(core_count) + ", use_wm_mask=" + str(use_wm_mask) + ", mfdir=\"" + str(mfdir)+ "\", csf_mask=" + str(csf_mask) + ", ear_mask=" + str(ear_mask) + ")'",
                        "job_name": "mf_" + p,
                        "ntasks": 1,
                        "cpus_per_task": core_count,
                        "mem_per_cpu": 8096,
                        "time": "20:00:00",
                        "mail_user": slurm_email,
                        "mail_type": "FAIL",
                        "output": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/' + mfdir + "/slurm-%j.out",
                        "error": folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/' + mfdir + "/slurm-%j.err",
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
                mf_solo(folder_path + "/", p, dictionary_path, CSD_bvalue = CSD_bvalue, core_count=core_count, use_wm_mask=use_wm_mask, csf_mask=csf_mask, ear_mask=ear_mask, mfdir=mfdir)
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
        """ Computes a white matter mask for each subject based on the T1 structural images or on the anisotropic power maps
        (obtained from the diffusion images) if the T1 images are not available. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/masks/.
        The T1 images can be gibbs ringing corrected.

        example : study.white_mask()

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param corr_gibbs: If true, Gibbs ringing correction is performed on the T1 images. default=True
        :param forceUsePowerMap: Force the use of an AnisotropicPower map for the white matter mask generation. default=False
        :param debug: If true, additional intermediate output will be saved. default=False
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 3h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
            patient_path = p
            if slurm:
                core_count = 1 if cpus is None else cpus
                p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import white_mask_solo; white_mask_solo(\"" + folder_path + "/\",\"" + p + "\"" + ",corr_gibbs=" + str(corr_gibbs) + ",forceUsePowerMap=" + str(forceUsePowerMap) + ",debug=" + str(debug) + ",core_count=" + str(core_count) + " )'",
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
                white_mask_solo(folder_path + "/", p, corr_gibbs=corr_gibbs, core_count=core_count, forceUsePowerMap=forceUsePowerMap, debug=debug)
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

    def noddi(self, folder_path=None, patient_list_m=None, use_wm_mask=False, slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None, lambda_iso_diff=3.e-9, lambda_par_diff=1.7e-9):
        """Computes the NODDI metrics for each subject. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi/.

        example : study.noddi()

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param force_brain_mask: Force the use of a brain mask even if a whitematter mask exist. default=False
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 10h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        :param lambda_iso_diff: user defined isotropic diffusivity for the CSF model. default=3.e-9
        :param lambda_par_diff: user defined axial diffusivity of the intra-neurite space. default=1.7e-9
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
            patient_path = p

            noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi"
            makedir(noddi_path,folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi/noddi_logs.txt",
                    log_prefix)

            if slurm:
                core_count = 1 if cpus is None else cpus
                p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import noddi_solo; noddi_solo(\"" + folder_path + "/\",\"" + p + "\",use_wm_mask=" + str(use_wm_mask) + ",core_count="+str(core_count)+ ",lambda_iso_diff="+str(lambda_iso_diff) +", lambda_par_diff="+str(lambda_par_diff) + ")'",
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
                noddi_solo(folder_path + "/",p,core_count=cpus,lambda_iso_diff=lambda_iso_diff, lambda_par_diff=lambda_par_diff,use_wm_mask=use_wm_mask)
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

    def noddi_amico(self, folder_path=None, patient_list_m=None, force_brain_mask=False, use_wm_mask=False, slurm=None, slurm_email=None, slurm_timeout=None, slurm_cpus=None, slurm_mem=None):
        """Computes the NODDI amico metrics for each subject. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/noddi/.

        example : study.noddi_amico()

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param force_brain_mask: Force the use of a brain mask even if a whitematter mask exist. default=False
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 10h by a custom timeout.
        :param slurm_cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
            patient_path = p

            noddi_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi_amico"
            makedir(noddi_path, folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/noddi_amico/noddi_amico_logs.txt", log_prefix)

            if slurm:
                p_job = {
                        "wrap": "python -c 'from elikopy.individual_subject_processing import noddi_amico_solo; noddi_amico_solo(\"" + folder_path + "/\",\"" + p + "\"" + ", use_wm_mask=" + str(use_wm_mask) + ")'",
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
                noddi_amico_solo(folder_path + "/",p,use_wm_mask=use_wm_mask)
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

    def diamond(self, folder_path=None, patient_list_m=None, reportOnly=False, use_wm_mask=False, customDiamond="", slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """Computes the DIAMOND metrics for each subject. The outputs are available in the directories <folder_path>/subjects/<subjects_ID>/dMRI/microstructure/diamond/.

        example : study.diamond()

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 14h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 4 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
            patient_path = p

            diamond_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/diamond"
            makedir(diamond_path,folder_path+'/subjects/'+patient_path+"/dMRI/microstructure/diamond/diamond_logs.txt",
                    log_prefix)

            if slurm:
                p_job = {
                        "wrap": "export OMP_NUM_THREADS="+str(core_count)+" ; export FSLPARALLEL="+str(core_count)+" ; python -c 'from elikopy.individual_subject_processing import diamond_solo; diamond_solo(\"" + folder_path + "/\",\"" + p + "\", reportOnly="+str(reportOnly) + ", core_count="+str(core_count) + ", use_wm_mask=" + str(use_wm_mask) + ", customDiamond=\"" + customDiamond + "\")'",
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
                diamond_solo(folder_path + "/",p,core_count=core_count,reportOnly=reportOnly,use_wm_mask=use_wm_mask,customDiamond=customDiamond)
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
        """ Performs tract base spatial statistics (TBSS) between the data in grp1 and grp2. The data type of each subject is specified by the subj_type.json file generated during the call to the patient_list function. The data type corresponds to the original directory of the subject (e.g. a subject that was originally in the folder data_2 is of type 2).
        It is mandatory to have performed DTI prior to tbss.
        This is function should not be used as it has been replaced by regall_FA, regall and randomise_all to allow for more flexibility.

        example : study.tbss(grp1=[1,2], grp2=[3,4])

        :param folder_path: the path to the root directory. default=study_folder
        :param grp1: List of number corresponding to the type of the subjects to put in the first group.
        :param grp2: List of number corresponding to the type of the subjects to put in the second group.
        :param starting_state: Manually set which step of TBSS to execute first. Could either be None, reg, post_reg, prestats, design or randomise. default=None
        :param last_state: Manually set which step of TBSS to execute last. Could either be None, preproc, reg, post_reg, prestats, design or randomise. default=None
        :param registration_type: Define the argument used by the tbss command tbss_2_reg. Could either by '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is align to every other one, identify the "most representative" one, and use this as the target image.
        :param postreg_type: Define the argument used by the tbss command tbss_3_postreg. Could either by '-S' or '-T'. If you wish to use the FMRIB58_FA mean FA image and its derived skeleton, instead of the mean of your subjects in the study, use the '-T' option. Otherwise, use the '-S' option.
        :param prestats_treshold: Thresholds the mean FA skeleton image at the chosen threshold during prestats. default=0.2
        :param randomise_numberofpermutation: Define the number of permutations. default=5000
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param slurm_tasks: Replace the default number of slurm cpus of 8 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
        """ Allows to obtain in a single Export folder the outputs of specific processing steps for all subjects.

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param raw: If true, copy the raw data of each subject in the Export folder. default=FALSE
        :param preprocessing: If true, copy the preprocessed data of each subject in the Export folder. default=FALSE
        :param dti: If true, copy the DTI outputs of each subject in the Export folder. default=FALSE
        :param noddi: If true, copy the NODDI outputs of each subject in the Export folder. default=FALSE
        :param diamond: If true, copy the DIAMOND outputs of each subject in the Export folder. default=FALSE
        :param mf: If true, copy the MF outputs of each subject in the Export folder. default=FALSE
        :param wm_mask: If true, copy the white matter mask of each subject in the Export folder. default=FALSE
        :param report: If true, copy the quality control reports of each subject in the Export folder. default=FALSE
        :param tractography: If true, copy the tractography outputs of each subject in the Export folder. default=FALSE
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
        """ Register all the subjects Fractional Anisotropy into a common space, skeletonisedd and non skeletonised. This is performed based on TBSS of FSL.
        It is mandatory to have performed DTI prior to regall_FA.

        :param folder_path: the path to the root directory. default=study_folder
        :param grp1: List of number corresponding to the type of the subjects to put in the first group.
        :param grp2: List of number corresponding to the type of the subjects to put in the second group.
        :param starting_state: Manually set which step of TBSS to execute first. Could either be None, reg, post_reg, prestats, design or randomise. default=None
        :param registration_type: Define the argument used by the tbss command tbss_2_reg. Could either by '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is align to every other one, identify the "most representative" one, and use this as the target image.
        :param postreg_type: Define the argument used by the tbss command tbss_3_postreg. Could either by '-S' or '-T'. If you wish to use the FMRIB58_FA mean FA image and its derived skeleton, instead of the mean of your subjects in the study, use the '-T' option. Otherwise, use the '-S' option.
        :param prestats_treshold: Thresholds the mean FA skeleton image at the chosen threshold during prestats. default=0.2
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 8 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
        """ Register all the subjects diffusion metrics specified in the argument metrics_dic into a common space using the transformation computed for the FA with the regall_FA function. This is performed based on TBSS of FSL.
        It is mandatory to have performed regall_FA prior to regall.

        :param folder_path: the path to the root directory. default=study_folder
        :param grp1: List of number corresponding to the type of the subjects to put in the first group.
        :param grp2: List of number corresponding to the type of the subjects to put in the second group.
        :param metrics_dic: Dictionnary containing the diffusion metrics to register in a common space. For each diffusion metric, the metric name is the key and the metric's folder is the value. default={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
               additional_atlases=None, slurm=None, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None):
        """ Performs tract base spatial statistics (TBSS) between the data in grp1 and grp2 (groups are specified during the call to regall_FA) for each diffusion metric specified in the argument metrics_dic.
        The mean value of the diffusion metrics across atlases regions can also be reported in CSV files using the regionWiseMean flag. The used atlases are : the Harvard-Oxford cortical and subcortical structural atlases, the JHU DTI-based white-matter atlases and the MNI structural atlas
        It is mandatory to have performed regall_FA prior to randomise_all.

        :param folder_path: the path to the root directory. default=study_folder
        :param randomise_numberofpermutation: Define the number of permutations. default=5000
        :param skeletonised: If True, randomize will be using only the white matter skeleton instead of the whole brain. default=True
        :param metrics_dic: Dictionnary containing the diffusion metrics to register in a common space. For each diffusion metric, the metric name is the key and the metric's folder is the value. default={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
        :param regionWiseMean: If true, csv containing atlas-based region wise mean will be generated.
        :param additional_atlases: Dic that define additional atlases to be used as segmentation template for csv generation (see regionWiseMean). Dictionary is in the form {'Atlas_name_1':["path to atlas 1 xml","path to atlas 1 nifti"],'Atlas_name_1':["path to atlas 2 xml","path to atlas 2 nifti"]}.
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
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
                    json.dumps(metrics_dic)) + ",additional_atlases=" + str(additional_atlases) + ")'",
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
            randomise_all(folder_path=folder_path, randomise_numberofpermutation=randomise_numberofpermutation, skeletonised=skeletonised, metrics_dic=metrics_dic,core_count=cpus, regionWiseMean=regionWiseMean, additional_atlases=additional_atlases)
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

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param fintra_threshold: Threshold applied on the fintra. default=0.99
        :param fbundle_threshold: Threshold applied on the fbundle. default=0.05
        :param use_brain_mask: Set to 0 values outside the brain mask. default=False
        :param use_wm_mask: Set to 0 values outside the white matter mask. default=False
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
            patient_path = p

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


    def patientlist_wrapper(self, function, func_args, folder_path=None, patient_list_m=None, filename=None, function_name=None, slurm=False, slurm_email=None, slurm_timeout=None, cpus=None, slurm_mem=None, slurm_path=None):
        """ A wrapper function that apply a function given as an argument to every subject of the study. The wrapped function must takes two arguments as input, the patient\_name and the path to the root of the study.

        :param folder_path: the path to the root directory. default=study_folder
        :param patient_list_m: Define a subset of subjects to process instead of all the available subjects. example : ['patientID1','patientID2','patientID3']. default=None
        :param function: The pointer to the function (only without slurm /!\)
        :param func_args: Additional arguments to pass to the wrapped function (only without slurm /!\)
        :param filename: The name of the file containing the wrapped function (only with slurm /!\)
        :param function_name: The name of the wrapped function (only with slurm /!\)
        :param slurm: Whether to use the Slurm Workload Manager or not (for computer clusters). default=value_during_init
        :param slurm_email: Email adress to send notification if a task fails. default=None
        :param slurm_timeout: Replace the default slurm timeout of 20h by a custom timeout.
        :param cpus: Replace the default number of slurm cpus of 1 by a custom number of cpus of using slum, or for standard processing, its the number of core available for processing.
        :param slurm_mem: Replace the default amount of ram allocated to the slurm task (8096MO by cpu) by a custom amount of ram.
        """
        folder_path = self._folder_path if folder_path is None else folder_path
        slurm = self._slurm if slurm is None else slurm
        slurm_email = self._slurm_email if slurm_email is None else slurm_email
        slurm_path = folder_path + '/' + "slurm-%j" if slurm_path is None else slurm_path

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
            patient_path = p

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
                    "output": slurm_path + ".out",
                    "error": slurm_path + ".err",
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
