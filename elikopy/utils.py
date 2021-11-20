import datetime
import time
import os
import json
import shutil
import matplotlib.pyplot

from future.utils import iteritems
import subprocess


def submit_job(job_info):
    """
    Submit a job to the Slurm Workload Manager using a crafted sbatch.

    :param job_info: The parameters to use in the sbatch.
    :return job_id: The id of the submited job.
    """
    # Construct sbatch command
    slurm_cmd = ["sbatch"]
    script=False
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
            script=True
            continue
        slurm_cmd.append("--%s=%s" % (key, value))
    if script:
        slurm_cmd.append(job_info["script"])
    print("[INFO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Generated slurm batch command: '%s'" % slurm_cmd)

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
            #break


def anonymise_nifti(rootdir,anonymize_json,rename):
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
                print ('json: ', os.path.join(subdir, file))
                f = open(os.path.join(subdir, file),'r+')
                data = json.load(f)

                print(data.get('PatientID'))
                print()
                name_key.update({os.path.splitext(file)[0]:data.get('PatientID')})

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
                    ID = name_key.get(os.path.splitext(os.path.splitext(file)[0])[0])
                else:
                    ID = name_key.get(os.path.splitext(file)[0])

                if ID:
                    new_file = ID + "_DTI" + ext
                    new_path = os.path.join(subdir, new_file)
                    old_path = os.path.join(subdir, file)
                    print("New path: " + new_path)
                    print("Old path: "  + old_path)
                    if rename:
                        os.rename(old_path,new_path)
                else:
                    print("ID is none " + file)
                    print(name_key)

                print()


def getJobsState(folder_path,job_list,step_name):
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

    export_path = folder_path + "/export_" + step.rsplit('/',1)[1]
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
        copy_path = folder_path + '/subjects/' + os.path.splitext(p)[0] + '/' + step
        shutil.copytree(copy_path, export_path, dirs_exist_ok=True)

    shutil.copyfile(folder_path + "/subjects/subj_list.json", export_path + "/subj_list.json")
    shutil.copyfile(folder_path + "/subjects/subj_error.json", export_path + "/subj_error.json")
    shutil.copyfile(folder_path + "/subjects/subj_type.json", export_path + "/subj_type.json")


def get_job_state(job_id):
    """
    Retrieve the state of a job through the sacct bash command offered by the lurm Workload Manager.
    :param job_id: The id of the job to retrieve the state of.
    :return state: The string value representing the state of the job.
    """
    cmd = "sacct --jobs=" + str(job_id) + " -n -o state"

    proc = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def makedir(dir_path,log_path,log_prefix):
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
            print ("Creation of the directory %s failed" % dir_path)
            f=open(log_path, "a+")
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % dir_path)
            f.close()
        else:
            print ("Successfully created the directory %s " % dir_path)
            f=open(log_path, "a+")
            f.write("["+log_prefix+"] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % dir_path)
            f.close()


def tbss_utils(folder_path, grp1, grp2, starting_state=None, last_state=None, registration_type="-T", postreg_type="-S", prestats_treshold=0.2, randomise_numberofpermutation=5000):
    """
    [Legacy] Performs tract base spatial statistics (TBSS) between the data in grp1 and grp2. The data type of each subject is specified by the subj_type.json file generated during the call to the patient_list function. The data type corresponds to the original directory of the subject (e.g. a subject that was originally in the folder data_2 is of type 2).
    It is mandatory to have performed DTI prior to tbss.

    :param folder_path: path to the root directory.
    :param grp1: List of number corresponding to the type of the subjects to put in the first group.
    :param grp2: List of number corresponding to the type of the subjects to put in the second group.
    :param starting_state: Manually set which step of TBSS to execute first. Could either be None, reg, post_reg, prestats, design or randomise. default=None
    :param last_state: Manually set which step of TBSS to execute last. Could either be None, preproc, reg, post_reg, prestats, design or randomise. default=None
    :param registration_type: Define the argument used by the tbss command tbss_2_reg. Could either by '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is align to every other one, identify the "most representative" one, and use this as the target image.
    :param postreg_type: Define the argument used by the tbss command tbss_3_postreg. Could either by '-S' or '-T'. If you wish to use the FMRIB58_FA mean FA image and its derived skeleton, instead of the mean of your subjects in the study, use the '-T' option. Otherwise, use the '-S' option.
    :param prestats_treshold: Thresholds the mean FA skeleton image at the chosen threshold during prestats. default=0.2
    :param randomise_numberofpermutation: Define the number of permutations. default=5000
    """
    starting_state = None if starting_state == "None" else starting_state
    last_state = None if last_state == "None" else last_state
    assert starting_state in (None, "reg", "postreg", "prestats", "design", "randomise"), 'invalid starting state!'
    assert last_state in (None, "preproc", "reg", "postreg", "prestats", "design", "randomise"), 'invalid last state!'
    assert registration_type in ("-T", "-t", "-n"), 'invalid registration type!'
    assert postreg_type in ("-S", "-T"), 'invalid postreg type!'

    print("You are using the legacy TBSS function. We recommend you to use regall_FA() function.")

    # create the output directory
    log_prefix = "TBSS"
    outputdir = folder_path + "/TBSS"
    makedir(outputdir, folder_path + "/logs.txt", log_prefix)

    import subprocess
    tbss_log = open(folder_path + "/TBSS/TBSS_logs.txt", "a+")

    from distutils.dir_util import copy_tree

    # open the subject and is_control lists
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)
    dest_subj_type = folder_path + "/subjects/subj_type.json"
    with open(dest_subj_type, 'r') as f:
        subj_type = json.load(f)
    numpatient = 0
    numcontrol = 0
    for p in patient_list:
        patient_path = os.path.splitext(p)[0]
        control_info = subj_type[patient_path]
        if control_info in grp1:
            numcontrol += 1
        if control_info in grp2:
            numpatient += 1

    if starting_state == None:

        makedir(folder_path + "/TBSS/FA", folder_path + "/logs.txt", log_prefix)
        makedir(folder_path + "/TBSS/origdata", folder_path + "/logs.txt", log_prefix)

        # transfer the FA files to the TBSS directory
        numpatient = 0
        numcontrol = 0
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of preproc\n")
        tbss_log.flush()
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]
            control_info = subj_type[patient_path]
            if control_info in grp1:
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
                    outputdir + "/origdata/control" + str(numcontrol) + "_" + patient_path + "_FA.nii.gz")
                pref = "control" + str(numcontrol) + "_"
                numcontrol += 1
            if control_info in grp2:
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
                    outputdir + "/origdata/case" + str(numpatient) + "_" + patient_path + "_FA.nii.gz")
                pref = "case" + str(numpatient) + "_"
                numpatient += 1

            x_val = int(subprocess.check_output("cd " + outputdir + "; fslval origdata/" + pref + patient_path + "_FA dim1", shell=True));
            x = x_val - 2
            y_val = int(subprocess.check_output("cd " + outputdir + "; fslval origdata/" + pref + patient_path + "_FA dim2", shell=True));
            y = y_val - 2
            z_val = int(subprocess.check_output("cd " + outputdir + "; fslval origdata/" + pref + patient_path + "_FA dim3", shell=True));
            z = z_val - 2

            #bashCommand = 'cd ' + outputdir + ' && tbss_1_preproc \"*_fa.nii.gz\"'
            cmd1="fslmaths origdata/" + pref + patient_path + "_FA -min 1 -dilD -ero -ero -roi 1 "+str(x)+" 1 "+str(y)+" 1 "+str(z)+" 0 1 FA/" + pref + patient_path + "_FA"

            # create mask (for use in FLIRT & FNIRT)
            cmd2="fslmaths FA/" + pref + patient_path + "_FA -bin FA/" + pref + patient_path + "_FA_mask"
            cmd3="fslmaths FA/" + pref + patient_path + "_FA_mask -dilD -dilD -sub 1 -abs -add FA/" + pref + patient_path + "_FA_mask FA/" + pref + patient_path + "_FA_mask -odt char"
            #bashCommand = 'cd ' + outputdir + ' && tbss_1_preproc \"*_fa.nii.gz\"'
            bashCommand = 'cd ' + outputdir + '; ' + cmd1 + '; ' + cmd2 + '; ' + cmd3
            bashcmd = bashCommand.split()
            print("Bash command is:\n{}\n".format(bashcmd))
            tbss_log.write(bashCommand+"\n")
            tbss_log.flush()
            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
            output, error = process.communicate()

        #bashCommand = 'cd ' + outputdir + '/FA; ' + "slicesdir `imglob *_FA.*`" + '; ' + cmd2 + '; ' + cmd3
        #bashcmd = bashCommand.split()
        #print("Bash command is:\n{}\n".format(bashcmd))
        #process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,
        #                           stderr=subprocess.STDOUT)
        #output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of preproc\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        copy_tree(folder_path + "/TBSS/origdata", folder_path + "/TBSS/backup/tbss_preproc/origdata")
        copy_tree(folder_path + "/TBSS/FA", folder_path + "/TBSS/backup/tbss_preproc/FA")

        if last_state=="preproc":
            tbss_log.close()
            return

    if starting_state in (None, "reg"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of reg\n")

        if starting_state == "reg":
            copy_tree(folder_path + "/TBSS/backup/tbss_preproc/origdata", folder_path + "/TBSS/origdata")
            copy_tree(folder_path + "/TBSS/backup/tbss_preproc/FA", folder_path + "/TBSS/FA")

        bashCommand = 'cd ' + outputdir + ' && tbss_2_reg '+ registration_type
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        tbss_log.write(bashCommand+"\n")
        tbss_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of reg\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/FA", folder_path + "/TBSS/backup/tbss_reg/FA")


        if last_state=="reg":
            tbss_log.close()
            return

    if starting_state in (None, "reg", "postreg"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of postreg\n")
        tbss_log.flush()

        if starting_state == "postreg":
            copy_tree(folder_path + "/TBSS/backup/tbss_preproc/origdata", folder_path + "/TBSS/origdata")
            copy_tree(folder_path + "/TBSS/backup/tbss_reg/FA", folder_path + "/TBSS/FA")

        bashCommand = 'cd ' + outputdir + ' && tbss_3_postreg ' + postreg_type
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        tbss_log.write(bashCommand+"\n")
        tbss_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of postreg\n")
        tbss_log.flush()

        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/backup/tbss_postreg/stats")

        if last_state=="postreg":
            tbss_log.close()
            return

    if starting_state in (None, "reg", "postreg", "prestats"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of prestats\n")
        tbss_log.flush()

        if starting_state == "prestats":
            copy_tree(folder_path + "/TBSS/backup/tbss_preproc/origdata", folder_path + "/TBSS/origdata")
            copy_tree(folder_path + "/TBSS/backup/tbss_reg/FA", folder_path + "/TBSS/FA")
            copy_tree(folder_path + "/TBSS/backup/tbss_postreg/stats", folder_path + "/TBSS/stats")

        bashCommand = 'cd ' + outputdir + ' && tbss_4_prestats ' + str(prestats_treshold)
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        tbss_log.write(bashCommand+"\n")
        tbss_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,
                                   stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of prestats\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/backup/tbss_prestats/stats")

        if last_state == "prestats":
            tbss_log.close()
            return

    if starting_state in (None, "reg", "postreg", "prestats", "design"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of design\n")
        tbss_log.flush()

        if starting_state == "design":
            copy_tree(folder_path + "/TBSS/backup/tbss_prestats/stats", folder_path + "/TBSS/stats")

        #bashCommand = 'cd ' + outputdir + '/stats ' + ' && design_ttest2 design ' + str(numcontrol) + ' ' + str(numpatient)
        bashCommand = 'cd ' + outputdir + '/stats ' + ' && design_ttest2 design ' + str(numpatient) + ' ' + str(numcontrol)
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        tbss_log.write(bashCommand+"\n")
        tbss_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of design\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/backup/design_ttest2/stats")

        if last_state=="design":
            tbss_log.close()
            return

    if starting_state == "randomise":
        copy_tree(folder_path + "/TBSS/backup/design_ttest2/stats", folder_path + "/TBSS/stats")

    bashCommand1 = 'cd ' + outputdir + '/stats ' + ' && randomise -i all_FA_skeletonised -o tbss -m mean_FA_skeleton_mask -d design.mat -t design.con -n ' + str(randomise_numberofpermutation) + ' --T2 --uncorrp'
    bashCommand2 = 'cd ' + outputdir + '/stats ' + ' && autoaq -i tbss_tfce_corrp_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report1_corrected_subcortical.txt && autoaq -i tbss_tfce_corrp_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report2_corrected_subcortical.txt && autoaq -i tbss_tfce_corrp_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report1_corrected_cortical.txt && autoaq -i tbss_tfce_corrp_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report2_corrected_cortical.txt'
    bashCommand3 = 'cd ' + outputdir + '/stats ' + ' && autoaq -i tbss_tfce_p_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report1_uncorrected_subcortical.txt && autoaq -i tbss_tfce_p_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report2_uncorrected_subcortical.txt && autoaq -i tbss_tfce_p_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report1_uncorrected_cortical.txt && autoaq -i tbss_tfce_p_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report2_uncorrected_cortical.txt'

    if starting_state in (None, "reg", "postreg", "prestats", "design", "randomise"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of randomise\n")
        tbss_log.flush()



        bashcmd1 = bashCommand1.split()
        print("Bash command is:\n{}\n".format(bashcmd1))
        tbss_log.write(bashCommand1+"\n")
        tbss_log.flush()
        process = subprocess.Popen(bashCommand1, universal_newlines=True, shell=True, stdout=tbss_log,
                                   stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of randomise\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/backup/randomise/stats")

        if last_state=="randomise":
            tbss_log.close()
            return

    tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of autoaq\n")
    tbss_log.flush()

    bashcmd2 = bashCommand2.split()
    print("Bash command is:\n{}\n".format(bashcmd2))
    tbss_log.write(bashCommand2+"\n")
    tbss_log.flush()
    process = subprocess.Popen(bashCommand2, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
    output, error = process.communicate()

    bashcmd3 = bashCommand3.split()
    print("Bash command is:\n{}\n".format(bashcmd3))
    tbss_log.write(bashCommand3 + "\n")
    tbss_log.flush()
    process = subprocess.Popen(bashCommand3, universal_newlines=True, shell=True, stdout=tbss_log,
                               stderr=subprocess.STDOUT)
    output, error = process.communicate()

    tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": End of autoaq\n")
    tbss_log.flush()

    tbss_log.close()


def synb0DisCo(folder_path,topuppath,patient_path,starting_step=None,topup=True,gpu=True):
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

    if starting_step == "None":
        starting_step = None
         
    assert starting_step in (None, "Registration", "Inference", "Apply", "topup")

    synb0path = topuppath + "/synb0-DisCo"

    if starting_step == None:
        """
        Step 1 - Normalize T1
        """

        mri_convert_T1 = "mri_convert " + synb0path + "/T1.nii.gz " + synb0path + "/T1.mgz"

        n3_correction = "mri_nu_correct.mni --i " + synb0path + "/T1.mgz --o " + synb0path + "/T1_N3.mgz --n 2"

        mri_convert_N3 = "mri_convert " + synb0path + "/T1_N3.mgz " + synb0path + "/T1_N3.nii.gz"

        mri_normalize = "mri_normalize -g 1 -mprage " + synb0path + "/T1_N3.mgz " + synb0path + "/T1_norm.mgz"

        mri_convert_norm = "mri_convert " + synb0path + "/T1_norm.mgz " + synb0path + "/T1_norm.nii.gz"

        bashCommand_step1 = mri_convert_T1 + "; " + n3_correction + "; " + mri_convert_N3 + "; " + mri_normalize + "; " + mri_convert_norm
        step1_log = open(synb0path + "/step1_logs.txt", "a+")
        step1_log.write("[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of step 1 \n\n" )
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

        bet = "bet " + synb0path + "/T1.nii.gz " + synb0path + "/T1_mask.nii.gz -R"# + " -f 0.4 -g -0.2"

        # epi_reg distorted b0 to T1; wont be perfect since B0 is distorted

        epi_reg_b0_dist = "epi_reg --epi=" + synb0path + "/b0.nii.gz  --t1=" + synb0path + "/T1.nii.gz --t1brain=" + synb0path + "/T1_mask.nii.gz --out=" + synb0path + "/epi_reg_d"

        # Convert FSL transform to ANTS transform
        c3d_affine_tool = "c3d_affine_tool -ref " + synb0path + "/T1.nii.gz -src " + synb0path + "/b0.nii.gz " + synb0path + "/epi_reg_d.mat -fsl2ras -oitk " + synb0path + "/epi_reg_d_ANTS.txt"

        # ANTs register T1 to atla
        antsRegistrationSyNQuick = "antsRegistrationSyNQuick.sh -d 3 -f " + folder_path + "/static_files/atlases/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz -m " + synb0path + "/T1.nii.gz -o " + synb0path + "/ANTS"

        # Apply linear transform to normalized T1 to get it into atlas space
        antsApplyTransforms_lin_T1 = "antsApplyTransforms -d 3 -i " + synb0path + "/T1_norm.nii.gz -r " + folder_path + "/static_files/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + synb0path + "/ANTS0GenericAffine.mat -o " + synb0path + "/T1_norm_lin_atlas_2_5.nii.gz"

        # Apply linear transform to distorted b0 to get it into atlas space
        antsApplyTransforms_lin_b0 = "antsApplyTransforms -d 3 -i " + synb0path + "/b0.nii.gz -r " + folder_path + "/static_files/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + synb0path + "/ANTS0GenericAffine.mat -t " + synb0path + "/epi_reg_d_ANTS.txt -o " + synb0path + "/b0_d_lin_atlas_2_5.nii.gz"

        # Apply nonlinear transform to normalized T1 to get it into atlas space
        antsApplyTransforms_nonlin_T1 = "antsApplyTransforms -d 3 -i " + synb0path + "/T1_norm.nii.gz -r " + folder_path + "/static_files/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + synb0path + "/ANTS1Warp.nii.gz -t " + synb0path + "/ANTS0GenericAffine.mat -o " + synb0path + "/T1_norm_nonlin_atlas_2_5.nii.gz"

        # Apply nonlinear transform to distorted b0 to get it into atlas space
        antsApplyTransforms_nonlin_b0 = "antsApplyTransforms -d 3 -i " + synb0path + "/b0.nii.gz -r " + folder_path + "/static_files/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz -n BSpline -t " + synb0path + "/ANTS1Warp.nii.gz -t " + synb0path + "/ANTS0GenericAffine.mat -t " + synb0path + "/epi_reg_d_ANTS.txt -o " + synb0path + "/b0_d_nonlin_atlas_2_5.nii.gz"

        bashCommand_step2 = bet + "; " + epi_reg_b0_dist + "; " + c3d_affine_tool + "; " + antsRegistrationSyNQuick + "; " + antsApplyTransforms_lin_T1 + "; " + antsApplyTransforms_lin_b0 + "; " + antsApplyTransforms_nonlin_T1 + "; " + antsApplyTransforms_nonlin_b0
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

        for i in range(1,numfold+1):
            torch.cuda.empty_cache()
            b0_output_path = synb0path + "/b0_u_lin_atlas_2_5_FOLD_" + str(i) + ".nii.gz"
            model_path = folder_path + "/static_files/dual_channel_unet/num_fold_" + str(i) + "_total_folds_" + str(numfold) + "_seed_1_num_epochs_100_lr_0.0001_betas_(0.9, 0.999)_weight_decay_1e-05_num_epoch_*.pth"
            model_path = glob.glob(model_path)[0]
            # Get model
            model = UNet3D(2, 1).to(device)
            model.load_state_dict(torch.load(model_path,map_location=device))

            # Inference
            step3_log.write("[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of inference " + str(i) +"\n\n")
            step3_log.flush()
            img_model = inference(T1_input_path, b0_input_path, model, device)

            # Save
            nii_template = nib.load(b0_input_path)
            nii = nib.Nifti1Image(util.torch2nii(img_model.detach().cpu()), nii_template.affine, nii_template.header)
            nib.save(nii, b0_output_path)

        step3_log.write("[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of step 3 \n\n")
        step3_log.close()

    if starting_step in (None, "Registration", "Inference", "Apply"):
        """
        Step 4 -  Apply
        """

        # Take mean
        mean_merge ="fslmerge -t " + synb0path + "/b0_u_lin_atlas_2_5_merged.nii.gz " + synb0path + "/b0_u_lin_atlas_2_5_FOLD_*.nii.gz"
        mean_math = "fslmaths " + synb0path + "/b0_u_lin_atlas_2_5_merged.nii.gz -Tmean " + synb0path + "/b0_u_lin_atlas_2_5.nii.gz"

        # Apply inverse xform to undistorted b0
        antsApplyTransforms_inv_xform = "antsApplyTransforms -d 3 -i " + synb0path + "/b0_u_lin_atlas_2_5.nii.gz -r " + synb0path + "/b0.nii.gz -n BSpline -t [" + synb0path + "/epi_reg_d_ANTS.txt,1] -t [" + synb0path + "/ANTS0GenericAffine.mat,1] -o " + synb0path + "/b0_u.nii.gz"

        # Smooth image
        smooth_math = "fslmaths " + synb0path + "/b0.nii.gz -s 1.15 " + synb0path + "/b0_d_smooth.nii.gz"

        # Merge for topup
        merge_image = "fslmerge -t " + synb0path + "/b0_all.nii.gz " + synb0path + "/b0_d_smooth.nii.gz " + synb0path + "/b0_u.nii.gz"


        bashCommand_step4 = mean_merge + "; " + mean_math + "; " + antsApplyTransforms_inv_xform + "; " + smooth_math + "; " + merge_image
        step4_log = open(synb0path + "/step4_logs.txt", "a+")
        step4_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of step 4 \n\n")
        process = subprocess.Popen(bashCommand_step4, universal_newlines=True, shell=True, stdout=step4_log,
                                   stderr=subprocess.STDOUT)


        output, error = process.communicate()

        with open(synb0path + '/' + 'acqparams_topup.txt') as f:
            topup_acq = [[float(x) for x in line2.split()] for line2 in f]

        topup_acq.append([topup_acq[0][0], - topup_acq[0][1], topup_acq[0][2], 0])

        print(topup_acq)

        with open(synb0path + '/' + "acqparams_topup.txt", 'w') as file:
            file.writelines(' '.join(str(j) for j in i) + '\n' for i in topup_acq)

        step4_log.write(
            "[SynB0DISCO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of step 4 \n\n")
        step4_log.close()

    if starting_step in (None, "Registration", "Inference", "Apply", "topup") and topup:
        run_topup = "topup -v --imain=" + synb0path + "/b0_all.nii.gz --datain=" + synb0path + "/acqparams_topup.txt --config=b02b0.cnf --iout=" + topuppath + "/" + patient_path + "_topup_iout_estimate --out=" + topuppath + "/" + patient_path + "_topup_estimate --subsamp=1,1,1,1,1,1,1,1,1 --miter=10,10,10,10,10,20,20,30,30 --lambda=0.00033,0.000067,0.0000067,0.000001,0.00000033,0.000000033,0.0000000033,0.000000000033,0.00000000000067 --scale=0 "+ '--fout="' + topuppath + '/' + patient_path + '_topup_fout_estimate" '
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


import os
import sys
import glob
import math

from random import shuffle

import numpy as np


#from torchvision import datasets, transforms

import nibabel as nib


import elikopy.utilsSynb0Disco as util


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
    img_model = util.unnormalize_img(img_model, max_img_b0_d, min_img_b0_d, 1, -1)

    # Remove padding
    img_model = img_model[:, :, 2:-1, 2:-1, 3:-2]

    # Return model
    return img_model


def regall_FA(folder_path, grp1, grp2, starting_state=None, registration_type="-T", postreg_type="-S", prestats_treshold=0.2, core_count=1):
    """ Register all the subjects Fractional Anisotropy into a common space, skeletonisedd and non skeletonised. This is performed based on TBSS of FSL.
    It is mandatory to have performed DTI prior to regall_FA.

    :param folder_path: path to the root directory.
    :param grp1: List of number corresponding to the type of the subjects to put in the first group.
    :param grp2: List of number corresponding to the type of the subjects to put in the second group.
    :param starting_state: Manually set which step of TBSS to execute first. Could either be None, reg, post_reg, prestats, design or randomise. default=None
    :param registration_type: Define the argument used by the tbss command tbss_2_reg. Could either by '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is align to every other one, identify the "most representative" one, and use this as the target image.
    :param postreg_type: Define the argument used by the tbss command tbss_3_postreg. Could either by '-S' or '-T'. If you wish to use the FMRIB58_FA mean FA image and its derived skeleton, instead of the mean of your subjects in the study, use the '-T' option. Otherwise, use the '-S' option.
    :param prestats_treshold: Thresholds the mean FA skeleton image at the chosen threshold during prestats. default=0.2
    :param core_count: Define the number of available core. default=1
    """
    starting_state = None if starting_state == "None" else starting_state
    assert starting_state in (None, "reg", "postreg", "prestats"), 'invalid starting state!'
    assert registration_type in ("-T", "-t", "-n"), 'invalid registration type!'
    assert postreg_type in ("-S", "-T"), 'invalid postreg type!'

    # create the output directory
    log_prefix = "registration"
    outputdir = folder_path + "/registration"
    makedir(outputdir, folder_path + "/logs.txt", log_prefix)

    import subprocess
    registration_log = open(folder_path + "/registration/registration_logs.txt", "a+")

    from distutils.dir_util import copy_tree
    import json

    # open the subject and is_control lists
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)
    dest_subj_type = folder_path + "/subjects/subj_type.json"
    with open(dest_subj_type, 'r') as f:
        subj_type = json.load(f)

    if starting_state == None:

        makedir(folder_path + "/registration/FA", folder_path + "/logs.txt", log_prefix)
        makedir(folder_path + "/registration/origdata", folder_path + "/logs.txt", log_prefix)

        # transfer the FA files to the TBSS directory
        numpatient = 0
        numcontrol = 0
        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of registration\n")
        registration_log.flush()
        type_dict = {}
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]
            control_info = subj_type[patient_path]

            if control_info not in grp1 and control_info not in grp2:
                continue

            if control_info in grp1:
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
                    outputdir + "/origdata/control" + str(numcontrol) + "_" + patient_path + "_FA.nii.gz")
                pref = "control" + str(numcontrol) + "_"
                numcontrol += 1
                type_dict[pref] = patient_path
            if control_info in grp2:
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
                    outputdir + "/origdata/case" + str(numpatient) + "_" + patient_path + "_FA.nii.gz")
                pref = "case" + str(numpatient) + "_"
                numpatient += 1
                type_dict[pref] = patient_path

            x_val = int(subprocess.check_output("cd " + outputdir + "; fslval origdata/" + pref + patient_path + "_FA dim1", shell=True));
            x = x_val - 2
            y_val = int(subprocess.check_output("cd " + outputdir + "; fslval origdata/" + pref + patient_path + "_FA dim2", shell=True));
            y = y_val - 2
            z_val = int(subprocess.check_output("cd " + outputdir + "; fslval origdata/" + pref + patient_path + "_FA dim3", shell=True));
            z = z_val - 2

            cmd1="fslmaths origdata/" + pref + patient_path + "_FA -min 1 -dilD -ero -ero -roi 1 "+str(x)+" 1 "+str(y)+" 1 "+str(z)+" 0 1 FA/" + pref + patient_path + "_FA"

            # create mask (for use in FLIRT & FNIRT)
            cmd2="fslmaths FA/" + pref + patient_path + "_FA -bin FA/" + pref + patient_path + "_FA_mask"
            cmd3="fslmaths FA/" + pref + patient_path + "_FA_mask -dilD -dilD -sub 1 -abs -add FA/" + pref + patient_path + "_FA_mask FA/" + pref + patient_path + "_FA_mask -odt char"
            bashCommand = 'cd ' + outputdir + '; ' + cmd1 + '; ' + cmd2 + '; ' + cmd3
            bashcmd = bashCommand.split()
            print("Bash command is:\n{}\n".format(bashcmd))
            registration_log.write(bashCommand+"\n")
            registration_log.flush()
            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=registration_log,stderr=subprocess.STDOUT)
            output, error = process.communicate()

        with open(folder_path + "/registration/"+"tbss_id.json", "w") as write_file:
            json.dump(type_dict, write_file)

        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of step 1\n")
        registration_log.flush()


    if starting_state in (None, "reg"):
        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of reg\n")

        bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + ' && tbss_2_reg '+ registration_type
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        registration_log.write(bashCommand+"\n")
        registration_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=registration_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of reg\n")
        registration_log.flush()

    if starting_state in (None, "reg", "postreg"):
        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of postreg\n")
        registration_log.flush()

        bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + ' && tbss_3_postreg ' + postreg_type
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        registration_log.write(bashCommand+"\n")
        registration_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=registration_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of postreg\n")
        registration_log.flush()

    if starting_state in (None, "reg", "postreg", "prestats"):
        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of prestats\n")
        registration_log.flush()

        bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + ' && tbss_4_prestats ' + str(prestats_treshold) + '&& cd ' + outputdir + '/stats ' + ' && design_ttest2 design ' + str(numpatient) + ' ' + str(numcontrol)
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        registration_log.write(bashCommand+"\n")
        registration_log.flush()
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=registration_log,
                                   stderr=subprocess.STDOUT)
        output, error = process.communicate()

        registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of prestats\n")
        registration_log.flush()

    registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": End of FA Registration \n")
    registration_log.flush()


def regall(folder_path, grp1, grp2, core_count=1 ,metrics_dic={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}):
    """ Register all the subjects diffusion metrics specified in the argument metrics_dic into a common space using the transformation computed for the FA with the regall_FA function. This is performed based on TBSS of FSL.
    It is mandatory to have performed regall_FA prior to regall.

    :param folder_path: path to the root directory.
    :param grp1: List of number corresponding to the type of the subjects to put in the first group.
    :param grp2: List of number corresponding to the type of the subjects to put in the second group.
    :param metrics_dic: Dictionnary containing the diffusion metrics to register in a common space. For each diffusion metric, the metric name is the key and the metric's folder is the value. default={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
    :param core_count: Define the number of available core. default=1
    """

    assert os.path.isdir(folder_path + "/registration/FA"),"No FA registration found! You first need to run regall_FA() before using this function!"

    # create the output directory
    log_prefix = "registration"
    outputdir = folder_path + "/registration"
    makedir(outputdir, folder_path + "/logs.txt", log_prefix)

    import subprocess
    registration_log = open(folder_path + "/registration/registration_logs.txt", "a+")

    registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of microstructural metrics registration\n")
    registration_log.flush()

    # open the subject and is_control lists
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)
    dest_subj_type = folder_path + "/subjects/subj_type.json"
    with open(dest_subj_type, 'r') as f:
        subj_type = json.load(f)

    with open(folder_path + "/registration/"+"tbss_id.json") as json_file:
        tbss_id = json.load(json_file)

    for key, value in metrics_dic.items():
        metric_bool = True
        type_dict = {}
        numpatient = 0
        numcontrol = 0
        makedir(folder_path + "/registration/" + key, folder_path + "/logs.txt", log_prefix)
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]
            control_info = subj_type[patient_path]
            if control_info in grp1:
                metric_path = folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/' + value + '/' + patient_path + key +".nii.gz"
                pref = "control" + str(numcontrol) + "_"
                numcontrol += 1
                if os.path.isfile(metric_path):
                    shutil.copyfile(
                        folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/' + value + '/' + patient_path + key + ".nii.gz",
                        outputdir + "/" + key + "/" + pref + patient_path + ".nii.gz")
                    type_dict[pref] = patient_path
            if control_info in grp2:
                metric_path = folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/' + value + '/' + patient_path + key +".nii.gz"
                pref = "case" + str(numpatient) + "_"
                numpatient += 1
                if os.path.isfile(metric_path):
                    shutil.copyfile(
                        folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/' + value + '/' + patient_path + key + ".nii.gz",
                        outputdir + "/" + key + "/" + pref + patient_path + ".nii.gz")
                    type_dict[pref] = patient_path

        # Verify that the matching between patient and FA is equivalent to the matching between patient and the computed metric
        if type_dict != tbss_id:
            metric_bool = False
            registration_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Unable to process the metric "+ key + " : " +value +"! Some subject have a FA but not the metric!\n")
            registration_log.write("Subject with FA: \n" +json.dumps(tbss_id, sort_keys=True, indent=4))
            registration_log.write("Subject with the metric: \n" + json.dumps(type_dict, sort_keys=True, indent=4))

            shutil.rmtree(outputdir + "/" + key + "/",ignore_errors=True)

        if metric_bool:
            bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + ' && tbss_non_FA ' + key
            bashcmd = bashCommand.split()
            print("Bash command is:\n{}\n".format(bashcmd))
            registration_log.write(bashCommand + "\n")
            registration_log.flush()
            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=registration_log, stderr=subprocess.STDOUT)
            output, error = process.communicate()

            bashCommand = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + ' && fslmaths stats/all_' + key + ' -Tmean stats/mean_' + key
            bashcmd = bashCommand.split()
            print("Bash command is:\n{}\n".format(bashcmd))
            registration_log.write(bashCommand + "\n")
            registration_log.flush()
            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=registration_log, stderr=subprocess.STDOUT)
            output, error = process.communicate()

    registration_log.close()


def randomise_all(folder_path,randomise_numberofpermutation=5000,skeletonised=True,metrics_dic={'FA':'dti','_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'},core_count=1, regionWiseMean=True, additional_atlases=None):
    """ Performs tract base spatial statistics (TBSS) between the data in grp1 and grp2 (groups are specified during the call to regall_FA) for each diffusion metric specified in the argument metrics_dic.
    The mean value of the diffusion metrics across atlases regions can also be reported in CSV files using the regionWiseMean flag. The used atlases are : the Harvard-Oxford cortical and subcortical structural atlases, the JHU DTI-based white-matter atlases and the MNI structural atlas
    It is mandatory to have performed regall_FA prior to randomise_all.

    :param folder_path: path to the root directory.
    :param randomise_numberofpermutation: Define the number of permutations. default=5000
    :param skeletonised: If True, randomize will be using only the white matter skeleton instead of the whole brain. default=True
    :param metrics_dic: Dictionnary containing the diffusion metrics to register in a common space. For each diffusion metric, the metric name is the key and the metric's folder is the value. default={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
    :param regionWiseMean: If true, csv containing atlas-based region wise mean will be generated.
    :param core_count: Number of allocated cpu core. default=1
    :param additional_atlases:  Define additional atlases to be used as segmentation template for csv generation (see regionWiseMean). Dictionary is in the form {'Atlas_name_1':["path to atlas 1 xml","path to atlas 1 nifti"],'Atlas_name_1':["path to atlas 2 xml","path to atlas 2 nifti"]}.
    """
    outputdir = folder_path + "/registration"
    log_prefix = "randomise"

    assert os.path.isdir(folder_path + "/registration/stats"), "You first need to run regall_FA() before using randomise!"

    randomise_log = open(folder_path + "/registration/stats/randomise_log.txt", "a+")

    if core_count==1:
        randomise_type= "randomise"
    else:
        randomise_type = "randomise_parallel"

    for key, value in metrics_dic.items():

        if skeletonised:
            outkey = key + '_skeletonised'
            bashCommand1 = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + '/stats ' + ' && ' + randomise_type + ' -i all_' + key + '_skeletonised -o '+ outkey +' -m mean_FA_skeleton_mask -d design.mat -t design.con -n ' + str(randomise_numberofpermutation) + ' --T2 --uncorrp'
        else:
            outkey = key
            bashCommand1 = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + '/stats ' + ' && ' + randomise_type + ' -i all_' + key + ' -o ' + key + ' -m mean_FA_mask -d design.mat -t design.con -n ' + str(randomise_numberofpermutation) + ' --T2 --uncorrp'

        bashCommand2 = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + '/stats ' + ' && autoaq -i ' + outkey + '_tfce_corrp_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report1_corrected_subcortical.txt && autoaq -i ' + outkey + '_tfce_corrp_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report2_corrected_subcortical.txt && autoaq -i ' + outkey + '_tfce_corrp_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report1_corrected_cortical.txt && autoaq -i ' + outkey + '_tfce_corrp_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report2_corrected_cortical.txt'
        bashCommand3 = 'export OMP_NUM_THREADS='+str(core_count)+' ; export FSLPARALLEL='+str(core_count)+' ; cd ' + outputdir + '/stats ' + ' && autoaq -i ' + outkey + '_tfce_p_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report1_uncorrected_subcortical.txt && autoaq -i ' + outkey + '_tfce_p_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report2_uncorrected_subcortical.txt && autoaq -i ' + outkey + '_tfce_p_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report1_uncorrected_cortical.txt && autoaq -i ' + outkey + '_tfce_p_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o ' + outkey + '_report2_uncorrected_cortical.txt'


        if randomise_numberofpermutation > 0:
            randomise_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Beginning of randomise\n")
            randomise_log.flush()



            bashcmd1 = bashCommand1.split()
            print("Bash command is:\n{}\n".format(bashcmd1))
            randomise_log.write(bashCommand1+"\n")
            randomise_log.flush()
            process = subprocess.Popen(bashCommand1, universal_newlines=True, shell=True, stdout=randomise_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()

            randomise_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of randomise\n")
            randomise_log.flush()


            randomise_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Beginning of autoaq\n")
            randomise_log.flush()

            bashcmd2 = bashCommand2.split()
            print("Bash command is:\n{}\n".format(bashcmd2))
            randomise_log.write(bashCommand2+"\n")
            randomise_log.flush()
            process = subprocess.Popen(bashCommand2, universal_newlines=True, shell=True, stdout=randomise_log,stderr=subprocess.STDOUT)
            output, error = process.communicate()

            bashcmd3 = bashCommand3.split()
            print("Bash command is:\n{}\n".format(bashcmd3))
            randomise_log.write(bashCommand3 + "\n")
            randomise_log.flush()
            process = subprocess.Popen(bashCommand3, universal_newlines=True, shell=True, stdout=randomise_log,
                                       stderr=subprocess.STDOUT)
            output, error = process.communicate()

            randomise_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": End of autoaq\n")
            randomise_log.flush()

        if regionWiseMean:
            randomise_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Starting region based analysis\n")
            randomise_log.flush()

            import pandas as pd
            import lxml.etree as etree
            from dipy.io.image import load_nifti

            # path to the atlas directory of FSL
            fsldir = os.getenv('FSLDIR')
            atlas_path = fsldir + "/data/atlases"

            # list of directory and their labels
            xmlName = [atlas_path + "/MNI.xml", atlas_path + "/HarvardOxford-Cortical.xml", atlas_path + "/HarvardOxford-Subcortical.xml", atlas_path + "/JHU-tracts.xml"]
            atlases = [atlas_path + "/MNI/MNI-prob-1mm.nii.gz", atlas_path + "/HarvardOxford/HarvardOxford-cort-prob-1mm.nii.gz", atlas_path + "/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz", atlas_path + "/JHU/JHU-ICBM-tracts-prob-1mm.nii.gz"]
            name = ["MNI", "HarvardCortical", "HarvardSubcortical", "JHUWhiteMatterTractography"]
            if additional_atlases:
                xmlName = xmlName + list(map(list, zip(*list(additional_atlases.values()))))[0]
                atlases = atlases + list(map(list, zip(*list(additional_atlases.values()))))[1]
                name = name + list(additional_atlases.keys())
            # open the data
            data, data_affine = load_nifti(outputdir + '/stats/all_' + key + '.nii.gz')

            for iteration in range(len(atlases)):

                # read the labels in xml file
                x = etree.parse(xmlName[iteration])
                labels = []
                for elem in x.iter():
                    if elem.tag == 'label':
                        labels.append([elem.attrib['index'], elem.text])
                labels = np.array(labels)[:, 1]

                # open the atlas
                atlas, atlas_affine = load_nifti(atlases[iteration])

                matrix = np.zeros((np.shape(data)[-1], np.shape(atlas)[-1]))
                for i in range(np.shape(atlas)[-1]):
                    for j in range(np.shape(data)[-1]):
                        patient = data[..., j]
                        region = atlas[..., i]
                        mean_fa = np.sum(patient * region) / np.sum(region)
                        matrix[j, i] = mean_fa
                df = pd.DataFrame(matrix, columns=labels)
                df.to_csv(outputdir + '/stats/regionWise_' + name[iteration] + key + '.csv')


    randomise_log.close()


def get_patient_list_by_types(folder_path,type=None):
    """Print the list of patient corresponding to a specfic type of patient.

    :param folder_path: Path to the root folder of the study.
    :param type: The selected type
    """

    import json, os

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
        print(patients_by_id.get(type,"Type not found!"))
    else:
        for key, value in patients_by_id.items():
            print("Patient list of cat " + str(key) + ": \n")
            print(value)
            print("\n")


def merge_all_reports(folder_path):
    """ Merge all subjects quality control reports into a single report.

    :param folder_path: Path to the root folder of the study.
    """
    from PyPDF2 import PdfFileWriter, PdfFileReader
    import json, os

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    writer = PdfFileWriter()

    for p in patient_list:
        patient_path = os.path.splitext(p)[0]
        pdf_path = folder_path + '/subjects/' + patient_path + '/quality_control.pdf'
        if (os.path.exists(pdf_path)):
            reader = PdfFileReader(pdf_path)
            for i in range(reader.numPages):
                page = reader.getPage(i)
                page.compressContentStreams()
                writer.addPage(page)

    with open(folder_path + '/quality_control_all_tmp.pdf', 'wb') as f:
        writer.write(f)

    #try to compress pdf with ghostscript
    bashCommand = "command -v gs && gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=" + folder_path + '/quality_control_all.pdf ' + folder_path + '/quality_control_all_tmp.pdf || mv ' + folder_path + '/quality_control_all_tmp.pdf '  + folder_path + '/quality_control_all.pdf'
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
    from PyPDF2 import PdfFileWriter, PdfFileReader
    import json, os

    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    if merge_wm_report:
        wm_writer = PdfFileWriter()

    if merge_legacy_report:
        legacy_writer = PdfFileWriter()

    for p in patient_list:
        patient_path = os.path.splitext(p)[0]

        if merge_wm_report:
            pdf_path = folder_path + '/subjects/' + patient_path + '/masks/quality_control/qc_report.pdf'
            if (os.path.exists(pdf_path)):
                reader = PdfFileReader(pdf_path)
                for i in range(reader.numPages):
                    page = reader.getPage(i)
                    page.compressContentStreams()
                    wm_writer.addPage(page)

        if merge_legacy_report:
            pdf_path = folder_path + '/subjects/' + patient_path + '/report/report_' + patient_path + '.pdf'
            if (os.path.exists(pdf_path)):
                reader = PdfFileReader(pdf_path)
                for i in range(reader.numPages):
                    page = reader.getPage(i)
                    page.compressContentStreams()
                    legacy_writer.addPage(page)

    if merge_wm_report:
        with open(folder_path + '/wm_mask_qc_report_all.pdf', 'wb') as f:
            wm_writer.write(f)

    if merge_legacy_report:
        with open(folder_path + '/legacy_report_all.pdf', 'wb') as f:
            legacy_writer.write(f)
