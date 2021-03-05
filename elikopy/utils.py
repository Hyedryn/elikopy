import datetime
import time
import os
import json
import shutil

from future.utils import iteritems
import subprocess


def submit_job(job_info):
    """

    :param job_info:
    :return:
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

    :param rootdir:
    :param anonymize_json:
    :param rename:
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

def export_files(folder_path, step):
    """
    Create an export folder in the root folder containing the results of step for each patient in a single folder

    :param folder_path: root folder
    :param step: step to export
    :return: nothing

    example : export_files('user/my_rootfolder', 'dMRI/microstructure/dti')
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

    for p in patient_list:
        copy_path = folder_path + '/subjects/' + os.path.splitext(p)[0] + '/' + step
        shutil.copytree(copy_path, export_path, dirs_exist_ok=True)

    shutil.copyfile(folder_path + "/subjects/subj_list.json", export_path + "/subj_list.json")
    shutil.copyfile(folder_path + "/subjects/subj_error.json", export_path + "/subj_error.json")
    shutil.copyfile(folder_path + "/subjects/is_control.json", export_path + "/is_control.json")


def get_job_state(job_id):
    """

    :param job_id:
    :return:
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



def tbss_utils(folder_path, grp1, grp2, starting_state=None, last_state=None, registration_type="-T", postreg_type="-S", prestats_treshold=0.2, randomise_corrected=False):
    """
    Perform tract base spatial statistics between the control data and case data. DTI needs to have been
    performed on the data first !!

    :return:
    :param folder_path: path to the root directory.
    :param grp1: List of number corresponding to the type of the patients to put in the first group.
    :param grp2: List of number corresponding to the type of the patients to put in the second group.
    :param starting_state: Manually set which step of TBSS to execute first. Could either be None, reg, post_reg, prestats, design or randomise.
    :param last_state: Manually set which step of TBSS to execute last. Could either be None, preproc, reg, post_reg, prestats, design or randomise.
    :param registration_type: Define the argument used by the tbss command tbss_2_reg. Could either by '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is align to every other one, identify the "most representative" one, and use this as the target image.
    :param postreg_type: Define the argument used by the tbss command tbss_3_postreg. Could either by '-S' or '-T'. If you wish to use the FMRIB58_FA mean FA image and its derived skeleton, instead of the mean of your subjects in the study, use the '-T' option. Otherwise, use the '-S' option.
    :param prestats_treshold: Thresholds the mean FA skeleton image at the chosen threshold during prestats.
    :param randomise_corrected: Define whether or not the p value must be FWE corrected.
    """
    starting_state = None if starting_state == "None" else starting_state
    last_state = None if last_state == "None" else last_state
    assert starting_state in (None, "reg", "postreg", "prestats", "design", "randomise"), 'invalid starting state!'
    assert last_state in (None, "preproc", "reg", "postreg", "prestats", "design", "randomise"), 'invalid last state!'
    assert registration_type in ("-T", "-t", "-n"), 'invalid registration type!'
    assert postreg_type in ("-S", "-T"), 'invalid postreg type!'

    # create the output directory
    log_prefix = "TBSS"
    outputdir = folder_path + "/TBSS"
    makedir(outputdir, folder_path + "/logs.txt", log_prefix)

    import subprocess
    tbss_log = open(folder_path + "/TBSS/TBSS_logs.txt", "a+")

    if starting_state == None:
        # open the subject and is_control lists
        dest_success = folder_path + "/subjects/subj_list.json"
        with open(dest_success, 'r') as f:
            patient_list = json.load(f)
        dest_subj_type = folder_path + "/subjects/subj_type.json"
        with open(dest_subj_type, 'r') as f:
            subj_type = json.load(f)

        # transfer the FA files to the TBSS directory
        numpatient = 0
        numcontrol = 0
        for p in patient_list:
            patient_path = os.path.splitext(p)[0]
            control_info = subj_type[patient_path]
            if control_info in grp1:
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
                    outputdir + "/control" + str(numcontrol) + "_" + patient_path + "_fa.nii.gz")
                numcontrol += 1
            if control_info in grp2:
                shutil.copyfile(
                    folder_path + '/subjects/' + patient_path + '/dMRI/microstructure/dti/' + patient_path + "_FA.nii.gz",
                    outputdir + "/case" + str(numpatient) + "_" + patient_path + "_fa.nii.gz")
                numpatient += 1

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of preproc\n")
        tbss_log.flush()

        bashCommand = 'cd ' + outputdir + ' && tbss_1_preproc \"*_fa.nii.gz\"'
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of preproc\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/origdata", folder_path + "/TBSS/tbss_preproc/origdata")
        copy_tree(folder_path + "/TBSS/FA", folder_path + "/TBSS/tbss_preproc/FA")

        if last_state=="preproc":
            tbss_log.close()
            return

    if starting_state in (None, "reg"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of reg\n")

        if starting_state == "reg":
            copy_tree(folder_path + "/TBSS/tbss_preproc/origdata", folder_path + "/TBSS/origdata")
            copy_tree(folder_path + "/TBSS/tbss_preproc/FA", folder_path + "/TBSS/FA")

        bashCommand = 'cd ' + outputdir + ' && tbss_2_reg '+ registration_type
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of reg\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/FA", folder_path + "/TBSS/tbss_reg/FA")
        #copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/tbss_reg/stats")

        if last_state=="reg":
            tbss_log.close()
            return

    if starting_state in (None, "reg", "postreg"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of postreg\n")
        tbss_log.flush()

        if starting_state == "postreg":
            copy_tree(folder_path + "/TBSS/tbss_preproc/origdata", folder_path + "/TBSS/origdata")
            copy_tree(folder_path + "/TBSS/tbss_reg/FA", folder_path + "/TBSS/FA")

        bashCommand = 'cd ' + outputdir + ' && tbss_3_postreg ' + postreg_type
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of postreg\n")
        tbss_log.flush()

        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/tbss_postreg/stats")

        if last_state=="postreg":
            tbss_log.close()
            return

    if starting_state in (None, "reg", "postreg", "prestats"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of prestats\n")
        tbss_log.flush()

        if starting_state == "prestats":
            copy_tree(folder_path + "/TBSS/tbss_preproc/origdata", folder_path + "/TBSS/origdata")
            copy_tree(folder_path + "/TBSS/tbss_reg/FA", folder_path + "/TBSS/FA")
            copy_tree(folder_path + "/TBSS/tbss_postreg/stats", folder_path + "/TBSS/stats")

        bashCommand = 'cd ' + outputdir + ' && tbss_4_prestats ' + prestats_treshold
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,
                                   stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of prestats\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/tbss_prestats/stats")

        if last_state=="prestats":
            tbss_log.close()
            return

    if starting_state in (None, "reg", "postreg", "prestats", "design"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of design\n")
        tbss_log.flush()

        if starting_state == "design":
            copy_tree(folder_path + "/TBSS/tbss_prestats/stats", folder_path + "/TBSS/stats")

        bashCommand = 'cd ' + outputdir + '/stats ' + ' && design_ttest2 design ' + numcontrol + ' ' + numpatient
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of design\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/design_ttest2/stats")

        if last_state=="design":
            tbss_log.close()
            return

    if randomise_corrected:
        bashCommand1 = 'cd ' + outputdir + '/stats ' + ' && randomise -i all_FA_skeletonised -o tbss -m mean_FA_skeleton_mask -d design.mat -t design.con -n 5000 --T2'
        bashCommand2 = 'cd ' + outputdir + '/stats ' + ' && autoaq -i tbss_tfce_corrp_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report1_subcortical.txt && autoaq -i tbss_tfce_corrp_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report2_subcortical.txt && autoaq -i tbss_tfce_corrp_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report1_cortical.txt && autoaq -i tbss_tfce_corrp_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report2_cortical.txt'
    else:
        bashCommand1 = 'cd ' + outputdir + '/stats ' + ' && randomise -i all_FA_skeletonised -o tbss -m mean_FA_skeleton_mask -d design.mat -t design.con -n 5000 --T2 --uncorrp'
        bashCommand2 = 'cd ' + outputdir + '/stats ' + ' && autoaq -i tbss_tfce_p_tstat1 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report1_subcortical.txt && autoaq -i tbss_tfce_p_tstat2 -a \"Harvard-Oxford Subcortical Structural Atlas\" -t 0.95 -o report2_subcortical.txt && autoaq -i tbss_tfce_p_tstat1 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report1_cortical.txt && autoaq -i tbss_tfce_p_tstat2 -a \"Harvard-Oxford Cortical Structural Atlas\" -t 0.95 -o report2_cortical.txt'

    if starting_state in (None, "reg", "postreg", "prestats", "design", "randomise"):
        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Beginning of randomise\n")
        tbss_log.flush()
        
        if starting_state == "randomise":
            copy_tree(folder_path + "/TBSS/design_ttest2/stats", folder_path + "/TBSS/stats")

        bashcmd1 = bashCommand1.split()
        print("Bash command is:\n{}\n".format(bashcmd1))
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,
                                   stderr=subprocess.STDOUT)
        output, error = process.communicate()

        tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": End of randomise\n")
        tbss_log.flush()

        # PERFORMS BACKUP FOR STARTING STATE
        from distutils.dir_util import copy_tree
        copy_tree(folder_path + "/TBSS/stats", folder_path + "/TBSS/randomise/stats")

        if last_state=="randomise":
            tbss_log.close()
            return

    tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of autoaq\n")
    tbss_log.flush()

    bashcmd2 = bashCommand2.split()
    print("Bash command is:\n{}\n".format(bashcmd2))
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tbss_log,stderr=subprocess.STDOUT)
    output, error = process.communicate()

    tbss_log.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": End of autoaq\n")
    tbss_log.flush()

    tbss_log.close()
