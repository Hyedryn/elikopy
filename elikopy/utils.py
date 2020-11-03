import datetime
import os
import json
import shutil

from future.utils import iteritems
import subprocess


def submit_job(job_info):
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
            
