import datetime
import os

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
