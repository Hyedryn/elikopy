
import os
import os
import shutil
import json
import numpy as np
import math


def preproc_solo(folder_path, p, eddy=False, denoising=False):

    print("[PREPROC SOLO] Beginning of individual preprocessing for patient %s \n" % p)
    from dipy.io.image import load_nifti, save_nifti
    from dipy.segment.mask import median_otsu
    from dipy.denoise.localpca import mppca

    patient_path = os.path.splitext(p)[0]

    nifti_path = folder_path + '/' + patient_path + '.nii.gz'
    data, affine = load_nifti(nifti_path)

    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]))
    save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
    save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_mask.nii.gz', b0_mask.astype(np.float32), affine)
    if not(denoising) and not(eddy):
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '.nii.gz', b0_mask.astype(np.float32),affine)
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32),affine)
        shutil.copyfile(folder_path + "/" + patient_path + ".bval", folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
        shutil.copyfile(folder_path + "/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")


    if denoising:
        denoising_path = folder_path + "/out/preproc/denoising"
        if not(os.path.exists(denoising_path)):
            try:
                os.mkdir(denoising_path)
            except OSError:
                print ("Creation of the directory %s failed" % denoising_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] Creation of the directory %s failed\n" % denoising_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % denoising_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] Successfully created the directory %s \n" % denoising_path)
                f.close()

        pr = math.ceil((np.shape(b0_mask)[3] ** (1 / 3) - 1) / 2)
        denoised = mppca(b0_mask, patch_radius=pr)
        save_nifti(denoising_path + '/' + patient_path + '_mask_denoised.nii.gz', denoised.astype(np.float32), affine)
        if not eddy:
            save_nifti(folder_path + '/out/preproc/final/' + patient_path + '.nii.gz', denoised.astype(np.float32),affine)
            save_nifti(folder_path + '/out/preproc/final/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
            shutil.copyfile(folder_path + "/" + patient_path + ".bval",folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
            shutil.copyfile(folder_path + "/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")

    if eddy:
        eddy_path = folder_path + "/out/preproc/eddy"
        if not(os.path.exists(eddy_path)):
            try:
                os.mkdir(eddy_path)
            except OSError:
                print ("Creation of the directory %s failed" % eddy_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] Creation of the directory %s failed\n" % eddy_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % eddy_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] Successfully created the directory %s \n" % eddy_path)
                f.close()


        if denoising:
            bashCommand = 'eddy --imain=' + folder_path  + '/out/preproc/denoising/' + patient_path + '_mask_denoised.nii.gz --mask=' + folder_path  + '/out/preproc/bet/' +  patient_path + '_bet.nii.gz --acqp="' + folder_path + '/acqparams.txt" --index="' + folder_path + '/index.txt" --bvecs="' + folder_path + '/' + patient_path + '.bvec" --bvals="' + folder_path + '/' + patient_path + '.bval" --out="' + folder_path + '/out/preproc/eddy/' + patient_path + '_mfc" --verbose'
        else:
            bashCommand = 'eddy --imain=' + folder_path  + '/' + patient_path + '.nii.gz --mask=' + folder_path  + '/out/preproc/bet/' +  patient_path + '_bet.nii.gz --acqp="' + folder_path + '/acqparams.txt" --index="' + folder_path + '/index.txt" --bvecs="' + folder_path + '/' + patient_path + '.bvec" --bvals="' + folder_path + '/' + patient_path + '.bval" --out="' + folder_path + '/out/preproc/eddy/' + patient_path + '_mfc" --verbose'

        import subprocess
        bashcmd = bashCommand.split()
        print("Bash command is:\n{}\n".format(bashcmd))
        process = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)

        #wait until eddy finish
        output, error = process.communicate()

        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + ".bval",folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")
        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + "_mfc.nii.gz",folder_path + "/out/preproc/final" + "/" + patient_path + ".nii.gz")
        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + "_bet_mfc.nii.gz",folder_path + "/out/preproc/final" + "/" + patient_path + "_binary_mask.nii.gz")

    print("[PREPROC SOLO] Successfully processed patient %s \n" % p)
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[PREPROC SOLO] Successfully processed patient %s \n" % p)
    f.close()


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
    print("Generated slurm batch command: '%s'" % slurm_cmd)

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
