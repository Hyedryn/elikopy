import datetime
import os
import os
import shutil
import json
import numpy as np
import math


def preproc_solo(folder_path, p, eddy=False, denoising=False, reslice=False):

    print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual preprocessing for patient %s \n" % p)
    f.close()
    from dipy.io.image import load_nifti, save_nifti
    from dipy.segment.mask import median_otsu
    from dipy.denoise.localpca import mppca

    patient_path = os.path.splitext(p)[0]

    nifti_path = folder_path + '/' + patient_path + '.nii.gz'
    data, affine, voxel_size = load_nifti(nifti_path, return_voxsize=True)

    if reslice:
        reslice_path = folder_path + "/out/preproc/reslice"
        if not(os.path.exists(reslice_path)):
            try:
                os.mkdir(reslice_path)
            except OSError:
                print ("Creation of the directory %s failed" % reslice_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % reslice_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % reslice_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % reslice_path)
                f.close()

        from dipy.align.reslice import reslice
        new_voxel_size = (2., 2., 2.)
        data, affine = reslice(data, affine, voxel_size, new_voxel_size)
        save_nifti(folder_path + '/out/preproc/reslice/' + patient_path + '.nii.gz', data, affine)
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f=open(folder_path + "/out/logs.txt", "a+")
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Reslice completed for patient %s \n" % p)
        f.close()

    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]))
    save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
    save_nifti(folder_path + '/out/preproc/bet/' + patient_path + '_mask.nii.gz', b0_mask.astype(np.float32), affine)
    print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Brain extraction completed for patient %s \n" % p)
    f.close()

    if not(denoising) and not(eddy):
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '.nii.gz', b0_mask.astype(np.float32),affine)
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32),affine)
        shutil.copyfile(folder_path + "/" + patient_path + ".bval", folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
        shutil.copyfile(folder_path + "/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")


    if denoising:
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of denoising for patient %s \n" % p)
        f=open(folder_path + "/out/logs.txt", "a+")
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Denoising launched for patient %s \n" % p)
        f.close()

        denoising_path = folder_path + "/out/preproc/denoising"
        if not(os.path.exists(denoising_path)):
            try:
                os.mkdir(denoising_path)
            except OSError:
                print ("Creation of the directory %s failed" % denoising_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % denoising_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % denoising_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % denoising_path)
                f.close()

        pr = math.ceil((np.shape(b0_mask)[3] ** (1 / 3) - 1) / 2)
        denoised = mppca(b0_mask, patch_radius=pr)
        save_nifti(denoising_path + '/' + patient_path + '_mask_denoised.nii.gz', denoised.astype(np.float32), affine)

        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of denoising for patient %s \n" % p)

        if not eddy:
            save_nifti(folder_path + '/out/preproc/final/' + patient_path + '.nii.gz', denoised.astype(np.float32),affine)
            save_nifti(folder_path + '/out/preproc/final/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
            shutil.copyfile(folder_path + "/" + patient_path + ".bval",folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
            shutil.copyfile(folder_path + "/" + patient_path + ".bvec",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")


    #Explicitly freeing memory
    import gc
    denoised = None
    b0_mask = None
    mask = None
    data= None
    affine = None
    gc.collect()


    if eddy:
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of eddy for patient %s \n" % p)

        eddy_path = folder_path + "/out/preproc/eddy"
        if not(os.path.exists(eddy_path)):
            try:
                os.mkdir(eddy_path)
            except OSError:
                print ("Creation of the directory %s failed" % eddy_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Creation of the directory %s failed\n" % eddy_path)
                f.close()
            else:
                print ("Successfully created the directory %s " % eddy_path)
                f=open(folder_path + "/out/logs.txt", "a+")
                f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully created the directory %s \n" % eddy_path)
                f.close()


        if denoising:
            bashCommand = 'eddy --imain="' + folder_path  + '/out/preproc/denoising/' + patient_path + '_mask_denoised.nii.gz" --mask="' + folder_path  + '/out/preproc/bet/' +  patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/acqparams.txt" --index="' + folder_path + '/index.txt" --bvecs="' + folder_path + '/' + patient_path + '.bvec" --bvals="' + folder_path + '/' + patient_path + '.bval" --out="' + folder_path + '/out/preproc/eddy/' + patient_path + '_mfc" --verbose'
        else:
            bashCommand = 'eddy --imain="' + folder_path  + '/out/preproc/bet/' + patient_path + '_mask.nii.gz" --mask="' + folder_path  + '/out/preproc/bet/' +  patient_path + '_binary_mask.nii.gz" --acqp="' + folder_path + '/acqparams.txt" --index="' + folder_path + '/index.txt" --bvecs="' + folder_path + '/' + patient_path + '.bvec" --bvals="' + folder_path + '/' + patient_path + '.bval" --out="' + folder_path + '/out/preproc/eddy/' + patient_path + '_mfc" --verbose'

        import subprocess
        bashcmd = bashCommand.split()
        #print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command \n{}\n".format(bashcmd))
        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f=open(folder_path + "/out/logs.txt", "a+")
        #f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command \n{}\n".format(bashcmd))
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command " + bashCommand)
        f.close()

        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)

        #wait until eddy finish
        output, error = process.communicate()

        print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f=open(folder_path + "/out/logs.txt", "a+")
        #f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Eddy launched for patient %s \n" % p + " with bash command \n{}\n".format(bashcmd))
        f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": End of eddy for patient %s \n" % p)
        f.close()

        data, affine = load_nifti(folder_path + "/out/preproc/eddy/" + patient_path + "_mfc.nii.gz")
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=range(0, np.shape(data)[3]))
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '_binary_mask.nii.gz', mask.astype(np.float32), affine)
        save_nifti(folder_path + '/out/preproc/final/' + patient_path + '.nii.gz', b0_mask.astype(np.float32), affine)

        shutil.copyfile(folder_path + "/" + patient_path + ".bval",folder_path + "/out/preproc/final" + "/" + patient_path + ".bval")
        shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + "_mfc.eddy_rotated_bvecs",folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")
        #shutil.copyfile(folder_path + "/out/preproc/eddy/" + patient_path + "_mfc.nii.gz",folder_path + "/out/preproc/final" + "/" + patient_path + ".nii.gz")
        #shutil.copyfile(folder_path + "/out/preproc/bet/" + patient_path + "_binary_mask.nii.gz",folder_path + "/out/preproc/final" + "/" + patient_path + "_binary_mask.nii.gz")



    print("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[PREPROC SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()

def dti_solo(folder_path, p):
    print("[DTI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual DTI processing for patient %s \n" % p)

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    import dipy.reconst.dti as dti

    patient_path = os.path.splitext(p)[0]
    # load the data======================================
    data, affine = load_nifti(folder_path + "/out/preproc/final" + "/" + patient_path + ".nii.gz")
    mask, _ = load_nifti(folder_path + "/out/preproc/final" + "/" + patient_path + "_binary_mask.nii.gz")
    bvals, bvecs = read_bvals_bvecs(folder_path + "/out/preproc/final" + "/" + patient_path + ".bval", folder_path + "/out/preproc/final" + "/" + patient_path + ".bvec")
    # create the model===================================
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    # FA ================================================
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    save_nifti(folder_path + "/out/dti/" + patient_path + "_fa.nii.gz", FA.astype(np.float32), affine)
    # colored FA ========================================
    RGB = dti.color_fa(FA, tenfit.evecs)
    save_nifti(folder_path + "/out/dti/" + patient_path + "_fargb.nii.gz", np.array(255 * RGB, 'uint8'), affine)
    # Mean diffusivity ==================================
    MD = dti.mean_diffusivity(tenfit.evals)
    save_nifti(folder_path + "/out/dti/" + patient_path + "_md.nii.gz", MD.astype(np.float32), affine)
    # Radial diffusivity ==================================
    RD = dti.radial_diffusivity(tenfit.evals)
    save_nifti(folder_path + "/out/dti/" + patient_path + "_rd.nii.gz", RD.astype(np.float32), affine)
    # Axial diffusivity ==================================
    AD = dti.axial_diffusivity(tenfit.evals)
    save_nifti(folder_path + "/out/dti/" + patient_path + "_ad.nii.gz", AD.astype(np.float32), affine)
    # eigen vectors =====================================
    save_nifti(folder_path + "/out/dti/" + patient_path + "_evecs.nii.gz", tenfit.evecs.astype(np.float32), affine)
    # eigen values ======================================
    save_nifti(folder_path + "/out/dti/" + patient_path + "_evals.nii.gz", tenfit.evals.astype(np.float32), affine)
    # diffusion tensor ====================================
    save_nifti(folder_path + "/out/dti/" + patient_path + "_dtensor.nii.gz", tenfit.quadratic_form.astype(np.float32), affine)

    print("[DTI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f=open(folder_path + "/out/logs.txt", "a+")
    f.write("[DTI SOLO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
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


def white_mask_solo(folder_path, p):

    print("[White maks solo] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Beginning of individual white mask processing for patient %s \n" % p)

    from dipy.align.imaffine import (AffineMap, MutualInformationMetric, AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
    from dipy.segment.tissue import TissueClassifierHMRF
    from dipy.io.image import load_nifti, save_nifti
    import subprocess

    patient_path = os.path.splitext(p)[0]
    # Read the moving image ====================================
    anat_path = folder_path + '/anat/' + patient_path + '_T1.nii.gz'
    bet_path = folder_path + '/out/whitemask/' + patient_path + '_T1_brain.nii.gz'
    bashCommand = 'bet2 ' + anat_path + ' ' + bet_path +' -f 1 -g 3'
    bashcmd = bashCommand.split()
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)
    output, error = process.communicate()
    anat_path = folder_path + '/out/whitemask/' + patient_path + '_T1_brain.nii.gz'
    bet_path = folder_path + '/out/whitemask/' + patient_path + '_T1_brain_brain.nii.gz'
    bashCommand = 'bet2 ' + anat_path + ' ' + bet_path + ' -f 0.4 -g 0.2'
    bashcmd = bashCommand.split()
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)
    output, error = process.communicate()
    moving_data, moving_affine = load_nifti(bet_path)
    moving = moving_data
    moving_grid2world = moving_affine
    # Read the static image ====================================
    static_data, static_affine = load_nifti(folder_path + "/out/preproc/final" + "/" + patient_path + ".nii.gz")
    static = np.squeeze(static_data)[..., 0]
    static_grid2world = static_affine
    # Reslice the moving image ====================================
    identity = np.eye(4)
    affine_map = AffineMap(identity, static.shape, static_grid2world, moving.shape, moving_grid2world)
    # translation the moving image ====================================
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = affine_map.affine
    translation = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)
    # Rigid transform the moving image ====================================
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)
    # affine transform the moving image ====================================
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)
    transformed = affine.transform(moving)
    # final result of registration ==========================================
    anat = transformed
    anat_affine = static_grid2world
    # make the white matter segmentation ===================================
    nclass = 3
    beta = 0.1
    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(anat, nclass, beta)
    # save the white matter mask ============================================
    white_mask = PVE[..., 2]
    white_mask[white_mask >= 0.05] = 1
    white_mask[white_mask < 0.05] = 0
    out_path = folder_path + '/out/whitemask/' + patient_path + '_whitemask.nii.gz'
    save_nifti(out_path, white_mask.astype(np.float32), anat_affine)

    print("[White maks solo] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f = open(folder_path + "/out/logs.txt", "a+")
    f.write("[White maks solo] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Successfully processed patient %s \n" % p)
    f.close()
