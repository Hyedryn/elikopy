import os
import subprocess
from pathlib import Path
import json
from dipy.io.image import load_nifti
import nibabel as nib
import numpy as np 

def MDTprotocol(folder_path, p, mdt_image=None):
    patient_path = p
    preproc_folder = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/'
    
    if mdt_image is None:
        mdt_image = os.path.join(folder_path,"mdt.simg")
    
    with open(folder_path + '/subjects/' + patient_path + '/dMRI/raw/'+ patient_path + "_raw_dmri.json","r") as f:
        raw_metadata = json.load(f)
    TR = raw_metadata["RepetitionTime"]
    TE = raw_metadata["EchoTime"]
    
    bvec_path = preproc_folder + patient_path + '_dmri_preproc.bvec'
    bval_path = preproc_folder + patient_path + '_dmri_preproc.bval'
    protocol_output_path = preproc_folder + patient_path + '_dmri_preproc.prtcl'
    
    baseline_cmd = "singularity exec --bind " + folder_path + " " + mdt_image + " "
    cmd = f"mdt-create-protocol --sequence-timing-units s {bvec_path} {bval_path} --TE {TE} --TR {TR} -o {protocol_output_path} "
    
    bashCommand = baseline_cmd + cmd 

    protocol_log = open(preproc_folder + "MDT_protocol_logs.txt", "a+")
    print(bashCommand, flush=True)
    protocol_log.write(bashCommand + "\n")
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=protocol_log,
                               stderr=subprocess.STDOUT)
    process.communicate()

    protocol_log.close()
    
    
def computeDiff(model_path, patient, model):

    fib1 = [1, 1, 2]
    fib2 = [2, 3, 3]
    
    V = nib.load(os.path.join(model_path, f'{patient}_{model}_CHARMEDRestricted1.phi.nii.gz'))
    img_shape = V.get_fdata().shape
    phi = np.zeros((img_shape[0], img_shape[1], img_shape[2], 2), dtype=float)
    theta = np.zeros((img_shape[0], img_shape[1], img_shape[2], 2), dtype=float)
    f_r = np.zeros((img_shape[0], img_shape[1], img_shape[2], 2), dtype=float)
    disp = np.zeros((img_shape[0], img_shape[1], img_shape[2], 3), dtype=float)

    for nf in range(3):
        V = nib.load(os.path.join(model_path, f'{patient}_{model}_CHARMEDRestricted{fib1[nf] - 1}.phi.nii.gz'))
        phi[:, :, :, 0] = V.get_fdata()
        
        V = nib.load(os.path.join(model_path, f'{patient}_{model}_CHARMEDRestricted{fib1[nf] - 1}.theta.nii.gz'))
        theta[:, :, :, 0] = V.get_fdata()
        
        V = nib.load(os.path.join(model_path, f'{patient}_{model}_CHARMEDRestricted{fib2[nf] - 1}.phi.nii.gz'))
        phi[:, :, :, 1] = V.get_fdata()
        
        V = nib.load(os.path.join(model_path, f'{patient}_{model}_CHARMEDRestricted{fib2[nf] - 1}.theta.nii.gz'))
        theta[:, :, :, 1] = V.get_fdata()
        
        V = nib.load(os.path.join(model_path, f'{patient}_{model}_w_res{fib1[nf] - 1}.w.nii.gz'))
        f_r[:, :, :, 0] = V.get_fdata()
        
        V = nib.load(os.path.join(model_path, f'{patient}_{model}_w_res{fib2[nf] - 1}.w.nii.gz'))
        f_r[:, :, :, 1] = V.get_fdata()

        for i in range(f_r.shape[0]):
            for j in range(f_r.shape[1]):
                for k in range(f_r.shape[2]):
                    if np.any(f_r[i, j, k, :]):
                        phi_values = phi[i, j, k, :]
                        theta_values = theta[i, j, k, :]
                        x, y, z = np.sin(phi_values) * np.cos(theta_values), np.sin(phi_values) * np.sin(theta_values), np.cos(phi_values)
                        fiber = np.vstack((-x, y, z)).T
                        disp[i, j, k, nf] = np.mean(np.sqrt(np.sum(np.diff(fiber, axis=0) ** 2, axis=1)))

                    else:
                        disp[i, j, k, nf] = 0

    disp_ave = np.mean(disp, axis=3)
    nii = nib.Nifti1Image(disp_ave, affine=np.eye(4))
    nib.save(nii, os.path.join(model_path, f'{patient}_{model}_disp.nii.gz'))
    

def MDTcomputation(folder_path, p, model="CHARMED_r3", core_count=1, mdt_image=None, maskType="brain_mask_dilated", method="Powell"):
    """
    Computes the SIFT2 algorithm using the fod and tractogram.

    Warning: DWI volumes often have a non-negligible B1 bias field, mostly due to high-density receiver coils.
    If left uncorrected, SIFT will incorrectly interpret this as a spatially-varying fibre density.
    Therefore bias field correction during the preprocessing is HIGHLY recommended when using SIFT.

    :param folder_path:
    :param p:
    :param SIFT2:
    :param msmtCSD:
    :param core_count:
    :return:
    """
    patient_path = p
    assert method in ['Powell', 'Nelder-Mead', 'Levenberg-Marquardt', 'Subplex'], "Invalid optimisation method"
    assert maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                            "wm_mask_Freesurfer_T1"], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1"

    mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_' + maskType + '.nii.gz'
    if os.path.isfile(mask_path):
        mask, _ = load_nifti(mask_path)
    else:
        mask, _ = load_nifti(
            folder_path + '/subjects/' + patient_path + '/masks/' + patient_path + "_brain_mask_dilated.nii.gz")
    microstructure_path = folder_path + '/subjects/' + patient_path + "/dMRI/microstructure/"
    model_path = os.path.join(microstructure_path, model)
    
    assert model in ["CHARMED_r3"]
    
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    if mdt_image is None:
        mdt_image = os.path.join(folder_path,"mdt_nvidia.simg")
    
    baseline_cmd = "singularity exec --nv --bind " + folder_path + " " + mdt_image + " "
    preproc_folder = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/'
    dwi_path = preproc_folder + patient_path + '_dmri_preproc.nii.gz'
    
    # Create protocol file
    #mdt_create_protocol
    protocol_path = preproc_folder + patient_path + '_dmri_preproc.prtcl'
    if not os.path.exists(protocol_path):
        MDTprotocol(folder_path, p, mdt_image=mdt_image)

    # Fit model
    #mdt_model_fit
    cmd = f"mdt-model-fit {model} {dwi_path} {protocol_path} {mask_path} -o {model_path} --method {method} "
    
    bashCommand = baseline_cmd + cmd 

    model_log = open(model_path + "/MDT_model_fit_logs.txt", "a+")
    print(bashCommand, flush=True)
    
    if not os.path.exists(os.path.join(model_path,model)):
        model_log.write(bashCommand + "\n")
        
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=model_log,
                                   stderr=subprocess.STDOUT)

        process.communicate()
    
    if os.path.exists(os.path.join(model_path,model)):
        for f in os.listdir(os.path.join(model_path,model)):
            if ".nii.gz" in f:
                img = nib.load(os.path.join(model_path,model,f))
                dim = img.header["dim"]
                if dim[0] == 4 and dim[4] == 1 and dim[5] == 1:
                    print(f"Adjusted header for file {f}")
                    img.header["dim"][0] = 3
                    img.header.set_data_shape((img.header["dim"][1],img.header["dim"][2],img.header["dim"][3]))
                    #print(img.header)
                    print(np.squeeze(img.get_fdata()).shape)
                    img = nib.Nifti1Image(np.squeeze(img.get_fdata()), img.affine, img.header)
                else:
                    print(f)
                nib.save(img, os.path.join(model_path, f'{patient_path}_{model}_{f}'))  

        model_log.close()
        
        if model == "CHARMED_r3":
            computeDiff(model_path,patient_path,model)
        
        json_status_file = os.path.join(folder_path,"subjects",p,f"{p}_status.json")
        if os.path.exists(json_status_file):
            with open(json_status_file,"r") as f:
                patient_status = json.load(f)
        else:
            patient_status = {}
        patient_status[model] = True
        with open(json_status_file,"w") as f:
            json.dump(patient_status, f, indent = 6)
    else:
        raise Exception("Unable to perform MDT")
    
