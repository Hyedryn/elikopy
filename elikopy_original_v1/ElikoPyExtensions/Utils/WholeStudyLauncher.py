import elikopy
import os
import subprocess
from pathlib import Path
import json
from dipy.io.image import load_nifti
import nibabel as nib
import sys
from elikopy.registration import regallDWIToT1wToT1wCommonSpace, applyTransformToAllMapsInFolder
from elikopy.utils import update_status
import traceback

wrapper_path = r"/CECI/proj/pilab/static_files_ELIKOPY/ElikoPyWrapper/"
atlasreg_path= wrapper_path + r"Registration/"
sys.path.append(atlasreg_path)

connM_path= wrapper_path + r"Tractography/"
sys.path.append(connM_path)

from registerT1toMNIusingT1Ref import applyTransformT1withT1Ref
from registerAtlasFromCommonSpaceToSubjectSpace import inverseTransformAtlas
from connectivityMatrixFromTractogram import connectivityMatrix


T1_MNI="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
T1_MNI_mask="${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
T1_MNI=r'/CECI/proj/pilab/static_files_ELIKOPY/T1_MNI/MNI_ICBM152_T1_NLIN_ASYM_09c_BRAIN.nii.gz'
T1_MNI_mask=r'/CECI/proj/pilab/static_files_ELIKOPY/T1_MNI/MNI_ICBM152_T1_NLIN_ASYM_09c_BRAINMASK.nii.gz'

def printError(ex_type, ex_value, ex_traceback):
    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" %ex_value)
    print("Stack trace : %s" %stack_trace, flush=True)

def processingPipeline(folder_path, p, slurm_email, singleShell=False, forced={}, excluded={}, longitudinal=False, DWI_type="B0FSL", core_count=1):
    study = elikopy.core.Elikopy(folder_path, slurm=False, slurm_email=slurm_email, cuda=False)
    
    longitudinal_txt = "" if not longitudinal else "_longitudinal"
    
    print(f"Starting processingPipeline for subject {p}")
    
    print(excluded)
    
    #dic_path = "/CECI/proj/pilab/static_files_ELIKOPY/mf_dic/dictionary-fixedraddist_scheme-StLucGE.mat"
    dic_path = "/CECI/proj/pilab/static_files_ELIKOPY/mf_dic/dictionary-fixedraddist_scheme-StLucGE_size-exhaustive.mat"
    error = False
    json_status_file = os.path.join(folder_path,"subjects",p,f"{p}_status.json")
    if os.path.exists(json_status_file):
        with open(json_status_file,"r") as f:
            patient_status = json.load(f)
    else:
        patient_status = {}
        
    if forced.get('preproc') is None:
        forced["preproc"] = False
    if forced.get('wm_mask_FSL_T1') is None:
        forced["wm_mask_FSL_T1"] = False
    if forced.get('wm_mask_AP') is None:
        forced["wm_mask_AP"] = False
    if forced.get('dti') is None:
        forced["dti"] = False
    if forced.get('odf_msmtcsd') is None:
        forced["odf_msmtcsd"] = False
    if forced.get('odf_csd') is None:
        forced["odf_csd"] = False
    if forced.get('noddi') is None:
        forced["noddi"] = False
    if forced.get('fingerprinting') is None:
        forced["fingerprinting"] = False
    if forced.get(f'regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}') is None:
        forced[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] = False
    if forced.get('tracking') is None:
        forced["tracking"] = False
    if forced.get(f'inverseTransformAtlas_{DWI_type}{longitudinal_txt}') is None:
        forced[f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}"] = False
    if forced.get('siftComputation') is None:
        forced["siftComputation"] = False
    if forced.get(f'connectivityMatrixSift_{DWI_type}{longitudinal_txt}') is None:
        forced[f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}"] = False
    if forced.get(f'connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}') is None:
        forced[f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}"] = False
    if forced.get(f'regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}') is None:
        forced[f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}"] = False
    if forced.get(f'regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}') is None:
        forced[f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}"] = False
    if forced.get(f'regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}') is None:
        forced[f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}"] = False
    if forced.get(f'regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}') is None:
        forced[f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}"] = False
    if forced.get(f'inverseT1{longitudinal_txt}') is None:
        forced[f"inverseT1{longitudinal_txt}"] = False
        
    
    if excluded.get('preproc') is None:
        excluded["preproc"] = False
    if excluded.get('wm_mask_FSL_T1') is None:
        excluded["wm_mask_FSL_T1"] = False
    if excluded.get('wm_mask_AP') is None:
        excluded["wm_mask_AP"] = False
    if excluded.get('dti') is None:
        excluded["dti"] = False
    if excluded.get('odf_msmtcsd') is None:
        excluded["odf_msmtcsd"] = False
    if excluded.get('odf_csd') is None:
        excluded["odf_csd"] = False
    if excluded.get('noddi') is None:
        excluded["noddi"] = False
    if excluded.get('fingerprinting') is None:
        excluded["fingerprinting"] = False
    if excluded.get(f'regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get('tracking') is None:
        excluded["tracking"] = False
    if excluded.get(f'inverseTransformAtlas_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get('siftComputation') is None:
        excluded["siftComputation"] = False
    if excluded.get(f'connectivityMatrixSift_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get(f'connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get(f'regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get(f'regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get(f'regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get(f'regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}') is None:
        excluded[f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}"] = False
    if excluded.get(f'inverseT1{longitudinal_txt}') is None:
        excluded[f"inverseT1{longitudinal_txt}"] = False
    
    
    
    if (not (patient_status.get('preproc') is not None and patient_status["preproc"] == True) or forced["preproc"]) and not excluded["preproc"]:
        try:
            print("preproc",flush=True)
            study.preproc(eddy=True,
                        topup=True,
                        forceSynb0DisCo=False,
                        denoising=True,  
                        reslice=False, 
                        gibbs=False, 
                        biasfield=False,
                        qc_reg=False,
                        starting_state=None, 
                        report=True, patient_list_m=[p], cpus=core_count, eddy_additional_arg=" --data_is_shelled ")
            patient_status["preproc"] = True
        except Exception as e:
            patient_status["preproc"] = False
            error = True
            ex_type, ex_value, ex_traceback = sys.exc_info()
            printError(ex_type, ex_value, ex_traceback)
            print(e, flush=True)
            
            
            
    with open(json_status_file,"w") as f:
            json.dump(patient_status, f, indent = 6)
            
    if patient_status.get('preproc') is not None and patient_status["preproc"] == True:
        
        if (not (patient_status.get('wm_mask_FSL_T1') is not None and patient_status["wm_mask_FSL_T1"] == True) or forced["wm_mask_FSL_T1"] ) and not excluded["wm_mask_FSL_T1"]:
            try:
                print("wm_mask_FSL_T1",flush=True)
                study.white_mask("wm_mask_FSL_T1", corr_gibbs=True, debug=False, patient_list_m=[p], cpus=core_count)
                patient_status["wm_mask_FSL_T1"] = True
            except Exception as e:
                patient_status["wm_mask_FSL_T1"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if (not (patient_status.get('wm_mask_AP') is not None and patient_status["wm_mask_AP"] == True) or forced["wm_mask_AP"]) and not excluded["wm_mask_AP"]:
            try:
                print("wm_mask_AP",flush=True)
                study.white_mask("wm_mask_AP", debug=False, patient_list_m=[p], cpus=core_count)
                patient_status["wm_mask_AP"] = True
            except Exception as e:
                patient_status["wm_mask_AP"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if (not (patient_status.get('dti') is not None and patient_status["dti"] == True) or forced["dti"]) and not excluded["dti"]:
            try:
                print("dti",flush=True)
                study.dti(patient_list_m=[p])
                patient_status["dti"] = True
            except Exception as e:
                patient_status["dti"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if (not (patient_status.get('odf_msmtcsd') is not None and patient_status["odf_msmtcsd"] == True) or forced["odf_msmtcsd"]) and not excluded["odf_msmtcsd"]:
            try:
                print("odf_msmtcsd",flush=True)
                study.odf_msmtcsd(num_peaks=2, peaks_threshold=0.25, cpus=core_count, patient_list_m=[p])
                patient_status["odf_msmtcsd"] = True
            except Exception as e:
                patient_status["odf_msmtcsd"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if (not (patient_status.get('odf_csd') is not None and patient_status["odf_csd"] == True) or forced["odf_csd"]) and not excluded["odf_csd"]:
            try:
                print("odf_csd",flush=True)
                study.odf_csd(num_peaks=2, peaks_threshold=0.25, patient_list_m=[p], cpus=core_count)
                patient_status["odf_csd"] = True
            except Exception as e:
                patient_status["odf_csd"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
                
        if (not (patient_status.get('noddi') is not None and patient_status["noddi"] == True) or forced["noddi"]) and not excluded["noddi"]:
            try:
                print("noddi",flush=True)
                study.noddi(cpus=core_count, patient_list_m=[p])
                patient_status["noddi"] = True
            except Exception as e:
                print(e, flush=True)
                patient_status["noddi"] = False
                error = True
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
           
        if patient_status.get('odf_msmtcsd') is not None and patient_status["odf_msmtcsd"] == True:
            if ((not (patient_status.get('fingerprinting') is not None and patient_status["fingerprinting"] == True) or forced["fingerprinting"]) and not singleShell) and not excluded["fingerprinting"]:
                try:
                    print("fingerprinting",flush=True)
                    study.fingerprinting(dictionary_path=dic_path, mfdir="mf", patient_list_m=[p], peaksType="MSMT-CSD", cpus=core_count)
                    patient_status["fingerprinting"] = True
                except Exception as e:
                    patient_status["fingerprinting"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
        if patient_status.get('wm_mask_AP') is not None and patient_status["wm_mask_AP"] == True:
            if (not (patient_status.get(f'regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] == True) or forced[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"]) and not excluded[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"]:
                try:
                    print(f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}",flush=True)
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type=DWI_type, longitudinal=longitudinal, maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask, metrics_dic={})
                    patient_status[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
                    
        if patient_status.get(f'regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}') is not None and patient_status.get('dti') is not None and patient_status[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] == True and patient_status["dti"] == True:
            if (not (patient_status.get(f'regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}"] == True) or forced[f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}"]) and not excluded[f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}"]:
                try:
                    print(f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}",flush=True)
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type=DWI_type, longitudinal=longitudinal, maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask, metrics_dic={'_FA': 'dti', 'RD': 'dti', 'AD': 'dti', 'MD': 'dti'})
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceDTI_{DWI_type}{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
        if (not (patient_status.get('tracking') is not None and patient_status["tracking"] == True) or forced["tracking"]) and not excluded["tracking"]:
            try:
                print("tracking",flush=True)
                if excluded["odf_msmtcsd"] == True or singleShell:
                    study.tracking(patient_list_m=[p],streamline_number=1000000,msmtCSD=False, cpus=core_count)
                else:
                    study.tracking(patient_list_m=[p],streamline_number=1000000,msmtCSD=True, cpus=core_count)
                patient_status["tracking"] = True
            except Exception as e:
                patient_status["tracking"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
        
        if patient_status.get(f'regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] == True and patient_status.get('tracking') is not None and patient_status["tracking"] == True:
            
            if (not (patient_status.get(f'inverseTransformAtlas_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}"] == True) or forced[f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}"]) and not excluded[f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}"]:
                try:
                    print(f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}",flush=True)
                    atlas_path = "/CECI/proj/pilab/static_files_ELIKOPY/T1_MNI/" + "BN_Atlas_246_1mm_MNI.nii.gz"
                    inverseTransformAtlas(folder_path, p, atlasPath=atlas_path, atlasName="BN_246_1mm", DWI_type=DWI_type, longitudinal=longitudinal)
                    atlas_path = "/CECI/proj/pilab/static_files_ELIKOPY/T1_MNI/" + "BN_Atlas_280_1mm_MNI_cerebellum.nii.gz"
                    inverseTransformAtlas(folder_path, p, atlasPath=atlas_path, atlasName="BN_280_1mm", DWI_type=DWI_type, longitudinal=longitudinal)
                    patient_status[f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"inverseTransformAtlas_{DWI_type}{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
            
            if (not (patient_status.get('siftComputation') is not None and patient_status["siftComputation"] == True) or forced["siftComputation"]) and not excluded["siftComputation"]:
                try:
                    print("siftComputation",flush=True)
                    print("Starting SIFT")
                    if singleShell:
                        study.sift(patient_list_m=[p], msmtCSD=False, streamline_number=500000, cpus=core_count)
                    else:
                        study.sift(patient_list_m=[p], msmtCSD=True, streamline_number=500000, cpus=core_count)
                    patient_status["siftComputation"] = True
                except Exception as e:
                    patient_status["siftComputation"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
        
            if patient_status.get('siftComputation') is not None and patient_status["siftComputation"] == True:
                if (not (patient_status.get(f'connectivityMatrixSift_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}"] == True)
                    or forced[f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}"]) and not excluded[f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}"]:
                    try:
                        print(f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}",flush=True)
                        print("Starting conn matrix")
                        fname=f"BN_246_1mm_InSubjectDWISpaceFrom_{DWI_type}"
                        connectivityMatrix(folder_path, p, label_fname=fname, input="SIFT", dilation_radius=1, inclusive=False, longitudinal=longitudinal)
                        connectivityMatrix(folder_path, p, label_fname=fname, input="SIFT", dilation_radius=1, inclusive=True, longitudinal=longitudinal)
                        fname=f"BN_280_1mm_InSubjectDWISpaceFrom_{DWI_type}"
                        connectivityMatrix(folder_path, p, label_fname=fname, input="SIFT", dilation_radius=1, inclusive=False, longitudinal=longitudinal)
                        connectivityMatrix(folder_path, p, label_fname=fname, input="SIFT", dilation_radius=1, inclusive=True, longitudinal=longitudinal)
                        
                        patient_status[f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}"] = True
                    except Exception as e:
                        patient_status[f"connectivityMatrixSift_{DWI_type}{longitudinal_txt}"] = False
                        error = True
                        ex_type, ex_value, ex_traceback = sys.exc_info()
                        printError(ex_type, ex_value, ex_traceback)
                        print(e, flush=True)
                    with open(json_status_file,"w") as f:
                        json.dump(patient_status, f, indent = 6)
                    
            if (not (patient_status.get(f'connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}"] == True)
                or forced[f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}"]) and not excluded[f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}"]:
                try:
                    print(f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}",flush=True)
                    fname=f"BN_246_1mm_InSubjectDWISpaceFrom_{DWI_type}"
                    print("Starting conn matrix")
                    fname=f"BN_246_1mm_InSubjectDWISpaceFrom_{DWI_type}"
                    connectivityMatrix(folder_path, p, label_fname=fname, input="TCKGEN", dilation_radius=1, inclusive=False, longitudinal=longitudinal)
                    connectivityMatrix(folder_path, p, label_fname=fname, input="TCKGEN", dilation_radius=1, inclusive=True,  longitudinal=longitudinal)
                    fname=f"BN_280_1mm_InSubjectDWISpaceFrom_{DWI_type}"
                    connectivityMatrix(folder_path, p, label_fname=fname, input="TCKGEN", dilation_radius=1, inclusive=False, longitudinal=longitudinal)
                    connectivityMatrix(folder_path, p, label_fname=fname, input="TCKGEN", dilation_radius=1, inclusive=True,  longitudinal=longitudinal)
                    patient_status[f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"connectivityMatrixTCKGEN_{DWI_type}{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
                    
        if patient_status.get('wm_mask_AP') is not None and patient_status.get('noddi') is not None and patient_status["wm_mask_AP"] == True and patient_status["noddi"] == True:
            if (not (patient_status.get(f'regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}"] == True)
                or forced[f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}"]) and not excluded[f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}"]:
                try:
                    print(f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}",flush=True)
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type=DWI_type, maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask, longitudinal=longitudinal, metrics_dic={'noddi_fiso': 'noddi', 'noddi_odi': 'noddi', 'noddi_icvf': 'noddi', 'noddi_fintra': 'noddi', 'noddi_fextra': 'noddi', 'noddi_fbundle': 'noddi'})
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceNoddi_{DWI_type}{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
            
        if patient_status.get('wm_mask_AP') is not None and patient_status.get('fingerprinting') is not None and patient_status["wm_mask_AP"] == True and patient_status["fingerprinting"] == True:
            if (not (patient_status.get(f'regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}"] == True)
                or forced[f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}"]) and not excluded[f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}"]:
                try:
                    print(f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}",flush=True)
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type=DWI_type, maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask,longitudinal=longitudinal, metrics_dic={'mf_fvf_f0': 'mf', 'mf_fvf_f1': 'mf', 'mf_fvf_tot': 'mf', 'mf_frac_f0': 'mf', 'mf_frac_f1': 'mf', 'mf_frac_csf': 'mf', 'mf_DIFF_ex_f0': 'mf', 'mf_DIFF_ex_f1': 'mf', 'mf_DIFF_ex_tot': 'mf'})
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceFingerprinting_{DWI_type}{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
        if patient_status.get('wm_mask_AP') is not None and patient_status.get('CHARMED_r3') is not None and patient_status["wm_mask_AP"] == True and patient_status["CHARMED_r3"] == True:
            if (not (patient_status.get(f'regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}') is not None and patient_status[f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}"] == True)
                or forced[f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}"]) and not excluded[f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}"]:
                try:
                    print(f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}",flush=True)
                    
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type=DWI_type, maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask,longitudinal=longitudinal,metrics_dic={'CHARMED_r3_FR': 'CHARMED_r3', 'CHARMED_r3_S0.s0': 'CHARMED_r3', 'CHARMED_r3_AIC': 'CHARMED_r3', 'CHARMED_r3_BIC': 'CHARMED_r3', 'CHARMED_r3_Tensor.theta': 'CHARMED_r3','CHARMED_r3_Tensor.d': 'CHARMED_r3', 'CHARMED_r3_Tensor.psi': 'CHARMED_r3', 'CHARMED_r3_Tensor.phi': 'CHARMED_r3', 'CHARMED_r3_Tensor.dperp1': 'CHARMED_r3', 'CHARMED_r3_Tensor.dperp0': 'CHARMED_r3','CHARMED_r3_Tensor.FA': 'CHARMED_r3', 'CHARMED_r3_Tensor.MD': 'CHARMED_r3', 'CHARMED_r3_Tensor.AD': 'CHARMED_r3', 'CHARMED_r3_Tensor.RD': 'CHARMED_r3','CHARMED_r3_w_': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted0.d': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted0.phi': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted0.theta': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted1.d': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted1.phi': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted1.theta': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted2.d': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted2.phi': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted2.theta': 'CHARMED_r3', 'CHARMED_r3_disp': 'CHARMED_r3'})
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"regallDWIToT1wToT1wCommonSpaceCHARMED_r3_{DWI_type}{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)

        if longitudinal>0 and patient_status.get(f'regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}') is not None and patient_status[
            f"regallDWIToT1wToT1wCommonSpace_{DWI_type}{longitudinal_txt}"] == True and patient_status.get(
                f'regallDWIToT1wToT1wCommonSpace') is not None and patient_status["regallDWIToT1wToT1wCommonSpace"] == True:

            from elikopy.utils import get_patient_ref
            p_ref = get_patient_ref(root=folder_path, patient=p, suffix_length=longitudinal)
            reg_T1RefToCommonSpace_precomputed = folder_path + '/subjects/' + p_ref + '/reg/' + 'mapping_T1w_to_T1wCommonSpace.p'
            if os.path.exists(reg_T1RefToCommonSpace_precomputed) and (not (patient_status.get(f'inverseT1{longitudinal_txt}') is not None and patient_status[
                f"inverseT1{longitudinal_txt}"] == True) or forced[
                    f"inverseT1{longitudinal_txt}"]) and not excluded[
                f"inverseT1{longitudinal_txt}"]:
                try:
                    print(f"inverseT1{longitudinal_txt}", flush=True)
                    applyTransformT1withT1Ref(folder_path, p, T1wCommonSpace_filepath= T1_MNI)
                    patient_status[f"inverseT1{longitudinal_txt}"] = True
                except Exception as e:
                    patient_status[f"inverseT1{longitudinal_txt}"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file, "w") as f:
                    json.dump(patient_status, f, indent=6)


        with open(json_status_file,"w") as f:
            json.dump(patient_status, f, indent = 6)
            
        print(patient_status, flush=True)
        
    if error == True:
        raise Exception("One of the subscripts returned an error.")
        