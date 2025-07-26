import os
import json


def cleanT1Derivatives(folder_path, p):

    T1_subject_path = os.path.join(folder_path, "subjects", p, "T1")
    reg_subject_path = os.path.join(folder_path, "subjects", p, "reg")
    mask_reg_subject_path = os.path.join(folder_path, "subjects", p, "masks", "reg")
    mask_subject_path = os.path.join(folder_path, "subjects", p, "masks")

    # Delete content of reg folder
    if os.path.exists(reg_subject_path):
        for f in os.listdir(reg_subject_path):
            os.remove(os.path.join(reg_subject_path, f))

    # Delete content of mask/reg folder
    if os.path.exists(mask_reg_subject_path):
        for f in os.listdir(mask_reg_subject_path):
            os.remove(os.path.join(mask_reg_subject_path, f))

    # Delete content of T1 folder except for the <p>_T1.nii.gz and <p>_T1.json files
    if os.path.exists(T1_subject_path):
        for f in os.listdir(T1_subject_path):
            if f != p + "_T1.nii.gz" and f != p + "_T1.json":
                os.remove(os.path.join(T1_subject_path, f))

    if os.path.exists(os.path.join(mask_subject_path, p + "_ap.nii.gz")):
        os.remove(os.path.join(mask_subject_path, p + "_ap.nii.gz"))
    if os.path.exists(os.path.join(mask_subject_path, p + "_segmentation_AP.nii.gz")):
        os.remove(os.path.join(mask_subject_path, p + "_segmentation_AP.nii.gz"))
    if os.path.exists(os.path.join(mask_subject_path, p + "_segmentation_FSL_T1.nii.gz")):
        os.remove(os.path.join(mask_subject_path, p + "_segmentation_FSL_T1.nii.gz"))
    if os.path.exists(os.path.join(mask_subject_path, p + "_wm_mask_AP.nii.gz")):
        os.remove(os.path.join(mask_subject_path, p + "_wm_mask_AP.nii.gz"))
    if os.path.exists(os.path.join(mask_subject_path, p + "_wm_mask_FSL_T1.nii.gz")):
        os.remove(os.path.join(mask_subject_path, p + "_wm_mask_FSL_T1.nii.gz"))

    to_update = ["regallDWIToT1wToT1wCommonSpaceNoddi", "regallDWIToT1wToT1wCommonSpaceFingerprinting",
                 "regallDWIToT1wToT1wCommonSpaceCHARMED_r3", "regallDWIToT1wToT1wCommonSpaceDTI",
                 "regallDWIToT1wToT1wCommonSpace", "wm_mask_AP", "inverseTransformAtlas", "wm_mask_FSL_T1"]
    if os.path.exists(os.path.join(folder_path, "subjects", p, f"{p}_status.json")):
        with open(os.path.join(folder_path, "subjects", p, f"{p}_status.json"), "r") as f:
            data = json.load(f)
        for key in to_update:
            data[key] = False
        with open(os.path.join(folder_path, "subjects", p, f"{p}_status.json"), "w") as f:
            json.dump(data, f)
