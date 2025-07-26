import os
import pickle
from elikopy.utils import get_patient_ref
from elikopy.registration import applyTransform

def applyTransformT1withT1Ref(folder_path, p, longitudinal=2, T1wCommonSpace_filepath="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"):
    print("[applyTransformT1withT1Ref] Processing subject " + p)
    T1_subject = folder_path + '/subjects/' + p + '/T1/' + p + "_T1_brain.nii.gz"
    reg_path = folder_path + '/subjects/' + p + '/reg/'

    T1_CommonSpace = os.path.expandvars(T1wCommonSpace_filepath)

    if not os.path.exists(reg_path + 'mapping_T1w_to_T1wRef.p'):
        raise ValueError("No mapping_T1w_to_T1wRef.p file found in the reg folder")
    with open(reg_path + 'mapping_T1w_to_T1wRef.p', 'rb') as handle:
        mapping_T1w_to_T1wRef = pickle.load(handle)

    p_ref = get_patient_ref(root=folder_path, patient=p, suffix_length=longitudinal)
    reg_T1RefToCommonSpace_precomputed = folder_path + '/subjects/' + p_ref + '/reg/' + 'mapping_T1w_to_T1wCommonSpace.p'
    if not os.path.exists(reg_T1RefToCommonSpace_precomputed):
        raise ValueError("No mapping_T1w_to_T1wCommonSpace.p file found in the reg folder of the reference subject")
    with open(reg_T1RefToCommonSpace_precomputed, 'rb') as handle:
        mapping_T1w_to_T1wCommonSpace = pickle.load(handle)

    applyTransform(T1_subject, mapping_T1w_to_T1wRef, mapping_2=mapping_T1w_to_T1wCommonSpace, mask_file=None,
                   static_file=T1_CommonSpace,
                   output_path=folder_path + '/subjects/' + p + '/T1/' + p + '_space-T1CommonSpace_transform-T1RefT1CommonSpace_type-brain_T1.nii.gz',
                   binary=False,
                   inverse=False, static_fa_file=T1_CommonSpace)