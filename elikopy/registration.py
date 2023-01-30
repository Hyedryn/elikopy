# Imports

import os
import numpy as np
import nibabel as nib
from dipy.viz import regtools
from dipy.segment.mask import applymask
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

import pickle


def getTransform(static_volume_file, moving_volume_file, mask_file=None, onlyAffine=False,
                 diffeomorph=True, sanity_check=False, DWI=False):
    '''


    Parameters
    ----------
    static_volume : 3D array of static volume
    moving_volume : 3D array of moving volume
    diffeomorph : if False then registration is only affine
    sanity_check : if True then prints figures

    Returns
    -------
    mapping : transform operation to send moving_volume to static_volume space

    '''

    static, static_affine = load_nifti(static_volume_file)
    static_grid2world = static_affine

    moving, moving_affine = load_nifti(moving_volume_file)
    moving_grid2world = moving_affine

    if DWI:
        moving = np.squeeze(moving)[..., 0]

    if mask_file is not None:
        mask, mask_affine = load_nifti(mask_file)
        moving = applymask(moving, mask)

    # Affine registration -----------------------------------------------------

    if sanity_check or onlyAffine:

        identity = np.eye(4)
        affine_map = AffineMap(identity,
                               static.shape, static_grid2world,
                               moving.shape, moving_grid2world)
        
        if sanity_check:
        
            resampled = affine_map.transform(moving)

            regtools.overlay_slices(static, resampled, None, 0,
                                    "Static", "Moving", "resampled_0.png")
            regtools.overlay_slices(static, resampled, None, 1,
                                    "Static", "Moving", "resampled_1.png")
            regtools.overlay_slices(static, resampled, None, 2,
                                    "Static", "Moving", "resampled_2.png")

        if onlyAffine:
            return affine_map

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # !!!
    level_iters = [10000, 1000, 100]
    # level_iters = [100, 10, 1]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=c_of_mass.affine)

    transform = RigidTransform3D()
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=translation.affine)

    transform = AffineTransform3D()
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=rigid.affine)

    # Diffeomorphic registration --------------------------

    if diffeomorph:

        metric = CCMetric(3)

        level_iters = [10, 10, 5]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

        mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                               affine.affine)

    else:

        mapping = affine

    if sanity_check:
        transformed = mapping.transform(moving)
        # transformed_static = mapping.transform_inverse(static)

        regtools.overlay_slices(static, transformed, None, 0,
                                "Static", "Transformed", "transformed.png")
        regtools.overlay_slices(static, transformed, None, 1,
                                "Static", "Transformed", "transformed.png")
        regtools.overlay_slices(static, transformed, None, 2,
                                "Static", "Transformed", "transformed.png")

    return mapping


def applyTransform(file_path, mapping, mapping_2=None, mask_file=None, static_file='', output_path='', binary=False,
                   inverse=False, mask_static=None, static_fa_file=''):
    '''


    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.
    mapping : TYPE
        DESCRIPTION.
    static_file : TYPE, optional
        Only necessary if output_path is specified. The default is ''.
    output_path : TYPE, optional
        If entered, saves result at specified location. The default is ''.
    binary : TYPE, optional
        DESCRIPTION. The default is False.
    inverse : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    transformed : TYPE
        DESCRIPTION.

    '''

    moving = nib.load(file_path)
    moving_data = moving.get_fdata()

    if mask_file is not None:
        mask, mask_affine = load_nifti(mask_file)
        moving_data = applymask(moving_data, mask)

    if inverse:
        transformed = mapping.transform_inverse(moving_data)
    else:
        transformed = mapping.transform(moving_data)

    if mapping_2 is not None:
        if inverse:
            transformed = mapping_2.transform_inverse(transformed)
        else:
            transformed = mapping_2.transform(transformed)

    if binary:
        transformed[transformed > .5] = 1
        transformed[transformed <= .5] = 0

    if len(output_path) > 0:
        print("OUT", output_path)
        static = nib.load(static_file)

        static_fa = nib.load(static_fa_file)

        print(static_fa.header)
        print(static.header)

        if mask_static is not None:
            mask_static_data, mask_static_affine = load_nifti(mask_static)
            transformed = applymask(transformed, mask_static_data)

        out = nib.Nifti1Image(transformed, static.affine, header=static_fa.header)
        out.to_filename(output_path)

    else:
        return transformed


def applyTransformToAllMapsInFolder(input_folder, output_folder, mapping, mapping_2=None, static_file=None,
                                    mask_file=None,
                                    keywordList=[], inverse=False, mask_static=None, static_fa_file=''):
    '''


    Parameters
    ----------
    Patient_static : string
    Patient_moving : string
    transform : transform object
    folder : folder containing ROIs of Patient_static

    Returns
    -------
    None.

    '''

    for filename in os.listdir(input_folder):
        if all(keyword in filename for keyword in keywordList):
            # print(filename)
            try:
                applyTransform(input_folder + filename, mapping, mapping_2=mapping_2, static_file=static_file,
                               output_path=output_folder + filename, mask_file=mask_file, binary=False, inverse=inverse,
                               mask_static=mask_static, static_fa_file=static_fa_file)
            except TypeError:
                continue


def regToT1fromB0(reg_path, T1_subject, DWI_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI, T1_MNI,
                  mask_static, FA_MNI):
    if os.path.exists(reg_path + 'mapping_DWI_B0_to_T1.p'):
        with open(reg_path + 'mapping_DWI_B0_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        mapping_DWI_to_T1 = getTransform(T1_subject, DWI_subject, mask_file=mask_file, onlyAffine=False,
                                         diffeomorph=False, sanity_check=False, DWI=True)
        with open(reg_path + 'mapping_DWI_B0_to_T1.p', 'wb') as handle:
            pickle.dump(mapping_DWI_to_T1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not (os.path.exists(folderpath + "/subjects/" + p + "/masks/reg/")):
        try:
            os.makedirs(folderpath + "/subjects/" + p + "/masks/reg/")
        except OSError:
            print("Creation of the directory %s failed" % folderpath + "/subjects/" + p + "/masks/reg/")

    for maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1"]:
        in_mask_path = folderpath + "/subjects/" + p + "/masks/" + p + "_" + maskType + ".nii.gz"
        reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_B0_" + maskType + ".nii.gz"
        if os.path.exists(in_mask_path):
            applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                           output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                           mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():
        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_B0/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                        static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                        mask_static=mask_static, static_fa_file=FA_MNI)


def regToT1fromWMFOD(reg_path, T1_subject, WM_FOD_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI,
                     T1_MNI, mask_static, FA_MNI):
    if os.path.exists(reg_path + 'mapping_DWI_WMFOD_to_T1.p'):
        with open(reg_path + 'mapping_DWI_WMFOD_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        mapping_DWI_to_T1 = getTransform(T1_subject, WM_FOD_subject, mask_file=mask_file, onlyAffine=False,
                                         diffeomorph=False, sanity_check=False, DWI=True)
        with open(reg_path + 'mapping_DWI_WMFOD_to_T1.p', 'wb') as handle:
            pickle.dump(mapping_DWI_to_T1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not (os.path.exists(folderpath + "/subjects/" + p + "/masks/reg/")):
        try:
            os.makedirs(folderpath + "/subjects/" + p + "/masks/reg/")
        except OSError:
            print("Creation of the directory %s failed" % folderpath + "/subjects/" + p + "/masks/reg/")

    for maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1"]:
        in_mask_path = folderpath + "/subjects/" + p + "/masks/" + p + "_" + maskType + ".nii.gz"
        reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_WMFOD_" + maskType + ".nii.gz"
        if os.path.exists(in_mask_path):
            applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                           output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                           mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():

        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_WMFOD/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                        static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                        mask_static=mask_static, static_fa_file=FA_MNI)


def regToT1fromAP(reg_path, T1_subject, AP_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI, T1_MNI,
                  mask_static, FA_MNI):
    if os.path.exists(reg_path + 'mapping_DWI_AP_to_T1.p'):
        with open(reg_path + 'mapping_DWI_AP_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        mapping_DWI_to_T1 = getTransform(T1_subject, AP_subject, mask_file=mask_file, onlyAffine=False,
                                         diffeomorph=False, sanity_check=False, DWI=False)
        with open(reg_path + 'mapping_DWI_AP_to_T1.p', 'wb') as handle:
            pickle.dump(mapping_DWI_to_T1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not (os.path.exists(folderpath + "/subjects/" + p + "/masks/reg/")):
        try:
            os.makedirs(folderpath + "/subjects/" + p + "/masks/reg/")
        except OSError:
            print("Creation of the directory %s failed" % folderpath + "/subjects/" + p + "/masks/reg/")

    for maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1"]:
        in_mask_path = folderpath + "/subjects/" + p + "/masks/" + p + "_" + maskType + ".nii.gz"
        reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_AP_" + maskType + ".nii.gz"
        if os.path.exists(in_mask_path):
            applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                           output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                           mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():

        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_AP/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                        static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                        mask_static=mask_static, static_fa_file=FA_MNI)


def regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type="AP", maskType=None, T1_filepath=None, T1wCommonSpace_filepath="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz", T1wCommonSpaceMask_filepath="${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz", metrics_dic={'_FA': 'dti', 'RD': 'dti', 'AD': 'dti', 'MD': 'dti'}):
    preproc_folder = folder_path + '/subjects/' + p + '/dMRI/preproc/'
    T1_CommonSpace = os.path.expandvars(T1wCommonSpace_filepath)
    FA_MNI = os.path.expandvars('${FSLDIR}/data/standard/FSL_HCP1065_FA_1mm.nii.gz')

    assert maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1", None], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1, None"

    assert DWI_type in ["AP", "WMFOD", "BO"], "The DWI_type parameter must be one of the following : AP, WMFOD, BO"

    mask_path = ""
    if maskType is not None and os.path.isfile(folder_path + '/subjects/' + p + "/masks/" + p + '_' + maskType + '.nii.gz'):
        mask_path = folder_path + '/subjects/' + p + "/masks/" + p + '_' + maskType + '.nii.gz'
    else:
        mask_path = None


    if T1_filepath is None:
        T1_subject = folder_path + '/subjects/' + p + '/T1/' + p + "_T1_brain.nii.gz"
    elif os.path.exists(os.path.join(T1_filepath,p + ".nii.gz")):
        T1_subject = os.path.join(T1_filepath,p + ".nii.gz")
    elif os.path.exists(os.path.join(T1_filepath,p + "_T1.nii.gz")):
        T1_subject = os.path.join(T1_filepath,p + "_T1.nii.gz")
    elif os.path.exists(os.path.join(T1_filepath,p + "_T1_brain.nii.gz")):
        T1_subject = os.path.join(T1_filepath,p + "_T1_brain.nii.gz")
    else:
        raise ValueError("No T1 file found in the T1_filepath folder")

    DWI_subject = preproc_folder + p + "_dmri_preproc.nii.gz"
    AP_subject = folder_path + '/subjects/' + p + '/masks/' + p + '_ap.nii.gz'
    WM_FOD_subject = folder_path + '/subjects/' + p + '/dMRI/ODF/MSMT-CSD/' + p + "_MSMT-CSD_WM_ODF.nii.gz"

    reg_path = folder_path + '/subjects/' + p + '/reg/'
    if not(os.path.exists(reg_path)):
        try:
            os.makedirs(reg_path)
        except OSError:
            print ("Creation of the directory %s failed" % reg_path)

    print("Start of getTransform for T1 to T1_MNI")
    mask_file = None

    if os.path.exists(reg_path + 'mapping_T1w_to_T1wCommonSpace.p'):
        with open(reg_path + 'mapping_T1w_to_T1wCommonSpace.p', 'rb') as handle:
            mapping_T1w_to_T1wCommonSpace = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        mapping_T1w_to_T1wCommonSpace = getTransform(T1_CommonSpace, T1_subject, mask_file=mask_file, onlyAffine=False, diffeomorph=True,
                                           sanity_check=False, DWI=False)
        with open(reg_path + 'mapping_T1w_to_T1wCommonSpace.p', 'wb') as handle:
            pickle.dump(mapping_T1w_to_T1wCommonSpace, handle, protocol=pickle.HIGHEST_PROTOCOL)

    applyTransform(T1_subject, mapping_T1w_to_T1wCommonSpace, mapping_2=None, mask_file=None, static_file=T1_CommonSpace,
                   output_path=folder_path + '/subjects/' + p + '/T1/' + p + '_T1_MNI_FS.nii.gz', binary=False,
                   inverse=False, static_fa_file=T1_CommonSpace)

    print("Start of getTransform for DWI to T1")
    if T1wCommonSpaceMask_filepath is not None:
        mask_static = os.path.expandvars(T1wCommonSpaceMask_filepath)
    else:
        mask_static = None

    if DWI_type == "B0":
        regToT1fromB0(reg_path, T1_subject, DWI_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI)
    elif DWI_type == "WMFOD":
        regToT1fromWMFOD(reg_path, T1_subject, WM_FOD_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI)
    elif DWI_type == "AP":
        regToT1fromAP(reg_path, T1_subject, AP_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI)
    else:
        print("DWI_type not recognized")

    print("End of DWI registration")


def regallFAToMNI(folderpath, p, metrics_dic={'_FA': 'dti', 'RD': 'dti', 'AD': 'dti', 'MD': 'dti'}):

    FA_MNI = os.path.expandvars('${FSLDIR}/data/standard/FSL_HCP1065_FA_1mm.nii.gz')
    static_volume_file = FA_MNI
    moving_volume_file = folderpath + '/subjects/' + p + '/dMRI/microstructure/dti/' + p + '_FA.nii.gz'
    mask_file = folderpath + '/subjects/' + p + '/masks/' + p + '_brain_mask.nii.gz'
    print("Start of getTransform")
    mapping = getTransform(static_volume_file, moving_volume_file, mask_file=mask_file, onlyAffine=False,
                           diffeomorph=False, sanity_check=True)

    for key, value in metrics_dic.items():

        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_MNI/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        applyTransformToAllMapsInFolder(input_folder, output_folder, mapping, static_volume_file, mask_file=mask_file,
                                        keywordList=[p, key], inverse=False)


