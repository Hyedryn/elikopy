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

from elikopy.utils import get_patient_ref


def getTransform(static_volume_file, moving_volume_file, mask_file=None, onlyAffine=False,
                 diffeomorph=True, sanity_check=False, DWI=False, affine_map=None):
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

        if affine_map is None:
            affine_map = np.eye(4)
        affine_map = AffineMap(affine_map,
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


def applyTransform(file_path, mapping, mapping_2=None, mapping_3=None, mask_file=None, static_file='', output_path='', binary=False,
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
    print("Applying transform to", file_path)
    moving = nib.load(file_path)
    moving_data = moving.get_fdata()
    print("Moving data shape:", moving_data.shape)

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

    if mapping_3 is not None:
        if inverse:
            transformed = mapping_3.transform_inverse(transformed)
        else:
            transformed = mapping_3.transform(transformed)

    if binary:
        transformed[transformed > .5] = 1
        transformed[transformed <= .5] = 0

    if len(output_path) > 0:
        static = nib.load(static_file)

        static_fa = nib.load(static_fa_file)

        if mask_static is not None:
            mask_static_data, mask_static_affine = load_nifti(mask_static)
            transformed = applymask(transformed, mask_static_data)

        out = nib.Nifti1Image(transformed, static.affine, header=static_fa.header)
        out.to_filename(output_path)

    else:
        return transformed


def applyTransformToAllMapsInFolder(input_folder, output_folder, mapping, mapping_2=None, mapping_3=None, static_file=None,
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
                applyTransform(input_folder + filename, mapping, mapping_2=mapping_2, mapping_3=mapping_3, static_file=static_file,
                               output_path=output_folder + filename, mask_file=mask_file, binary=False, inverse=inverse,
                               mask_static=mask_static, static_fa_file=static_fa_file)
            except TypeError:
                continue



def regToT1fromB0FSL(reg_path, T1_subject, DWI_B0_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI, T1_MNI,
                  mask_static, FA_MNI, longitudinal_transform=None):
    if os.path.exists(reg_path + 'mapping_DWI_B0FSL_to_T1.p'):
        with open(reg_path + 'mapping_DWI_B0FSL_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)

        T1_subject_raw = os.path.join(folderpath, 'subjects', p, 'T1', p + '_T1.nii.gz')
        T1_brain_subject = os.path.join(folderpath, 'subjects', p, 'T1', p + '_T1_brain.nii.gz')
        b0fsl_reg_path = os.path.join(folderpath, 'subjects', p, 'reg', 'B0FSL_to_T1')
        os.makedirs(b0fsl_reg_path, exist_ok=True)
        cmd = f"epi_reg --epi={DWI_B0_subject} --t1={T1_subject_raw} --t1brain={T1_brain_subject} --out={b0fsl_reg_path}/{p}_B0toT1 "
        cmd_part2 = f"c3d_affine_tool -ref {T1_subject_raw} -src {DWI_B0_subject} {b0fsl_reg_path}/{p}_B0toT1.mat -fsl2ras -oitk {b0fsl_reg_path}/{p}_B0toT1_ANTS.txt -o {b0fsl_reg_path}/{p}_B0toT1_ANTS.mat"
        cmd += " && " + cmd_part2
        import subprocess
        process = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        affine_map_fslb0 = np.loadtxt(f"{b0fsl_reg_path}/{p}_B0toT1_ANTS.mat")
        mapping_DWI_to_T1 = getTransform(T1_subject, DWI_B0_subject, mask_file=mask_file, onlyAffine=True, diffeomorph=False, sanity_check=False, DWI=False, affine_map=affine_map_fslb0)
        with open(reg_path + 'mapping_DWI_B0FSL_to_T1.p', 'wb') as handle:
            pickle.dump(mapping_DWI_to_T1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not (os.path.exists(folderpath + "/subjects/" + p + "/masks/reg/")):
        try:
            os.makedirs(folderpath + "/subjects/" + p + "/masks/reg/")
        except OSError:
            print("Creation of the directory %s failed" % folderpath + "/subjects/" + p + "/masks/reg/")

    for maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1"]:
        in_mask_path = folderpath + "/subjects/" + p + "/masks/" + p + "_" + maskType + ".nii.gz"
        if os.path.exists(in_mask_path):
            if longitudinal_transform is not None:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_B0FSL_" + maskType + "_longitudinal.nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=longitudinal_transform, mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI,
                               output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                               mask_static=mask_static, static_fa_file=FA_MNI)
            else:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_B0FSL_" + maskType + ".nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                               output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                               mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():
        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        if longitudinal_transform is not None:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_B0FSL_longitudinal/'
        else:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_B0FSL/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        if longitudinal_transform is not None:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=longitudinal_transform,
                                            mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI, mask_file=mask_file,
                                            keywordList=[p, key], inverse=False, mask_static=mask_static,
                                            static_fa_file=FA_MNI)
        else:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                            static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                            mask_static=mask_static, static_fa_file=FA_MNI)




def regToT1fromB0(reg_path, T1_subject, DWI_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI, T1_MNI,
                  mask_static, FA_MNI, longitudinal_transform=None):
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
        if os.path.exists(in_mask_path):
            if longitudinal_transform is not None:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_B0_" + maskType + "_longitudinal.nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=longitudinal_transform, mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI,
                               output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                               mask_static=mask_static, static_fa_file=FA_MNI)
            else:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_B0_" + maskType + ".nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                               output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                               mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():
        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        if longitudinal_transform is not None:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_B0_longitudinal/'
        else:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_B0/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        if longitudinal_transform is not None:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=longitudinal_transform,
                                            mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI, mask_file=mask_file,
                                            keywordList=[p, key], inverse=False, mask_static=mask_static,
                                            static_fa_file=FA_MNI)
        else:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                            static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                            mask_static=mask_static, static_fa_file=FA_MNI)


def regToT1fromWMFOD(reg_path, T1_subject, WM_FOD_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI,
                     T1_MNI, mask_static, FA_MNI, longitudinal_transform=None):
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
        if os.path.exists(in_mask_path):
            if longitudinal_transform is not None:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_WMFOD_" + maskType + "_longitudinal.nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=longitudinal_transform, mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI,
                                 output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                                 mask_static=mask_static, static_fa_file=FA_MNI)
            else:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_WMFOD_" + maskType + ".nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                               output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                               mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():

        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        if longitudinal_transform is not None:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_WMFOD_longitudinal/'
        else:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_WMFOD/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        if longitudinal_transform is not None:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=longitudinal_transform,
                                            mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI, mask_file=mask_file,
                                            keywordList=[p, key], inverse=False, mask_static=mask_static,
                                            static_fa_file=FA_MNI)
        else:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                            static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                            mask_static=mask_static, static_fa_file=FA_MNI)


def regToT1fromAP(reg_path, T1_subject, AP_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI, T1_MNI,
                  mask_static, FA_MNI, longitudinal_transform=None):
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

        if os.path.exists(in_mask_path):
            if longitudinal_transform is not None:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_AP_" + maskType + "_longitudinal.nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=longitudinal_transform, mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI,
                               output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                               mask_static=mask_static, static_fa_file=FA_MNI)
            else:
                reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_AP_" + maskType + ".nii.gz"
                applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                               output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                               mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():

        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        if longitudinal_transform is not None:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_AP_longitudinal/'
        else:
            output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_AP/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        if longitudinal_transform is not None:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=longitudinal_transform,
                                            mapping_3=mapping_T1_to_T1MNI, static_file=T1_MNI, mask_file=mask_file,
                                            keywordList=[p, key], inverse=False, mask_static=mask_static,
                                            static_fa_file=FA_MNI)
        else:
            applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                            static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                            mask_static=mask_static, static_fa_file=FA_MNI)


def regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type="B0FSL", maskType="brain_mask", T1_filepath=None, T1wCommonSpace_filepath="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz", T1wCommonSpaceMask_filepath="${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz", metrics_dic={'_FA': 'dti', 'RD': 'dti', 'AD': 'dti', 'MD': 'dti'}, longitudinal=False):
    preproc_folder = folder_path + '/subjects/' + p + '/dMRI/preproc/'
    T1_CommonSpace = os.path.expandvars(T1wCommonSpace_filepath)
    FA_MNI = os.path.expandvars('${FSLDIR}/data/standard/FSL_HCP1065_FA_1mm.nii.gz')

    assert maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1", None], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1, None"

    assert DWI_type in ["AP", "WMFOD", "B0", "B0FSL"], "The DWI_type parameter must be one of the following : AP, WMFOD, B0, B0FSL"

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
    DWI_B0_subject = os.path.join(folder_path, 'subjects', p, 'dMRI', 'preproc', p + '_dwiref.nii.gz')

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


    if longitudinal != False and longitudinal>0:
        print("[Longitudinal Registration] Start of getTransform for T1 to T1_ref")
        p_ref = get_patient_ref(root=folder_path, patient=p, suffix_length=longitudinal)
        T1_ref_subject = folder_path + '/subjects/' + p_ref + '/T1/' + p_ref + "_T1_brain.nii.gz"

        if os.path.exists(reg_path + 'mapping_T1w_to_T1wRef.p'):
            with open(reg_path + 'mapping_T1w_to_T1wRef.p', 'rb') as handle:
                mapping_T1w_to_T1wRef = pickle.load(handle)
        else:
            if not (os.path.exists(reg_path)):
                try:
                    os.makedirs(reg_path)
                except OSError:
                    print("Creation of the directory %s failed" % reg_path)
            if p == p_ref:
                print("[Longitudinal Registration] Skipping, longitudinal registration with the same subject")
                mapping_T1w_to_T1wRef = getTransform(T1_ref_subject, T1_subject, mask_file=mask_file,
                                                             onlyAffine=True, diffeomorph=False,
                                                             sanity_check=False, DWI=False)
            else:
                mapping_T1w_to_T1wRef = getTransform(T1_ref_subject, T1_subject, mask_file=mask_file,
                                                             onlyAffine=False, diffeomorph=False,
                                                             sanity_check=False, DWI=False)
            with open(reg_path + 'mapping_T1w_to_T1wRef.p', 'wb') as handle:
                pickle.dump(mapping_T1w_to_T1wRef, handle, protocol=pickle.HIGHEST_PROTOCOL)

        applyTransform(T1_subject, mapping_T1w_to_T1wRef, mapping_2=None, mask_file=None,
                       static_file=T1_ref_subject,
                       output_path=folder_path + '/subjects/' + p + '/T1/' + p + '_space-T1Ref_type-brain_T1.nii.gz', binary=False,
                       inverse=False, static_fa_file=T1_ref_subject)

        reg_T1RefToCommonSpace_precomputed = folder_path + '/subjects/' + p_ref + '/reg/' + 'mapping_T1w_to_T1wCommonSpace.p'
        if not os.path.exists(reg_T1RefToCommonSpace_precomputed):
            raise ValueError("No mapping_T1w_to_T1wCommonSpace.p file found in the reg folder of the reference subject")
        with open(reg_T1RefToCommonSpace_precomputed, 'rb') as handle:
            mapping_T1w_to_T1wCommonSpace = pickle.load(handle)
    else:
        mapping_T1w_to_T1wRef = None



    print("Start of getTransform for DWI to T1")
    if T1wCommonSpaceMask_filepath is not None:
        mask_static = os.path.expandvars(T1wCommonSpaceMask_filepath)
    else:
        mask_static = None

    if DWI_type == "B0":
        regToT1fromB0(reg_path, T1_subject, DWI_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI, longitudinal_transform=mapping_T1w_to_T1wRef)
    elif DWI_type == "WMFOD":
        regToT1fromWMFOD(reg_path, T1_subject, WM_FOD_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI, longitudinal_transform=mapping_T1w_to_T1wRef)
    elif DWI_type == "AP":
        regToT1fromAP(reg_path, T1_subject, AP_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI, longitudinal_transform=mapping_T1w_to_T1wRef)
    elif DWI_type == "B0FSL":
        regToT1fromB0FSL(reg_path, T1_subject, DWI_B0_subject, mask_path, metrics_dic, folder_path, p,
                      mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI,
                      longitudinal_transform=mapping_T1w_to_T1wRef)

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


