import numpy as np
from dipy.io.image import load_nifti_data
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
import matplotlib.pyplot as plt
import os



def preprocess_data(Patient_Input_Directory, B_files_Input_Directory, Patient_basename):
  os.chdir(Patient_Input_Directory)
  data = load_nifti_data(Patient_basename + '.nii.gz')

  os.chdir(B_files_Input_Directory)
  bvals, bvecs = read_bvals_bvecs(Patient_basename + '.bval', Patient_basename + '.bvec')
  gtab = gradient_table(bvals, bvecs, atol=1)

  maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,
                               autocrop=False, dilate=1)

  tenmodel = dti.TensorModel(gtab)
  tenfit = tenmodel.fit(maskdata)

  FA = tenfit.fa
  MD = tenfit.md

  plt.imshow(FA[:, :, 30], cmap='Greys')
  plt.colorbar()

my_function("Emil", "Tobias", "Linus")
