.. _preprocessing-dmri:

========================================
Preprocessing of diffusion images
========================================

The preprocessing stage aims at correcting the non idealities affecting the diffusion data
before computing the diffusion metrics. With the exception of skull stripping, all processing steps are optional and can be applied at the
user discretion.

.. image:: pictures/preprocessing.PNG
	:width: 800
	:alt: Overview of the main preprocessing steps.

To preproccess the dMRI data, the following line of code is used. However, this results in the default preprocessing that encompass only the skull striping step. To perform more advanced preprocessing, we need to dive into the arguments of the preproc function.

.. code-block:: python

	study.preproc()


The arguments of the preproc function are given in the API : LINK.
In this page, only the main arguments are explained in order to grasp the key aspects of preprocessing using ElikoPy.

-------
Reslice
-------

Description
^^^^^^^^^^^

If the raw data is not in its ’native’ resolution, a reslicing process might be required. Usually, the
MRI scanner performs automatic interpolation on the data in order to beautify the data
since clinicians usually have a preference for high resolution images. However, the intrinsic
resolution is not augmented by this interpolation. While somewhat useful to clinicians, the
interpolation is usually not desirable for research. Using it means more computation time
and uncorrelated noise becoming correlated which reduces the performances of MPPCA
denoising algorithms. Moreover, interpolation is not desirable when performing Gibbs
ringing correction. Reslicing is therefore a way to mitigate the
effect of mandatory interpolation during the acquisition.


Related parameters
^^^^^^^^^^^^^^^^^^
The reslicing step during the preprocessing can be activated using the reslice argument.

.. code-block:: python

	study.preproc(reslice=True)

----------------
Brain Extraction
----------------

Description
^^^^^^^^^^^

.. image:: pictures/preproc_bet.jpg
	:width: 800
	:alt: Original b0 images and binary mask obtained using median_otsu are shown in the left and middle panels, while the thresholded histogram used by median otsu is shown in the right panel.

The brain is extracted from the skull and other tissues surrounding the brain to increase
the processing efficiency of subsequent steps and it is generally required before using
other image processing algorithms. At the end of the preprocessing, a final brain mask readjusted in regard of all the applied
preprocessing steps is also provided as output.

The mask is computed using median_otsu from DiPy.

Related parameters
^^^^^^^^^^^^^^^^^^

The brain extraction is the only mandatory step and cannot be disabled. However, it is possible to change the parameters of the method

* **bet_median_radius** - Radius (in voxels) of the applied median filter during brain extraction. default=2
* **bet_numpass** - Number of pass of the median filter during brain extraction. default=1
* **bet_dilate** - Number of iterations for binary dilation during brain extraction. default=2

.. code-block:: python

	study.preproc(bet_median_radius=2, bet_numpass=2, bet_dilate=2)

---------------
MPPCA Denoising
---------------

Description
^^^^^^^^^^^

.. image:: pictures/preproc_mppca.jpg
	:width: 800
	:alt: Original and denoised b0 images are shown in the left and middle panels, while the difference between these images is shown in the right panel. An unstructured spatial distribution of the right image indicates extraction of random thermal noise.

To reduce Rician noise typically found in MR images, the input images are denoised
using the Marchenko-Pastur PCA technique as implemented in DiPy. Since the noise in
diffusion data is spatially dependent in the case of multichannel receive coils, Principal component analysis of Marchenko-Pastur (MPPCA) noise-only
distribution provides an accurate and fast method of noise evaluation and reduction. This methods has been chosen since it is a fast denoising algorithm
that does not blur the image or create artifact.

Related parameters
^^^^^^^^^^^^^^^^^^

The denoising step during the preprocessing can be activated using the denoising argument.

.. code-block:: python

	study.preproc(denoising=True)

------------------------
Gibbs Ringing Correction
------------------------

Description
^^^^^^^^^^^

.. image:: pictures/preproc_gibbs.jpg
	:width: 800
	:alt: Gibbs ringing correction, uncorrected and b0 images corrected for Gibbs ringing are shown in the left and middle panels, while the difference between these images is shown in the right panel. Gibbs ringing artifacts typically occur at interfaces with sharp changes in intensity.

In general, in the context of diffusion-weighted imaging, derived diffusion-based estimates
are greatly affected by Gibbs oscillations. To correct for this,
gibbs_removal from DiPy is used. This algorithm models the truncation of k-space as a
convolution with a sinc-function in the image space. The severity of ringing artifacts thus
depends on how the sampling of the sinc function occurs. The gibbs_removal function
reinterpolate the image based on local, subvoxel-shifts to sample the ringing pattern at
the zero-crossings of the oscillating sinc-function.

Related parameters
^^^^^^^^^^^^^^^^^^

-------------------------------
Susceptibility field estimation
-------------------------------

Description
^^^^^^^^^^^

Related parameters
^^^^^^^^^^^^^^^^^^

--------------------------
Eddy and motion correction
--------------------------

Description
^^^^^^^^^^^

Related parameters
^^^^^^^^^^^^^^^^^^

---------------------
Bias Field Correction
---------------------

Description
^^^^^^^^^^^

Related parameters
^^^^^^^^^^^^^^^^^^

------
Report
------

