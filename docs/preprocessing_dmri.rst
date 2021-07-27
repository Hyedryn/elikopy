.. _preprocessing-dmri:

========================================
Project as an easy way to manage a study
========================================

The preprocessing stage aims at correcting the non idealities affecting the diffusion data
before computing the diffusion metrics. With the exception of skull stripping, all processing steps are optional and can be applied at the
user discretion.

-------
Reslice
-------

Description
^^^^^^^^^^^

If the raw data is not in its ’native’ resolution, a reslicing process is required. Usually, the
MRI scanner performs automatic interpolation on the data in order to beautify the data
since clinicians usually have a preference for high resolution images. However, the intrinsic
resolution is not augmented by this interpolation. While somewhat useful to clinicians, the
interpolation is usually not desirable for research. Using it means more computation time
and uncorrelated noise becoming correlated which reduces the performances of MPPCA
denoising algorithms. Moreover, interpolation is not desirable when performing Gibbs
ringing correction. Reslicing is therefore a way to mitigate the
effect of mandatory interpolation during the acquisition. Furthermore, it allows to ensure
isotropic voxels which facilitate most tractography algorithms.


Related parameters
^^^^^^^^^^^^^^^^^^


----------------
Brain Extraction
----------------

Description
^^^^^^^^^^^

.. image:: pictures/preproc_bet.jpg
	:width: 800
	:alt: Original b0 images and binary mask obtained with using median_otsu are shown in the left and middle panels, while the thresholded histogram used by median otsu is shown in the right panel.

	Original b0 images and binary mask obtained with using median_otsu are shown in the left and middle panels, while the thresholded histogram used by median otsu is shown in the right panel.

The brain is extracted from the skull and other tissues surrounding the brain to increase
the processing efficiency of subsequent steps. In addition to the gain of processing speed
for several algorithms, removal of non brain tissues is a fundamental step in enabling
the processing of diffusion images since brain regions must generally be skull-stripped before using
other image processing algorithms.

At the end of the preprocessing, a final brain mask readjusted in regard of all the applied
preprocessing steps is also provided as output.

The mask is computed using median_otsu from DiPy. Median_otsu has
been selected since the method is fast, robust, designed for DW-MRI images and directly
available in Python. First, a non linear median filter is applied to reduce the noise and
improve the brain segmentation. Second, the Otsu’s method is used to
separate the brain (foreground) from the background. Using the intensity histogram of the
brain along multiple volumes, the algorithm returns an intensity threshold that separates
the voxels between foreground and background. This threshold is computed by maximizing
the inter-class variance. The skull gives rise to a weak signal in diffusion MRI making this
technique possible, however it is not applicable on T1 images. It is noteworthy that the
performance and quality of the image extraction tools depend on the quality of the MR
images, therefore the first computed mask (before correction) is dilated using mathematical
morphology.

Related parameters
^^^^^^^^^^^^^^^^^^


---------------
MPPCA Denoising
---------------

Description
^^^^^^^^^^^

.. image:: pictures/preproc_mppca.jpg
	:width: 800
	:alt: Original and denoised b0 images are shown in the left and middle panels, while the difference between these images is shown in the right panel. An unstructured spatial distribution of the right image indicates extraction of random thermal noise.

	Original and denoised b0 images are shown in the left and middle panels, while the difference between these images is shown in the right panel. An unstructured spatial distribution of the right image indicates extraction of random thermal noise.

To reduce Rician noise typically found in MR images, the input images are denoised
using the Marchenko-Pastur PCA technique as implemented in DiPy. Since the noise in
diffusion data is spatially dependent in the case of multichannel receive coils, Principal component analysis of Marchenko-Pastur (MPPCA) noise-only
distribution provides an accurate and fast method of noise evaluation and reduction. This methods has been chosen since it is a fast denoising algorithm
that does not blur the image or create artifact.

Related parameters
^^^^^^^^^^^^^^^^^^


------------------------
Gibbs Ringing Correction
------------------------

Description
^^^^^^^^^^^

.. image:: pictures/preproc_gibbs.jpg
	:width: 800
	:alt: Gibbs ringing correction, uncorrected and b0 images corrected for Gibbs ringing are shown in the left and middle panels, while the difference between these images is shown in the right panel. Gibbs ringing artifacts typically occur at interfaces with sharp changes in intensity.

	Gibbs ringing correction, uncorrected and b0 images corrected for Gibbs ringing are shown in the left and middle panels, while the difference between these images is shown in the right panel. Gibbs ringing artifacts typically occur at interfaces with sharp changes in intensity.


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

Description
^^^^^^^^^^^

Related parameters
^^^^^^^^^^^^^^^^^^

