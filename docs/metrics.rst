.. _metrics:

============================
Diffusion models and metrics
============================

There are currently four diffusion models that are supported by the ElikoPy pipeline. These models are listed below with their accessible parameters.

* **Diffusion Tensor Imaging (DTI)**
* **Neurite Orientation Dispersion and Density Imaging (NODDI)**
    - lambda_iso_diff - isotropic diffusivity for the CSF model
    - lambda_par_diff - axial diffusivity of the intra-neurite space
* **DIstribution of 3D Anisotropic MicrOstructural eNvironment in Diffusion compartment imaging (DIAMOND)**
* **Microstructure Fingerprinting (MF)**
    - dictionary_path - Path to the dictionary of fingerprints (mandatory)
    - CSD_bvalue - If the DIAMOND outputs are not available, the fascicles directions are estimated using a CSD with the images at the b-values specified in this argument.

.. code-block:: python

	study.dti()
	study.noddi(lambda_iso_diff=3.e-9, lambda_par_diff=1.7e-9)
	study.diamond()
	study.fingerprinting(dictionary_path="my_dictionary", CSD_bvalue=None)


The metrics outputted by the functions are listed below.

Diffusion Tensor Imaging (DTI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fractional anisotropy (FA).
- Axial diffusivity (AD).
- Radial diffusivity (RD).
- Mean diffusivity (MD).
- Colored FA i.e. RGB map, a color is attributed to each voxel depending on the direction of the first eigenvalue and the intensity of the color depends on the FA value (fargb).
- Residual (residual).
- Eigenvectors of the diffusion tensor (evecs).
- Eigenvalues of the diffusion tensor (evals).
- Diffusion Tensor (dtensor).

Neurite Orientation Dispersion and Density Imaging (NODDI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Thresholded intra-cellular volume fraction $\nu_{ic}$ (icvf).
- Fiber orientation dispersion index (odi).
- Mean of the watson distribution of the Intra-cellular model  (mu).
- Fiber bundles volume fraction (fbundle).
- Extra-cellular volume fraction (fextra).
- Intra-cellular volume fraction (fintra).
- Free water volume fraction (fiso).
- Mean squared error (mse).
- R squared (R2).

DIstribution of 3D Anisotropic MicrOstructural eNvironment in Diffusion compartment imaging (DIAMOND)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Tensor orientations of the fiber population 0 (t0).
- Tensor orientations of the fiber population 1 (t1).
- Residuals (residuals).
- Intermediary step of the t0 output (mtm\_t0).
- Intermediary step of the t1 output (mtm\_t1).
- Intermediary step of the fractions output (mtm\_fractions).
- Volume with a null b-value (b0).
- DTI estimate (dti).
- Automose model selection map. Gray and white matter correspond to positive values and CSF to negative values (aicu).
- Fraction of voxel attributed to each compartment (fractions).
- Shape parameters of the mv-$\Gamma$ distribution, Homogeneity index (kappa).
- Log of kappa (logkappa).
- The heterogeneity indexes defined as $H E I=\frac{2}{\pi} \arctan \left(\frac{1}{\kappa}\right)$ (hei).
- Number of fascicles by voxel (mosemap).

Microstructure Fingerprinting (MF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Extra-axonal diffusivity of fascicle 0 (DIFF\_ex\_f0).
- Extra-axonal diffusivity of fascicle 1 (DIFF\_ex\_f1).
- Total extra-axonal diffusivity (DIFF\_ex\_tot).
- Volume fraction of cerebrospinal fluid (frac\_csf).
- Volume fractions of fascicle 0 (frac\_f0).
- Volume fractions of fascicle 1 (frac\_f1).
- Fiber volume fractions of the fascicle 0 (fvf\_f0).
- Fiber volume fractions of the fascicle 1 (fvf\_f1).
- Total fiber volume fraction of all fascicles (fvf\_tot).
- Mask (M0).
- Mean squared error (MSE).
- Peak map fascicle 0 (peak\_f0).
- Peak map fascicle 1 (peak\_f1).
- R squared (coefficient of determination, square of Pearson) (R2).
