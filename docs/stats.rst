.. _stats-others:

====================
Statistical Analysis
====================

The microstructural diffusion metrics estimated after preprocessing can be used to compute
basic group comparison statistics in order to rapidly localize brain changes related to development,
degeneration and disease. The group comparison can be performed between any two groups that are in different DATA_N directories.
For example, to perform a group comparison between the data in DATA_1 DATA_2 and the data in DATA_3 DATA_4:

.. code-block:: python

	grp1=[1,2]
	grp2=[3,4]

These statistical analyzes can be performed with three ElikoPy functions based on
Tract-Based Spatial Statistics of FSL : `TBSS FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide>`_.

* **regall_FA** : This function performs a carefully tuned nonlinear coregistration of the FA maps. The FA maps of all subjects are registered in the same space for the whole brain and also the skeleton.
* **regall** : This function applies to the other diffusion metrics the nonlinear warps and skeleton projection vectors computed by the regall_FA function. All the diffusion metrics specified in a dictionary are coregistered using the FA coregistration.
* **randomise_all** : This function based on randomise from FSL performs nonparametric permutation inference between the two groups defined in the regall_FA function. The user can choose to do the statistics on the whole brain or only in the skeleton. This function also outputs CSV files that contain for each subject the metrics mean and standard deviation across regions of atlases

.. note::
	The DTI is a prerequisite for regall_FA

.. note::
	Regall_FA is a prerequisite to reall and randomise_all

.. code-block:: python

    # coregistration of the FA maps
	study.regall_FA(grp1=grp1,grp2=grp2)

    # coregistration of the metrics in the additional_metrics dictionary
	additional_metrics={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
	study.regall(grp1=grp1,grp2=grp2, metrics_dic=additional_metrics)

    # Compute statistical results for the diffusion metrics specified in the metrics dictionary
	metrics={'dti':'FA','_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
	study.randomise_all(metrics_dic=metrics)

As a result of the above lines of code the following results are provided. For each diffusion metric:

* A nii of the coregistered subjects metric for the whole brain AND the skeleton.
* CSV files with the mean and standard deviation for each subject of the metric across regions of atlases.
* Nii maps of the p and t values (FWE corrected and not corrected) for the statistical differences between the two compared groups (positive and negative)

The default dictionaries supported by ElikoPy are (`Atlases <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases>`_):

* The Harvard-Oxford cortical and subcortical structural 1mm probabilistic atlases
* The JHU DTI-based white matter tractography 1mm probabilistic atlas
* The MNI structural 1mm probabilistic atlas

It is also possible to specify a custom atlas using the additional_atlases argument of randomise_all

.. code-block:: python

	study.randomise_all(metrics_dic=metrics, additional_atlases={'Atlas_name_1':["path to atlas 1 xml","path to atlas 1 nifti"],'Atlas_name_2':["path to atlas 2 xml","path to atlas 2 nifti"]})


Other useful arguments
^^^^^^^^^^^^^^^^^^^^^^

randomise_all:

- randomise_numberofpermutation - Define the number of permutations
- skeletonised - If True, randomize will be using only the white matter skeleton instead of the whole brain.

regall_FA:

- registration_type - Could either be '-T', '-t' or '-n'. If '-T' is used, a FMRIB58_FA standard-space image is used. If '-t' is used, a custom image is used. If '-n' is used, every FA image is aligned to every other one, identifying the "most representative" one, and using it as the target image.

