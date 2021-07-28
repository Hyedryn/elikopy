.. _stats-others:

====================
Statistical Analysis
====================

The microstructural diffusion metrics estimated after preprocessing can be used to compute
basic group comparison statistics in order to rapidly localize brain changes related to development,
degeneration and disease. The group comparison can be performed between any two groups that are in different DATA_N directories.
For example, to perform a group comparison between the data in DATA_1 DATA_2 and the data in DATA_3 DATA_4:

.. code-block:: python
	:linenos:
	:lineno-start: 1

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
	:linenos:
	:lineno-start: 3

    # coregistration of the FA maps
	study.regall_FA(grp1=grp1,grp2=grp2)

    # coregistration of the metrics in the additional_metrics dictionary
	additional_metrics={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
	study.regall(grp1=grp1,grp2=grp2, metrics_dic=additional_metrics)

    # Compute statistical results for the diffusion metrics specified in the metrics dictionary
	metrics={'dti':'FA','_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}
	study.randomise_all(metrics_dic=metrics)

