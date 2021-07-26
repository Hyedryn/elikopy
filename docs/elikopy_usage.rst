.. _elikopy-usage:

====================================
Typical usage for processing a study
====================================

---------------
Container Usage
---------------

blablabla

-----------
Python Code
-----------
	
	
Header and initialisation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
	:linenos:
	:lineno-start: 1
	
	import elikopy 
	import elikopy.utils
	
	f_path="/PROJECTS/" 
	dic_path="/PROJECTS/static_files/mf_dic/fixed_rad_dist.mat"
	
	study = elikopy.core.Elikopy(f_path, slurm=False)
	
	
Preprocessing and generation of the whitematter mask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
	:linenos:
	:lineno-start: 8
	
	study.preproc()
	study.white_mask()

Microstructural metrics computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
	:linenos:
	:lineno-start: 10
	
	study.dti()
	study.noddi()
	study.diamond()
	study.fingerprinting()
	
Statistical Analysis
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
	:linenos:
	:lineno-start: 14
	
	grp1=[1]
	grp2=[2]
	
	metrics={'_noddi_odi':'noddi','_mf_fvf_tot':'mf','_diamond_kappa':'diamond'}

	study.regall_FA(grp1=grp1,grp2=grp2)
	study.regall(grp1=grp1,grp2=grp2, metrics_dic=metrics)
	study.randomise_all(metrics_dic=metrics)
	
Data Exportation
^^^^^^^^^^^^^^^^

.. code-block:: python
	:linenos:
	:lineno-start: 22
	
	study.export(raw=False, preprocessing=False, dti=True, 
		noddi=False, diamond=False, mf=False, wm_mask=False, report=True)