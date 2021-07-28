.. _examples:

========
Examples
========


.. code-block:: python

	import elikopy


	f_path="/CHEMIN/VERS/LETUDE"


	patient_list=None
	#patient_list=["Case1","Case2","Case3","Control1","Control2","Control3"]


	study = elikopy.core.Elikopy(f_path)

	#Génération de la liste des sujets
	study.patient_list()


	# Preprocessing
	study.preproc(eddy=True,topup=True,denoising=True,reslice=False,gibbs=False,biasfield=False,patient_list_m=patient_list,starting_state=None)

	study.white_mask()

	# Microstructure

	study.dti(patient_list_m=patient_list, use_wm=False)
	study.noddi(use_wm=False)

	dic_path="/home/users/microstructure/fixed_rad_dist.mat"
	study.fingerprinting(dic_path,use_wm=False)


	# Stats

	grp1=[1,2]
	grp2=[3,4]


	study.regall_FA(grp1=grp1, grp2=grp2, registration_type='-T', postreg_type='-S')

	additional_metrics={'_noddi_odi':'noddi','_mf_fvf_tot':'mf'}
	study.regall(grp1=grp1,grp2=grp2, metrics_dic=additional_metrics)


	metrics={'dti':'FA','_noddi_odi':'noddi','_mf_fvf_tot':'mf'}
	study.randomise_all(metrics_dic=metrics,randomise_numberofpermutation=5000, skeletonised=True, additional_atlases={'AtlasName':["path to xml","path to nifti"], 'AtlasName2':["path to xml2","path to nifti2"]})

	# Export

	study.export(preprocessing=True,dti=True,noddi=True,mf=True,wm_mask=False,report=False,raw=False)




