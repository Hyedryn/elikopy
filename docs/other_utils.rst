.. _other-utils:

=============================
Additionnal utility functions
=============================

Get a list of patient by types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If no type are provided, print a list of patient for each type present in the study.
If a specific type is provided, only print the list of patient corresponding to that type.

.. code-block:: python

	elikopy.utils.get_patient_list_by_types(folder_path, type=None)	

Merge all global reports
^^^^^^^^^^^^^^^^^^^^^^^^

Merge all subjects quality control reports into a single report.

.. code-block:: python

	elikopy.utils.merge_all_reports(folder_path)

Dicom to NifTi
^^^^^^^^^^^^^^

Convert dicom data into compressed nifti. 
Converted dicoms are then moved to a sub-folder named original_data. 
The niftis are named patientID_ProtocolName_SequenceName.

.. code-block:: python

	elikopy.core.dicom_to_nifti(folder_path)
	
Anonymise NifTi
^^^^^^^^^^^^^^^

Anonymise all nifti present in rootdir by removing the PatientName and PatientBirthDate 
(only month and day) in the json and renaming the nifti file name to the PatientID.

* **rootdir** - Folder containing all the nifti to anonimyse.
* **anonymize_json** - If true, edit the json to remove the PatientName and replace the PatientBirthDate by the year of birth.
* **rename** - If true, rename the nifti to the PatientID. 

.. code-block:: python

	elikopy.core.anonymise_nifti(rootdir,anonymize_json,rename)
