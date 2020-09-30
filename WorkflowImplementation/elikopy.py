"""
 
"""


def dicom_to_nifti(folder_path):
    """Convert dicom data into nifti. Converted dicom are then moved to a sub-folder named original_data
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    Returns
    -------
    /
    Notes
    -----
    Dicom can be organised into sub folder or in the root folder
    Examples
    --------
    dicom_to_nifti("C:\Memoire\example_data\")
    """


def patient_list(folder_path):
    """Verify the validity of all the nifti present in the root folder. If some nifti does not posses an associated bval
     and bvec file, they are discarded and the user is notified by the mean of a summary file named patient_error.txt generated
     in the out sub-directory. All the valid patient are stored in a file named patient_list.json
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    Notes
    -----
    /
    Examples
    --------
    patient_list("C:\Memoire\example_data\")
    """


def preproc(folder_path, eddy=False, denoising=False):
    """Perform bet and optionnaly eddy and denoising. Generated data are stored in bet, eddy, denoising and final directory
    located in the folder out/preproc
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    eddy: If True, eddy is called
    denoising: If True, denoising is called
    Notes
    -----
    All the function executed after this function MUST take input data from folder_path/out/preproc/final
    Examples
    --------
    preproc("C:\Memoire\example_data\")
    """


def dti(folder_path):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """

def total_workflow(folder_path, dicomToNifti=False, eddy=False, denoising=False, dti=False):
    """Perform dti and store the data in the out/dti folder.
    Parameters
    ----------
    folder_path: Path to root folder containing all the dicom
    """
