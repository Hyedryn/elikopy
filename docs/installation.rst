.. _installation:

============
Installation
============

-------------------------
Manual Installation Steps
-------------------------

You will need a Linux system (CentOS 7 is recommended) to run ElikoPy and all its dependencies natively using a manual installation. 
Doing a manual installation is not recommended if you have only a limited knownledge in computer science. 

Installation of the dependencies
================================

You must first install dependency to your system. Some dependencies are optional while others are mandatory.

FSL installation (mandatory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FSL is a mandatory comprehensive library dependency used among other steps for the preprocessing of diffusion images. 
FSL is available, ready to run, on `the official FSL installation page <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_.

FreeSurfer installation (optionnal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FreeSurfer is a software package for the analysis and visualization of structural and functional neuroimaging data 
for cross-sectional or longitudinal studies. This software is mandatory when correcting from susceptibility distortion
using T1 structural images in the preprocessing. To install it, visit the `FreeSurfer Downloads page <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_ 
and pick a package archive suitable to the environment you are in.

ANTs installation (optionnal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ANTs computes high-dimensional mappings to capture the statistics of brain structure and function. This software is mandatory 
when correcting from susceptibility distortion using T1 structural images in the preprocessing. ANTs can be compiled from 
source or installed via a pre-built package using their `Github page <https://github.com/ANTsX/ANTs>`_.

C3D installation (optionnal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C3D is a command-line tool for converting 3D images between common file formats. The tool also includes a growing list of commands 
for image manipulation, such as thresholding and resampling. This software is mandatory  when correcting from susceptibility distortion 
using T1 structural images in the preprocessing. A precompiled version of C3D is availabe on `Sourceforge <https://sourceforge.net/projects/c3d/>`_.

Microstructure Fingerprinting installation (recommended) (optionnal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Microstructure Fingerprinting estimate the white matter microstructural properties from a dictionary of Monte Carlo diffusion MRI fingerprints. To install it,
first download a copy of the `MF repository <https://github.com/rensonnetg/microstructure_fingerprinting>`_.

.. code-block:: none

	git clone git@github.com:rensonnetg/microstructure_fingerprinting.git
	
Then, navigate to the folder where this repository was cloned or downloaded (the folder containing the setup.py file) and install the package as follows.

.. code-block:: none

	cd microstructure_fingerprinting
	python setup.py install --user
	

DIAMOND installation (optionnal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unfortunatly, the DIAMOND code is not publicly available. If you do not have it in your possession, you will not be able to use this algorithm.
If you have it, simply add the executable to your path.

Installation of ElikoPy
=======================

ElikoPy requires Python v3.7+ to run.To install it, first download a copy or clone the `ElikoPy repository <https://github.com/Hyedryn/elikopy>`_.

.. code-block:: none

	git clone git@github.com:Hyedryn/elikopy.git


After cloning the repo, you can either firstly install all the python dependencies including optional dependencies used to speed up the code.

.. code-block:: none

	pip install -r requirements.txt --user

Or you can install directly the library with only the mandatory dependencies (if you performed the previous step, you still need to perform this step).

.. code-block:: none

	python3 setup.py install --user

.. note::
	When using ElikoPy, do not forget to reference it among all of the used dependencies.


----------------------------
Container Installation Steps
----------------------------

To ease the installation of ElikoPy, a Singularity container is provided in the `ElikoPy repository <https://github.com/Hyedryn/elikopy>`_.
To learn more about Singularity, you can visit their `official website <https://sylabs.io/singularity/>`_.

.. code-block:: none

	git clone https://github.com/Hyedryn/elikopy.git
	cd /path/to/repo
	sudo singularity build /path/to/elikopy.sif Singularity_elikopy
	
After building the container, ElikoPy can be run using the following command: 

.. code-block:: none

	singularity run -e --contain
	-B /path/to/study/directory/:/PROJECTS
	-B /tmp:/tmp
	-B /path/to/freesurfer/license.txt:/Software/freesurfer/license.txt
	-B /path/to/cuda:/usr/local/cuda
	--nv
	/path/to/elikopy.sif
	/path/to/script.py
	
The script.py file contains the Python code that will be executed inside the container. The path to the root directory in your python code must always be "/PROJECTS/" due to the folder binding.
	
.. note::
	Binding the freesurfer license is optional and is only needed for Synb0-DisCo.
	
.. note::
	Binding the cuda path is optional and is only needed to speed-up Synb0-DisCo or perform inter slice motion correction with Eddy FSL.

---------------------------------
Using ElikoPy on the CECI Cluster
---------------------------------

UCLouvain student who wish to use ElikoPy on the CECI cluster can use the existing installation present in the pilab project directory. 
First, the following line needs to be added to your *.bash_profile*.

.. code-block:: none

	source /CECI/proj/pilab/Software/config_elikopy.bash
	
Then, execute the following line of code to install ElikoPy:

.. code-block:: none

	source /CECI/proj/pilab/Software/install_elikopy.bash
	
If you wish to update your ElikoPy installation, you just need to execute again the preceding line of code.

Authorized user can update the local ElikoPy repository present in the PiLab directory using the following script. The local repository is updated using the master branch of the remote Github repository.

.. code-block:: none

	source /CECI/proj/pilab/Software/update_elikopy.bash

These steps should be sufficient for the *lemaitre3* and *manneback* clusters. When using other clusters, some additional modules may need to be loaded 
(see the related `CECI documentation <https://support.ceci-hpc.be/doc/_contents/UsingSoftwareAndLibraries/UsingPreInstalledSoftware/index.html>`_ for more information ).
We also strongly recommend you to familiarize yourself with slurm job when using ElikoPy on the CECI cluster.
