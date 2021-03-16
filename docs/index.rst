.. ElikoPy documentation master file, created by
   sphinx-quickstart on Thu Feb 18 16:18:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ElikoPy's documentation!
===================================

ElikoPy is Python library aiming at easing the processing of diffusion imaging for microstructural analysis. This Python library is based on

- DIPY, a python library for the analysis of MR diffusion imaging.
- Microstructure fingerprinting, a python library doing estimation of white matter microstructural properties from a dictionary of Monte Carlo diffusion MRI fingerprints.
- FSL, a comprehensive library of analysis tools for FMRI, MRI and DTI brain imaging data.
- DIAMOND, a c software that is characterizing brain tissue by assessment of the distribution of anisotropic microstructural environments in diffusion-compartment imaging.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
------------------

* :ref:`genindex`


Features
--------

1. Preprocessing of your dMRI studies. The preprocessing includes as optional steps the following correction: reslice, denoising (mppca), gibbs ringing correction, eddy current correction (topup) and mouvement correction (eddy).
2. Generation of a synthesized b0 for diffusion distortion correction using the based on the Synb0-DISCO repository (https://github.com/MASILab/Synb0-DISCO ). This synthesized b0 is usefull for topup if the acquistion was only performed with one phase encoding direction.
3. The library can compute the DTI, Noddi, DIAMOND and the novel Microstucturefingerprinting metric.
4. A paper on this library will soon provide a complete assesment of the perfomance of the library.

Installation
------------

| ElikoPy requires Python v3.7+ to run.
| After cloning the repo, you can either firstly install all the python dependencies including optionnal dependency used to speed up the code:

   pip install -r requirements.txt --user
|
| Or you can install directly the library with only the mandatory dependencies (if you performed the previous step, you still need to perform this step):

   python3 setup.py install --user

| Microstructure Fingerprinting is currently not avaible in the standard python repo, you can clone and install this library manually.

   git clone git@github.com:rensonnetg/microstructure_fingerprinting.git

   cd microstructure_fingerprinting

   python setup.py install

| FSL also needs to be installed and availabe in our path if you want to perform eddy current correction, mouvement correction or tbss.

| Ants, FSL, Freesurfer, pyTorch, torchfusion and Convert3D Tool from ITK-Snap needs to be installed if you want to generate a second direction of encoding for your b0 in order to performs topup even if only a single direction of encoding were taken during the acquisition pahse of your data.

| Unfortunatly, the DIAMOND code is not publically available. If you do not have it in your possesion, you will not be able to use this algorithm. If you have it, simply add the executable to your path.

Contribute
----------

Want to contribute? Great!

Do not hesitate to open issue or pull request!

- Issue Tracker: https://github.com/Hyedryn/elikopy/issues
- Source Code: https://github.com/Hyedryn/elikopy

Support
-------

| If you are having issues, please let us know.
| You can contacted us by email:
|     quentin.dessain@student.uclouvain.be
|     mathieu.simon@student.uclouvain.be

License
-------

The project is licensed under the GNU AGPLv3 license.
