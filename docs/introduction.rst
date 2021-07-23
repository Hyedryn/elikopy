.. _introduction:

=======================
Introduction to ElikoPy
=======================

ElikoPy is Python library aiming at easing the processing of diffusion imaging 
for microstructural analysis. This Python library is based on

- DIPY, a python library for the analysis of MR diffusion imaging.
- Microstructure fingerprinting, a python library doing estimation of white matter microstructural properties from a dictionary of Monte Carlo diffusion MRI fingerprints.
- FSL, a comprehensive library of analysis tools for FMRI, MRI and DTI brain imaging data.
- DIAMOND, a c software that is characterizing brain tissue by assessment of the distribution of anisotropic microstructural environments in diffusion-compartment imaging.
- Dmipy, a python library estimating diffusion MRI-based microstructure features, used to fit and recover the parameters of multi-compartment microstructure models



Why use ElikoPy?
======================

Because ElikoPy is nice!
	
Features
========

1. Preprocessing of your dMRI data. The preprocessing includes as optional steps the following corrections: reslice, denoising (MP-PCA), gibbs ringing correction, susceptibility and eddy current induced distortions correction, motion correction (volume-to-volume and slice-to-volume).
2. Generation of a synthesized b0 for diffusion distortion correction using the based on the Synb0-DISCO repository (https://github.com/MASILab/Synb0-DISCO ). This synthesized b0 is usefull for topup if the acquistion was only performed with one phase encoding direction.
3. The library can compute the DTI, Noddi, DIAMOND and the novel Microstucturefingerprinting metric.
4. Complete quality reports to review each step of the processing
5. Tissue segmentation from T1 images
6. Ability to run TBSS (FSL) on the dataset
7. A paper on this library will soon provide a complete assesment of the perfomances of the library.

Use Cases
=========

--------------------
Reproducible science
--------------------

ElikoPy provides a standard processing experience!

---------------------------
Bla bla bla bla bla bla bla
---------------------------

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod 
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, 
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo 
consequat. Duis aute irure dolor in reprehenderit in voluptate velit 
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat 
cupidatat non proident, sunt in culpa qui officia deserunt mollit anim 
id est laborum.



