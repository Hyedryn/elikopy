.. _introduction:

=======================
Introduction to ElikoPy
=======================

ElikoPy is Python library aiming at easing the processing of diffusion imaging 
for microstructural analysis. ElikoPy expands state of the art pipeline frameworks by providing a complete 
quality assessment and quality reports for each processed subject, providing a standardized
framework to ensure reproducibility and consistency, reducing error propagation and
improve sensitivity. It also grants the possibility for clinicians to perform fast preprocessing
for a large variety of studies with minimal knowledge.

The ElikoPy library was developed during a Master's thesis.

.. note::
	If you wish to learn more about the library and its validation, we invite you to read our `Master's thesis <http://hdl.handle.net/2078.1/thesis:30673>`_.


Why use ElikoPy?
======================

Diffusion weighted magnetic resonance imaging (DW-MRI) is a rapidly evolving, non radiating and non
invasive technique that allows to capture information on the brain microstructure through the restricted
diffusion of water molecules. DW-MRI has seen a growing interest in the recent years motivating the
acquisition of large multi-scanner multi-site data sets. The substantial acquisition time of this type of MRI
sequence has also encouraged the extensive use of Echo Planar Imaging which suffers from additional
artifacts and noise. Several tools have been developed in order to correct those individual problems
but they come with the disadvantages of processing only one subject at a time and requiring different
softwares making them cumbersome to use. This work presents and evaluates the performances of the
ElikoPy library, a complete diffusion MRI processing pipeline that reduces common sources of artifact and
captures information on the brain microstructure through multiple microstructural diffusion models. ElikoPy
has been designed to deal with large databases and to be robust to different types of acquisitions.
	
Features
========

1. Preprocessing of your dMRI data.
2. Generation of a synthesized b0 for susceptibility distortion correction using the `Synb0-DISCO repository <https://github.com/MASILab/Synb0-DISCO>`_. This synthesized b0 is useful for topup if the acquistion was only performed with one phase encoding direction.
3. The library can compute the DTI, Noddi, DIAMOND and the novel Microstucture fingerprinting models.
4. Complete quality reports to review each step of the processing.
5. Tissue segmentation from T1 images.
6. Ability to run subject and group wise statistics on the dataset.



