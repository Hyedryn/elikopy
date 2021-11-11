===================================
Welcome to ElikoPy's documentation!
===================================

ElikoPy is Python library aiming at easing the processing of diffusion imaging for microstructural analysis. This Python library is based on

- DIPY, a python library for the analysis of MR diffusion imaging.
- Microstructure fingerprinting, a python library doing estimation of white matter microstructural properties from a dictionary of Monte Carlo diffusion MRI fingerprints.
- FSL, a comprehensive library of analysis tools for FMRI, MRI and DTI brain imaging data.
- DIAMOND, a c software that is characterizing brain tissue by assessment of the distribution of anisotropic microstructural environments in diffusion-compartment imaging.
- Dmipy, a python library estimating diffusion MRI-based microstructure features, used to fit and recover the parameters of multi-compartment microstructure models

This guide aims to give an introduction to ElikoPy and a brief installation instructions.
   
              
.. toctree::
   :maxdepth: 2
   :caption: Getting Started
              
   Introduction to ElikoPy <introduction>
   Installation <installation>
   Project Structure <elikopy_project>
   Using ElikoPy <elikopy_usage>
   
.. toctree::
   :maxdepth: 2 
   :caption: Detailed Guides

   Preprocessing of diffusion images <preprocessing_dmri>
   White matter mask <preprocessing_T1>

.. toctree::
   :maxdepth: 2

   Microstructural metrics <metrics>

.. toctree::
   :maxdepth: 2

   Statistics <stats>
   
.. toctree::
   :maxdepth: 2

   Utils function <other_utils>
 
.. toctree::
   :maxdepth: 2

   Examples <examples>

.. toctree::
   :maxdepth: 2
   :caption: About ElikoPy
   
   Contributing <contributing>
   
.. toctree::
   :maxdepth: 2

   API <elikopy>
   License <license>
