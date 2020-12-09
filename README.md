# ElikoPy


ElikoPy is Python library aiming at easing the processing of diffusion imaging for microstructural analysis. 
This Python library is based on
  - DIPY, a python library for the analysis of MR diffusion imaging.
  - Microstructure fingerprinting, a python library doing estimation of white matter microstructural properties from a dictionary of Monte Carlo diffusion MRI fingerprints.
  - FSL, a comprehensive library of analysis tools for FMRI, MRI and DTI brain imaging data.
  - DIAMOND, a c software that is characterizing brain tissue by assessment of the distribution of anisotropic microstructural environments in diffusion‐compartment imaging.

### Installation

ElikoPy requires [Python](https://www.python.org/) v3.7+ to run.

After cloning the repo, you can either firstly install all the python dependencies including optionnal dependency used to speed up the code:

```sh
$ pip install -r requirements.txt --user
```
Or you can install directly the library with only the mandatory dependencies (if you performed the previous step, you still need to perform this step):

```sh
$ python3 setup.py install --user
```

Microstructure Fingerprinting is currently not avaible in the standard python repo, you can clone and install this library manually.

```sh
$ git clone git@github.com:rensonnetg/microstructure_fingerprinting.git
$ cd microstructure_fingerprinting
$ python setup.py install
```

FSL also needs to be installed and availabe in our path if you want to perform mouvement correction or tbss.

Unfortunatly, the DIAMOND code is not publically available. If you do not have it in your possesion, you will not be able to use this algorithm. If you have it, simply add the executable to your path. 

### Usage

Todo

### Development

Want to contribute? Great!

Do not hesitate to open issue or pull request!
### Todos

 - Fully implement TBSS
 - Fully implement Quality Control Metrics
 - Implement visualisation functions
 - Release a complete and accurate documentation for the library


**Free Software, Hell Yeah!**
