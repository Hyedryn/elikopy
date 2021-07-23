.. _installation:

============
Installation
============

ElikoPy requires Python v3.7+ to run.
After cloning the repo, you can either firstly install all the python dependencies including optionnal dependency used to speed up the code::

.. code-block:: none

   pip install -r requirements.txt --user

Or you can install directly the library with only the mandatory dependencies (if you performed the previous step, you still need to perform this step)::

.. code-block:: none

   python3 setup.py install --user

Microstructure Fingerprinting is currently not avaible in the standard python repo, you can clone and install this library manually::

.. code-block:: none

   git clone git@github.com:rensonnetg/microstructure_fingerprinting.git
   cd microstructure_fingerprinting
   python setup.py install

.. note::
	Note that FSL also needs to be installed and availabe in our path if you want to perform eddy current correction, mouvement correction or tbss.
	Ants, FSL, Freesurfer, pyTorch, torchvision and Convert3D Tool from ITK-Snap needs to be installed if you want to generate a second direction of encoding for your b0 in order to performs topup even if only a single direction of encoding were taken during the acquisition pahse of your data.
	Unfortunatly, the DIAMOND code is not publically available. If you do not have it in your possesion, you will not be able to use this algorithm. If you have it, simply add the executable to your path.
