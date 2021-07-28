.. _preprocessing-T1:

================
T1 Preprocessing
================

Providing a white matter mask is a useful step to accelerate microstructural features
computation and more easily do tractography. The white_mask function of the ElikoPy
library has been elaborated to perform this important step.

.. image:: pictures/T1_pipeline_steps.jpg
	:width: 800
	:alt: T1 preprocessing pipeline.

On the one hand, when a T1 image is available, a white matter mask can be computed
from this data. Therefore, the T1 image is first preprocessed then segmented. Finally the
segmented white matter mask is projected into the space of the preprocessed diffusion
image.

On the other hand, when no T1 images are available, the white matter mask is directly
computed from a segmentation of the diffusion data using Anisotropic Power (AP) map. In this case, no registrations are necessary.


.. image:: pictures/APvsT1.jpg
	:width: 800
	:alt: AP map is shown in the left panels, while a T1 weighted structural image obtained from the same subject is shown in the right panel.