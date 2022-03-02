# Diffusion Kurtosis Imaging tensor (DKI) estimation
Please refer to the diffusion MRI processing pipeline (https://github.com/NYU-DiffusionMRI/DESIGNER) for DKI fitting (https://github.com/NYU-DiffusionMRI/DESIGNER/blob/master/utils/dki_fit.m).

# WMTI-Watson-Estimator

Given DKI maps:   
  - mean/axial/radial diffusivity [md/ad/rd] in μm<sup>2</sup>/ms 
  - mean/axial/radial kurtosis [mk/ak/rk]    

Output WMTI-Watson model parameter maps:  
  - f: axonal water Fraction.  
  - Da: Axonal Diffusivity.  
  - Depar: Extra-axonal PARallel Diffusivity.  
  - Deperp: Extra-axonal PERPendicular Diffusivity.  
  - c2: mean cos<sup>2</sup> of the axon orientation dispersion
  
# Recommended Usage
```
from WMTI_RNN_Estimator import WMTI_RNN_Estimator
import os

wmti_names=['f', 'Da', 'Depar', 'Deperp', 'c2']  
dki_names=['md', 'ad', 'rd', 'mk', 'ak', 'rk']  

dki_path = 'path/to/dki'
output_path = 'path/to/wmti'
mask = 'path/to/brainmask'
fa_threshold = 0

estimator = WMTI_RNN_Estimator()

estimator.read_dki_nii(dki_path, output_path, dki_names=dki_names, 
                        wmti_names=wmti_names, fa_threshold=fa_threshold, mask=mask)
estimator.estimate()
estimator.wmti_maps()
```
# Evaluation*

