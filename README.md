# Parameter estimation for WMTI-Watson model of white matter using encoder-decoder recurrent neural network
See pre-print: Yujian Diao and Ileana Ozana Jelescu , 2022. Parameter estimation for WMTI-Watson model of white matter  
using encoder-decoder recurrent neural network. arXiv:2203.00595[physics]. (https://arxiv.org/abs/2203.00595)


# WMTI-Watson Estimator

Given DKI maps:  
*(For Diffusion Kurtosis Imaging tensor (DKI) estimation please refer to DKI fitting (https://github.com/NYU-DiffusionMRI/DESIGNER/blob/master/utils/dki_fit.m) in the diffusion 
MRI processing pipeline (https://github.com/NYU-DiffusionMRI/DESIGNER).)*
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
# Evaluation

If you have WMTI-Watson parametric maps fitted by other methods (e.g. NLLS), you can provide the path to 'wmti_path' in estimator.read_dki_nii() and run estimator.test(). The evaluation result will be stored under 'output_path/tmp'. If filtering=True   
is set in estimator.test(), the evaluaiton will be only performed on voxels with physical parametric values.

# Re-training
If you want to retrain the model on your own dataset, you can use the following command line,  
model$main.py --mode=test --dataset=data_filename.mat --datapath=path/to/data --model_folder=path/to/output  
--num_epochs=900 --train_perc=training_data_ratio --val_perc=validation_ratio  
  
Note the training dataset which is a Matlab mat-file should contain two variables, 'dki'(samples x 6) and 'wmti_paras'(samples x 5)  
with orders discribed above.
