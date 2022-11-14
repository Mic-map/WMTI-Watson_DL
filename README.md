# Parameter estimation for WMTI-Watson model of white matter using encoder-decoder recurrent neural network
See article: Yujian Diao and Ileana Ozana Jelescu, Parameter estimation for WMTI-Watson model of white matter  
using encoder-decoder recurrent neural network. Magn Reson Med 2022. (https://doi.org/10.1002/mrm.29495)


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
#The physical bounds for filtering the parameter estimation.
wmti_bounds=[[0.01, 1], [0.01, 3.9], [0.01, 3], [0.01, 3], [0.33, 1]]

dki_path = 'path/to/dki'
output_path = 'path/to/wmti'
mask = 'path/to/brainmask'

#Change the FA threshold when you only want to do estimation for voxels with a higher FA  
#Usually FA in white matter is higher
fa_threshold = 0

estimator = WMTI_RNN_Estimator(wmti_bounds=wmti_bounds)

estimator.read_dki_nii(dki_path, output_path, dki_names=dki_names, 
                        wmti_names=wmti_names, fa_threshold=fa_threshold, mask=mask)
estimator.estimate()
estimator.wmti_maps()

#When you only want to keep the estimate within the physical bounds.
estimator.wmti_maps(filtering=True)
```
     
If you have problems in downloading the embedding data automatically, please download it manually following the link:   
https://drive.google.com/file/d/1w0SeNVUnl218BJlzucY5uThSgMufN1Q7/view?usp=sharing and put it in the "model" folder.
# Evaluation
```

'''
If you have WMTI-Watson parametric maps fitted by other methods (e.g. NLLS), you can provide the path to 'wmti_path'  
The evaluation result (comparison between RNN and NLLS estimation) will be stored under 'output_path/tmp'. 
The bounds of WMTI parameters ('f', 'Da', 'Depar', 'Deperp', 'c2') in our training data are:   
[0, 1], [0, 4], [0, 3], [0, 3], [0.33, 1].    
Those bounds have to be consistent when fitting the WMTI model using other methods like NLLS.
'''

wmti_nlls = 'path/to/NLLS/estimation'
estimator.read_dki_nii(dki_path, output_path, wmti_path=wmti_nlls, fa_threshold=fa_threshold, mask=mask) 
estimator.test()    

#When you only want to compare RNN and NLLS estimation on voxels respecting the physical bounds.
#Parameter estimates outside the physical bounds will be ignored
estimator.test(filtering=True) 
```
# Re-training
If you want to retrain the model on your own dataset, you can use the following command line,  
model$main.py --mode=test --dataset=data_filename.mat --datapath=path/to/data --model_folder=path/to/output  
--num_epochs=900 --train_perc=training_data_ratio --val_perc=validation_ratio  
  
Note the training dataset which is a Matlab mat-file should contain two variables, 'dki'(samples x 6) and 'wmti_paras'(samples x 5)  
with orders discribed above.
