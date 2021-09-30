#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:14:21 2021

@author: yuanchenwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:31:16 2021

@author: yuanchenwang
"""

CONDITION = "04"
"""
01,02,03,04, OT, Pl
"""
 
import math
import numpy as np
import nibabel as nib
from scipy import stats
import statsmodels.stats as smstats


file_prefix = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/resultPerm/"
mask_filename = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Analysis/mask_all.nii"
output_prefix = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/resultPerm/output/"

actual_prefix = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/mvpaR/"
actual_filename = actual_prefix + CONDITION + "/res_accuracy_minus_chance.img"
actual = nib.load(actual_filename).get_fdata()

mask = nib.load(mask_filename)
maskdata = mask.get_fdata()
affine = mask.affine

# number of permutation
n = 500
accuracy = np.empty((n,53,63,46), float)

for i in range(n):
    filepath = file_prefix + CONDITION + "/perm" + str(i+1) + "/res_accuracy_minus_chance.img"
    temp = nib.load(filepath).get_fdata()
    accuracy[i] = np.array(temp)

t_score = np.empty((53,63,46), float)
p_value = np.empty((53,63,46), float)

for i in range(53):
    for j in range(63):
        for k in range(46):
            voxel_acc = accuracy[:,i,j,k]
            test = stats.ttest_1samp(voxel_acc, actual[i,j,k])
            t_score[i,j,k] = test.statistic
            p_value[i,j,k] = test.pvalue

flatten_p = np.ndarray.flatten(p_value)
significant = np.reshape(smstats.multitest.fdrcorrection(flatten_p, alpha=0.001)[0],(53,63,46))
t_sig = np.empty((53,63,46), float)
for i in range(53):
    for j in range(63):
        for k in range(46):
            if significant[i,j,k]:# and t_score[i,j,k]>0:
                t_sig[i,j,k] = t_score[i,j,k]

output_path = output_prefix + CONDITION + "Sig.nii"
img = nib.Nifti1Image(t_sig, affine)
nib.save(img, output_path)  
                