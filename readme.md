# Oxytocin and Self-other distinction <img src="https://raw.githubusercontent.com/andlab-um/OT_face/main/demo.png" align="right" width="561px">

[![GitHub repo size](https://img.shields.io/github/languages/code-size/andlab-um/OT_face?color=brightgreen&label=repo%20size&logo=github)](https://github.com/andlab-um/OT_face)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fcercor%2Fbhac167-blue)](https://doi.org/10.1093/cercor/bhac167)<br />
[![Twitter URL](https://img.shields.io/twitter/url?label=%40ANDlab3&style=social&url=https%3A%2F%2Ftwitter.com%2Flizhn7)](https://twitter.com/ANDlab3)

**Code and data for: <br />**
**Wang, Y., Wang, R., & Wu, H. (2022). The role of oxytocin in modulating self–other distinction in human brain: A pharmacological fMRI study.** *Cerebral Cortex*. <br />
[DOI: 10.1093/cercor/bhac167](https://doi.org/10.1093/cercor/bhac167).
___

## Highlights
- Univariate analysis revealed that oxytocin increases the activity of neural networks involved in facial information processing and self-referential: cuneus，IFG，FFG
- MVPA further revealed a higher decoding accuracy of oxytocin group in MFG & IFG
- Higher representational similarity was found in the oxytocin group in the facial processing neural networks based on RSA

___

## Structure

**This repository contains:**
```
root
 ├── data               
 │    ├── OT_questionnaire data.xlsx  # behavior data  
 │    └── fmri_data_link.md          
 ├── code                
 │    ├── R
 │    │    ├── PCAcode.r
 │    │    ├── analysis.r
 │    │    ├── data.csv
 │    │    ├── pca.csv
 │    │    └── pp.r
 │    └── Python
 │    │    ├── pca.py
 │    │    ├── questionnaier.py
 │    │    ├── roi_signal_change_aal.py
 │    │    └── tsne.py
 ├──  models            
 │    ├── MATLAB
 │    │    ├── TDTmvpa.m
 │    │    ├── mvpaR_main.m
 │    │    └── parsave.m
 │    ├── Python
 │    │    ├── mvpaR.py
 │    │    ├── mvpa_perm_all_cond.py
 │    │    ├── mvpa_perm_all_cond_neurosynth.py
 │    │    ├── rsa_roi.py
 │    │    └── images
 │    │        ├── OT_masked_accuracies.nii
 │    │        ├── OT_p_adjusted.nii
 │    │        ├── OT_p_unadjusted.nii
 │    │        ├── PL_masked_accuracies.nii
 │    │        ├── PL_p_adjusted.nii
 │    │        ├── PL_p_unadjusted.nii
 │    │        ├── diff_masked_accuracies.nii
 │    │        ├── diff_p_adjusted.nii
 │    │        └── diff_p_unadjusted.nii
 │    └── README.md
 ├──  MVPA_plots         
 │    ├── Wholebrain
 │    │    ├── niis
 │    │         ├── OT_masked_accuracies.nii
 │    │         ├── OT_p_adjusted.nii
 │    │         ├── OT_p_unadjusted.nii
 │    │         ├── PL_masked_accuracies.nii
 │    │         ├── PL_p_adjusted.nii
 │    │         ├── PL_p_unadjusted.nii
 │    │         ├── diff_masked_accuracies.nii
 │    │         ├── diff_p_adjusted.nii
 │    │         └── diff_p_unadjusted.nii
 │    │    ├── OT_lh_caud.jpg
 │    │    ├── OT_lh_lat.jpg
 │    │    ├── OT_lh_med.jpg
 │    │    ├── OT_lh_ros.jpg
 │    │    ├── OT_rh_caud.jpg
 │    │    ├── OT_rh_lat.jpg
 │    │    ├── OT_rh_med.jpg
 │    │    ├── OT_rh_ros.jpg
 │    │    ├── PL_lh_caud.jpg
 │    │    ├── PL_lh_lat.jpg
 │    │    ├── PL_lh_med.jpg
 │    │    ├── PL_lh_ros.jpg
 │    │    ├── PL_rh_caud.jpg
 │    │    ├── PL_rh_lat.jpg
 │    │    ├── PL_rh_med.jpg
 │    │    ├── PL_rh_ros.jpg
 │    │    ├── diff_lh_caud.jpg
 │    │    ├── diff_lh_lat.jpg
 │    │    ├── diff_lh_med.jpg
 │    │    ├── diff_lh_ros.jpg
 │    │    ├── diff_rh_caud.jpg
 │    │    ├── diff_rh_lat.jpg
 │    │    ├── diff_rh_med.jpg
 │    │    └── diff_rh_ros.jpg
 │    └── Neurosynth
 │    │    ├── niis
 │    │         ├── OT_masked_accuracies.nii
 │    │         ├── OT_p_adjusted.nii
 │    │         ├── OT_p_unadjusted.nii
 │    │         ├── PL_masked_accuracies.nii
 │    │         ├── PL_p_adjusted.nii
 │    │         ├── PL_p_unadjusted.nii
 │    │         ├── diff_masked_accuracies.nii
 │    │         ├── diff_p_adjusted.nii
 │    │         └── diff_p_unadjusted.nii
 │    │    ├── diff_lh_caud.jpg
 │    │    ├── diff_lh_lat.jpg
 │    │    ├── diff_lh_med.jpg
 │    │    ├── diff_lh_ros.jpg
 │    │    ├── diff_rh_caud.jpg
 │    │    ├── diff_rh_lat.jpg
 │    │    ├── diff_rh_med.jpg
 │    │    └── diff_rh_ros.jpg
 └── README.md
```
**Note 1**: Before running the codes, change the directories to the path of corresponding locations. <br />
**Note 2**: All fMRI data with descriptions of the variables is put in openfmri. <br />

___

## Citation

    @article{wang2022role,
      title={The role of oxytocin in modulating self--other distinction in human brain: a pharmacological fMRI study},
      author={Wang, Yuanchen and Wang, Ruien and Wu, Haiyan},
      journal={Cerebral Cortex},
      year={2022}
    }
    
___
