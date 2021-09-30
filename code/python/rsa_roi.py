#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:03:33 2020

@author: yuanchenwang
"""

CONDITION = "OT"

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from heatmap import heatmap,annotate_heatmap
from nilearn import datasets, plotting
from nilearn.image import index_img, mean_img
import numpy as np
import pandas as pd
import nibabel as nib
from neurora.stuff import get_affine, datamask
from neurora.nps_cal import nps_fmri, nps_fmri_roi
from neurora.rsa_plot import plot_rdm
from neurora.rdm_cal import fmriRDM_roi, fmriRDM
from neurora.corr_cal_by_rdm import fmrirdms_corr
from neurora.nii_save import corr_save_nii, stats_save_nii
from neurora.stats_cal import stats_fmri
import nilearn.image as niimg
from neurora.rdm_corr import rdm_correlation_spearman,rdm_correlation_pearson
import seaborn as sns
from roi_signal_change_aal import z_obs
import scipy

# sys.stdout = open("log.txt", "w")
output_prefix = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/python/roi/inter"
# /Users/yuanchenwang/Library/Mobile\ Documents/com\~apple\~CloudDocs/Projects/fmri/main/Participants_list_OT_fMRI_new.xlsx 

mask_filename = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Analysis/mask_all.nii"
mask = nib.load(mask_filename).get_fdata()
mask_flat = np.ndarray.flatten(mask)
affine = nib.load(mask_filename).affine

aal3 = niimg.resample_img(nib.load("/Users/yuanchenwang/Documents/MATLAB/AAL3/AAL3v1.nii.gz"),affine,mask.shape).get_fdata()

roi_names = ["left amygdala","right amygdala","left ifg_oper","right ifg_oper","left ifg_tri","right ifg_tri",
                 "left fusiform", "right fusiform","left anterior_cingulate","right anterior_cingulate",
                 "left posterior_cingulate","right posterior_cingulate","left insula","right insula",
                 "left Supp. motor area","right Supp. motor area"]
roi_nums = np.array([45,46,7,8,9,10,59,60,35,36,39,40,33,34,15,16])


root = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/fmri_data/fMRI_haiyan_processed/s"

treatment = pd.read_excel("/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/main/Participants_list_OT_fMRI_new.xlsx", sheet_name="Sheet1")

#%%

sc = "selfchild"
oc = "otherchild"
sa = "selfadult"
oa = "otheradult"

categories = [sc,oc,sa,oa]
n_con = len(categories)

ids = treatment['ID']
group = treatment['Drug_name']
n_sub_total = len(ids)


# heatmap(model_RSM, categories, categories, cmap="YlGn")

n_sub_OT = 0
n_sub_Pl = 0
for i in range(n_sub_total):  # n_sub
    sub_path = root + str(ids[i]) + "/"
    if  os.path.isdir(sub_path) :
        if group[i] == "OT":
            n_sub_OT += 1
        elif group[i] == "Pl":
            n_sub_Pl += 1

corrs = []

nx, ny, nz = mask.shape

# fmri_all_data = []
fmri_OT = np.full([n_con, n_sub_OT, nx, ny, nz], np.nan)
fmri_Pl = np.full([n_con, n_sub_Pl, nx, ny, nz], np.nan)

t1 = 0
t2 = 0
for i in range(n_sub_total):  # n_sub_total
    sub_path = root + str(ids[i]) + "/"
    
    if  os.path.isdir(sub_path) :
        print("\n" + sub_path)
        
        beta_sc = [(sub_path + "beta_0001.img"),(sub_path + "beta_0011.img")]
        beta_oc = [(sub_path + "beta_0002.img"),(sub_path + "beta_0012.img")]
        beta_sa = [(sub_path + "beta_0003.img"),(sub_path + "beta_0013.img")]
        beta_oa = [(sub_path + "beta_0004.img"),(sub_path + "beta_0014.img")]
        
        if group[i] == "OT":
            img = mean_img(beta_sc)
            fmri_OT[0,t1] = datamask(img.get_fdata(), mask)
            img = mean_img(beta_oc)
            fmri_OT[1,t1] = datamask(img.get_fdata(), mask)
            img = mean_img(beta_sa)
            fmri_OT[2,t1] = datamask(img.get_fdata(), mask)
            img = mean_img(beta_oa)
            fmri_OT[3,t1] = datamask(img.get_fdata(), mask)
            t1 += 1
        elif group[i] == "Pl":
            img = mean_img(beta_sc)
            fmri_Pl[0,t2] = datamask(img.get_fdata(), mask)
            img = mean_img(beta_oc)
            fmri_Pl[1,t2] = datamask(img.get_fdata(), mask)
            img = mean_img(beta_sa)
            fmri_Pl[2,t2] = datamask(img.get_fdata(), mask)
            img = mean_img(beta_oa)
            fmri_Pl[3,t2] = datamask(img.get_fdata(), mask)
            t2 += 1
        
        # fmri_all_data[:,t,:,:,:] = fmri_data

   
#%%     
alpha = 0.05  # significance level

rdm_OT = []
rdm_Pl = []
for j in range(len(roi_names)):
    print(f'\nCalculating roi {roi_nums[j]}: {roi_names[j]}')
    roi_mask = (aal3 == roi_nums[j])
                    
    rdm_roi_OT = fmriRDM_roi(fmri_OT, roi_mask, method="euclidean", sub_opt=0)
    rdm_OT = rdm_OT + [rdm_roi_OT]
    rdm_roi_Pl = fmriRDM_roi(fmri_Pl, roi_mask, method="euclidean", sub_opt=0)
    rdm_Pl = rdm_Pl + [rdm_roi_Pl]
    

#%%
spearman_roi_OT = np.zeros((len(roi_names),len(roi_names)))
spearman_roi_Pl = np.zeros((len(roi_names),len(roi_names)))
pearson_roi_OT = np.zeros((len(roi_names),len(roi_names)))
pearson_roi_Pl = np.zeros((len(roi_names),len(roi_names)))
for i in range(len(roi_names)):
    for j in range(0,i):
        spearman_roi_OT[i,j] = rdm_correlation_spearman(rdm_OT[i],rdm_OT[j])[0]
        spearman_roi_Pl[i,j] = rdm_correlation_spearman(rdm_Pl[i],rdm_Pl[j])[0]
        pearson_roi_OT[i,j] = rdm_correlation_pearson(rdm_OT[i],rdm_OT[j])[0]
        pearson_roi_Pl[i,j] = rdm_correlation_pearson(rdm_Pl[i],rdm_Pl[j])[0]

#%%
corrmask = np.zeros_like(spearman_roi_OT)
corrmask[np.triu_indices_from(corrmask)] = True

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(1,2,figsize=[20,12])
    fig.suptitle("Spearman Correalation Coefficient")
    ax[0].set_title("PL")
    ax[1].set_title("OT")
    sns.heatmap(spearman_roi_Pl,ax=ax[0],mask=corrmask,square=True,cbar=False,yticklabels=roi_names,cmap="rocket_r")
    sns.heatmap(spearman_roi_OT,ax=ax[1],mask=corrmask,square=True,cbar=False,yticklabels=False,cmap="rocket_r")
    ax[0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(1,2,figsize=[20,12])
    fig.suptitle("Pearson Correalation Coefficient")
    ax[0].set_title("PL")
    ax[1].set_title("OT")
    sns.heatmap(pearson_roi_Pl,ax=ax[0],mask=corrmask,square=True,cbar=False,yticklabels=roi_names,cmap="rocket_r")
    sns.heatmap(pearson_roi_OT,ax=ax[1],mask=corrmask,square=True,cbar=False,yticklabels=False,cmap="rocket_r")
    ax[0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/rsa_OTPL.png",bbox_inches='tight')
    
#%%
z_scores = z_obs(pearson_roi_OT,pearson_roi_Pl)
p_values = scipy.stats.norm.sf(abs(z_scores))*2

annote = np.full((16,16)," ")
for i in range(16):
    for j in range(16):
        if p_values[i][j]<=0.001:
            annote[i][j] = "***"
        elif p_values[i][j]<=0.01:
            annote[i][j] = "**"
        elif p_values[i][j]<=0.05:
            annote[i][j] = "*"
    

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(figsize=[10,12])
    #ax.set_title("Z-scores of correlation differences")
    sns.heatmap(z_scores,ax=ax,mask=corrmask,square=True,yticklabels=roi_names,cmap="rocket_r",
                cbar=False, annot=annote, fmt = '', annot_kws={"size": 30})
    ax.set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,np.max(z_scores))
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.8, 0.3, 0.025, 0.5])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/rsa.png",bbox_inches='tight')
    
# with sns.plotting_context("poster"):
#     fig, ax = plt.subplots(figsize=[10,12])
#     sns.heatmap(p_values,ax=ax,mask=corrmask,vmax=0.05,square=True,yticklabels=roi_names,cmap="rocket_r")
#     ax.set_xticklabels(roi_names,rotation=50, ha="right")
    