#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:14:12 2021

@author: yuanchenwang
"""

import os
import math
import numpy as np
import nibabel as nib
import pandas as pd
import pingouin as pg
from tabulate import tabulate
import seaborn as sns
from scipy import stats
import statsmodels.stats as smstats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from scipy import stats
import nilearn.image as img
import scipy

sns.set_style("ticks",{'font.family': 'Times'})

enmax_palette = ["#DF4058", "#0070C0", "#70AD47", "FFD579"]
color_codes_wanted = ['child', 'adult', 'self','other']

c = lambda x: enmax_palette[color_codes_wanted.index(x)]

colors = sns.color_palette([c("adult"),c("child"),c("adult"),c("child")])

mask_filename = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Analysis/mask_all.nii"
mask = nib.load(mask_filename).get_fdata()
mask_flat = np.ndarray.flatten(mask)
affine = nib.load(mask_filename).affine

aal3 = img.resample_img(nib.load("/Users/yuanchenwang/Documents/MATLAB/AAL3/AAL3v1.nii.gz"),affine,mask.shape).get_fdata()

roi_names = ["left amygdala","right amygdala","left ifg_oper","right ifg_oper","left ifg_tri","right ifg_tri",
                 "left fusiform", "right fusiform","left anterior_cingulate","right anterior_cingulate",
                 "left posterior_cingulate","right posterior_cingulate","left insula","right insula",
                 "left Supp. motor area","right Supp. motor area"]
roi_nums = np.array([45,46,7,8,9,10,59,60,35,36,39,40,33,34,15,16])


root = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/fmri_data/fMRI_haiyan_processed/s"

treatment = pd.read_excel("/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/main/Participants_list_OT_fMRI_new.xlsx", sheet_name="Sheet1")


#%%
def z_obs(corr1, corr2):
    zcorr1 = 0.5*(np.log(np.abs(corr1)+1)-np.log(1-np.abs(corr1)))
    zcorr2 = 0.5*(np.log(np.abs(corr2)+1)-np.log(1-np.abs(corr2)))
    z_obs = (zcorr1 - zcorr2)/np.sqrt(1/(60-3)+1/(58-3))
    return np.abs(z_obs)
#%%

df = pd.DataFrame(columns = ['ROI', 'Treatment', 'Subject', 'beta', 'Signal Change','constant'])

for j in range(len(roi_nums)):
    roi = np.array(aal3==roi_nums[j])
    roi_flat = np.ndarray.flatten(roi)
    
    constant = {}
    for i in range(105,166+1):
        path1 = root+str(i)+"/beta_0021.img"
        path2 = root+str(i)+"/beta_0022.img"
        if (os.path.isfile(path1)):
            avg = (np.ndarray.flatten(nib.load(path1).get_fdata()) + np.ndarray.flatten(nib.load(path2).get_fdata()))/2
            constant[str(i)] = np.nanmean(avg[roi_flat!=0])
    
    for i in range(105,166+1):
        path1 = root+str(i)+"/beta_0001.img"
        path2 = root+str(i)+"/beta_0011.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"01-sc","Signal Change":change1,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"01-sc","Signal Change":change2,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
        
        path1 = root+str(i)+"/beta_0002.img"
        path2 = root+str(i)+"/beta_0012.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"02-oc","Signal Change":change1,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"02-oc","Signal Change":change2,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
            
        path1 = root+str(i)+"/beta_0003.img"
        path2 = root+str(i)+"/beta_0013.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"03-sa","Signal Change":change1,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"03-sa","Signal Change":change2,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
        
        path1 = root+str(i)+"/beta_0004.img"
        path2 = root+str(i)+"/beta_0014.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat!=0])/constant[str(i)]*100
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"04-oa","Signal Change":change1,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
            df = df.append({"ROI": roi_names[j],"Subject":i,"beta":"04-oa","Signal Change":change2,"Treatment" : treatment.loc[treatment.ID==i,'Drug_name'].values[0], "constant":constant[str(i)]},ignore_index=True)
            
#%%

aov_table = [["ROI","Treatment","Child face: p","Self face: p","Interaction: p","sa-oa difference: t-test"]]        
for j in range(len(roi_nums)):
    tempdf = df.loc[df.ROI==roi_names[j]].copy()
    tempdf["child face"] = (tempdf["beta"]=="01-sc") | (tempdf["beta"]=="02-oc")
    tempdf["self face"] = (tempdf["beta"]=="01-sc") | (tempdf["beta"]=="03-sa")
    aov_OT = pg.rm_anova(data = tempdf.loc[tempdf.Treatment=="OT"], dv="Signal Change",within=["child face","self face"], subject="Subject")
    aov_Pl = pg.rm_anova(data = tempdf.loc[tempdf.Treatment=="Pl"], dv="Signal Change",within=["child face","self face"], subject="Subject")
    with sns.plotting_context("poster"):
        fig, ax = plt.subplots(figsize=[10,10])
        ax.set_title(f"{roi_names[j]}",pad=15, size=30)
        ax=sns.barplot(x="Treatment", y="Signal Change", data=tempdf, hue="beta",  alpha=0.9, 
                       capsize=0.11, errwidth=2, n_boot=500,
                       palette={"01-sc":"#C15E53","02-oc":"#ef8b69","03-sa":"#1F819F","04-oa":"#A2B093"})
        handles, _ = ax.get_legend_handles_labels()
        ax=sns.swarmplot(x="Treatment", y="Signal Change", data=tempdf,hue="beta",dodge=True,
                         palette={"01-sc":"#C15E53","02-oc":"#ef8b69","03-sa":"#1F819F","04-oa":"#A2B093"})
        ax.get_legend().remove()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["OT","PL"])
        ax.set_xlabel("Treatment",size=33)
        ax.set_ylabel("Percent Signal Change",size=33)
        labels = ["Self-Child","Other-Child","Self-Adult","Other-Adult"]
        lgd=ax.legend(handles, labels,title='Facial Conditions',loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=1, fontsize=25)
        fig.savefig(f"figures07/{roi_names[j]}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    sa_oa_diff = np.array(tempdf.loc[tempdf.beta=="03-sa","Signal Change"])-np.array(tempdf.loc[tempdf.beta=="04-oa","Signal Change"])
    sa_oa_test = stats.ttest_ind(sa_oa_diff[tempdf.loc[tempdf.beta=="04-oa","Treatment"]=="OT"],sa_oa_diff[tempdf.loc[tempdf.beta=="04-oa","Treatment"]=="Pl"])
    aov_table = aov_table+[[roi_names[j],"OT",aov_OT.loc[0,"p-GG-corr"].round(4),aov_OT.loc[1,"p-GG-corr"].round(4),aov_OT.loc[2,"p-GG-corr"].round(4),sa_oa_test.pvalue.round(4)]]
    aov_table = aov_table+[[roi_names[j],"PL",aov_Pl.loc[0,"p-GG-corr"].round(4),aov_Pl.loc[1,"p-GG-corr"].round(4),aov_Pl.loc[2,"p-GG-corr"].round(4)]]
    
aov_table_str=(tabulate(aov_table, headers='firstrow'))
print(aov_table_str)

#%%
matrixOT1 = []
for roi in roi_names:
    matrixOT1 = matrixOT1 + [df.loc[df.ROI==roi].loc[df.Treatment=="OT"].loc[df.beta=="01-sc",'Signal Change']]
matrixOT1 = np.array(matrixOT1)

matrixOT2 = []
for roi in roi_names:
    matrixOT2 = matrixOT2 + [df.loc[df.ROI==roi].loc[df.Treatment=="OT"].loc[df.beta=="02-oc",'Signal Change']]
matrixOT2 = np.array(matrixOT2)

matrixOT3 = []
for roi in roi_names:
    matrixOT3 = matrixOT3 + [df.loc[df.ROI==roi].loc[df.Treatment=="OT"].loc[df.beta=="03-sa",'Signal Change']]
matrixOT3 = np.array(matrixOT3)

matrixOT4 = []
for roi in roi_names:
    matrixOT4 = matrixOT4 + [df.loc[df.ROI==roi].loc[df.Treatment=="OT"].loc[df.beta=="04-oa",'Signal Change']]
matrixOT4 = np.array(matrixOT4)

corrOT1 = np.corrcoef(matrixOT1)
corrOT2 = np.corrcoef(matrixOT2)
corrOT3 = np.corrcoef(matrixOT3)
corrOT4 = np.corrcoef(matrixOT4)

matrixPl1 = []
for roi in roi_names:
    matrixPl1 = matrixPl1 + [df.loc[df.ROI==roi].loc[df.Treatment=="Pl"].loc[df.beta=="01-sc",'Signal Change']]
matrixPl1 = np.array(matrixPl1)

matrixPl2 = []
for roi in roi_names:
    matrixPl2 = matrixPl2 + [df.loc[df.ROI==roi].loc[df.Treatment=="Pl"].loc[df.beta=="02-oc",'Signal Change']]
matrixPl2 = np.array(matrixPl2)

matrixPl3 = []
for roi in roi_names:
    matrixPl3 = matrixPl3 + [df.loc[df.ROI==roi].loc[df.Treatment=="Pl"].loc[df.beta=="03-sa",'Signal Change']]
matrixPl3 = np.array(matrixPl3)

matrixPl4 = []
for roi in roi_names:
    matrixPl4 = matrixPl4 + [df.loc[df.ROI==roi].loc[df.Treatment=="Pl"].loc[df.beta=="04-oa",'Signal Change']]
matrixPl4 = np.array(matrixPl4)

corrPl1 = np.corrcoef(matrixPl1)
corrPl2 = np.corrcoef(matrixPl2)
corrPl3 = np.corrcoef(matrixPl3)
corrPl4 = np.corrcoef(matrixPl4)

corrmask = np.zeros_like(corrOT1)
corrmask[np.triu_indices_from(corrmask)] = True

#%%
with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[15,15])
    fig.suptitle("OT-PL correlation")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(corrOT1-corrPl1,ax=ax[0,0],vmin=-0.2,vmax=0.2,mask=corrmask,square=True,cbar=False,cmap="RdBu_r",xticklabels=False,yticklabels=roi_names)
    sns.heatmap(corrOT2-corrPl2,ax=ax[0,1],vmin=-0.2,vmax=0.2,mask=corrmask,square=True,cbar=False,cmap="RdBu_r",xticklabels=False,yticklabels=False)
    sns.heatmap(corrOT3-corrPl3,ax=ax[1,0],vmin=-0.2,vmax=0.2,mask=corrmask,square=True,cbar=False,cmap="RdBu_r",yticklabels=roi_names)
    sns.heatmap(corrOT4-corrPl4,ax=ax[1,1],vmin=-0.2,vmax=0.2,mask=corrmask,square=True,cbar=False,cmap="RdBu_r",yticklabels=False)
    ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(-0.2,0.2)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)

#%%
    
with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[15,15])
    fig.suptitle("OT correlation between voxels")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(corrOT1,ax=ax[0,0],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=roi_names)
    sns.heatmap(corrOT2,ax=ax[0,1],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=False)
    sns.heatmap(corrOT3,ax=ax[1,0],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
    sns.heatmap(corrOT4,ax=ax[1,1],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[15,15])
    fig.suptitle("PL correlation between voxels")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(corrPl1,ax=ax[0,0],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=roi_names)
    sns.heatmap(corrPl2,ax=ax[0,1],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=False)
    sns.heatmap(corrPl3,ax=ax[1,0],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
    sns.heatmap(corrPl4,ax=ax[1,1],vmin=0,vmax=1,mask=corrmask,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)

#%%




#%%
ques_data = pd.read_excel('/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/main/OT_questionnaire data.xlsx',
                          header=1)
roi_ques = ques_data.drop(columns=["编号","姓名"])
ques_var = roi_ques.columns.values

for roi in roi_names:
    temp = []
    for sub in ques_data["编号"]%1000:
        tempdf = df.loc[df.ROI==roi].loc[df.Subject==sub]
        if (tempdf.shape[0] != 0):
            temp = temp + [np.nanmean(tempdf["constant"])]
        else:
            temp = temp + [np.nan]
    roi_ques[roi] = temp
corr = np.array(roi_ques.corr())[30:,0:30]

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(figsize=[20,25])
    ax.set_title("Constant")
    sns.heatmap(corr,ax=ax, vmin=0,vmax=0.5,square=True,cbar_kws={"orientation": "horizontal"},cmap="rocket_r")
    ax.set_yticklabels(roi_names,rotation=15, ha="right")
    ax.set_xticklabels(ques_var,rotation=45, ha="right")
    
#%%
roi_ques = ques_data.drop(columns=["编号","姓名"])
ques_var = roi_ques.columns.values

names = []

for roi in roi_names:
    temp = []
    for sub in ques_data["编号"]%1000:
        tempdf = df.loc[df.ROI==roi].loc[df.Subject==sub]
        if (tempdf.shape[0] != 0):
            temp = temp + [np.nanmean(tempdf.loc[tempdf.beta=="01-sc","Signal Change"])]
        else:
            temp = temp + [np.nan]
    roi_ques[str(roi+" 01-sc")] = temp
    names = names+[str(roi+" 01-sc")]
    
    temp = []
    for sub in ques_data["编号"]%1000:
        tempdf = df.loc[df.ROI==roi].loc[df.Subject==sub]
        if (tempdf.shape[0] != 0):
            temp = temp + [np.nanmean(tempdf.loc[tempdf.beta=="02-oc","Signal Change"])]
        else:
            temp = temp + [np.nan]
    roi_ques[str(roi+" 02-oc")] = temp
    names = names+[str(roi+" 02-oc")]
    
    temp = []
    for sub in ques_data["编号"]%1000:
        tempdf = df.loc[df.ROI==roi].loc[df.Subject==sub]
        if (tempdf.shape[0] != 0):
            temp = temp + [np.nanmean(tempdf.loc[tempdf.beta=="03-sa","Signal Change"])]
        else:
            temp = temp + [np.nan]
    roi_ques[str(roi+" 03-sa")] = temp
    names = names+[str(roi+" 03-sa")]
    
    temp = []
    for sub in ques_data["编号"]%1000:
        tempdf = df.loc[df.ROI==roi].loc[df.Subject==sub]
        if (tempdf.shape[0] != 0):
            temp = temp + [np.nanmean(tempdf.loc[tempdf.beta=="04-oa","Signal Change"])]
        else:
            temp = temp + [np.nan]
    roi_ques[str(roi+" 04-oa")] = temp
    names = names+[str(roi+" 04-oa")]

all_corr = np.abs(np.array(roi_ques.corr())[30:,0:30])





#%%

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[20,15])
    fig.suptitle("Correlation between ROI activation and psychometric data")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(all_corr[0::4],ax=ax[0,0],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=roi_names)
    sns.heatmap(all_corr[1::4],ax=ax[0,1],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=False)
    sns.heatmap(all_corr[2::4],ax=ax[1,0],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
    sns.heatmap(all_corr[3::4],ax=ax[1,1],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    ax[1,0].set_xticks(range(0,30))
    ax[1,1].set_xticks(range(0,30))
    ax[1,0].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[1,1].set_xticklabels(ques_var,rotation=50, ha="right")
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)

# with sns.plotting_context("poster"):
#     fig, ax = plt.subplots(figsize=[25,30])
#     ax.set_title("All conditions")
#     sns.heatmap(all_corr,ax=ax, vmin=0,vmax=0.5,square=True,cbar_kws={"orientation": "horizontal"},cmap="rocket_r")
#     ax.set_yticklabels(names, rotation=15, ha="right")
#     ax.set_xticklabels(ques_var,rotation=45, ha="right")

# corr_diff = np.zeros(all_corr.shape)
# for i in range(corr_diff.shape[0]):
#     for j in range(corr_diff.shape[1]):
#         corr_diff[i,j] = all_corr[i,j] - corr[int(i/4),j]

# with sns.plotting_context("poster"):
#     fig, ax = plt.subplots(figsize=[20,30])
#     ax.set_title("All conditions - constant (difference)")
#     sns.heatmap(corr_diff,ax=ax, vmin=-0.5,vmax=0.5,square=True,cbar_kws={"orientation": "horizontal"}, cmap="RdBu_r")
#     ax.set_yticklabels(names, rotation=15, ha="right")
#     ax.set_xticklabels(ques_var,rotation=45, ha="right")

#%%

z_child = z_obs(all_corr[0::4], all_corr[1::4])
z_adult = z_obs(all_corr[2::4], all_corr[3::4])
p_child = scipy.stats.norm.sf(abs(z_child))*2
p_adult = scipy.stats.norm.sf(abs(z_adult))*2

annote_c = np.full((len(roi_names),ques_var.shape[0])," ")
annote_a = np.full((len(roi_names),ques_var.shape[0])," ")
for i in range(len(roi_names)):
    for j in range(ques_var.shape[0]):
        if p_child[i][j]<=0.05:
            annote_c[i][j] = "*"
        if p_adult[i][j]<=0.05:
            annote_a[i][j] = "*"

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,1,figsize=[20,30])
    # fig.suptitle("Self-Other correlation comparison",fontsize=40)
    ax[0].set_title("Child", fontsize=40, pad = 30)
    ax[1].set_title("Adult", fontsize=40, pad = 30)
    sns.heatmap(z_child,ax=ax[0],vmin=0,vmax=2.5,square=True,cbar=False,cmap="rocket_r",
                yticklabels=roi_names, annot=annote_c, fmt = '', annot_kws={"size": 30})
    sns.heatmap(z_adult,ax=ax[1],vmin=0,vmax=2.5,square=True,cbar=False,cmap="rocket_r",
                yticklabels=roi_names, annot=annote_a, fmt = '', annot_kws={"size": 30})
    ax[0].set_xticks(range(0,30))
    ax[1].set_xticks(range(0,30))
    ax[0].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[1].set_xticklabels(ques_var,rotation=50, ha="right")
    norm = plt.Normalize(0,2.5)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/roi_psy.png",bbox_inches='tight')



#%%
roi_ques = ques_data.drop(columns=["姓名"])
roi_ques = roi_ques.rename(columns={"编号": "ID"})
roi_ques["ID"] = roi_ques["ID"]%1000

roi_ques_trmt = roi_ques.merge(treatment[["ID","Drug_name"]], on="ID")
roi_ques_OT = roi_ques_trmt[roi_ques_trmt["Drug_name"]=="OT"].drop(columns=["Drug_name"])
roi_ques_PL = roi_ques_trmt[roi_ques_trmt["Drug_name"]=="Pl"].drop(columns=["Drug_name"])

#%%
names=[]
for roi in roi_names:
    betas = ["01-sc","02-oc","03-sa","04-oa"]
    temp = []
    for b in betas:
        names = names+[str(roi+" "+b)]
        for sub in roi_ques_OT["ID"]:
            tempdf = df.loc[df.ROI==roi].loc[df.Subject==sub]
            if (tempdf.shape[0] != 0):
                temp = temp + [np.nanmean(tempdf.loc[tempdf.beta==b,"Signal Change"])]
            else:
                temp = temp + [np.nan]
        roi_ques_OT[str(roi+" "+b)] = temp
        temp=[]
        
        for sub in roi_ques_PL["ID"]:
            tempdf = df.loc[df.ROI==roi].loc[df.Subject==sub]
            if (tempdf.shape[0] != 0):
                temp = temp + [np.nanmean(tempdf.loc[tempdf.beta==b,"Signal Change"])]
            else:
                temp = temp + [np.nan]
        roi_ques_PL[str(roi+" "+b)] = temp
        temp=[]
        
roi_ques_OT = roi_ques_OT.drop(columns=["ID"])
roi_ques_PL = roi_ques_PL.drop(columns=["ID"])

roi_ques_corr_OT = np.abs(np.array(roi_ques_OT.corr())[30:,0:30])
roi_ques_corr_PL = np.abs(np.array(roi_ques_PL.corr())[30:,0:30])


#%%
with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[40,30])
    fig.suptitle("Correlation between ROI activation and psychometric data (OT)",fontsize=60)
    ax[0,0].set_title("01-sc",fontsize=40)
    ax[0,1].set_title("02-oc",fontsize=40)
    ax[1,0].set_title("03-sa",fontsize=40)
    ax[1,1].set_title("04-oa",fontsize=40)
    sns.heatmap(roi_ques_corr_OT[0::4],ax=ax[0,0],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
    sns.heatmap(roi_ques_corr_OT[1::4],ax=ax[0,1],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    sns.heatmap(roi_ques_corr_OT[2::4],ax=ax[1,0],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
    sns.heatmap(roi_ques_corr_OT[3::4],ax=ax[1,1],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    ax[0,0].set_xticks(range(0,30))
    ax[0,1].set_xticks(range(0,30))
    ax[1,0].set_xticks(range(0,30))
    ax[1,1].set_xticks(range(0,30))
    ax[0,0].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[0,1].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[1,0].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[1,1].set_xticklabels(ques_var,rotation=50, ha="right")
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    
with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[40,30])
    fig.suptitle("Correlation between ROI activation and psychometric data (PL)",fontsize=60)
    ax[0,0].set_title("01-sc",fontsize=40)
    ax[0,1].set_title("02-oc",fontsize=40)
    ax[1,0].set_title("03-sa",fontsize=40)
    ax[1,1].set_title("04-oa",fontsize=40)
    sns.heatmap(roi_ques_corr_PL[0::4],ax=ax[0,0],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
    sns.heatmap(roi_ques_corr_PL[1::4],ax=ax[0,1],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    sns.heatmap(roi_ques_corr_PL[2::4],ax=ax[1,0],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
    sns.heatmap(roi_ques_corr_PL[3::4],ax=ax[1,1],vmin=0,vmax=1,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    ax[0,0].set_xticks(range(0,30))
    ax[0,1].set_xticks(range(0,30))
    ax[1,0].set_xticks(range(0,30))
    ax[1,1].set_xticks(range(0,30))
    ax[0,0].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[0,1].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[1,0].set_xticklabels(ques_var,rotation=50, ha="right")
    ax[1,1].set_xticklabels(ques_var,rotation=50, ha="right")
    norm = plt.Normalize(0,1)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    
#%%
z_obs1 = z_obs(roi_ques_corr_OT[0::4],roi_ques_corr_PL[0::4])
z_obs2 = z_obs(roi_ques_corr_OT[1::4],roi_ques_corr_PL[1::4])
z_obs3 = z_obs(roi_ques_corr_OT[2::4],roi_ques_corr_PL[2::4])
z_obs4 = z_obs(roi_ques_corr_OT[3::4],roi_ques_corr_PL[3::4])

p1 = scipy.stats.norm.sf(abs(z_obs1))*2
p2 = scipy.stats.norm.sf(abs(z_obs2))*2
p3 = scipy.stats.norm.sf(abs(z_obs3))*2
p4 = scipy.stats.norm.sf(abs(z_obs4))*2


annote1 = np.full((16,30)," ")
annote2 = np.full((16,30)," ")
annote3 = np.full((16,30)," ")
annote4 = np.full((16,30)," ")
for i in range(16):
    for j in range(30):
        if p1[i][j]<=0.05:
            annote1[i][j] = "*"
        if p2[i][j]<=0.05:
            annote2[i][j] = "*"
        if p3[i][j]<=0.05:
            annote3[i][j] = "*"
        if p4[i][j]<=0.05:
            annote4[i][j] = "*"

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[40,30])
    # fig.suptitle("OT-PL significance (correlation between ROI activation and psychometric data)",fontsize=60)
    ax[0,0].set_title("Self-Child",fontsize=45,pad=30)
    ax[0,1].set_title("Other-Child",fontsize=45,pad=30)
    ax[1,0].set_title("Self-Adult",fontsize=45,pad=30)
    ax[1,1].set_title("Other-Adult",fontsize=45,pad=30)
    sns.heatmap(z_obs1,ax=ax[0,0],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",
                annot=annote1, fmt = '', annot_kws={"size": 30},yticklabels=roi_names)
    sns.heatmap(z_obs2,ax=ax[0,1],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",
                annot=annote2, fmt = '', annot_kws={"size": 30},yticklabels=False)
    sns.heatmap(z_obs3,ax=ax[1,0],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",
                annot=annote3, fmt = '', annot_kws={"size": 30},yticklabels=roi_names)
    sns.heatmap(z_obs4,ax=ax[1,1],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",
                annot=annote4, fmt = '', annot_kws={"size": 30},yticklabels=False)
    ax[0,0].set_xticks(range(0,30))
    ax[0,1].set_xticks(range(0,30))
    ax[1,0].set_xticks(range(0,30))
    ax[1,1].set_xticks(range(0,30))
    ax[0,0].set_xticklabels(ques_var,rotation=50, ha="right",fontsize=33)
    ax[0,1].set_xticklabels(ques_var,rotation=50, ha="right",fontsize=33)
    ax[1,0].set_xticklabels(ques_var,rotation=50, ha="right",fontsize=33)
    ax[1,1].set_xticklabels(ques_var,rotation=50, ha="right",fontsize=33)
    norm = plt.Normalize(0,2)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/otpl_roipsy.png",bbox_inches='tight')

#%%
df_acc = pd.read_csv("/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Rstudio/pca_rt_acc.csv")
df_merge = pd.merge(df,df_acc,on="Subject").drop('treatment', 1)

#OT
df_acc_OT1 = df.loc[df.Treatment=="OT"].loc[df.beta=="01-sc"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="OT"].loc[df.beta=="01-sc"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_OT1 = pd.concat([df_acc_OT1,temp],axis=1)
df_acc_OT1.columns = ['Subject']+roi_names
df_acc_OT1 = pd.merge(df_acc,df_acc_OT1).drop(columns=['treatment','Subject'])

df_acc_OT2 = df.loc[df.Treatment=="OT"].loc[df.beta=="02-oc"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="OT"].loc[df.beta=="02-oc"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_OT2 = pd.concat([df_acc_OT2,temp],axis=1)
df_acc_OT2.columns = ['Subject']+roi_names
df_acc_OT2 = pd.merge(df_acc,df_acc_OT2).drop(columns=['treatment','Subject'])

df_acc_OT3 = df.loc[df.Treatment=="OT"].loc[df.beta=="03-sa"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="OT"].loc[df.beta=="03-sa"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_OT3 = pd.concat([df_acc_OT3,temp],axis=1)
df_acc_OT3.columns = ['Subject']+roi_names
df_acc_OT3 = pd.merge(df_acc,df_acc_OT3).drop(columns=['treatment','Subject'])

df_acc_OT4 = df.loc[df.Treatment=="OT"].loc[df.beta=="04-oa"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="OT"].loc[df.beta=="04-oa"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_OT4 = pd.concat([df_acc_OT4,temp],axis=1)
df_acc_OT4.columns = ['Subject']+roi_names
df_acc_OT4 = pd.merge(df_acc,df_acc_OT4).drop(columns=['treatment','Subject'])

#pl
df_acc_Pl1 = df.loc[df.Treatment=="Pl"].loc[df.beta=="01-sc"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="Pl"].loc[df.beta=="01-sc"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_Pl1 = pd.concat([df_acc_Pl1,temp],axis=1)
df_acc_Pl1.columns = ['Subject']+roi_names
df_acc_Pl1 = pd.merge(df_acc,df_acc_Pl1).drop(columns=['treatment','Subject'])

df_acc_Pl2 = df.loc[df.Treatment=="Pl"].loc[df.beta=="02-oc"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="Pl"].loc[df.beta=="02-oc"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_Pl2 = pd.concat([df_acc_Pl2,temp],axis=1)
df_acc_Pl2.columns = ['Subject']+roi_names
df_acc_Pl2 = pd.merge(df_acc,df_acc_Pl2).drop(columns=['treatment','Subject'])

df_acc_Pl3 = df.loc[df.Treatment=="Pl"].loc[df.beta=="03-sa"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="Pl"].loc[df.beta=="03-sa"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_Pl3 = pd.concat([df_acc_Pl3,temp],axis=1)
df_acc_Pl3.columns = ['Subject']+roi_names
df_acc_Pl3 = pd.merge(df_acc,df_acc_Pl3).drop(columns=['treatment','Subject'])

df_acc_Pl4 = df.loc[df.Treatment=="Pl"].loc[df.beta=="04-oa"].loc[df.ROI==roi,("Subject")].reset_index(drop=True)
for roi in roi_names:
    temp = df.loc[df.Treatment=="Pl"].loc[df.beta=="04-oa"].loc[df.ROI==roi,("Signal Change")].reset_index(drop=True)
    df_acc_Pl4 = pd.concat([df_acc_Pl4,temp],axis=1)
df_acc_Pl4.columns = ['Subject']+roi_names
df_acc_Pl4 = pd.merge(df_acc,df_acc_Pl4).drop(columns=['treatment','Subject'])

#%%
labels = ['pca','RT Child','RT adult','Accuracy Child','Accuracy Adult']

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[30,15])
    fig.suptitle("OT correlation of behavioral data")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(np.abs(np.array(df_acc_OT1.corr())[0:5,5:]),ax=ax[0,0],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=labels)
    sns.heatmap(np.abs(np.array(df_acc_OT2.corr())[0:5,5:]),ax=ax[0,1],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=False)
    sns.heatmap(np.abs(np.array(df_acc_OT3.corr())[0:5,5:]),ax=ax[1,0],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",yticklabels=labels)
    sns.heatmap(np.abs(np.array(df_acc_OT4.corr())[0:5,5:]),ax=ax[1,1],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,0.5)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/behav_psy_OT.png",bbox_inches='tight')

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[30,15])
    fig.suptitle("PL correlation of behavioral data")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(np.abs(np.array(df_acc_Pl1.corr())[0:5,5:]),ax=ax[0,0],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=labels)
    sns.heatmap(np.abs(np.array(df_acc_Pl2.corr())[0:5,5:]),ax=ax[0,1],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=False)
    sns.heatmap(np.abs(np.array(df_acc_Pl3.corr())[0:5,5:]),ax=ax[1,0],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",yticklabels=labels)
    sns.heatmap(np.abs(np.array(df_acc_Pl4.corr())[0:5,5:]),ax=ax[1,1],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
    ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,0.5)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/behav_psy_PL.png",bbox_inches='tight')

#%%
z_br1 = z_obs(np.array(df_acc_OT1.corr())[0:5,5:],np.array(df_acc_Pl1.corr())[0:5,5:])
z_br2 = z_obs(np.array(df_acc_OT2.corr())[0:5,5:],np.array(df_acc_Pl2.corr())[0:5,5:])
z_br3 = z_obs(np.array(df_acc_OT3.corr())[0:5,5:],np.array(df_acc_Pl3.corr())[0:5,5:])
z_br4 = z_obs(np.array(df_acc_OT4.corr())[0:5,5:],np.array(df_acc_Pl4.corr())[0:5,5:])

p_br1 = scipy.stats.norm.sf(abs(z_br1))*2
p_br2 = scipy.stats.norm.sf(abs(z_br2))*2
p_br3 = scipy.stats.norm.sf(abs(z_br3))*2
p_br4 = scipy.stats.norm.sf(abs(z_br4))*2

ann_br1 = np.full((5,16)," ")
ann_br2 = np.full((5,16)," ")
ann_br3 = np.full((5,16)," ")
ann_br4 = np.full((5,16)," ")
for i in range(5):
    for j in range(16):
        if p_br1[i][j]<=0.05:
            ann_br1[i][j] = "*"
        if p_br2[i][j]<=0.05:
            ann_br2[i][j] = "*"
        if p_br3[i][j]<=0.05:
            ann_br3[i][j] = "*"
        if p_br4[i][j]<=0.05:
            ann_br4[i][j] = "*"

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[30,15])
    fig.suptitle("Correlation difference: behavioral data and ROI activation")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(z_br1,ax=ax[0,0],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",yticklabels=labels,
                annot=ann_br1, fmt = '', annot_kws={"size": 30})
    sns.heatmap(z_br2,ax=ax[0,1],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",yticklabels=False,
                annot=ann_br2, fmt = '', annot_kws={"size": 30})
    sns.heatmap(z_br3,ax=ax[1,0],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",yticklabels=labels,
                annot=ann_br3, fmt = '', annot_kws={"size": 30})
    sns.heatmap(z_br4,ax=ax[1,1],vmin=0,vmax=2,square=True,cbar=False,cmap="rocket_r",yticklabels=False,
                annot=ann_br4, fmt = '', annot_kws={"size": 30})
    ax[0,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[0,1].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,2)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/behav_psy.png",bbox_inches='tight')


#%%

subdf_OT = pd.concat([df_merge.loc[df_merge.Treatment=="OT"].loc[df_merge.beta=="01-sc","Signal Change"].reset_index(drop=True),
                      df_merge.loc[df_merge.Treatment=="OT"].loc[df_merge.beta=="02-oc","Signal Change"].reset_index(drop=True),
                      df_merge.loc[df_merge.Treatment=="OT"].loc[df_merge.beta=="03-sa","Signal Change"].reset_index(drop=True),
                      df_merge.loc[df_merge.Treatment=="OT"].loc[df_merge.beta=="04-oa"].drop(columns=["ROI","Treatment","Subject","beta"]).reset_index(drop=True)],
                     axis=1,ignore_index=True)
subdf_OT.columns=['01-sc','02-oc','03-sa','04-oa','Constant','pca','RT Child','RT adult','Accuracy Child','Accuracy Adult'] 

subdf_Pl = pd.concat([df_merge.loc[df_merge.Treatment=="Pl"].loc[df_merge.beta=="01-sc","Signal Change"].reset_index(drop=True),
                      df_merge.loc[df_merge.Treatment=="Pl"].loc[df_merge.beta=="02-oc","Signal Change"].reset_index(drop=True),
                      df_merge.loc[df_merge.Treatment=="Pl"].loc[df_merge.beta=="03-sa","Signal Change"].reset_index(drop=True),
                      df_merge.loc[df_merge.Treatment=="Pl"].loc[df_merge.beta=="04-oa"].drop(columns=["ROI","Treatment","Subject","beta"]).reset_index(drop=True)],
                     axis=1,ignore_index=True)
subdf_Pl.columns=['01-sc','02-oc','03-sa','04-oa','Constant','pca','RT Child','RT adult','Accuracy Child','Accuracy Adult'] 

corr_OT = np.abs(np.array(subdf_OT.corr())[0:4,4:])
corr_Pl = np.abs(np.array(subdf_Pl.corr())[0:4,4:])

xlabels = ['Constant','pca','RT Child','RT adult','Accuracy Child','Accuracy Adult']
ylabels = ['01-sc','02-oc','03-sa','04-oa']

#%%
with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,1,figsize=[15,10])
    fig.suptitle("Correlation with behavior data")
    ax[0].set_title("OT")
    ax[1].set_title("PL")
    sns.heatmap(corr_OT,ax=ax[0],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=ylabels)
    sns.heatmap(corr_Pl,ax=ax[1],vmin=0,vmax=0.5,square=True,cbar=False,cmap="rocket_r",xticklabels=xlabels,yticklabels=ylabels)
    ax[1].set_xticklabels(xlabels,rotation=50, ha="right")
    ax[0].set_yticklabels(ylabels,rotation=10, ha="right")
    ax[1].set_yticklabels(ylabels,rotation=10, ha="right")
    norm = plt.Normalize(0,0.5)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.8, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)


#%%
z_obs1 = z_obs(corrOT1,corrPl1)
z_obs2 = z_obs(corrOT2,corrPl2)
z_obs3 = z_obs(corrOT3,corrPl3)
z_obs4 = z_obs(corrOT4,corrPl4)

p1 = scipy.stats.norm.sf(abs(z_obs1))*2
p2 = scipy.stats.norm.sf(abs(z_obs2))*2
p3 = scipy.stats.norm.sf(abs(z_obs3))*2
p4 = scipy.stats.norm.sf(abs(z_obs4))*2


annote1 = np.full((16,16)," ")
annote2 = np.full((16,16)," ")
annote3 = np.full((16,16)," ")
annote4 = np.full((16,16)," ")
for i in range(16):
    for j in range(16):
        if p1[i][j]<=0.05:
            annote1[i][j] = "*"
        if p2[i][j]<=0.05:
            annote2[i][j] = "*"
        if p3[i][j]<=0.05:
            annote3[i][j] = "*"
        if p4[i][j]<=0.05:
            annote4[i][j] = "*"

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(2,2,figsize=[15,15])
    # fig.suptitle("OT-PL correlation significance")
    ax[0,0].set_title("01-sc")
    ax[0,1].set_title("02-oc")
    ax[1,0].set_title("03-sa")
    ax[1,1].set_title("04-oa")
    sns.heatmap(z_obs1,ax=ax[0,0],vmin=0,vmax=2,mask=corrmask,square=True,cbar=False,cmap="rocket_r",
                annot=annote1, fmt = '', annot_kws={"size": 30},xticklabels=False,yticklabels=roi_names)
    sns.heatmap(z_obs2,ax=ax[0,1],vmin=0,vmax=2,mask=corrmask,square=True,cbar=False,cmap="rocket_r",
                annot=annote2, fmt = '', annot_kws={"size": 30},xticklabels=False,yticklabels=False)
    sns.heatmap(z_obs3,ax=ax[1,0],vmin=0,vmax=2,mask=corrmask,square=True,cbar=False,cmap="rocket_r",
                annot=annote3, fmt = '', annot_kws={"size": 30},yticklabels=roi_names)
    sns.heatmap(z_obs4,ax=ax[1,1],vmin=0,vmax=2,mask=corrmask,square=True,cbar=False,cmap="rocket_r",
                annot=annote4, fmt = '', annot_kws={"size": 30},yticklabels=False)
    ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
    ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
    norm = plt.Normalize(0,2)
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.15, 0.025, 0.6])
    fig.colorbar(sm,cax=cax)
    fig.savefig("figures/roi_corr.png",bbox_inches='tight')

# with sns.plotting_context("poster"):
#     fig, ax = plt.subplots(2,2,figsize=[15,15])
#     fig.suptitle("OT-PL correlation significance")
#     ax[0,0].set_title("01-sc")
#     ax[0,1].set_title("02-oc")
#     ax[1,0].set_title("03-sa")
#     ax[1,1].set_title("04-oa")
#     sns.heatmap(p1,ax=ax[0,0],vmax=0.05,mask=corrmask,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=roi_names)
#     sns.heatmap(p2,ax=ax[0,1],vmax=0.05,mask=corrmask,square=True,cbar=False,cmap="rocket_r",xticklabels=False,yticklabels=False)
#     sns.heatmap(p3,ax=ax[1,0],vmax=0.05,mask=corrmask,square=True,cbar=False,cmap="rocket_r",yticklabels=roi_names)
#     sns.heatmap(p4,ax=ax[1,1],vmax=0.05,mask=corrmask,square=True,cbar=False,cmap="rocket_r",yticklabels=False)
#     ax[1,0].set_xticklabels(roi_names,rotation=50, ha="right")
#     ax[1,1].set_xticklabels(roi_names,rotation=50, ha="right")
#     norm = plt.Normalize(0,0.05)
#     sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
#     sm.set_array([])
#     plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
#     cax = plt.axes([0.95, 0.15, 0.025, 0.6])
#     fig.colorbar(sm,cax=cax)
