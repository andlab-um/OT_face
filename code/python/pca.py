#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:29:43 2021

@author: yuanchenwang
"""

import os
import math
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.stats as smstats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from scipy import stats
import pingouin as pg

sns.set_style("ticks",{'font.family': 'Times'})

#%%

root = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/fmri_data/fMRI_haiyan_processed/s"

beta_imgs = ["01","02","03","04","11","12","13","14"]

treatment = pd.read_excel("/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/main/Participants_list_OT_fMRI_new.xlsx", sheet_name="Sheet1")

mask_filename = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Analysis/mask_all.nii"
mask = nib.load(mask_filename).get_fdata()
mask_flat = np.ndarray.flatten(mask)
affine = nib.load(mask_filename).affine

all_imgs = []
OT_imgs = []
Pl_imgs = []
labels = np.full(472,0) # 0 for pl
l_index = 0
for i in range(105,166+1):
    for j in beta_imgs:
        path = root+str(i)+"/beta_00"+str(j)+".img"
        if (os.path.isfile(path)):
            image = np.ndarray.flatten(nib.load(path).get_fdata())
            masked_img = image[mask_flat!=0]
            all_imgs = all_imgs + [masked_img.tolist()]
            if (np.array(treatment[treatment["ID"]==i]["Drug_name"])[0]=="OT"):
                OT_imgs = OT_imgs + [masked_img.tolist()]
                labels[l_index] = 1
            else:
                Pl_imgs = Pl_imgs + [masked_img.tolist()]
        l_index = l_index+1
        
arr_allimgs = np.array(all_imgs)
arr_OTimgs = np.array(OT_imgs)
arr_Plimgs = np.array(Pl_imgs)



#%%
nc = 3

pca_all = PCA(n_components=nc)
pca_OT = PCA(n_components=nc)
pca_Pl = PCA(n_components=nc)
pca_all.fit(arr_allimgs)
pca_OT.fit(arr_OTimgs)
pca_Pl.fit(arr_Plimgs)

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

output_root = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Analysis/PCA/"

for comp in range(nc):
    index=0
    pc_all = np.full((53,63,46), np.nan)
    pc_OT = np.full((53,63,46), np.nan)
    pc_Pl = np.full((53,63,46), np.nan)
    for i in range(53):
        for j in range(63):
            for k in range(46):
                if mask[i,j,k]!=0:
                    pc_Pl[i,j,k] = pca_Pl.components_[comp,index]
                    pc_OT[i,j,k] = pca_OT.components_[comp,index]
                    pc_all[i,j,k] = pca_all.components_[comp,index]
                    index = index+1
    output_path_all= output_root + "PC" + str(comp+1) + "-all.nii"
    output_path_OT= output_root + "PC" + str(comp+1) + "-OT.nii"
    output_path_Pl= output_root + "PC" + str(comp+1) + "-Pl.nii"
    nib.save(nib.Nifti1Image(pc_all, affine),output_path_all)
    nib.save(nib.Nifti1Image(pc_OT, affine),output_path_OT)
    nib.save(nib.Nifti1Image(pc_Pl, affine),output_path_Pl)
        
# pc1_all = np.empty((53,63,46), float)
# for i in range(53):
#     for j in range(63):
#         for k in range(46):
#             if mask[i,j,k]!=0:
#                 pc1_all[i,j,k] = pca_OT.components_[1,index]
#                 index = index+1
                
# output_path = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Analysis/PCA/pc1.nii"
# img = nib.Nifti1Image(pc1_all, affine)
# nib.save(img, output_path)  
    
#%%
projections = pca_all.transform(arr_allimgs)
    
pd.DataFrame(projections).to_csv("/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Rstudio/pca.csv",
                                               index=None)

#%%
beta1 = ["01","11"]
beta2 = ["02","12"]
beta3 = ["03","13"]
beta4 = ["04","14"]

betas= np.array([beta1,beta2,beta3,beta4])

imgs = [[],[],[],[]]

for b in range(4):
    temp_img = []
    for i in range(105,166+1):
        beta = betas[b]
        for j in beta:
            path = root+str(i)+"/beta_00"+str(j)+".img"
            if (os.path.isfile(path)):
                image = np.ndarray.flatten(nib.load(path).get_fdata())
                masked_img = image[mask_flat!=0]
                temp_img = temp_img + [masked_img.tolist()]
    imgs[b] = temp_img

#%%
arr_imgs = np.array(imgs)
pcas = [PCA(n_components=nc),PCA(n_components=nc),PCA(n_components=nc),PCA(n_components=nc)]

for b in range(4):
    pcas[b].fit(arr_imgs[b])              
    for comp in range(nc):
        index=0
        pc_out = np.full((53,63,46), np.nan)
        for i in range(53):
            for j in range(63):
                for k in range(46):
                    if mask[i,j,k]!=0:
                        pc_out[i,j,k] = pcas[b].components_[comp,index]
                        index = index+1
        output_path= output_root + str(b+1)+ "PC" + str(comp+1) + ".nii"
        nib.save(nib.Nifti1Image(pc_out, affine),output_path)

#%%
arr = np.array([pca_all.components_[0], pca_OT.components_[0], pca_Pl.components_[0],pcas[0].components_[0],pcas[1].components_[0],pcas[2].components_[0],pcas[3].components_[0]])
corr = np.corrcoef(arr)

#%%
with sns.plotting_context("poster"):
    fig, ax = plt.subplots(figsize=[10,10])
    sns.scatterplot(x=projections[:,0],y=projections[:,1], hue=labels)

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(figsize=[10,10])
    sns.heatmap(corr, ax=ax)

#%%
root = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri"

df = pd.read_excel(f'{root}/main/OT_fMRI_all data.csv.xls',sheet_name="OT_fMRI_all data")
t_df = pd.read_excel(f'{root}/main/Participants_list_OT_fMRI_new.xlsx',sheet_name="Sheet1")

behav = df[['Run','sub','face','STIM.CRESP','STIM.RESP','STIM.RT']]
treatment = t_df.drop('Group_name',axis = 1)

behav = behav.dropna()
treatment = treatment.dropna()

#%%
child_acc = []
adult_acc = []
both_acc = []

child_rt = []
adult_rt = []
both_rt = []

all_rt = []
subject = []

trmt = []
for i in range(105,166+1):
    sub_behav = behav[behav['sub']==i]
    if (len(sub_behav)!=0):
        for j in range(2):
            sub_run = sub_behav[sub_behav['Run']==j+1]
            
            sub_c = sub_run[(sub_run['face']=="sc")|(sub_run['face']=="oc")]
            sub_a = sub_run[(sub_run['face']=="sa")|(sub_run['face']=="oa")]
            
            child_rt = child_rt + [np.mean(sub_c['STIM.RT'])]
            adult_rt = adult_rt + [np.mean(sub_a['STIM.RT'])]
            both_rt = both_rt + [np.mean(sub_run['STIM.RT'])]
            
            child_acc = child_acc + [len(sub_c[sub_c['STIM.CRESP']-1==sub_c['STIM.RESP']])/len(sub_c)]
            adult_acc = adult_acc + [len(sub_a[sub_a['STIM.CRESP']-1==sub_a['STIM.RESP']])/len(sub_a)]
            both_acc = both_acc + [len(sub_run[sub_run['STIM.CRESP']-1==sub_run['STIM.RESP']])/len(sub_run)]
            
            if (np.array(treatment[treatment["ID"]==i]["Drug_name"])[0]=="OT"):
                trmt = trmt + [1]
            else:
                trmt = trmt + [0]
            all_rt = all_rt+[np.mean(sub_behav['STIM.RT'])]
            subject = subject+[i]

pc_run = []
for i in range(len(trmt)):
    pc_run = pc_run+[np.mean(projections[range(4*i,4*(i+1))])]

dic = {"Subject": subject,"treatment": trmt, "pca":pc_run,"child_rt":child_rt,"adult_rt":adult_rt,"child_acc": child_acc,"adult_acc":adult_acc}
pd.DataFrame(dic).to_csv("/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Rstudio/pca_rt_acc.csv",
                                               index=None)


#%%
rts = [child_rt, adult_rt]
names = []
for i in range(2):
    for j in range(len(rts[i])):
        if i==0: #child
            if (trmt[j]==1):
                names = names + ["OT child"]
            else:
                names = names + ["PL child"]
        else:
            if (trmt[j]==1):
                names = names + ["OT adult"]
            else:
                names = names + ["PL adult"]
rt_df = pd.DataFrame({"names":names,"rt":child_rt+adult_rt})

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(figsize=[10,10])
    sns.barplot(x="names",y="rt",data=rt_df, ax=ax, order=["OT adult", "OT child","PL adult","PL child"])
    ax.set_ylim(800,1000)


#%%
rt_OTadult = np.array(adult_rt)[np.array(trmt)==1]
rt_PLadult = np.array(adult_rt)[np.array(trmt)==0]
rt_OTchild = np.array(child_rt)[np.array(trmt)==1]
rt_PLchild = np.array(child_rt)[np.array(trmt)==0]

stats.ttest_ind(rt_OTadult, rt_PLadult)
stats.ttest_ind(rt_OTchild, rt_PLchild)


#%%
rt_01 = []
rt_02 = []
rt_03 = []
rt_04 = []

rt = []
face = []
child_face = []
self_face = []

subject=[]
trmt = []
for i in range(105,166+1):
    sub_behav = behav[behav['sub']==i]
    if (len(sub_behav)!=0):
        for j in range(2):
            sub_run = sub_behav[sub_behav['Run']==j+1]
            
            sub_01 = sub_run[(sub_run['face']=="sc")]
            sub_02 = sub_run[(sub_run['face']=="oc")]
            sub_03 = sub_run[(sub_run['face']=="sa")]
            sub_04 = sub_run[(sub_run['face']=="oa")]
            
            rt = rt + [np.mean(sub_01['STIM.RT'])]
            rt = rt + [np.mean(sub_02['STIM.RT'])]
            rt = rt + [np.mean(sub_03['STIM.RT'])]
            rt = rt + [np.mean(sub_04['STIM.RT'])]
            
            face = face + ["sc"]
            face = face + ["oc"]
            face = face + ["sa"]
            face = face + ["oa"]
            # both_rt = both_rt + [np.mean(sub_run['STIM.RT'])]
            
            child_face = child_face + [1]
            child_face = child_face + [1]
            child_face = child_face + [0]
            child_face = child_face + [0]
            
            self_face = self_face + [1]
            self_face = self_face + [0]
            self_face = self_face + [1]
            self_face = self_face + [0]
            
            if (np.array(treatment[treatment["ID"]==i]["Drug_name"])[0]=="OT"):
                trmt = trmt + ["OT","OT","OT","OT"]
            else:
                trmt = trmt + ["PL","PL","PL","PL"]
            # all_rt = all_rt+[np.mean(sub_behav['STIM.RT'])]
            subject = subject+[i,i,i,i]
            
rt_dic = {"Subject": subject,"Treatment": trmt, "RT":rt,"face":face,"child face":child_face,"self face":self_face}

rts = pd.DataFrame(rt_dic)

#%%
title_size = 40
axis_size  = 36

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(figsize=[10,10])
    #ax.set_title(f"Reaction time", fontsize=title_size, pad = 40)
    ax=sns.barplot(x="Treatment", y="RT", 
                   data=rts,  alpha=0.9, hue="face",
                   capsize=0.11, errwidth=2, n_boot=500,
                   palette={"sc":"#C15E53","oc":"#ef8b69","sa":"#1F819F","oa":"#A2B093"})
    handles, _ = ax.get_legend_handles_labels()
    ax=sns.swarmplot(x="Treatment", y="RT", data=rts,hue="face",dodge=True,
                     palette={"sc":"#C15E53","oc":"#ef8b69","sa":"#1F819F","oa":"#A2B093"})
    ax.get_legend().remove()
    ax.set_xticks([0, 1])
    # ax.set_xticklabels(["OT","PL"])
    ax.set_ylabel("Reaction time (ms)", fontsize=axis_size)
    ax.set_xlabel("Treatments", fontsize=axis_size)
    ax.set_ylim([500,1500])
    labels = ["Self-Child","Other-Child","Self-Adult","Other-Adult"]
    lgd=ax.legend(handles, labels,title='Facial Conditions',loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=1, fontsize=25)
    fig.savefig(f"figures/RT.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

#%%
OT_diff_a = rts[(rts["Treatment"]=="OT") & (rts["face"]=="sa")]["RT"].values - rts[(rts["Treatment"]=="OT") & (rts["face"]=="oa")]["RT"].values
PL_diff_a = rts[(rts["Treatment"]=="PL") & (rts["face"]=="sa")]["RT"].values - rts[(rts["Treatment"]=="PL") & (rts["face"]=="oa")]["RT"].values

stats.ttest_ind(OT_diff_a[~np.isnan(OT_diff_a)], PL_diff_a[~np.isnan(PL_diff_a)])

#%%
aov_OT = pg.rm_anova(data = rts.loc[rts.Treatment=="OT"], dv="RT",within=["child face","self face"], subject="Subject")
aov_PL = pg.rm_anova(data = rts.loc[rts.Treatment=="PL"], dv="RT",within=["child face","self face"], subject="Subject")


#%%
#RT2

subject=[]
trmt = []
rt2 = []
condition = []
child_face = []
response = []

for i in range(105,166+1):
    # recognized as self, child picture
    if (len(behav[behav['sub']==i])!=0):
        subject = subject + [i,i,i,i]
        
        subdf = behav.loc[behav["STIM.RESP"]==1].loc[(behav["face"]=="sc") | (behav["face"]=="oc")].loc[behav["sub"]==i]
        rt2 = rt2 + [np.mean(subdf["STIM.RT"])]
        condition = condition + ["s-c"]
        
        subdf = behav.loc[behav["STIM.RESP"]==2].loc[(behav["face"]=="sc") | (behav["face"]=="oc")].loc[behav["sub"]==i]
        rt2 = rt2 + [np.mean(subdf["STIM.RT"])]
        condition = condition + ["o-c"]
        
        subdf = behav.loc[behav["STIM.RESP"]==1].loc[(behav["face"]=="sa") | (behav["face"]=="oa")].loc[behav["sub"]==i]
        rt2 = rt2 + [np.mean(subdf["STIM.RT"])]
        condition = condition + ["s-a"]
        
        subdf = behav.loc[behav["STIM.RESP"]==2].loc[(behav["face"]=="sa") | (behav["face"]=="oa")].loc[behav["sub"]==i]
        rt2 = rt2 + [np.mean(subdf["STIM.RT"])]
        condition = condition + ["o-a"]
        
        child_face = child_face + [1,1,0,0]
        response = response + [1,0,1,0]
        
        if (np.array(treatment[treatment["ID"]==i]["Drug_name"])[0]=="OT"):
            trmt = trmt + ["OT","OT","OT","OT"]
        else:
            trmt = trmt + ["PL","PL","PL","PL"]

rt2_dic = {"Subject":subject, "Treatment": trmt, "RT2":rt2,"condition":condition, "child": child_face, "response": response}
rts2 = pd.DataFrame(rt2_dic)

#%%

with sns.plotting_context("poster"):
    fig, ax = plt.subplots(figsize=[10,10])
    ax.set_title(f"Reaction time (Response)", fontsize=title_size, pad = 40)
    ax=sns.barplot(x="Treatment", y="RT2", 
                   data=rts2,  alpha=0.9, hue="condition",
                   capsize=0.11, errwidth=2, n_boot=500,
                   palette={"s-c":"#C15E53","o-c":"#ef8b69","s-a":"#1F819F","o-a":"#A2B093"})
    handles, _ = ax.get_legend_handles_labels()
    ax=sns.swarmplot(x="Treatment", y="RT2", data=rts2,hue="condition",dodge=True,
                     palette={"s-c":"#C15E53","o-c":"#ef8b69","s-a":"#1F819F","o-a":"#A2B093"})
    ax.get_legend().remove()
    ax.set_xticks([0, 1])
    # ax.set_xticklabels(["OT","PL"])
    ax.set_ylabel("Reaction time (ms)", fontsize=axis_size)
    ax.set_xlabel("Treatments", fontsize=axis_size)
    ax.set_ylim([500,1500])
    labels = ["Self-Child","Other-Child","Self-Adult","Other-Adult"]
    lgd=ax.legend(handles, labels,title='Facial Conditions',loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1)
    fig.savefig(f"figures/RT2.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
#%%
OT_diff_a = rts2[(rts2["Treatment"]=="OT") & (rts2["condition"]=="s-a")]["RT2"].values - rts2[(rts2["Treatment"]=="OT") & (rts2["condition"]=="o-a")]["RT2"].values
PL_diff_a = rts2[(rts2["Treatment"]=="PL") & (rts2["condition"]=="s-a")]["RT2"].values - rts2[(rts2["Treatment"]=="PL") & (rts2["condition"]=="o-a")]["RT2"].values

stats.ttest_ind(OT_diff_a[~np.isnan(OT_diff_a)], PL_diff_a[~np.isnan(PL_diff_a)])

#%%
aov2_OT = pg.rm_anova(data = rts2.loc[rts2.Treatment=="OT"], dv="RT2",within=["child","response"], subject="Subject")
aov2_PL = pg.rm_anova(data = rts2.loc[rts2.Treatment=="PL"], dv="RT2",within=["child","response"], subject="Subject")
