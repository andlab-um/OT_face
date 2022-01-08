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
from IPython.display import display
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

df = pd.DataFrame(columns=['ROI', 'Treatment', 'Subject', 'beta', 'Signal Change', 'constant'])

for j in range(len(roi_nums)):
    roi = np.array(aal3 == roi_nums[j])
    roi_flat = np.ndarray.flatten(roi)

    constant = {}
    for i in range(105, 166 + 1):
        path1 = root + str(i) + "/beta_0021.img"
        path2 = root + str(i) + "/beta_0022.img"
        if (os.path.isfile(path1)):
            avg = (np.ndarray.flatten(nib.load(path1).get_fdata()) + np.ndarray.flatten(
                nib.load(path2).get_fdata())) / 2
            constant[str(i)] = np.nanmean(avg[roi_flat != 0])

    for i in range(105, 166 + 1):
        path1 = root + str(i) + "/beta_0001.img"
        path2 = root + str(i) + "/beta_0011.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "01-sc", "Signal Change": change1,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "01-sc", "Signal Change": change2,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)

        path1 = root + str(i) + "/beta_0002.img"
        path2 = root + str(i) + "/beta_0012.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "02-oc", "Signal Change": change1,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "02-oc", "Signal Change": change2,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)

        path1 = root + str(i) + "/beta_0003.img"
        path2 = root + str(i) + "/beta_0013.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "03-sa", "Signal Change": change1,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "03-sa", "Signal Change": change2,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)

        path1 = root + str(i) + "/beta_0004.img"
        path2 = root + str(i) + "/beta_0014.img"
        if (os.path.isfile(path1)):
            change1 = np.nanmean(np.ndarray.flatten(nib.load(path1).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            change2 = np.nanmean(np.ndarray.flatten(nib.load(path2).get_fdata())[roi_flat != 0]) / constant[
                str(i)] * 100
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "04-oa", "Signal Change": change1,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)
            df = df.append({"ROI": roi_names[j], "Subject": i, "beta": "04-oa", "Signal Change": change2,
                            "Treatment": treatment.loc[treatment.ID == i, 'Drug_name'].values[0],
                            "constant": constant[str(i)]}, ignore_index=True)


k=2