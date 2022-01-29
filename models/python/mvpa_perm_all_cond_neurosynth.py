N_PL = 29
N_OT = 30

import multiprocessing as mp
import numpy as np
import nibabel as nib
from tqdm import tqdm
import statsmodels.stats.multitest as fdr

np.random.seed(2021)

# set variables later used in parallel computing
file_prefix = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/MVPA-results/roi/"
mask_filename = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/Analysis/mask_all.nii"
output_prefix = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/python/mvpa_perm/final/roi/"

# mask of data
mask = nib.load(mask_filename)
maskdata = mask.get_fdata()
affine = mask.affine

# number of permutated images
n = 200
# # empty holder for all permutated images
acc_OT = np.empty((n,53,63,46), float)
acc_PL = np.empty((n,53,63,46), float)

# load the actual accuracy, +50 as the baseline accuracy is 50% (1/2)
actual_OT = np.array(nib.load(file_prefix + "OT/actual/res_accuracy_minus_chance.img").get_fdata()) + 50
actual_PL = np.array(nib.load(file_prefix + "PL/actual/res_accuracy_minus_chance.img").get_fdata()) + 50

filtered_OT = actual_OT + 0
filtered_OT[filtered_OT<50] = 50
filtered_PL = actual_PL + 0
filtered_PL[filtered_PL<50] = 50
actual_diff = np.subtract(filtered_OT, filtered_PL)

# load all permutated images
for i in range(n):
    filepath = file_prefix + "OT/perm" + str(i+1) + "/res_accuracy_minus_chance.img"
    temp = nib.load(filepath).get_fdata()
    acc_OT[i] = np.array(temp) + 50

    filepath = file_prefix + "PL/perm" + str(i+1) + "/res_accuracy_minus_chance.img"
    temp = nib.load(filepath).get_fdata()
    acc_PL[i] = np.array(temp) + 50

# number of bootstraps
boot_n = 10000

def bootstrap(b):
    # selected permutations for this simulation
    boot_choice = acc_OT[np.random.choice(range(n), size=N_OT*8)]
    # the mean accuracy of the simulation
    mean_acc = np.mean(boot_choice, axis=0)
    # ext: "extreme", array for if the mean simulation accuracy is more extreme than the "actual"
    # 1 on the voxel if this simulation is >= "actual" accuracy (depend on conditions)
    # 0 o.w.
    ext_OT = np.array(mean_acc>=actual_OT, dtype="int")

    boot_choice = acc_PL[np.random.choice(range(n), size=N_PL * 8)]
    mean_acc = np.mean(boot_choice, axis=0)
    ext_PL = np.array(mean_acc >= actual_PL, dtype="int")

    boot_choice = acc_OT[np.random.choice(range(n), size=N_OT * 8)]
    mean_OT = np.mean(boot_choice, axis=0)
    mean_OT[mean_OT < 50] = 50
    boot_choice = acc_PL[np.random.choice(range(n), size=N_PL * 8)]
    mean_PL = np.mean(boot_choice, axis=0)
    mean_PL[mean_OT < 50] = 50
    mean_diff = np.subtract(mean_OT, mean_PL)
    ext_diff = np.array(np.abs(mean_diff) >= np.abs(actual_diff), dtype="int")

    return list([ext_OT, ext_PL, ext_diff])

def run_process():
    count_OT = np.ones([53, 63, 46])
    count_PL = np.ones([53, 63, 46])
    count_diff = np.ones([53, 63, 46])
    print("Calculating...")
    with mp.Pool(processes=28) as pool:
        with tqdm(total=boot_n) as pbar:
            for i, exts in enumerate(pool.imap_unordered(bootstrap, range(boot_n))):
                count_OT += exts[0]
                count_PL += exts[1]
                count_diff += exts[2]
                pbar.update()
    p_OT = count_OT / (boot_n + 1)
    p_PL = count_PL / (boot_n + 1)
    p_diff = count_diff / (boot_n + 1)
    return p_OT, p_PL, p_diff



if __name__ == '__main__':
    def save_figures(output_subpath, actual, p_value):
        output_path = output_subpath + "_p_unadjusted.nii"
        img = nib.Nifti1Image(p_value, affine)
        nib.save(img, output_path)

        # fdr correction of the p-value (B-H method)
        filtered_p = p_value[maskdata != 0].flatten()
        rejected, adjusted = fdr.fdrcorrection(filtered_p, alpha=0.05)
        recon_rejected = np.zeros((53, 63, 46))
        recon_adjusted = np.ones((53, 63, 46))
        filtered_acc = np.zeros((53, 63, 46))
        l = 0
        for i in range(53):
            for j in range(63):
                for k in range(46):
                    if maskdata[i, j, k] != 0:
                        recon_adjusted[i, j, k] = adjusted[l]
                        if rejected[l]:
                            filtered_acc[i, j, k] = actual[i, j, k]
                            recon_rejected[i, j, k] = 1
                        l += 1

        # save the adjusted p-values
        output_path = output_subpath + "_p_adjusted.nii"
        img = nib.Nifti1Image(recon_adjusted, affine)
        nib.save(img, output_path)

        output_path = output_subpath + "_masked_accuracies.nii"
        img = nib.Nifti1Image(filtered_acc, affine)
        nib.save(img, output_path)


    p_OT, p_PL, p_diff = run_process()
    save_figures(f"{output_prefix}OT", actual_OT, p_OT)
    save_figures(f"{output_prefix}PL", actual_PL, p_PL)
    save_figures(f"{output_prefix}diff", actual_diff, p_diff)

