#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST batch analysis of red/blue features and their coincidence across multiple TIFF images,
optionally restricted to a rectangular region of interest (ROI).

For each image:
- Optionally crop to ROI
- Threshold red/blue
- Label features
- Compute coincidence (red features overlapping blue)
- Compute simple per-spot summed intensities and summarise by mean/SD per image
- Compute feature densities (counts per µm^2) for red, blue, coincident spots

Across all images:
- Write a single image_summary_metrics.tsv in top_root_directory (updated after each image)
- Write summary_CV_metrics.tsv with CVs across all images/ROIs
- Write concentration_density_summary.tsv with per-concentration mean/SD of densities
- Save plots of red/blue/coincident densities vs concentration with:
    - per-image points
    - mean ± SD
    - linear fit and R^2
"""

import os
import glob
import numpy as np
from skimage.io import imread
from skimage import filters, measure
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =========================
# User parameters
# =========================

# Size of pixels (nm)
pixel_size = 591  # nm

# Root directory where you want the summary TSVs and plots saved
top_root_directory = r"/Users/Mathew/Documents/Current analysis/Ryan_conc/"

# List of directories to analyse (can contain many .tif/.tiff files)
pathList = [
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/0pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/1pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/3.16pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/10pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/31.6pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/100pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/316pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/1000pM/",
]

# Matching concentrations (same order as pathList)
concList = [0, 1, 3.16, 10, 31.6, 100,316,1000]

# Thresholds (you can choose Otsu or fixed)
USE_OTSU = False
RED_FIXED_THRESHOLD = 200
BLUE_FIXED_THRESHOLD = 200

# ---- Region of interest (ROI) ----
# If USE_ROI is True, crop each image to [ymin:ymax, xmin:xmax]
USE_ROI = False
ROI_X = (4000, 8000)   # (xmin, xmax) in pixels
ROI_Y = (4000, 8000)   # (ymin, ymax) in pixels

# Precompute pixel area in µm^2
pixel_size_um = pixel_size / 1000.0
pixel_area_um2 = pixel_size_um ** 2

# =========================
# Helper functions (fast)
# =========================

def load_image(toload):
    return imread(toload)

def threshold_image_otsu(input_image):
    thr = filters.threshold_otsu(input_image)
    binary = input_image > thr
    return thr, binary

def threshold_image_fixed(input_image, threshold_number):
    thr = threshold_number
    binary = input_image > thr
    return thr, binary

def label_binary(binary_image):
    labelled = measure.label(binary_image)
    n = labelled.max()
    return n, labelled

def feature_coincidence_fast(binary_ref, binary_other):
    """
    Fast coincidence:
    - Label binary_ref (e.g. red)
    - Determine which labels have any overlap with binary_other (e.g. blue)
    - Return:
        coincident_count: number of coincident features
        fraction_coinc:  fraction of ref features that are coincident
    """
    n_ref, labels_ref = label_binary(binary_ref)
    if n_ref == 0:
        return 0, 0.0

    overlap = binary_ref & binary_other
    overlapping_labels = np.unique(labels_ref[overlap])
    overlapping_labels = overlapping_labels[overlapping_labels > 0]  # remove background

    coincident_count = int(len(overlapping_labels))
    fraction_coinc = coincident_count / n_ref if n_ref > 0 else 0.0
    return coincident_count, fraction_coinc

def per_spot_summed_intensity(label_image, intensity_image):
    """
    Compute per-spot summed intensity given a label image and the corresponding intensity image.
    Returns a 1D array of length n_labels (excluding label 0).
    """
    n_labels = label_image.max()
    if n_labels == 0:
        return np.array([])

    labels_flat = label_image.ravel()
    intens_flat = intensity_image.ravel()

    sums = np.bincount(labels_flat, weights=intens_flat, minlength=n_labels + 1)
    return sums[1:]  # drop background

def mean_and_sd(values):
    """
    Safe mean and SD. Returns (mean, sd) or (nan, nan) if empty.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return np.nan, np.nan
    m = float(arr.mean())
    if arr.size > 1:
        s = float(arr.std(ddof=1))
    else:
        s = np.nan
    return m, s

# =========================
# Global collector for all images/ROIs
# =========================

summary_rows = []  # one row per image/ROI across all paths

# =========================
# Main processing loop
# =========================

for directory, conc in zip(pathList, concList):
    directory = os.path.abspath(directory)

    tiff_files = sorted(glob.glob(os.path.join(directory, "*.tif")) +
                        glob.glob(os.path.join(directory, "*.tiff")))

    if not tiff_files:
        print(f"No TIFF files found in {directory}")
        continue

    for filepath in tiff_files:
        img_name = os.path.basename(filepath)
        print(f"\nProcessing image: {img_name} (conc = {conc} pM)")

        img_base = os.path.splitext(img_name)[0]
        img_out_dir = os.path.join(directory, img_base + "_results")
        os.makedirs(img_out_dir, exist_ok=True)

        # Load image
        img = load_image(filepath)

        # Extract channels
        if img.ndim == 3 and img.shape[0] <= 4:  # assume (C, Y, X)
            red_full = img[0, :, :]
            blue_full = img[1, :, :]
        elif img.ndim == 3 and img.shape[-1] <= 4:  # assume (Y, X, C)
            red_full = img[:, :, 0]
            blue_full = img[:, :, 1]
        else:
            raise ValueError(f"Unexpected image shape for {img_name}: {img.shape}")

        # Apply ROI if requested
        if USE_ROI:
            ymin, ymax = ROI_Y
            xmin, xmax = ROI_X

            ny_full, nx_full = red_full.shape
            ymin = max(0, ymin)
            ymax = min(ny_full, ymax)
            xmin = max(0, xmin)
            xmax = min(nx_full, xmax)

            if ymin >= ymax or xmin >= xmax:
                raise ValueError(
                    f"Invalid ROI after clipping for {img_name}: "
                    f"ROI_X={ROI_X}, ROI_Y={ROI_Y}, image shape={red_full.shape}"
                )

            red = red_full[ymin:ymax, xmin:xmax]
            blue = blue_full[ymin:ymax, xmin:xmax]
            roi_info = f"ROI x={xmin}:{xmax}, y={ymin}:{ymax}"
        else:
            red = red_full
            blue = blue_full
            roi_info = "Full image"

        print(f"Analysing region: {roi_info}")

        # ROI area in µm^2
        ny, nx = red.shape
        roi_area_um2 = ny * nx * pixel_area_um2

        # Thresholding
        if USE_OTSU:
            red_threshold, red_binary = threshold_image_otsu(red)
            blue_threshold, blue_binary = threshold_image_otsu(blue)
        else:
            red_threshold, red_binary = threshold_image_fixed(red, RED_FIXED_THRESHOLD)
            blue_threshold, blue_binary = threshold_image_fixed(blue, BLUE_FIXED_THRESHOLD)

        # Label features
        red_number, red_labelled = label_binary(red_binary)
        blue_number, blue_labelled = label_binary(blue_binary)
        print("%d features were detected in the Red ROI." % red_number)
        print("%d features were detected in the Blue ROI." % blue_number)

        # Coincidence analysis (fast)
        coincident_count, fraction_coinc = feature_coincidence_fast(red_binary, blue_binary)
        print(
            "%.2f of red ROI features had coincidence with features in blue ROI. "
            "Number of coincident spots: %d"
            % (fraction_coinc, coincident_count)
        )

        # Densities (features per µm^2)
        if roi_area_um2 > 0:
            red_density = red_number / roi_area_um2
            blue_density = blue_number / roi_area_um2
            coincident_density = coincident_count / roi_area_um2
        else:
            red_density = np.nan
            blue_density = np.nan
            coincident_density = np.nan

        # Simple per-spot summed intensities
        red_spot_sums = per_spot_summed_intensity(red_labelled, red)
        blue_spot_sums = per_spot_summed_intensity(blue_labelled, blue)

        red_sum_mean, red_sum_sd = mean_and_sd(red_spot_sums)
        blue_sum_mean, blue_sum_sd = mean_and_sd(blue_spot_sums)

        row = {
            'image_path': filepath,
            'image_name': img_name,
            'folder': directory,
            'concentration_pM': conc,
            'roi_info': roi_info,
            'red_threshold': red_threshold,
            'blue_threshold': blue_threshold,

            # Raw counts
            'red_count': red_number,
            'blue_count': blue_number,
            'coincident_count': coincident_count,
            'fraction_red_coincident': fraction_coinc,

            # Densities (features per µm^2)
            'red_density_per_um2': red_density,
            'blue_density_per_um2': blue_density,
            'coincident_density_per_um2': coincident_density,

            # Basic intensity metrics: per-spot summed intensities
            'red_spot_sum_mean': red_sum_mean,
            'red_spot_sum_sd': red_sum_sd,
            'blue_spot_sum_mean': blue_sum_mean,
            'blue_spot_sum_sd': blue_sum_sd,
        }

        summary_rows.append(row)

        # Write/update global image_summary_metrics.tsv in top_root_directory after each image
        summary_df = pd.DataFrame(summary_rows)
        os.makedirs(top_root_directory, exist_ok=True)
        summary_path = os.path.join(top_root_directory, "image_summary_metrics.tsv")
        summary_df.to_csv(summary_path, sep='\t', index=False)
        print(f"Updated per-image summary metrics at {summary_path}")

# =========================
# Global CV summary and concentration plots (after all images/ROIs)
# =========================

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)

    # -------------------------
    # 1) Global CV summary (densities + intensity)
    # -------------------------
    cv_metrics = [
        'red_density_per_um2',
        'blue_density_per_um2',
        'coincident_density_per_um2',
        'red_spot_sum_mean',
        'blue_spot_sum_mean'
    ]

    cv_rows = []
    for col in cv_metrics:
        values = summary_df[col].dropna().values.astype(float)
        if len(values) == 0:
            mean_val = np.nan
            sd_val = np.nan
            cv_val = np.nan
        else:
            mean_val = float(values.mean())
            sd_val = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            cv_val = sd_val / mean_val if mean_val != 0 else np.nan

        cv_rows.append({
            'metric': col,
            'mean_across_images_or_rois': mean_val,
            'sd_across_images_or_rois': sd_val,
            'cv_across_images_or_rois': cv_val
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_path = os.path.join(top_root_directory, "summary_CV_metrics.tsv")
    os.makedirs(os.path.dirname(cv_path), exist_ok=True)
    cv_df.to_csv(cv_path, sep='\t', index=False)
    print(f"Saved CV metrics across images/ROIs to {cv_path}")

    # -------------------------
    # 2) Per-concentration summaries and plots (densities vs conc)
    # -------------------------

    # Ensure concentration_pM is numeric
    summary_df['concentration_pM'] = pd.to_numeric(summary_df['concentration_pM'])

    # Metrics we want vs concentration (densities)
    density_metrics = [
        'red_density_per_um2',
        'blue_density_per_um2',
        'coincident_density_per_um2'
    ]
    conc_group = summary_df.groupby('concentration_pM')

    conc_summary_rows = []
    for conc_val, group in conc_group:
        row = {'concentration_pM': conc_val}
        for m in density_metrics:
            vals = group[m].dropna().values.astype(float)
            if len(vals) == 0:
                mean_val = np.nan
                sd_val = np.nan
            else:
                mean_val = float(vals.mean())
                sd_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f'{m}_mean'] = mean_val
            row[f'{m}_sd'] = sd_val
        conc_summary_rows.append(row)

    conc_summary_df = pd.DataFrame(conc_summary_rows).sort_values('concentration_pM')

    # Save concentration vs mean/SD table (densities)
    conc_summary_path = os.path.join(top_root_directory, "concentration_density_summary.tsv")
    conc_summary_df.to_csv(conc_summary_path, sep='\t', index=False)
    print(f"Saved concentration vs mean/SD density summary to {conc_summary_path}")

    # -------------------------
    # 3) Plots: densities vs concentration with linear fit and R^2
    # -------------------------

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11
    })

    for metric in density_metrics:
        plt.figure(figsize=(5, 4))

        # Scatter all image points: x = conc, y = metric
        plt.scatter(summary_df['concentration_pM'],
                    summary_df[metric],
                    alpha=0.5,
                    s=30,
                    color='0.5',
                    edgecolors='none',
                    label='Individual images')

        # Mean ± SD per concentration
        x = conc_summary_df['concentration_pM'].values
        y_mean = conc_summary_df[f'{metric}_mean'].values
        y_sd = conc_summary_df[f'{metric}_sd'].values

        # Mean points with error bars
        plt.errorbar(x, y_mean, yerr=y_sd,
                     fmt='s',
                     color='black',
                     ecolor='black',
                     elinewidth=1.5,
                     capsize=4,
                     label='Mean ± SD')

        # Linear fit on mean values (ignoring NaNs)
        valid = ~np.isnan(x) & ~np.isnan(y_mean)
        if valid.sum() >= 2:
            slope, intercept, r_value, p_value, stderr = linregress(x[valid], y_mean[valid])
            x_fit = np.linspace(x[valid].min(), x[valid].max(), 200)
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, color='red', linewidth=1.5,
                     label=f'Linear fit, $ R^2 $ = {r_value**2:.2f}')
        else:
            slope = intercept = r_value = np.nan

        # Labels and styling
        plt.xlabel('Concentration (pM)')
        ylabel = metric.replace('_density_per_um2', ' density / µm²').replace('_', ' ')
        plt.ylabel(ylabel)
        plt.title(metric.replace('_density_per_um2', ' density vs concentration').replace('_', ' '))
        plt.tight_layout()
        plt.legend(frameon=False)

        # Save figure
        fig_name = f"{metric}_vs_concentration.png"
        fig_path = os.path.join(top_root_directory, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot {fig_path}")

else:
    print("No images processed; no summary files created.")
