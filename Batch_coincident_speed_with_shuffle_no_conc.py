#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST batch analysis of red/blue features and their coincidence across multiple TIFF images,
optionally restricted to a rectangular region of interest (ROI).

For each image:
- Optionally crop to ROI
- Threshold red/blue (Otsu, Yen, or fixed)
- Label features
- Compute coincidence (red features overlapping blue)
- Compute simple per-spot summed intensities and summarise by mean/SD per image
- Compute feature densities (counts per µm^2) for red, blue, coincident spots
- Compute rotated/translated (90° + 1000 px, wrap-around) control coincidence

Across all images:
- Write a single image_summary_metrics.tsv in top_root_directory (updated after each image)
"""

import os
import glob
import numpy as np
from skimage.io import imread
from skimage import filters, measure, transform
import pandas as pd

# =========================
# User parameters
# =========================

# Size of pixels (nm)
pixel_size = 591  # nm

# Root directory where you want the summary TSVs saved
top_root_directory = r"/Users/Mathew/Documents/Current analysis/Ryan_conc/"

# List of directories to analyse (can contain many .tif/.tiff files)
pathList = [
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/0pM/",
    r"/Users/Mathew/Documents/Current analysis/Ryan_conc/1pM/",
]

# ---- Thresholding options ----
# Choose: "otsu", "yen", or "fixed"
THRESHOLD_MODE = "fixed"
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

def threshold_image(image, mode="fixed", fixed_value=None):
    """
    Threshold an image using 'otsu', 'yen', or 'fixed'.

    Returns:
      thr   - threshold value (float)
      binary - boolean mask
    """
    mode = mode.lower()
    if mode == "otsu":
        thr = filters.threshold_otsu(image)
    elif mode == "yen":
        thr = filters.threshold_yen(image)
    elif mode == "fixed":
        if fixed_value is None:
            raise ValueError("fixed_value must be provided when mode='fixed'")
        thr = fixed_value
    else:
        raise ValueError(f"Unknown threshold mode '{mode}'. Use 'otsu', 'yen', or 'fixed'.")
    binary = image > thr
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

def rotate_translate_mask_wrap(binary_mask, angle_deg=90, shift_x=1000, shift_y=0):
    """
    Rotate a binary mask by angle_deg around its center, then translate by (shift_x, shift_y) pixels
    with wrap-around (toroidal) boundary conditions.

    Returns a binary mask with the same shape.
    """
    # Rotate with constant fill
    rotated = transform.rotate(
        binary_mask.astype(float),
        angle=angle_deg,
        resize=False,
        center=None,
        order=0,             # nearest-neighbour for binary
        mode='constant',
        cval=0.0,
        preserve_range=True
    )

    # Translate with wrap-around
    tform = transform.AffineTransform(translation=(shift_x, shift_y))
    shifted = transform.warp(
        rotated,
        inverse_map=tform.inverse,
        order=0,
        mode='wrap',        # wrap-around
        preserve_range=True
    )

    return shifted > 0.5

# =========================
# Global collector for all images/ROIs
# =========================

summary_rows = []  # one row per image/ROI across all paths

# =========================
# Main processing loop
# =========================

for directory in pathList:
    directory = os.path.abspath(directory)

    tiff_files = sorted(glob.glob(os.path.join(directory, "*.tif")) +
                        glob.glob(os.path.join(directory, "*.tiff")))

    if not tiff_files:
        print(f"No TIFF files found in {directory}")
        continue

    for filepath in tiff_files:
        img_name = os.path.basename(filepath)
        print(f"\nProcessing image: {img_name}")

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

        # Thresholding (same mode for both channels)
        red_threshold, red_binary = threshold_image(
            red,
            mode=THRESHOLD_MODE,
            fixed_value=RED_FIXED_THRESHOLD
        )
        blue_threshold, blue_binary = threshold_image(
            blue,
            mode=THRESHOLD_MODE,
            fixed_value=BLUE_FIXED_THRESHOLD
        )

        # Label features
        red_number, red_labelled = label_binary(red_binary)
        blue_number, blue_labelled = label_binary(blue_binary)
        print("%d features were detected in the Red ROI." % red_number)
        print("%d features were detected in the Blue ROI." % blue_number)

        # Coincidence analysis (fast) - "normal" coincidence
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

        # Rotated + translated (wrap) control coincidence
        if roi_area_um2 > 0 and red_number > 0 and blue_number > 0:
            blue_rt = rotate_translate_mask_wrap(blue_binary, angle_deg=90, shift_x=1000, shift_y=0)
            coincident_count_rt, _ = feature_coincidence_fast(red_binary, blue_rt)
            coincident_density_rt = coincident_count_rt / roi_area_um2
        else:
            coincident_count_rt = 0
            coincident_density_rt = np.nan

        # Simple per-spot summed intensities
        red_spot_sums = per_spot_summed_intensity(red_labelled, red)
        blue_spot_sums = per_spot_summed_intensity(blue_labelled, blue)

        red_sum_mean, red_sum_sd = mean_and_sd(red_spot_sums)
        blue_sum_mean, blue_sum_sd = mean_and_sd(blue_spot_sums)

        row = {
            'image_path': filepath,
            'image_name': img_name,
            'folder': directory,
            'roi_info': roi_info,
            'red_threshold': red_threshold,
            'blue_threshold': blue_threshold,
            'threshold_mode': THRESHOLD_MODE,

            # Raw counts
            'red_count': red_number,
            'blue_count': blue_number,
            'coincident_count': coincident_count,
            'coincident_count_rot90px1000': coincident_count_rt,
            'fraction_red_coincident': fraction_coinc,

            # Densities (features per µm^2)
            'red_density_per_um2': red_density,
            'blue_density_per_um2': blue_density,
            'coincident_density_per_um2': coincident_density,
            'coincident_density_rot90px1000_per_um2': coincident_density_rt,

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

if not summary_rows:
    print("No images processed; no summary file created.")
