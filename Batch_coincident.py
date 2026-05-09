#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch analysis of red/blue features and their coincidence across multiple TIFF images,
optionally restricted to a rectangular region of interest (ROI).

For each image:
- Optionally crop to ROI
- Threshold red/blue
- Label features
- Compute coincidence (red features overlapping blue)
- Save binary masks, coincident mask, density maps, and coincident-feature metrics

Across all images:
- Summarise per-image counts and mean metrics in a TSV (updated after each image)
- Compute CVs of key metrics across images in another TSV
"""

import os
import glob
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters, measure
from PIL import Image
import pandas as pd
from scipy.spatial import distance

# =========================
# User parameters
# =========================

# Size of pixels (nm)
pixel_size = 591

# Root directory where you want summary results saved
root_directory = r"/Users/Mathew/Documents/Current analysis/Ryan_conc/"

# List of directories to analyse (can contain many .tif/.tiff files)
pathList = []

pathList.append(r"/Users/Mathew/Documents/Current analysis/Ryan_conc/0pM/")
pathList.append(r"/Users/Mathew/Documents/Current analysis/Ryan_conc/1pM/")
pathList.append(r"/Users/Mathew/Documents/Current analysis/Ryan_conc/3.16pM/")
pathList.append(r"/Users/Mathew/Documents/Current analysis/Ryan_conc/10pM/")
pathList.append(r"/Users/Mathew/Documents/Current analysis/Ryan_conc/31.6pM/")
pathList.append(r"/Users/Mathew/Documents/Current analysis/Ryan_conc/100pM/")

# Thresholds (you can choose Otsu or fixed)
USE_OTSU = False
RED_FIXED_THRESHOLD = 200
BLUE_FIXED_THRESHOLD = 200

# ---- Region of interest (ROI) ----
# If USE_ROI is True, crop each image to [ymin:ymax, xmin:xmax]
USE_ROI = True
ROI_X = (4000, 8000)   # (xmin, xmax) in pixels
ROI_Y = (4000, 8000)   # (ymin, ymax) in pixels

# =========================
# Helper functions
# =========================

def load_image(toload):
    image = imread(toload)
    return image

def threshold_image_otsu(input_image):
    threshold_value = filters.threshold_otsu(input_image)
    binary_image = input_image > threshold_value
    return threshold_value, binary_image

def threshold_image_fixed(input_image, threshold_number):
    threshold_value = threshold_number
    binary_image = input_image > threshold_value
    return threshold_value, binary_image

def label_image(input_image):
    labelled_image = measure.label(input_image)
    number_of_features = labelled_image.max()
    return number_of_features, labelled_image

def analyse_labelled_image(labelled_image, original_image):
    measure_image = measure.regionprops_table(
        labelled_image,
        intensity_image=original_image,
        properties=('area',
                    'perimeter',
                    'centroid',
                    'orientation',
                    'major_axis_length',
                    'minor_axis_length',
                    'mean_intensity',
                    'max_intensity')
    )
    measure_dataframe = pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

def coincidence_analysis_pixels(binary_image1, binary_image2):
    pixel_overlap_image = binary_image1 & binary_image2
    pixel_overlap_count = pixel_overlap_image.sum()
    if binary_image1.sum() > 0:
        pixel_fraction = pixel_overlap_image.sum() / binary_image1.sum()
    else:
        pixel_fraction = 0.0
    return pixel_overlap_image, pixel_overlap_count, pixel_fraction

def feature_coincidence(binary_image1, binary_image2):
    """
    Treat binary_image1 as the reference (e.g. red).

    Returns:
      coinc_list: labels (including 0 for background) that have overlap
      coinc_pixels: number of overlapping pixels per label in coinc_list
      fraction_coinc: fraction of binary_image1 features that have any overlap
      coincident_features_image: bool mask of overlapping features in binary_image1
      non_coincident_features_image: bool mask of non-overlapping features in binary_image1
      fract_pixels_overlap: per-feature fraction of its pixels that overlap
      coincident_count: number of coincident features (labels > 0)
    """
    number_of_features, labelled_image1 = label_image(binary_image1)

    if number_of_features == 0:
        empty = np.zeros_like(binary_image1, dtype=bool)
        return (np.array([]), np.array([]), 0.0,
                empty, ~empty, [], 0)

    coincident_image = binary_image1 & binary_image2
    coincident_labels = labelled_image1 * coincident_image

    # Unique labels in coincident image
    coinc_list, coinc_pixels = np.unique(coincident_labels, return_counts=True)

    # Map label -> total size in original labelled_image1
    label_list, label_pixels = np.unique(labelled_image1, return_counts=True)
    label_to_size = dict(zip(label_list, label_pixels))

    fract_pixels_overlap = []
    for label, overlap_pixels in zip(coinc_list, coinc_pixels):
        if label == 0:
            continue  # skip background
        total_pixels = label_to_size[label]
        fract_pixels_overlap.append(overlap_pixels / total_pixels)

    total_labels = labelled_image1.max()
    total_labels_coinc = np.sum(coinc_list != 0)
    fraction_coinc = total_labels_coinc / total_labels if total_labels > 0 else 0.0

    # Coincident features in binary_image1
    coinc_labels = coinc_list[coinc_list != 0]
    coincident_features_image = np.isin(labelled_image1, coinc_labels)
    non_coincident_features_image = np.isin(labelled_image1, coinc_labels, invert=True)

    coincident_count = int(total_labels_coinc)  # number of coincident spots

    return (coinc_list,
            coinc_pixels,
            fraction_coinc,
            coincident_features_image,
            non_coincident_features_image,
            fract_pixels_overlap,
            coincident_count)

def minimum_distance(measurements1, measurements2):
    s1 = measurements1[["centroid-0", "centroid-1"]].to_numpy()
    s2 = measurements2[["centroid-0", "centroid-1"]].to_numpy()
    minimum_lengths = distance.cdist(s1, s2).min(axis=1)
    return minimum_lengths

def plot_feature_density_maps(red_labelled,
                              blue_labelled,
                              coincident_features_image,
                              directory,
                              nbins_x=32,
                              nbins_y=32,
                              filename="density_maps.png"):
    """
    Generate and save density maps of feature centroids for:
      - red_labelled
      - blue_labelled
      - coincident_features_image (binary)
    """
    from skimage import measure

    # Red centroids
    if red_labelled.max() > 0:
        red_props = measure.regionprops_table(red_labelled, properties=('centroid',))
        red_y = red_props['centroid-0']
        red_x = red_props['centroid-1']
    else:
        red_y = np.array([])
        red_x = np.array([])

    # Blue centroids
    if blue_labelled.max() > 0:
        blue_props = measure.regionprops_table(blue_labelled, properties=('centroid',))
        blue_y = blue_props['centroid-0']
        blue_x = blue_props['centroid-1']
    else:
        blue_y = np.array([])
        blue_x = np.array([])

    # Coincident centroids
    coincident_labelled = measure.label(coincident_features_image)
    if coincident_labelled.max() > 0:
        coinc_props = measure.regionprops_table(coincident_labelled, properties=('centroid',))
        coinc_y = coinc_props['centroid-0']
        coinc_x = coinc_props['centroid-1']
    else:
        coinc_y = np.array([])
        coinc_x = np.array([])

    ny, nx = red_labelled.shape
    x_edges = np.linspace(0, nx, nbins_x + 1)
    y_edges = np.linspace(0, ny, nbins_y + 1)

    if len(red_x) > 0:
        red_density, _, _ = np.histogram2d(red_y, red_x, bins=[y_edges, x_edges])
    else:
        red_density = np.zeros((nbins_y, nbins_x))

    if len(blue_x) > 0:
        blue_density, _, _ = np.histogram2d(blue_y, blue_x, bins=[y_edges, x_edges])
    else:
        blue_density = np.zeros((nbins_y, nbins_x))

    if len(coinc_x) > 0:
        coinc_density, _, _ = np.histogram2d(coinc_y, coinc_x, bins=[y_edges, x_edges])
    else:
        coinc_density = np.zeros((nbins_y, nbins_x))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    vmax = max(red_density.max(), blue_density.max(), coinc_density.max())
    if vmax == 0:
        vmax = 1

    im0 = axes[0].imshow(red_density, cmap='Reds', origin='lower',
                         extent=[0, nx, 0, ny], vmin=0, vmax=vmax)
    axes[0].set_title('Red feature density')
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')

    im1 = axes[1].imshow(blue_density, cmap='Blues', origin='lower',
                         extent=[0, nx, 0, ny], vmin=0, vmax=vmax)
    axes[1].set_title('Blue feature density')
    axes[1].set_xlabel('x (pixels)')
    axes[1].set_yticklabels([])

    im2 = axes[2].imshow(coinc_density, cmap='Purples', origin='lower',
                         extent=[0, nx, 0, ny], vmin=0, vmax=vmax)
    axes[2].set_title('Coincident feature density')
    axes[2].set_xlabel('x (pixels)')
    axes[2].set_yticklabels([])

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Feature count per bin')

    out_path = os.path.join(directory, filename)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def analyse_coincident_features(red_image,
                                blue_image,
                                red_labelled,
                                coincident_features_image,
                                directory,
                                pixel_size_nm=None,
                                csv_name="coincident_features.csv",
                                hist_prefix="coincident_hist_"):
    """
    Analyse coincident features (red objects that overlap blue).
    Produces:
      - CSV with x, y, red_intensity, blue_intensity, total_intensity,
        length_pixels, area_pixels, eccentricity, (optional area_um2).
      - Histograms for area, length, red/blue/total intensity.
    """
    from skimage import measure

    # Restrict red_labelled to coincident regions
    coincident_labels = red_labelled * coincident_features_image.astype(red_labelled.dtype)
    coincident_labelled = measure.label(coincident_labels)
    n_features = coincident_labelled.max()

    if n_features == 0:
        print("No coincident features found; skipping coincident feature analysis.")
        return

    props_red = measure.regionprops_table(
        coincident_labelled,
        intensity_image=red_image,
        properties=('label',
                    'centroid',
                    'area',
                    'major_axis_length',
                    'eccentricity',
                    'intensity_image')
    )
    df_red = pd.DataFrame(props_red)
    red_sums = [np.sum(arr) for arr in df_red['intensity_image']]
    df_red['red_intensity'] = red_sums
    df_red.drop(columns=['intensity_image'], inplace=True)

    props_blue = measure.regionprops_table(
        coincident_labelled,
        intensity_image=blue_image,
        properties=('label',
                    'intensity_image')
    )
    df_blue = pd.DataFrame(props_blue)
    blue_sums = [np.sum(arr) for arr in df_blue['intensity_image']]
    df_blue['blue_intensity'] = blue_sums
    df_blue.drop(columns=['intensity_image'], inplace=True)

    df = pd.merge(df_red, df_blue, on='label', how='inner')
    df.rename(columns={
        'centroid-0': 'y',
        'centroid-1': 'x',
        'area': 'area_pixels',
        'major_axis_length': 'length_pixels'
    }, inplace=True)
    df['total_intensity'] = df['red_intensity'] + df['blue_intensity']

    if pixel_size_nm is not None:
        pixel_size_um = pixel_size_nm / 1000.0
        pixel_area_um2 = pixel_size_um ** 2
        df['area_um2'] = df['area_pixels'] * pixel_area_um2

    cols = ['label', 'x', 'y',
            'red_intensity', 'blue_intensity', 'total_intensity',
            'length_pixels', 'area_pixels', 'eccentricity']
    if 'area_um2' in df.columns:
        cols.append('area_um2')
    df = df[cols]

    csv_path = os.path.join(directory, csv_name)
    df.to_csv(csv_path, sep='\t', index=False)
    print(f"Saved coincident feature table to {csv_path}")

    # Histograms
    def save_hist(column, xlabel, fname, log=False):
        plt.figure(figsize=(4, 3))
        values = df[column].values
        plt.hist(values, bins=30, color='gray', edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        if log:
            plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(directory, fname), dpi=300)
        plt.close()

    save_hist('area_pixels', 'Area (pixels)', f"{hist_prefix}area_pixels.png", log=False)
    save_hist('length_pixels', 'Length (pixels)', f"{hist_prefix}length_pixels.png", log=False)
    save_hist('red_intensity', 'Red summed intensity', f"{hist_prefix}red_intensity.png", log=True)
    save_hist('blue_intensity', 'Blue summed intensity', f"{hist_prefix}blue_intensity.png", log=True)
    save_hist('total_intensity', 'Total summed intensity', f"{hist_prefix}total_intensity.png", log=True)

    print("Saved coincident feature histograms.")

# =========================
# Summary tables at root level
# =========================

summary_rows = []   # one row per image/ROI
summary_path = os.path.join(root_directory, "image_summary_metrics.tsv")

# =========================
# Main processing loop
# =========================

for path in pathList:
    directory = os.path.abspath(path)
    root_directory=path

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

            # Clip ROI to image bounds
            ny, nx = red_full.shape
            ymin = max(0, ymin)
            ymax = min(ny, ymax)
            xmin = max(0, xmin)
            xmax = min(nx, xmax)

            if ymin >= ymax or xmin >= xmax:
                raise ValueError(f"Invalid ROI after clipping for {img_name}: "
                                 f"ROI_X={ROI_X}, ROI_Y={ROI_Y}, image shape={red_full.shape}")

            red = red_full[ymin:ymax, xmin:xmax]
            blue = blue_full[ymin:ymax, xmin:xmax]
            roi_info = f"ROI x={xmin}:{xmax}, y={ymin}:{ymax}"
        else:
            red = red_full
            blue = blue_full
            roi_info = "Full image"

        print(f"Analysing region: {roi_info}")

        # Thresholding
        if USE_OTSU:
            red_threshold, red_binary = threshold_image_otsu(red)
            blue_threshold, blue_binary = threshold_image_otsu(blue)
        else:
            red_threshold, red_binary = threshold_image_fixed(red, RED_FIXED_THRESHOLD)
            blue_threshold, blue_binary = threshold_image_fixed(blue, BLUE_FIXED_THRESHOLD)

        # Save binary masks (ROI only)
        Image.fromarray((red_binary.astype(np.uint8) * 255)).save(
            os.path.join(img_out_dir, 'Red_Binary.tif')
        )
        Image.fromarray((blue_binary.astype(np.uint8) * 255)).save(
            os.path.join(img_out_dir, 'Blue_Binary.tif')
        )

        # Label features
        red_number, red_labelled = label_image(red_binary)
        blue_number, blue_labelled = label_image(blue_binary)
        print("%d features were detected in the Red ROI." % red_number)
        print("%d features were detected in the Blue ROI." % blue_number)

        # Measure metrics (ROI)
        red_measurements = analyse_labelled_image(red_labelled, red)
        blue_measurements = analyse_labelled_image(blue_labelled, blue)

        # Coincidence analysis (ROI)
        (coinc_list,
         coinc_pixels,
         fraction_coinc,
         coincident_features_image,
         noncoincident_features_image,
         fraction_pixels_overlap,
         coincident_count) = feature_coincidence(red_binary, blue_binary)

        if len(fraction_pixels_overlap) > 0:
            avg_overlap = sum(fraction_pixels_overlap) / len(fraction_pixels_overlap)
        else:
            avg_overlap = 0.0

        print("%.2f of red ROI features had coincidence with features in blue ROI. "
              "Average overlap was %2f. Number of coincident spots: %d"
              % (fraction_coinc, avg_overlap, coincident_count))

        # Save coincident mask (ROI)
        Image.fromarray((coincident_features_image.astype(np.uint8) * 255)).save(
            os.path.join(img_out_dir, 'Coincident_Features.tif')
        )

        # Density maps (ROI)
        plot_feature_density_maps(
            red_labelled=red_labelled,
            blue_labelled=blue_labelled,
            coincident_features_image=coincident_features_image,
            directory=img_out_dir,
            nbins_x=32,
            nbins_y=32,
            filename="density_maps.png"
        )

        # Detailed coincident-feature analysis (ROI)
        analyse_coincident_features(
            red_image=red,
            blue_image=blue,
            red_labelled=red_labelled,
            coincident_features_image=coincident_features_image,
            directory=img_out_dir,
            pixel_size_nm=pixel_size,
            csv_name="coincident_features.csv",
            hist_prefix="coincident_hist_"
        )

        # Per-image summary row (for this ROI)
        def mean_and_std(series):
            if len(series) == 0:
                return np.nan, np.nan
            m = float(series.mean())
            if len(series) > 1:
                s = float(series.std(ddof=1))
            else:
                s = np.nan
            return m, s

        # Red metrics
        r_area_mean, r_area_sd = mean_and_std(red_measurements['area'])
        r_len_mean, r_len_sd = mean_and_std(red_measurements['major_axis_length'])
        r_int_mean, r_int_sd = mean_and_std(red_measurements['mean_intensity'])
        r_max_mean, r_max_sd = mean_and_std(red_measurements['max_intensity'])

        # Blue metrics
        b_area_mean, b_area_sd = mean_and_std(blue_measurements['area'])
        b_len_mean, b_len_sd = mean_and_std(blue_measurements['major_axis_length'])
        b_int_mean, b_int_sd = mean_and_std(blue_measurements['mean_intensity'])
        b_max_mean, b_max_sd = mean_and_std(blue_measurements['max_intensity'])

        summary_rows.append({
            'image_path': filepath,
            'image_name': img_name,
            'roi_info': roi_info,
            'red_threshold': red_threshold,
            'blue_threshold': blue_threshold,

            'red_count': red_number,
            'blue_count': blue_number,
            'coincident_count': coincident_count,
            'fraction_red_coincident': fraction_coinc,
            'avg_fraction_pixels_overlap': avg_overlap,

            'red_area_mean': r_area_mean,
            'red_area_sd': r_area_sd,
            'red_length_mean': r_len_mean,
            'red_length_sd': r_len_sd,
            'red_mean_intensity_mean': r_int_mean,
            'red_mean_intensity_sd': r_int_sd,
            'red_max_intensity_mean': r_max_mean,
            'red_max_intensity_sd': r_max_sd,

            'blue_area_mean': b_area_mean,
            'blue_area_sd': b_area_sd,
            'blue_length_mean': b_len_mean,
            'blue_length_sd': b_len_sd,
            'blue_mean_intensity_mean': b_int_mean,
            'blue_mean_intensity_sd': b_int_sd,
            'blue_max_intensity_mean': b_max_mean,
            'blue_max_intensity_sd': b_max_sd,
        })

        # Update per-image/ROI summary after each image
        summary_df = pd.DataFrame(summary_rows)
        os.makedirs(root_directory, exist_ok=True)
        summary_df.to_csv(summary_path, sep='\t', index=False)
        print(f"Updated per-image summary metrics at {summary_path}")

# =========================
# Root-level CV summary (after all images/ROIs)
# =========================

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)

    cv_metrics = [
        'red_count',
        'blue_count',
        'coincident_count',
        'red_area_mean',
        'blue_area_mean',
        'red_mean_intensity_mean',
        'blue_mean_intensity_mean'
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
    cv_path = os.path.join(root_directory, "summary_CV_metrics.tsv")
    cv_df.to_csv(cv_path, sep='\t', index=False)
    print(f"Saved CV metrics across images/ROIs to {cv_path}")
else:
    print("No images processed; no summary files created.")
