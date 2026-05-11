#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST batch analysis of red/blue features and their coincidence across pairs of TIFF images.

Assumes file naming like:
  r02c07f01p01-ch1sk1fk1fl1.tiff  -> red  (ch1)
  r02c07f01p01-ch2sk1fk1fl1.tiff  -> blue (ch2)

For each image pair:
  - Optionally crop to ROI
  - Threshold red/blue (Otsu, Yen, or fixed)
  - Label features
  - Compute coincidence (red features overlapping blue)
  - Compute per-spot intensity and morphology metrics, including a per-spot coincidence flag
  - Compute feature densities (counts per µm²) for red, blue, coincident spots
  - Compute rotated/translated (90° + 1000 px, wrap-around) control coincidence
  - Save binary masks and per-spot CSVs for each image pair

Across all image pairs:
  - Write image_summary_metrics.tsv in top_root_directory (all folders combined)
  - Write summary_CV_metrics.tsv with CVs across all image pairs
  - Write global_mean_density_intensity.tsv with global mean/SD of selected metrics
  - Write per_folder_count_and_density_stats.tsv (mean/SD/CV of counts and densities per folder)
  - For each folder, also write:
      * image_summary_metrics.tsv  (only that folder)
      * global_mean_density_intensity.tsv  (only that folder)
"""

# -------------------------
# Imports
# -------------------------

import os                     # for filesystem paths, directory creation, etc.
import glob                   # for file pattern matching (e.g. *.tif)
import numpy as np            # for numerical operations
from skimage.io import imread, imsave  # for reading/saving images
from skimage import filters, measure, transform  # image processing tools
import pandas as pd           # for tabular data handling (DataFrames, TSV/CSV IO)

# -------------------------
# User parameters
# -------------------------

# Size of pixels in nanometres (nm); used to convert to µm for densities/areas
pixel_size = 591  # nm

# Root directory where you want GLOBAL summary TSVs saved
top_root_directory = r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_Lysates/"

# List of directories to analyse (each can contain many .tif/.tiff files)
# Each directory is expected to contain paired ch1/ch2 files.
pathList = [
    r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_Lysates/1/",
    r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_Lysates/11/",
]

# ---- Thresholding options ----

# Choose threshold mode: "otsu", "yen", or "fixed"
THRESHOLD_MODE = "fixed"

# If THRESHOLD_MODE == "fixed", these values will be used as thresholds
RED_FIXED_THRESHOLD = 200
BLUE_FIXED_THRESHOLD = 200

# ---- Region of interest (ROI) ----

# If USE_ROI is True, each image will be cropped to:
#   y in [ROI_Y[0] : ROI_Y[1]], x in [ROI_X[0] : ROI_X[1]]
USE_ROI = False
ROI_X = (20, 1060)   # (xmin, xmax) in pixels
ROI_Y = (20, 1060)   # (ymin, ymax) in pixels

# -------------------------
# Geometry: pixel area
# -------------------------

# Convert pixel size from nm to µm
pixel_size_um = pixel_size / 1000.0
# Area of one pixel in µm²
pixel_area_um2 = pixel_size_um ** 2

# -------------------------
# Helper functions
# -------------------------

def load_image(toload):
    """Load an image from disk using skimage.io.imread."""
    return imread(toload)

def threshold_image(image, mode="fixed", fixed_value=None):
    """
    Threshold an image using 'otsu', 'yen', or 'fixed'.

    Parameters
    ----------
    image : 2D numpy array
        Grayscale image.
    mode : str
        'otsu', 'yen', or 'fixed'.
    fixed_value : float or int, optional
        Threshold value to use if mode == 'fixed'.

    Returns
    -------
    thr : float
        Threshold value used.
    binary : 2D bool array
        Binary mask of foreground pixels (True) above threshold.
    """
    # Normalise mode string
    mode = mode.lower()

    # Choose threshold based on mode
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

    # Binary mask: True where image > threshold
    binary = image > thr

    # Return threshold and binary mask as boolean
    return thr, binary.astype(bool)

def label_binary(binary_image):
    """
    Label connected components in a binary image.

    Parameters
    ----------
    binary_image : 2D bool array

    Returns
    -------
    n : int
        Number of labelled objects (excluding background).
    labelled : 2D int array
        Label image where 0 is background and 1..n are objects.
    """
    # measure.label assigns unique integer labels to connected components
    labelled = measure.label(binary_image)
    # Max label ID is the number of objects
    n = labelled.max()
    return n, labelled

def feature_coincidence_fast(binary_ref, binary_other):
    """
    Fast coincidence computation between two binary masks.

    Concept:
      - Label binary_ref (e.g. red channel).
      - Check which labels in binary_ref overlap ANY True pixel in binary_other (e.g. blue).
      - Return the count and fraction of coincident ref features.

    Parameters
    ----------
    binary_ref : 2D bool array
        Reference binary mask (e.g. red).
    binary_other : 2D bool array
        Other binary mask to test for overlap (e.g. blue).

    Returns
    -------
    coincident_count : int
        Number of reference features that overlap binary_other.
    fraction_coinc : float
        Fraction of reference features that are coincident.
    """
    # Label the reference mask
    n_ref, labels_ref = label_binary(binary_ref)

    # If no features, all coincidence metrics are zero
    if n_ref == 0:
        return 0, 0.0

    # Overlap mask: True where both ref and other are True
    overlap = binary_ref & binary_other

    # Get unique labels of ref that overlap (excluding background label 0)
    overlapping_labels = np.unique(labels_ref[overlap])
    overlapping_labels = overlapping_labels[overlapping_labels > 0]

    # Number of coincident features
    coincident_count = int(len(overlapping_labels))

    # Fraction of ref features that are coincident
    fraction_coinc = coincident_count / n_ref if n_ref > 0 else 0.0

    return coincident_count, fraction_coinc

def per_spot_summed_intensity(label_image, intensity_image):
    """
    Compute per-spot summed intensity for each label.

    Parameters
    ----------
    label_image : 2D int array
        Label image (0 = background, 1..n = features).
    intensity_image : 2D float or int array
        Original grayscale intensities.

    Returns
    -------
    sums : 1D float array of length n_labels
        Sum of intensities within each labelled feature (label 1..n).
    """
    # Number of labels
    n_labels = label_image.max()
    if n_labels == 0:
        # No features: return empty array
        return np.array([])

    # Flatten labels and intensities for bincount
    labels_flat = label_image.ravel()
    intens_flat = intensity_image.ravel()

    # np.bincount with weights: sum intensities per label index
    sums = np.bincount(labels_flat, weights=intens_flat, minlength=n_labels + 1)

    # sums[0] corresponds to background, so drop it
    return sums[1:]

def mean_and_sd(values):
    """
    Compute mean and sample standard deviation for 1D data.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    m : float
        Mean of values, or nan if empty.
    s : float
        Sample standard deviation (ddof=1) or nan if size < 2.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return np.nan, np.nan

    # Mean
    m = float(arr.mean())

    # Standard deviation: only defined (ddof=1) if >= 2 values
    if arr.size > 1:
        s = float(arr.std(ddof=1))
    else:
        s = np.nan

    return m, s

def rotate_translate_mask_wrap(binary_mask, angle_deg=90, shift_x=1000, shift_y=0):
    """
    Rotate a binary mask by a given angle, then translate, with wrap-around.

    Steps:
      1. Rotate around center by angle_deg (nearest neighbour).
      2. Translate by (shift_x, shift_y) pixels.
      3. Use 'wrap' boundary mode so features leaving one side re-enter on the opposite.

    Parameters
    ----------
    binary_mask : 2D bool array
    angle_deg : float
        Rotation angle in degrees.
    shift_x : float
        Translation in X direction (columns).
    shift_y : float
        Translation in Y direction (rows).

    Returns
    -------
    shifted_binary : 2D bool array
        Transformed binary mask (same shape as input).
    """
    # Rotate mask as float (0/1); order=0 to keep nearest-neighbour (no interpolation)
    rotated = transform.rotate(
        binary_mask.astype(float),
        angle=angle_deg,
        resize=False,
        center=None,
        order=0,
        mode='constant',
        cval=0.0,
        preserve_range=True
    )

    # Define translation transform
    tform = transform.AffineTransform(translation=(shift_x, shift_y))

    # Apply warp with 'wrap' mode (toroidal boundary conditions)
    shifted = transform.warp(
        rotated,
        inverse_map=tform.inverse,
        order=0,
        mode='wrap',
        preserve_range=True
    )

    # Threshold back to bool > 0.5
    return shifted > 0.5

def compute_label_overlap_flags(labels_a, binary_b):
    """
    For each label in labels_a, determine if it overlaps any True pixel in binary_b.

    Parameters
    ----------
    labels_a : 2D int array
        Label image (0 = background).
    binary_b : 2D bool array
        Binary mask to test for overlap.

    Returns
    -------
    has_overlap : 1D bool array of length (max_label + 1)
        has_overlap[label_id] is True if that label overlaps at least one True pixel in binary_b.
        Index 0 corresponds to background (usually False).
    """
    # If image is empty, just return array of length 1 with False
    if labels_a.size == 0:
        return np.zeros(1, dtype=bool)

    # Only consider positions where binary_b is True
    overlap_mask = binary_b

    # Get labels at overlapping positions
    labels_flat = labels_a[overlap_mask].ravel()

    # If no overlap pixels, then no label has overlap
    if labels_flat.size == 0:
        return np.zeros(labels_a.max() + 1, dtype=bool)

    # Maximum label ID
    max_label = labels_a.max()

    # Count how many overlapping pixels per label id
    counts = np.bincount(labels_flat, minlength=max_label + 1)

    # has_overlap is True where counts > 0
    has_overlap = counts > 0
    return has_overlap

def extract_regionprops_table(label_image, intensity_image, coincident_label_flags=None):
    """
    Compute per-spot metrics using skimage.measure.regionprops, including a coincidence flag.

    Parameters
    ----------
    label_image : 2D int array
        Label image of spots (0 = background, 1..n = spots).
    intensity_image : 2D float or int array
        Corresponding intensity image (same shape).
    coincident_label_flags : 1D bool array, optional
        Boolean array indexed by label_id; True means that label is coincident.
        If None, all spots are treated as non-coincident.

    Returns
    -------
    df : pandas.DataFrame
        Columns:
          label_id         : int
          centroid_y       : float (row coordinate)
          centroid_x       : float (column coordinate)
          area_pixels      : int (spot area in pixels)
          area_um2         : float (spot area in µm²)
          sum_intensity    : float (sum of intensities in the spot)
          mean_intensity   : float (mean intensity)
          max_intensity    : float (max intensity)
          is_coincident    : 0/1 (whether this spot overlaps the other channel)
    """
    # regionprops returns a list of RegionProperties objects, one per label
    props = measure.regionprops(label_image, intensity_image=intensity_image)

    rows = []  # list to collect per-spot dictionaries

    for p in props:
        # Centroid (y, x)
        y, x = p.centroid

        # Area in pixels
        area_px = p.area

        # Area in µm² = area_px * pixel_area_um2
        area_um2 = area_px * pixel_area_um2

        # Integer label id
        label_id = int(p.label)

        # Determine if this label has overlap, if flags provided
        if coincident_label_flags is not None and label_id < len(coincident_label_flags):
            is_coincident = int(bool(coincident_label_flags[label_id]))
        else:
            is_coincident = 0

        # Append per-spot metrics to rows list
        rows.append({
            "label_id": label_id,
            "centroid_y": float(y),
            "centroid_x": float(x),
            "area_pixels": int(area_px),
            "area_um2": float(area_um2),
            "sum_intensity": float(p.intensity_image.sum()),
            "mean_intensity": float(p.mean_intensity),
            "max_intensity": float(p.max_intensity),
            "is_coincident": is_coincident,
        })

    # Convert list of dicts to DataFrame
    return pd.DataFrame(rows)

# -------------------------
# Global collector
# -------------------------

# Will store one dict (row) per image pair across all folders
summary_rows = []

# -------------------------
# Main processing loop
# -------------------------

# Loop over each directory in pathList
for directory in pathList:
    # Convert to absolute path (for cleaner output and grouping)
    directory = os.path.abspath(directory)

    # Find all channel-1 (red) files: match "*ch1*.tif" or "*ch1*.tiff"
    ch1_files = sorted(
        glob.glob(os.path.join(directory, "*ch1*.tif")) +
        glob.glob(os.path.join(directory, "*ch1*.tiff"))
    )

    # If no ch1 files found in this directory, skip it
    if not ch1_files:
        print(f"No ch1 TIFF files found in {directory}")
        continue

    # Process each red (ch1) file
    for red_path in ch1_files:
        # Base filename of red image
        red_name = os.path.basename(red_path)

        # Construct expected blue filename by replacing 'ch1' with 'ch2'
        blue_name = red_name.replace("ch1", "ch2")
        # Construct full path for blue file
        blue_path = os.path.join(directory, blue_name)

        # If corresponding blue file doesn't exist, warn and skip this pair
        if not os.path.exists(blue_path):
            print(f"WARNING: No matching ch2 file for {red_name} (expected {blue_name}); skipping.")
            continue

        # Status message
        print(f"\nProcessing pair: red={red_name}, blue={blue_name}")

        # Base name for output folder: replace 'ch1' with 'ch1_ch2' and remove extension
        pair_base = os.path.splitext(red_name)[0].replace("ch1", "ch1_ch2")

        # Create results folder for this pair in the same directory
        img_out_dir = os.path.join(directory, pair_base + "_results")
        os.makedirs(img_out_dir, exist_ok=True)

        # Load red and blue images from disk
        red_full = load_image(red_path)
        blue_full = load_image(blue_path)

        # Sanity check: shapes must match to compare pixels
        if red_full.shape != blue_full.shape:
            print(
                f"WARNING: Shape mismatch for pair {red_name} / {blue_name}: "
                f"{red_full.shape} vs {blue_full.shape}. Skipping."
            )
            continue

        # Ensure images are 2D grayscale (no channel dimension present)
        if red_full.ndim != 2 or blue_full.ndim != 2:
            raise ValueError(
                f"Expected 2D grayscale images for {red_name} and {blue_name}, "
                f"got {red_full.shape} and {blue_full.shape}"
            )

        # -------------------------
        # Apply ROI if requested
        # -------------------------
        if USE_ROI:
            # Unpack ROI ranges
            ymin, ymax = ROI_Y
            xmin, xmax = ROI_X

            # Ensure ROI is within image bounds
            ny_full, nx_full = red_full.shape
            ymin = max(0, ymin)
            ymax = min(ny_full, ymax)
            xmin = max(0, xmin)
            xmax = min(nx_full, xmax)

            # Check ROI validity after clipping
            if ymin >= ymax or xmin >= xmax:
                raise ValueError(
                    f"Invalid ROI after clipping for pair {red_name}/{blue_name}: "
                    f"ROI_X={ROI_X}, ROI_Y={ROI_Y}, image shape={red_full.shape}"
                )

            # Crop both channels to ROI
            red = red_full[ymin:ymax, xmin:xmax]
            blue = blue_full[ymin:ymax, xmin:xmax]

            # Text description of ROI used
            roi_info = f"ROI x={xmin}:{xmax}, y={ymin}:{ymax}"
        else:
            # Use full images if ROI not enabled
            red = red_full
            blue = blue_full
            roi_info = "Full image"

        print(f"Analysing region: {roi_info}")

        # -------------------------
        # Compute ROI area in µm²
        # -------------------------

        # ny = #rows (Y dimension), nx = #columns (X dimension)
        ny, nx = red.shape

        # ROI area (pixels) × pixel_area_um2
        roi_area_um2 = ny * nx * pixel_area_um2

        # -------------------------
        # Threshold red and blue channels
        # -------------------------

        if THRESHOLD_MODE == "fixed":
            # Use fixed thresholds
            red_threshold, red_binary = threshold_image(
                red,
                mode="fixed",
                fixed_value=RED_FIXED_THRESHOLD
            )
            blue_threshold, blue_binary = threshold_image(
                blue,
                mode="fixed",
                fixed_value=BLUE_FIXED_THRESHOLD
            )
        else:
            # Use adaptive thresholds (Otsu or Yen)
            red_threshold, red_binary = threshold_image(red, mode=THRESHOLD_MODE)
            blue_threshold, blue_binary = threshold_image(blue, mode=THRESHOLD_MODE)

        # -------------------------
        # Label features in each channel
        # -------------------------

        # Label red binary mask
        red_number, red_labelled = label_binary(red_binary)
        # Label blue binary mask
        blue_number, blue_labelled = label_binary(blue_binary)

        print(f"{red_number} features were detected in the Red ROI.")
        print(f"{blue_number} features were detected in the Blue ROI.")

        # -------------------------
        # Coincidence analysis (red vs blue)
        # -------------------------

        # Compute coincidence metrics: how many red labels overlap blue, and fraction
        coincident_count, fraction_coinc = feature_coincidence_fast(red_binary, blue_binary)

        print(
            f"{fraction_coinc:.2f} of red ROI features had coincidence with features in blue ROI. "
            f"Number of coincident spots: {coincident_count}"
        )

        # -------------------------
        # Densities (features per µm²)
        # -------------------------

        if roi_area_um2 > 0:
            # Feature density = count / area
            red_density = red_number / roi_area_um2
            blue_density = blue_number / roi_area_um2
            coincident_density = coincident_count / roi_area_um2
        else:
            # If area is zero (should not happen), set to NaN
            red_density = np.nan
            blue_density = np.nan
            coincident_density = np.nan

        # -------------------------
        # Rotated + translated (wrap) control coincidence
        # -------------------------

        # Only meaningful if we have some features in both channels and a non-zero area
        if roi_area_um2 > 0 and red_number > 0 and blue_number > 0:
            # Create rotated+translated blue mask
            blue_rt = rotate_translate_mask_wrap(
                blue_binary,
                angle_deg=90,
                shift_x=1000,
                shift_y=0
            )
            # Count coincidences between original red and rotated blue
            coincident_count_rt, _ = feature_coincidence_fast(red_binary, blue_rt)
            coincident_density_rt = coincident_count_rt / roi_area_um2
        else:
            coincident_count_rt = 0
            coincident_density_rt = np.nan

        # -------------------------
        # Per-spot summed intensities (simple intensity metric)
        # -------------------------

        red_spot_sums = per_spot_summed_intensity(red_labelled, red)
        blue_spot_sums = per_spot_summed_intensity(blue_labelled, blue)

        # Compute mean and SD of these sums per image pair
        red_sum_mean, red_sum_sd = mean_and_sd(red_spot_sums)
        blue_sum_mean, blue_sum_sd = mean_and_sd(blue_spot_sums)

        # -------------------------
        # Per-spot coincidence flags (for CSVs)
        # -------------------------

        # For each red label, does it overlap any blue pixels?
        red_label_overlap_flags = compute_label_overlap_flags(red_labelled, blue_binary)
        # For each blue label, does it overlap any red pixels?
        blue_label_overlap_flags = compute_label_overlap_flags(blue_labelled, red_binary)

        # -------------------------
        # Save binary images for this pair
        # -------------------------

        # Paths for binary images in the results folder
        red_bin_path = os.path.join(img_out_dir, "red_binary.tif")
        blue_bin_path = os.path.join(img_out_dir, "blue_binary.tif")

        # Save as uint8 images (0 or 255) for better compatibility
        imsave(red_bin_path, (red_binary.astype(np.uint8) * 255))
        imsave(blue_bin_path, (blue_binary.astype(np.uint8) * 255))

        # -------------------------
        # Save per-spot CSVs for this pair (with is_coincident)
        # -------------------------

        # Compute regionprops tables for red and blue
        red_props_df = extract_regionprops_table(
            red_labelled, red, coincident_label_flags=red_label_overlap_flags
        )
        blue_props_df = extract_regionprops_table(
            blue_labelled, blue, coincident_label_flags=blue_label_overlap_flags
        )

        # File paths for per-spot CSVs
        red_props_path = os.path.join(img_out_dir, "red_spots.csv")
        blue_props_path = os.path.join(img_out_dir, "blue_spots.csv")

        # Save CSVs (one row per spot)
        red_props_df.to_csv(red_props_path, index=False)
        blue_props_df.to_csv(blue_props_path, index=False)

        # -------------------------
        # Collect one summary row for this image pair
        # -------------------------

        row = {
            'red_image_path': red_path,
            'blue_image_path': blue_path,
            'red_image_name': red_name,
            'blue_image_name': blue_name,
            'folder': directory,
            'roi_info': roi_info,
            'red_threshold': red_threshold,
            'blue_threshold': blue_threshold,
            'threshold_mode': THRESHOLD_MODE,

            # Raw feature counts
            'red_count': red_number,
            'blue_count': blue_number,
            'coincident_count': coincident_count,
            'coincident_count_rot90px1000': coincident_count_rt,
            'fraction_red_coincident': fraction_coinc,

            # Densities (features per µm²)
            'red_density_per_um2': red_density,
            'blue_density_per_um2': blue_density,
            'coincident_density_per_um2': coincident_density,
            'coincident_density_rot90px1000_per_um2': coincident_density_rt,

            # Basic intensity metrics (summed intensities per spot)
            'red_spot_sum_mean': red_sum_mean,
            'red_spot_sum_sd': red_sum_sd,
            'blue_spot_sum_mean': blue_sum_mean,
            'blue_spot_sum_sd': blue_sum_sd,
        }

        # Append row to global list
        summary_rows.append(row)

        # After each pair, update the GLOBAL per-image summary file in top_root_directory
        summary_df = pd.DataFrame(summary_rows)
        os.makedirs(top_root_directory, exist_ok=True)
        global_summary_path = os.path.join(top_root_directory, "image_summary_metrics.tsv")
        summary_df.to_csv(global_summary_path, sep='\t', index=False)
        print(f"Updated global per-image summary metrics at {global_summary_path}")

# -------------------------
# Global and per-folder summaries
# -------------------------

if summary_rows:
    # Convert collected rows into a DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # ----------------------------------------
    # 1) Global CV summary (densities + intensity)
    # ----------------------------------------

    # Metrics for which we'll compute global mean/SD/CV across image pairs
    cv_metrics = [
        'red_density_per_um2',
        'blue_density_per_um2',
        'coincident_density_per_um2',
        'coincident_density_rot90px1000_per_um2',
        'red_spot_sum_mean',
        'blue_spot_sum_mean'
    ]

    cv_rows = []  # will hold dicts, one per metric

    for col in cv_metrics:
        # Extract numeric values (dropping NaNs)
        values = summary_df[col].dropna().values.astype(float)

        if len(values) == 0:
            # If no data, all stats are NaN
            mean_val = np.nan
            sd_val = np.nan
            cv_val = np.nan
        else:
            # Mean across image pairs
            mean_val = float(values.mean())
            if len(values) > 1:
                # Sample SD
                sd_val = float(values.std(ddof=1))
                # Coefficient of variation = SD / mean
                cv_val = sd_val / mean_val if mean_val != 0 else np.nan
            else:
                sd_val = np.nan
                cv_val = np.nan

        cv_rows.append({
            'metric': col,
            'mean_across_image_pairs': mean_val,
            'sd_across_image_pairs': sd_val,
            'cv_across_image_pairs': cv_val
        })

    # Create DataFrame of CV stats and save to TSV in root
    cv_df = pd.DataFrame(cv_rows)
    cv_path = os.path.join(top_root_directory, "summary_CV_metrics.tsv")
    os.makedirs(os.path.dirname(cv_path), exist_ok=True)
    cv_df.to_csv(cv_path, sep='\t', index=False)
    print(f"Saved global CV metrics across image pairs to {cv_path}")

    # ----------------------------------------
    # 2) Global mean/SD summary for selected metrics
    # ----------------------------------------

    # These metrics will have global (across all pairs) mean/SD stored
    global_metrics = [
        'red_count',
        'blue_count',
        'coincident_count',
        'fraction_red_coincident',
        'red_density_per_um2',
        'blue_density_per_um2',
        'coincident_density_per_um2',
        'coincident_density_rot90px1000_per_um2',
        'red_spot_sum_mean',
        'blue_spot_sum_mean'
    ]

    global_rows = []  # list of dicts, one row per metric

    for col in global_metrics:
        # Drop NaNs and convert to float
        values = summary_df[col].dropna().values.astype(float)

        if len(values) == 0:
            mean_val = np.nan
            sd_val = np.nan
        else:
            mean_val = float(values.mean())
            # Sample SD only if >=2 values, else NaN
            sd_val = float(values.std(ddof=1)) if len(values) > 1 else np.nan

        global_rows.append({
            'metric': col,
            'global_mean': mean_val,
            'global_sd': sd_val
        })

    # Save global mean/SD summary to root
    global_df = pd.DataFrame(global_rows)
    global_path = os.path.join(top_root_directory, "global_mean_density_intensity.tsv")
    global_df.to_csv(global_path, sep='\t', index=False)
    print(f"Saved global mean/SD summary to {global_path}")

    # ----------------------------------------
    # 3) Per-folder mean/SD/CV of counts AND densities
    # ----------------------------------------

    folder_stats_rows = []  # one dict per folder

    # Group entire summary table by folder path
    for folder, group in summary_df.groupby('folder'):
        # Initialise row with folder name/path
        stats_row = {'folder': folder}

        # ---- Counts per image (red, blue, coincident) ----
        for metric in ['red_count', 'blue_count', 'coincident_count']:
            vals = group[metric].dropna().values.astype(float)

            if len(vals) == 0:
                mean_val = np.nan
                sd_val = np.nan
                cv_val = np.nan
            else:
                mean_val = float(vals.mean())
                if len(vals) > 1:
                    sd_val = float(vals.std(ddof=1))
                    cv_val = sd_val / mean_val if mean_val != 0 else np.nan
                else:
                    sd_val = np.nan
                    cv_val = np.nan

            # Base name: 'red', 'blue', or 'coincident'
            base = metric.replace('_count', '')

            # Store per-folder count stats
            stats_row[f'{base}_count_mean'] = mean_val
            stats_row[f'{base}_count_sd'] = sd_val
            stats_row[f'{base}_count_cv'] = cv_val

        # ---- Densities per image (features per µm²) ----
        density_metrics = [
            'red_density_per_um2',
            'blue_density_per_um2',
            'coincident_density_per_um2',
            'coincident_density_rot90px1000_per_um2'
        ]

        for metric in density_metrics:
            vals = group[metric].dropna().values.astype(float)

            if len(vals) == 0:
                mean_val = np.nan
                sd_val = np.nan
                cv_val = np.nan
            else:
                mean_val = float(vals.mean())
                if len(vals) > 1:
                    sd_val = float(vals.std(ddof=1))
                    cv_val = sd_val / mean_val if mean_val != 0 else np.nan
                else:
                    sd_val = np.nan
                    cv_val = np.nan

            # Use metric name directly as base, e.g. "red_density_per_um2"
            base = metric

            # Store per-folder density stats
            stats_row[f'{base}_mean'] = mean_val
            stats_row[f'{base}_sd'] = sd_val
            stats_row[f'{base}_cv'] = cv_val

        # Append this folder's stats row
        folder_stats_rows.append(stats_row)

    # Save all per-folder stats to a TSV in root
    folder_stats_df = pd.DataFrame(folder_stats_rows)
    folder_stats_path = os.path.join(top_root_directory, "per_folder_count_and_density_stats.tsv")
    folder_stats_df.to_csv(folder_stats_path, sep='\t', index=False)
    print(f"Saved per-folder count and density statistics to {folder_stats_path}")

    # ----------------------------------------
    # 4) Per-folder summary files within each folder
    # ----------------------------------------

    for folder, group in summary_df.groupby('folder'):
        # Ensure folder exists (it should, but just in case)
        os.makedirs(folder, exist_ok=True)

        # ---- Per-folder image_summary_metrics.tsv ----
        folder_summary_path = os.path.join(folder, "image_summary_metrics.tsv")
        group.to_csv(folder_summary_path, sep='\t', index=False)
        print(f"Saved per-image summary metrics for folder {folder} to {folder_summary_path}")

        # ---- Per-folder global_mean_density_intensity.tsv ----
        folder_global_rows = []
        for col in global_metrics:
            vals = group[col].dropna().values.astype(float)
            if len(vals) == 0:
                mean_val = np.nan
                sd_val = np.nan
            else:
                mean_val = float(vals.mean())
                sd_val = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan

            folder_global_rows.append({
                'metric': col,
                'global_mean': mean_val,
                'global_sd': sd_val
            })

        folder_global_df = pd.DataFrame(folder_global_rows)
        folder_global_path = os.path.join(folder, "global_mean_density_intensity.tsv")
        folder_global_df.to_csv(folder_global_path, sep='\t', index=False)
        print(f"Saved per-folder mean/SD summary for folder {folder} to {folder_global_path}")

else:
    # If no image pairs were processed at all, report and exit
    print("No image pairs processed; no summary files created.")
