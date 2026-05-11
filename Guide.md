Paste everything below into nano Guide.md, save, commit, and it will render correctly on GitHub:

Red/Blue Coincidence Analysis Pipeline
This document explains how to use the Python script for batch analysis of red/blue feature coincidence in paired TIFF images (Opera Phenix data). It assumes filenames like:

...-ch1....tif = red channel
...-ch2....tif = blue channel
The script:

Pairs ch1 and ch2 images in each folder
Detects spots in red and blue
Quantifies coincidence between channels
Computes densities (per µm²)
Computes per-spot metrics and coincidence flags
Saves per-image results and per-folder/global summaries
1. Requirements
You need Python 3 and the following packages:

numpy
pandas
scikit-image (skimage)
matplotlib (not strictly required in the current version, but often installed with scikit-image)
scipy (not used for plotting now, but may be present from earlier versions)
Install them (if needed) with:

pip install numpy pandas scikit-image
Copy Code
If you use conda:

conda install numpy pandas scikit-image
Copy Code
2. File and folder structure
2.1 Input images
The script expects one file per channel:

Red (ch1): e.g.
r02c07f01p01-ch1sk1fk1fl1.tiff
Blue (ch2): e.g.
r02c07f01p01-ch2sk1fk1fl1.tiff
For each red file, the script looks for the corresponding blue file by replacing ch1 with ch2 in the filename.

Images must:

Be 2D grayscale TIFFs (no channel dimension inside the file).
Have matching shapes for red and blue in each pair.
2.2 Data folders (pathList)
You define a list of folders to analyse, for example:

pathList = [
    r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_EV/Test/",
    r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_EV/Test2/",
]
Copy Code
Each folder in pathList:

Contains TIFF files with ch1 and ch2 in their names.
Will get:
One *_results subfolder per image pair.
A per-folder image_summary_metrics.tsv.
A per-folder global_mean_density_intensity.tsv.
2.3 Root folder (top_root_directory)
You also set a “root” directory where the script writes global summary files:

top_root_directory = r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_EV/"
Copy Code
This root will receive:

image_summary_metrics.tsv (all image pairs from all folders combined)
summary_CV_metrics.tsv
global_mean_density_intensity.tsv
per_folder_count_and_density_stats.tsv
Make sure the path (including spaces) exactly matches the real folder name.

3. What the script does (conceptual overview)
For each folder in pathList:

Finds all red (ch1) TIFF files.
For each red file, finds the matching blue (ch2) file.
Loads red and blue images.
Optionally crops to a rectangular ROI.
Thresholds each channel (fixed value, Otsu, or Yen).
Labels spots in each binary mask.
Computes:
Red spot count
Blue spot count
Number of red spots overlapping blue (coincident)
Fraction of red spots that are coincident
Densities (counts per µm²) for red, blue, and coincident spots
A rotated / translated blue control and its coincidence with red
Computes per-spot properties for each channel:
Centroid (x, y)
Area (pixels and µm²)
Sum / mean / max intensity
Whether each spot is coincident or not
Saves:
Binary images (red_binary.tif, blue_binary.tif)
Per-spot CSVs (red_spots.csv, blue_spots.csv)
One row of summary metrics per image pair.
After processing all folders and image pairs, it:

Combines all rows into a global table.
Writes global summary files in top_root_directory.
Writes per-folder summary files back into each data folder.
4. Key parameters you may want to edit
Near the top of the script you’ll see the user parameters.

4.1 Pixel size
pixel_size = 591  # nm
Copy Code
Pixel size in nanometres.
Used to calculate area in µm² for densities and spot areas.
If your imaging settings change, update this value.
4.2 Root directory and data folders
top_root_directory = r".../RawData_EV/"

pathList = [
    r".../RawData_EV/Test/",
    r".../RawData_EV/Test2/",
]
Copy Code
top_root_directory is where global TSVs are written.
pathList contains folders to analyse.
Paths with spaces must be exact.
4.3 Thresholding mode
THRESHOLD_MODE = "fixed"   # or "otsu" or "yen"
RED_FIXED_THRESHOLD = 200
BLUE_FIXED_THRESHOLD = 200
Copy Code
"fixed": use the given numeric thresholds.
"otsu": compute threshold per image using Otsu’s method.
"yen": compute threshold per image using Yen’s method.
If you use "otsu" or "yen", the RED_FIXED_THRESHOLD and BLUE_FIXED_THRESHOLD values are ignored.

4.4 ROI (Region of Interest)
USE_ROI = False
ROI_X = (20, 1060)   # (xmin, xmax) in pixels
ROI_Y = (20, 1060)   # (ymin, ymax) in pixels
Copy Code
If USE_ROI = False, the script analyses the full image.
If USE_ROI = True, each image is cropped to the given rectangle.
ROI is clipped to stay within image bounds.
5. Outputs
5.1 Per-image / per-pair outputs
For each red/blue image pair, a folder like:

<that_folder>/<base_name>_results/
Copy Code
is created, containing:

red_binary.tif

Binary mask (0/255) of red spots after thresholding.
blue_binary.tif

Binary mask (0/255) of blue spots after thresholding.
red_spots.csv

One row per red spot with columns:
label_id
centroid_y, centroid_x
area_pixels, area_um2
sum_intensity, mean_intensity, max_intensity
is_coincident (1 if this red spot overlaps the blue mask)
blue_spots.csv

Same columns as red_spots.csv, but for blue spots, and is_coincident indicates overlap with the red mask.
5.2 Per-image summary table (global)
In top_root_directory:

image_summary_metrics.tsv
This file has one row per image pair (per folder), with columns such as:

Paths and names:
red_image_path, blue_image_path, red_image_name, blue_image_name, folder, roi_info
Thresholds:
red_threshold, blue_threshold, threshold_mode
Counts:
red_count, blue_count, coincident_count,
coincident_count_rot90px1000 (control)
fraction_red_coincident
Densities (per µm²):
red_density_per_um2, blue_density_per_um2
coincident_density_per_um2
coincident_density_rot90px1000_per_um2
Basic intensity metrics:
red_spot_sum_mean, red_spot_sum_sd
blue_spot_sum_mean, blue_spot_sum_sd
5.3 Global summary tables (root)
In top_root_directory you also get:

summary_CV_metrics.tsv

One row per metric (e.g. red_density_per_um2, blue_spot_sum_mean).
Columns:
metric
mean_across_image_pairs
sd_across_image_pairs
cv_across_image_pairs
global_mean_density_intensity.tsv

One row per metric (e.g. red_count, red_density_per_um2, red_spot_sum_mean).
Columns:
metric
global_mean
global_sd
per_folder_count_and_density_stats.tsv

One row per input folder (per pathList entry).
Columns include:
folder
red_count_mean, red_count_sd, red_count_cv
blue_count_mean, blue_count_sd, blue_count_cv
coincident_count_mean, coincident_count_sd, coincident_count_cv
red_density_per_um2_mean, red_density_per_um2_sd, red_density_per_um2_cv
blue_density_per_um2_mean, …
coincident_density_per_um2_mean, …
coincident_density_rot90px1000_per_um2_mean, …
5.4 Per-folder summary files (inside each data folder)
Inside each folder in pathList, e.g.:

.../RawData_EV/Test/
.../RawData_EV/Test2/
you will also get:

image_summary_metrics.tsv

Same structure as the global file, but only rows for that specific folder.
global_mean_density_intensity.tsv

Means and SDs computed only over that folder’s images.
6. How to run the script
Open a terminal.

Navigate to the folder containing your script, for example:

cd /path/to/your/script
Copy Code
Make sure the script is executable (optional):

chmod +x red_blue_coincidence.py
Copy Code
Run the script with Python:

python red_blue_coincidence.py
Copy Code
or, if executable and with a proper shebang:

./red_blue_coincidence.py
Copy Code
While running, you’ll see messages like:

Processing pair: red=..., blue=...
Analysing region: ...
X features were detected in the Red ROI.
Updates about summary files being written.
7. Common adjustments
7.1 Changing thresholds
If you find you are over- or under-detecting spots:

To try Otsu automatically:

THRESHOLD_MODE = "otsu"
Copy Code
To refine fixed thresholds:

THRESHOLD_MODE = "fixed"
RED_FIXED_THRESHOLD = 150  # example
BLUE_FIXED_THRESHOLD = 180
Copy Code
7.2 Changing ROI
If you want to exclude borders or artefacts:

USE_ROI = True
ROI_X = (100, 900)
ROI_Y = (100, 900)
Copy Code
Make sure the ROI is within the image dimensions.

7.3 Changing rotation control
If you want a different control transformation (e.g. 180° rotation, different shift):

blue_rt = rotate_translate_mask_wrap(
    blue_binary,
    angle_deg=180,
    shift_x=500,
    shift_y=0
)
Copy Code
8. Interpreting the key metrics
red_count, blue_count

Number of detected spots per image in each channel.
coincident_count

Number of red spots that overlap at least one blue pixel.
fraction_red_coincident

coincident_count / red_count.
Fraction of red spots that are “positive” for blue.
*_density_per_um2

Counts normalized by ROI area, giving feature density per µm².
coincident_density_rot90px1000_per_um2

Coincident density when the blue mask is rotated 90° and shifted by 1000 pixels with wrap-around.
Acts as a geometric/random control.
red_spot_sum_mean, blue_spot_sum_mean

Mean per-spot summed intensity in each channel.
Coarse measure of brightness per detected spot.
red_spots.csv / blue_spots.csv per spot:

is_coincident = 1 for spots that overlap the other channel’s binary mask.
Useful for exporting to other analysis tools or visualisation.
9. Troubleshooting
No ch1 files found:

Check naming: the script looks for *ch1*.tif or *ch1*.tiff.
Check that paths in pathList are correct and accessible.
Shape mismatch warnings:

Indicates red and blue images of a pair don’t have exactly the same dimensions.
Check the acquisition/export pipeline to ensure both channels are registered and cropped identically.
“Expected 2D grayscale images” error:

Your TIFF might contain multiple channels in one file.
This script assumes one channel per file. You’d need to split channels beforehand or modify the script.
No summary files created:

The terminal should print “No image pairs processed; no summary files created.”
Check that there are matching ch1 and ch2 pairs and that paths are correct.
10. Extending the script
Possible extensions:

Add more per-spot regionprops (eccentricity, perimeter, major/minor axis).
Add QC images that overlay detections on original intensities.
Group folders by experimental conditions and compute group-level stats.
If you plan to extend it, the main places to touch are:

extract_regionprops_table (for more per-spot columns).
The lists global_metrics, cv_metrics, and density_metrics (for which summary metrics are computed).
The pathList and folder naming conventions (for different experiment layouts).
