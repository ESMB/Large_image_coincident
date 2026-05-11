<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Red/Blue Coincidence Analysis Pipeline</title>
</head>
<body>

<h1>Red/Blue Coincidence Analysis Pipeline</h1>

<p>
This document explains how to use the Python script for batch analysis of red/blue feature
coincidence in paired TIFF images (Opera Phenix data). It assumes filenames like:
</p>

<ul>
  <li><code>...-ch1....tif</code> = red channel</li>
  <li><code>...-ch2....tif</code> = blue channel</li>
</ul>

<p>The script:</p>

<ul>
  <li>Pairs <code>ch1</code> and <code>ch2</code> images in each folder</li>
  <li>Detects spots in red and blue</li>
  <li>Quantifies coincidence between channels</li>
  <li>Computes densities (per µm²)</li>
  <li>Computes per-spot metrics and coincidence flags</li>
  <li>Saves per-image results and per-folder/global summaries</li>
</ul>

<hr>

<h2>1. Requirements</h2>

<p>You need Python 3 and the following packages:</p>

<ul>
  <li><code>numpy</code></li>
  <li><code>pandas</code></li>
  <li><code>scikit-image</code> (<code>skimage</code>)</li>
  <li><code>matplotlib</code> (not strictly required in the current version, but often installed with <code>scikit-image</code>)</li>
  <li><code>scipy</code> (not used for plotting now, but may be present from earlier versions)</li>
</ul>

<p>Install them (if needed) with:</p>

<pre><code class="language-bash">pip install numpy pandas scikit-image
</code></pre>

<p>If you use <code>conda</code>:</p>

<pre><code class="language-bash">conda install numpy pandas scikit-image
</code></pre>

<hr>

<h2>2. File and folder structure</h2>

<h3>2.1 Input images</h3>

<p>The script expects <strong>one file per channel</strong>:</p>

<ul>
  <li>Red (<code>ch1</code>): e.g. <code>r02c07f01p01-ch1sk1fk1fl1.tiff</code></li>
  <li>Blue (<code>ch2</code>): e.g. <code>r02c07f01p01-ch2sk1fk1fl1.tiff</code></li>
</ul>

<p>
For each red file, the script looks for the corresponding blue file by replacing
<code>ch1</code> with <code>ch2</code> in the filename.
</p>

<p>Images must:</p>
<ul>
  <li>Be 2D grayscale TIFFs (no channel dimension inside the file).</li>
  <li>Have matching shapes for red and blue in each pair.</li>
</ul>

<h3>2.2 Data folders (<code>pathList</code>)</h3>

<p>You define a list of folders to analyse, for example:</p>

<pre><code class="language-python">pathList = [
    r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_EV/Test/",
    r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_EV/Test2/",
]
</code></pre>

<p>Each folder in <code>pathList</code>:</p>

<ul>
  <li>Contains TIFF files with <code>ch1</code> and <code>ch2</code> in their names.</li>
  <li>Will get:
    <ul>
      <li>One <code>*_results</code> subfolder per image pair.</li>
      <li>A per-folder <code>image_summary_metrics.tsv</code>.</li>
      <li>A per-folder <code>global_mean_density_intensity.tsv</code>.</li>
    </ul>
  </li>
</ul>

<h3>2.3 Root folder (<code>top_root_directory</code>)</h3>

<p>You also set a “root” directory where the script writes global summary files:</p>

<pre><code class="language-python">top_root_directory = r"/Volumes/T7/010526_Sample_Panel_OperaPhenix /RawData_EV/"
</code></pre>

<p>This root will receive:</p>

<ul>
  <li><code>image_summary_metrics.tsv</code> (all image pairs from all folders combined)</li>
  <li><code>summary_CV_metrics.tsv</code></li>
  <li><code>global_mean_density_intensity.tsv</code></li>
  <li><code>per_folder_count_and_density_stats.tsv</code></li>
</ul>

<p>
Make sure the path (including spaces) exactly matches the real folder name.
</p>

<hr>

<h2>3. What the script does (conceptual overview)</h2>

<p>For <strong>each folder</strong> in <code>pathList</code>:</p>

<ol>
  <li>Finds all red (<code>ch1</code>) TIFF files.</li>
  <li>For each red file, finds the matching blue (<code>ch2</code>) file.</li>
  <li>Loads red and blue images.</li>
  <li>Optionally crops to a rectangular ROI.</li>
  <li>Thresholds each channel (fixed value, Otsu, or Yen).</li>
  <li>Labels spots in each binary mask.</li>
  <li>Computes:
    <ul>
      <li>Red spot count</li>
      <li>Blue spot count</li>
      <li>Number of red spots overlapping blue (coincident)</li>
      <li>Fraction of red spots that are coincident</li>
      <li>Densities (counts per µm²) for red, blue, and coincident spots</li>
      <li>A rotated / translated blue control and its coincidence with red</li>
    </ul>
  </li>
  <li>Computes per-spot properties for each channel:
    <ul>
      <li>Centroid (x, y)</li>
      <li>Area (pixels and µm²)</li>
      <li>Sum / mean / max intensity</li>
      <li>Whether each spot is coincident or not</li>
    </ul>
  </li>
  <li>Saves:
    <ul>
      <li>Binary images (<code>red_binary.tif</code>, <code>blue_binary.tif</code>)</li>
      <li>Per-spot CSVs (<code>red_spots.csv</code>, <code>blue_spots.csv</code>)</li>
      <li>One row of summary metrics per image pair.</li>
    </ul>
  </li>
</ol>

<p>After processing <strong>all folders and image pairs</strong>, it:</p>

<ol start="10">
  <li>Combines all rows into a global table.</li>
  <li>Writes global summary files in <code>top_root_directory</code>.</li>
  <li>Writes per-folder summary files back into each data folder.</li>
</ol>

<hr>

<h2>4. Key parameters you may want to edit</h2>

<p>Near the top of the script you’ll see the user parameters.</p>

<h3>4.1 Pixel size</h3>

<pre><code class="language-python">pixel_size = 591  # nm
</code></pre>

<ul>
  <li>Pixel size in nanometres.</li>
  <li>Used to calculate area in µm² for densities and spot areas.</li>
  <li>If your imaging settings change, update this value.</li>
</ul>

<h3>4.2 Root directory and data folders</h3>

<pre><code class="language-python">top_root_directory = r".../RawData_EV/"

pathList = [
    r".../RawData_EV/Test/",
    r".../RawData_EV/Test2/",
]
</code></pre>

<ul>
  <li><code>top_root_directory</code> is where global TSVs are written.</li>
  <li><code>pathList</code> contains folders to analyse.</li>
  <li>Paths with spaces must be exact.</li>
</ul>

<h3>4.3 Thresholding mode</h3>

<pre><code class="language-python">THRESHOLD_MODE = "fixed"   # or "otsu" or "yen"
RED_FIXED_THRESHOLD = 200
BLUE_FIXED_THRESHOLD = 200
</code></pre>

<ul>
  <li><code>"fixed"</code>: use the given numeric thresholds.</li>
  <li><code>"otsu"</code>: compute threshold per image using Otsu’s method.</li>
  <li><code>"yen"</code>: compute threshold per image using Yen’s method.</li>
</ul>

<p>
If you use <code>"otsu"</code> or <code>"yen"</code>, the <code>RED_FIXED_THRESHOLD</code> and
<code>BLUE_FIXED_THRESHOLD</code> values are ignored.
</p>

<h3>4.4 ROI (Region of Interest)</h3>

<pre><code class="language-python">USE_ROI = False
ROI_X = (20, 1060)   # (xmin, xmax) in pixels
ROI_Y = (20, 1060)   # (ymin, ymax) in pixels
</code></pre>

<ul>
  <li>If <code>USE_ROI = False</code>, the script analyses the full image.</li>
  <li>If <code>USE_ROI = True</code>, each image is cropped to the given rectangle.</li>
  <li>ROI is clipped to stay within image bounds.</li>
</ul>

<hr>

<h2>5. Outputs</h2>

<h3>5.1 Per-image / per-pair outputs</h3>

<p>For each red/blue image pair, a folder like:</p>

<pre><code>&lt;that_folder&gt;/&lt;base_name&gt;_results/
</code></pre>

<p>is created, containing:</p>

<ol>
  <li><code>red_binary.tif</code>
    <ul>
      <li>Binary mask (0/255) of red spots after thresholding.</li>
    </ul>
  </li>
  <li><code>blue_binary.tif</code>
    <ul>
      <li>Binary mask (0/255) of blue spots after thresholding.</li>
    </ul>
  </li>
  <li><code>red_spots.csv</code>
    <ul>
      <li>One row per red spot with columns:
        <ul>
          <li><code>label_id</code></li>
          <li><code>centroid_y</code>, <code>centroid_x</code></li>
          <li><code>area_pixels</code>, <code>area_um2</code></li>
          <li><code>sum_intensity</code>, <code>mean_intensity</code>, <code>max_intensity</code></li>
          <li><code>is_coincident</code> (1 if this red spot overlaps the blue mask)</li>
        </ul>
      </li>
    </ul>
  </li>
  <li><code>blue_spots.csv</code>
    <ul>
      <li>Same columns as <code>red_spots.csv</code>, but for blue spots, and
      <code>is_coincident</code> indicates overlap with the red mask.</li>
    </ul>
  </li>
</ol>

<h3>5.2 Per-image summary table (global)</h3>

<p>In <code>top_root_directory</code>:</p>

<ul>
  <li><code>image_summary_metrics.tsv</code></li>
</ul>

<p>This file has one row per image pair (per folder), with columns such as:</p>

<ul>
  <li>Paths and names:
    <ul>
      <li><code>red_image_path</code>, <code>blue_image_path</code>, <code>red_image_name</code>, <code>blue_image_name</code>, <code>folder</code>, <code>roi_info</code></li>
    </ul>
  </li>
  <li>Thresholds:
    <ul>
      <li><code>red_threshold</code>, <code>blue_threshold</code>, <code>threshold_mode</code></li>
    </ul>
  </li>
  <li>Counts:
    <ul>
      <li><code>red_count</code>, <code>blue_count</code>, <code>coincident_count</code>,</li>
      <li><code>coincident_count_rot90px1000</code> (control)</li>
      <li><code>fraction_red_coincident</code></li>
    </ul>
  </li>
  <li>Densities (per µm²):
    <ul>
      <li><code>red_density_per_um2</code>, <code>blue_density_per_um2</code></li>
      <li><code>coincident_density_per_um2</code></li>
      <li><code>coincident_density_rot90px1000_per_um2</code></li>
    </ul>
  </li>
  <li>Basic intensity metrics:
    <ul>
      <li><code>red_spot_sum_mean</code>, <code>red_spot_sum_sd</code></li>
      <li><code>blue_spot_sum_mean</code>, <code>blue_spot_sum_sd</code></li>
    </ul>
  </li>
</ul>

<h3>5.3 Global summary tables (root)</h3>

<p>In <code>top_root_directory</code> you also get:</p>

<ol>
  <li><code>summary_CV_metrics.tsv</code>
    <ul>
      <li>One row per metric (e.g. <code>red_density_per_um2</code>, <code>blue_spot_sum_mean</code>).</li>
      <li>Columns:
        <ul>
          <li><code>metric</code></li>
          <li><code>mean_across_image_pairs</code></li>
          <li><code>sd_across_image_pairs</code></li>
          <li><code>cv_across_image_pairs</code></li>
        </ul>
      </li>
    </ul>
  </li>
  <li><code>global_mean_density_intensity.tsv</code>
    <ul>
      <li>One row per metric (e.g. <code>red_count</code>, <code>red_density_per_um2</code>, <code>red_spot_sum_mean</code>).</li>
      <li>Columns:
        <ul>
          <li><code>metric</code></li>
          <li><code>global_mean</code></li>
          <li><code>global_sd</code></li>
        </ul>
      </li>
    </ul>
  </li>
  <li><code>per_folder_count_and_density_stats.tsv</code>
    <ul>
      <li>One row per input folder (per <code>pathList</code> entry).</li>
      <li>Columns include:
        <ul>
          <li><code>folder</code></li>
          <li><code>red_count_mean</code>, <code>red_count_sd</code>, <code>red_count_cv</code></li>
          <li><code>blue_count_mean</code>, <code>blue_count_sd</code>, <code>blue_count_cv</code></li>
          <li><code>coincident_count_mean</code>, <code>coincident_count_sd</code>, <code>coincident_count_cv</code></li>
          <li><code>red_density_per_um2_mean</code>, <code>red_density_per_um2_sd</code>, <code>red_density_per_um2_cv</code></li>
          <li><code>blue_density_per_um2_mean</code>, …</li>
          <li><code>coincident_density_per_um2_mean</code>, …</li>
          <li><code>coincident_density_rot90px1000_per_um2_mean</code>, …</li>
        </ul>
      </li>
    </ul>
  </li>
</ol>

<h3>5.4 Per-folder summary files (inside each data folder)</h3>

<p>Inside each folder in <code>pathList</code>, e.g.:</p>

<ul>
  <li><code>.../RawData_EV/Test/</code></li>
  <li><code>.../RawData_EV/Test2/</code></li>
</ul>

<p>you will also get:</p>

<ol>
  <li><code>image_summary_metrics.tsv</code>
    <ul>
      <li>Same structure as the global file, but only rows for that specific folder.</li>
    </ul>
  </li>
  <li><code>global_mean_density_intensity.tsv</code>
    <ul>
      <li>Means and SDs computed only over that folder’s images.</li>
    </ul>
  </li>
</ol>

<hr>

<h2>6. How to run the script</h2>

<ol>
  <li>Open a terminal.</li>
  <li>Navigate to the folder containing your script, for example:
    <pre><code class="language-bash">cd /path/to/your/script
</code></pre>
  </li>
  <li>Make sure the script is executable (optional):
    <pre><code class="language-bash">chmod +x red_blue_coincidence.py
</code></pre>
  </li>
  <li>Run the script with Python:
    <pre><

