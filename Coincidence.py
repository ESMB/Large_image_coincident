#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:00:15 2021

@author: Mathew
"""

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters,measure
from PIL import Image
import pandas as pd
from scipy.spatial import distance
import czifile


# Size of pixels (nm)
pixel_size=591
# Where to save overall results
root_directory=r"/Users/Mathew/Documents/Current analysis/Ryan_conc/1pM/"  

# Paths to analyse below


pathList=[]

pathList.append(r"/Users/Mathew/Documents/Current analysis/Ryan_conc/1pM/")
# pathList.append(r"/Volumes/T7/Ryan_250326_aSyn_oligomer_488_647_20x_water_4z__2026-03-25T16_19_09-Measurement 1/max/No amplification/")
# pathList.append(r"/Volumes/T7/Ryan_250326_aSyn_oligomer_488_647_20x_water_4z__2026-03-25T16_19_09-Measurement 1/max/No oligomer/")
# pathList.append(r"/Volumes/T7/Ryan_250326_aSyn_oligomer_488_647_20x_water_4z__2026-03-25T16_19_09-Measurement 1/max/NB_olig/")



filename="B4_1pM.tif"


# Function to load images:

def load_image(toload):
    
    image=imread(toload)
    
    return image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_otsu(input_image):
    threshold_value=filters.threshold_otsu(input_image)    
    binary_image=input_image>threshold_value

    return threshold_value,binary_image


# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image
    
# Function to show the particular image:
def show(input_image,color=''):
    if(color=='Red'):
        plt.imshow(input_image,cmap="Reds")
        plt.show()
    elif(color=='Blue'):
        plt.imshow(input_image,cmap="Blues")
        plt.show()
    elif(color=='Green'):
        plt.imshow(input_image,cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show() 
    
        
# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# This is to look at coincidence purely in terms of pixels

def coincidence_analysis_pixels(binary_image1,binary_image2):
    pixel_overlap_image=binary_image1&binary_image2         
    pixel_overlap_count=pixel_overlap_image.sum()
    pixel_fraction=pixel_overlap_image.sum()/binary_image1.sum()
    
    return pixel_overlap_image,pixel_overlap_count,pixel_fraction

# Look at coincidence in terms of features. Needs binary image input 

def feature_coincidence(binary_image1,binary_image2):
    number_of_features,labelled_image1=label_image(binary_image1)          # Labelled image is required for this analysis
    coincident_image=binary_image1 & binary_image2        # Find pixel overlap between the two images
    coincident_labels=labelled_image1*coincident_image   # This gives a coincident image with the pixels being equal to label
    coinc_list, coinc_pixels = np.unique(coincident_labels, return_counts=True)     # This counts number of unique occureences in the image
    # Now for some statistics
    total_labels=labelled_image1.max()
    total_labels_coinc=len(coinc_list)
    fraction_coinc=total_labels_coinc/total_labels
    
    # Now look at the fraction of overlap in each feature
    # First of all, count the number of unique occurances in original image
    label_list, label_pixels = np.unique(labelled_image1, return_counts=True)
    fract_pixels_overlap=[]
    for i in range(len(coinc_list)):
        overlap_pixels=coinc_pixels[i]
        label=coinc_list[i]
        total_pixels=label_pixels[label]
        fract=1.0*overlap_pixels/total_pixels
        fract_pixels_overlap.append(fract)
    
    
    # Generate the images
    coinc_list[0]=1000000   # First value is zero- don't want to count these. 
    coincident_features_image=np.isin(labelled_image1,coinc_list)   # Generates binary image only from labels in coinc list
    coinc_list[0]=0
    non_coincident_features_image=~np.isin(labelled_image1,coinc_list)  # Generates image only from numbers not in coinc list.
    
    return coinc_list,coinc_pixels,fraction_coinc,coincident_features_image,non_coincident_features_image,fract_pixels_overlap

def analyse_coincident_features(red_image,
                                blue_image,
                                red_labelled,
                                coincident_features_image,
                                directory,
                                pixel_size_nm=None,
                                csv_name="coincident_features.csv",
                                hist_prefix="coincident_hist_"):
    """
    Analyse features in the coincident image.

    For each coincident red feature:
      - x, y (centroid coordinates in pixels)
      - red_intensity: sum of red intensity within the feature
      - blue_intensity: sum of blue intensity within the same region
      - total_intensity: red + blue
      - area: area in pixels (and in physical units if pixel_size_nm is given)
      - length: major axis length (pixels)
      - eccentricity

    Saves:
      - A CSV with one row per coincident feature.
      - Histograms of area, red_intensity, blue_intensity, total_intensity, length.

    Parameters
    ----------
    red_image : 2D ndarray
        Original red channel image (intensity).
    blue_image : 2D ndarray
        Original blue channel image (intensity).
    red_labelled : 2D ndarray (int)
        Label image of red features (output of measure.label on red_binary).
    coincident_features_image : 2D ndarray (bool)
        Binary mask of coincident features (from feature_coincidence).
        True where a red feature overlaps blue.
    directory : str
        Directory to save outputs.
    pixel_size_nm : float or None
        Pixel size in nm. If provided, area_um2 will be added.
    csv_name : str
        Name of the CSV file to save.
    hist_prefix : str
        Prefix for histogram image filenames.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from skimage import measure

    # Restrict red_labelled to coincident features only
    coincident_labels = red_labelled * coincident_features_image.astype(red_labelled.dtype)

    # Relabel to ensure labels are 1..N without gaps
    coincident_labelled = measure.label(coincident_labels)
    n_features = coincident_labelled.max()

    if n_features == 0:
        print("No coincident features found; skipping coincident feature analysis.")
        return

    # Regionprops on coincident label image for geometry & red intensity
    props_red = measure.regionprops_table(
        coincident_labelled,
        intensity_image=red_image,
        properties=('label',
                    'centroid',
                    'area',
                    'major_axis_length',
                    'eccentricity',
                    'intensity_image')  # we will use intensity_image to compute summed intensity
    )

    df_red = pd.DataFrame(props_red)

    # Compute summed red intensity explicitly from the intensity_image masks
    # regionprops_table gives 'intensity_image' flattened per region as a 1D object array
    # for convenience; we can sum each.
    red_sums = []
    for arr in df_red['intensity_image']:
        red_sums.append(np.sum(arr))
    df_red['red_intensity'] = red_sums
    df_red.drop(columns=['intensity_image'], inplace=True)

    # Regionprops for blue intensity using same labels (coincident_labelled)
    props_blue = measure.regionprops_table(
        coincident_labelled,
        intensity_image=blue_image,
        properties=('label',
                    'intensity_image')
    )
    df_blue = pd.DataFrame(props_blue)

    blue_sums = []
    for arr in df_blue['intensity_image']:
        blue_sums.append(np.sum(arr))
    df_blue['blue_intensity'] = blue_sums
    df_blue.drop(columns=['intensity_image'], inplace=True)

    # Merge red & blue by label
    df = pd.merge(df_red, df_blue, on='label', how='inner')

    # Rename and compute derived metrics
    df.rename(columns={
        'centroid-0': 'y',
        'centroid-1': 'x',
        'area': 'area_pixels',
        'major_axis_length': 'length_pixels'
    }, inplace=True)

    df['total_intensity'] = df['red_intensity'] + df['blue_intensity']

    # Physical area if pixel size is given
    if pixel_size_nm is not None:
        pixel_size_um = pixel_size_nm / 1000.0
        pixel_area_um2 = pixel_size_um ** 2
        df['area_um2'] = df['area_pixels'] * pixel_area_um2

    # Reorder/select columns
    cols = ['label', 'x', 'y',
            'red_intensity', 'blue_intensity', 'total_intensity',
            'length_pixels', 'area_pixels', 'eccentricity']
    if 'area_um2' in df.columns:
        cols.append('area_um2')

    df = df[cols]

    # Save CSV
    csv_path = directory + csv_name
    df.to_csv(csv_path, sep='\t', index=False)
    print(f"Saved coincident feature table to {csv_path}")

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------
    def save_hist(column, xlabel, fname, log=False):
        plt.figure(figsize=(4, 3))
        values = df[column].values
        plt.hist(values, bins=30, color='gray', edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        if log:
            plt.yscale('log')
        plt.tight_layout()
        plt.savefig(directory + fname, dpi=300)
        plt.close()

    # Area histogram
    save_hist('area_pixels', 'Area (pixels)', f"{hist_prefix}area_pixels.png", log=False)

    # Length histogram
    save_hist('length_pixels', 'Length (pixels)', f"{hist_prefix}length_pixels.png", log=False)

    # Intensity histograms
    save_hist('red_intensity', 'Red summed intensity', f"{hist_prefix}red_intensity.png", log=True)
    save_hist('blue_intensity', 'Blue summed intensity', f"{hist_prefix}blue_intensity.png", log=True)
    save_hist('total_intensity', 'Total summed intensity', f"{hist_prefix}total_intensity.png", log=True)

    print("Saved coincident feature histograms.")

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

    Parameters
    ----------
    red_labelled : 2D ndarray (int)
        Label image for red features (from skimage.measure.label).
    blue_labelled : 2D ndarray (int)
        Label image for blue features.
    coincident_features_image : 2D ndarray (bool or int)
        Binary mask of coincident features (e.g. from feature_coincidence).
    directory : str
        Path to save the output figure.
    nbins_x : int, optional
        Number of bins in x-direction for the density map.
    nbins_y : int, optional
        Number of bins in y-direction for the density map.
    filename : str, optional
        Output filename for the figure (within directory).

    Returns
    -------
    None
        Saves a PNG figure to disk.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import measure

    # 1) Red feature centroids
    if red_labelled.max() > 0:
        red_props = measure.regionprops_table(red_labelled, properties=('centroid',))
        red_y = red_props['centroid-0']
        red_x = red_props['centroid-1']
    else:
        red_y = np.array([])
        red_x = np.array([])

    # 2) Blue feature centroids
    if blue_labelled.max() > 0:
        blue_props = measure.regionprops_table(blue_labelled, properties=('centroid',))
        blue_y = blue_props['centroid-0']
        blue_x = blue_props['centroid-1']
    else:
        blue_y = np.array([])
        blue_x = np.array([])

    # 3) Coincident feature centroids
    coincident_labelled = measure.label(coincident_features_image)
    if coincident_labelled.max() > 0:
        coinc_props = measure.regionprops_table(coincident_labelled, properties=('centroid',))
        coinc_y = coinc_props['centroid-0']
        coinc_x = coinc_props['centroid-1']
    else:
        coinc_y = np.array([])
        coinc_x = np.array([])

    # Image size
    ny, nx = red_labelled.shape

    # Define bin edges
    x_edges = np.linspace(0, nx, nbins_x + 1)
    y_edges = np.linspace(0, ny, nbins_y + 1)

    # 2D histograms = density maps
    if len(red_x) > 0:
        red_density,  _, _  = np.histogram2d(red_y,  red_x,  bins=[y_edges, x_edges])
    else:
        red_density = np.zeros((nbins_y, nbins_x))

    if len(blue_x) > 0:
        blue_density, _, _  = np.histogram2d(blue_y, blue_x, bins=[y_edges, x_edges])
    else:
        blue_density = np.zeros((nbins_y, nbins_x))

    if len(coinc_x) > 0:
        coinc_density, _, _ = np.histogram2d(coinc_y, coinc_x, bins=[y_edges, x_edges])
    else:
        coinc_density = np.zeros((nbins_y, nbins_x))

    # Plot the three density maps
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Common colour scale
    vmax = max(red_density.max(), blue_density.max(), coinc_density.max())
    if vmax == 0:
        vmax = 1  # avoid zero colour range

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

    # Single colourbar
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Feature count per bin')

    plt.savefig(directory + filename, dpi=500)
    plt.close(fig)


for i in range(len(pathList)):
    
    directory=pathList[i]+"/"
    
    # For fixed thresholds:
    # aptamer_threshold=14506
    # nucleus_threshold=2172
    # antibody_threshold=9684
    
    # Load .czi images
    
    img = load_image(directory+filename)
  
    
    red=img[0,:,:]
    blue=img[1,:,:]
    


    # Run red
    
    red_threshold,red_binary=threshold_image_fixed(red,200)

    im = Image.fromarray(red_binary)
    im.save(directory+'Red_Binary.tif')
    
    
     
    red_number,red_labelled=label_image(red_binary)
    print("%d feautres were detected in the Red image."%red_number)
    # red_measurements=analyse_labelled_image(red_labelled,red)
    # red_measurements.to_csv(directory + '/' + 'all_red_metrics.csv', sep = '\t')
    
    # Run functions for blue
    
    blue_threshold,blue_binary=threshold_image_fixed(blue,200)

    im = Image.fromarray(blue_binary)
    im.save(directory+'Blue_Binary.tif')
    
    
     
    blue_number,blue_labelled=label_image(blue_binary)
    print("%d feautres were detected in the Blue image."%blue_number)
    # blue_measurements=analyse_labelled_image(blue_labelled,blue)
    # blue_measurements.to_csv(directory + '/' + 'all_blue_metrics.csv', sep = '\t')
    
    

    coinc_list,coinc_pixels,fraction_coinc,coincident_features_image,noncoincident_features_image,fraction_pixels_overlap=feature_coincidence(red_binary,blue_binary)
    print("%.2f of red features had coincidence with features in blue image. Average overlap was %2f."%(fraction_coinc,sum(fraction_pixels_overlap)/len(fraction_pixels_overlap)))
    
    coinc_im = Image.fromarray((coincident_features_image.astype(np.uint8) * 255))
    coinc_im.save(directory + 'Coincident_Features.tif')
   
    # Plot density maps

    plot_feature_density_maps(red_labelled,
                           blue_labelled,
                           coincident_features_image,
                           directory,
                           nbins_x=100,
                           nbins_y=100,
                           filename="density_maps.png")
    
    # Analyse coincident features
    analyse_coincident_features(
        red_image=red,
        blue_image=blue,
        red_labelled=red_labelled,
        coincident_features_image=coincident_features_image,
        directory=directory,
        pixel_size_nm=pixel_size,  # or None if you don't want physical units
        csv_name="coincident_features.csv",
        hist_prefix="coincident_hist_"
    )
    
