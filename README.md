# Deep-Sea ROV Image Restoration

This repository contains code to restore images and videos taken by deep-sea ROVs (Remotely Operated Vehicles) at depths without natural light (most below a depth of 2000m). Specifically, it focuses on images and videos from the Norwegian Petroleum Directorate (NPD) and University of Bergen (UiB) research cruises at the mid-ocean ridges of Mohn's Ridge and Knipovich Ridge.   
The restoration process aims to provide a better starting point for both visual human interpretation and machine object detection. For videos, each frame is processed individually.

## Dataset

The ROV data from NPD/UiB data is publicly available. For more information visit the [NPD-website](https://www.npd.no/en/facts/seabed-minerals/data-acquisition-and-analyses/).   
This repository includes a few test images and short video clips for demonstration purposes:
- 2019 HD images and video (1920x1080p).
- 2018 SD images and video (704x576p).

## Image restoration model

The image restoration steps include:

| #  | Technique                              | Description                                                              |
|----|----------------------------------------|--------------------------------------------------------------------------|
| 1  | **Noise Reduction**                    | Bilateral filter-based smoothing                                         |
| 2  | **Dehazing**                           | Uses Underwater Dark Channel Prior (UDCP) to reduce scattering and absorption |
| 3  | **White Balancing**                    | Restores color balance using the "Gray World Assumption" method          |
| 4  | **Contrast Enhancement**               | Uses CLAHE* on LAB for contrast without altering color balance or hue     |
| 5  | **Color Correction**                   | Uses CLAHE* on RGB to adjust contrast with potential color balance/hue changes |
| 6  | **Edge Enhancement**                   | Employs Canny edge detection                                             |
| 7  | **Sharpening**                         | Enhances details using Unsharp Mask                                       |

\(*) CLAHE stands for Contrast Limited Adaptive Histogram Equalization and is a commonly used method for contrast enhancement.

## Repository Structure

The repository contains:

**Toolbox Scripts:**
- `plot_toolbox.py`: Contains functions related to plotting and inspecting the results.
- `restoration_toolbox.py`: Contains functions for image restoration.

**Jupyter Notebooks:**
- `process_images.ipynb`: Notebook to process images.
- `process_videos.ipynb`: Notebook to process videos.
- `concatenate_videos.ipynb`: Notebook to concatenate videos. For comparison between original vs. processed videos.

---

## Demo images
Original images are on the left and processed images are on the right.  
These images were processed with a relatively mild/safe setting to be suitable for a wide variety of scenes.

<p float="left">
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/test-images/ore_far.png" width="48%" alt="Original Image" />
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/result-images/ore_far_processed.png" width="48%" alt="Processed Image" />
</p>

<p float="left">
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/test-images/red_anemones_close.png" width="48%" alt="Original Image" />
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/result-images/red_anemones_close_processed.png" width="48%" alt="Segmented Image" />
</p>

<p float="left">
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/test-images/helio_far.png" width="48%" alt="Original Image" />
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/result-images/helio_far_processed.png" width="48%" alt="Segmented Image" />
</p>

<p float="left">
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/test-images/chimneys_columns.png" width="48%" alt="Original Image" />
  <img src="https://github.com/t-haakens/deep-sea-rov-image-restoration/blob/main/2019/HD/result-images/chimneys_columns_processed.png" width="48%" alt="Segmented Image" />
</p>

> **Note**: If GitHub isn't rendering the images, they can be seen or downloaded from the folders:
> - [2019/HD/test-images](https://github.com/t-haakens/deep-sea-rov-image-restoration/tree/main/2019/HD/test-images)
> - [2019/HD/result-images](https://github.com/t-haakens/deep-sea-rov-image-restoration/tree/main/2019/HD/result-images)

## Demo video
A demo video showing a side-by-side comparison of the original and processed videos can be seen or downloaded here.  
The in-browser player has a resolution limit of 720p, so download is preferred for better quality.
- [Video_comparison_of_Original_and_Processed](https://drive.google.com/file/d/1QvRMG4zKXWs5xGXra5jUs8M6EtK4EkM-/view?usp=sharing)  

---

## Getting Started

### Environment Setup using Conda

1. Create the conda environment:
   ```bash
   conda create -n restoration_env
   ```
2. Activate the created environment:
   ```bash
   conda activate restoration_env
   ```
3. Install the required packages:
   ```bash
   conda install numpy matplotlib opencv
   ```
If you need specific versions of the packages, they are:
   ```
   numpy==1.25.0
   matplotlib==3.7.1
   opencv==4.8.0.74
   ```
---

## Further work

- Make model dynamically fine-tune parameters settings for each image based on image statistics.   
The current model runs 1 parameter setting (a compromise) for all images in the folder.

## Contribution

Feel free to contribute, raise issues or suggest enhancements!
