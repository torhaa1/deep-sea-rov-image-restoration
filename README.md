# Deep-Sea ROV Image Restoration

This repository contains code to restore images and videos taken by deep-sea ROVs (Remotely Operated Vehicles) at depths without natural light. Specifically, it focuses on images and videos from the NPD and UiB research cruises at Mohn's Ridge and Knipovich Ridge.

## Dataset

The NPD/UiB data used is publicly available. This repository includes a few test images and a short video clip for demonstration purposes:
- HD images and video from 2019
- SD images and video from 2018

## Image restoration model

The image restoration steps include:

1. **Noise Reduction:** Smoothing
2. **UDCP (Underwater Dark Channel Prior):** mitigate underwater scattering and absorption
3. **White Balancing:** Color balance restoration using the method "Gray World Assumption"
4. **Contrast Enhancement:** CLAHE on LAB (doesn't change color balance and hue)
5. **Color Correction:** CLAHE on RGB (can alter color balance and hue)
6. **Edge Enhancement:** Canny
7. **Sharpening:** Unsharp Mask Sharpening


The restoration process aims to provide a better starting point for both visual human interpretation and machine object detection. For videos, each frame is processed individually.

## Repository Structure

The repository contains:

**Toolbox Scripts:**
- `plot_toolbox.py`: (You can add a brief description if you want)
- `restoration_toolbox.py`: (You can add a brief description if you want)

**Jupyter Notebooks:**
- `process_images.ipynb`: Notebook to process images.
- `process_videos.ipynb`: Notebook to process videos.
- `concatenate_videos.ipynb`: Notebook to concatenate processed videos.

## Getting Started

(You might want to include instructions on how to run the scripts or use the notebooks, any dependencies they need to install, etc.)

## Contribution

Feel free to contribute, raise issues or suggest enhancements!

## License

(You can mention the license here if you have one, for instance, "MIT License")

