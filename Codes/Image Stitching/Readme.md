# Image Stitching Using SIFT and RANSAC

## Overview
This project implements an image stitching pipeline using feature-based methods. The main steps include:
- Keypoint detection using SIFT
- Feature matching using BFMatcher
- Homography estimation with RANSAC
- Image transformation and stitching

The program reads a set of images, extracts keypoints, finds correspondences, and stitches them into a panorama.

## Installation & Setup
### Prerequisites
Ensure you have Conda installed. If not, download it from [here](https://docs.conda.io/en/latest/miniconda.html).

### Setting Up the Environment
Run the following commands to create and activate the environment:
```bash
conda create -n "Assignment-1"
conda activate image_stitching
conda install -c conda-forge menpo opencv numpy
```

## Running the Code
1. Ensure you have a `Test/Landscapre/` directory containing the images to be stitched.
2. Run the script using:
   ```bash
   python image_stitching.py
   ```
3. The results, including keypoint visualizations, matched keypoints, and the final panorama, will be stored in the `results/` directory.

## Methodology
1. **Keypoint Detection**
   - The SIFT algorithm is used to extract keypoints and descriptors from input images.
   - Visualizations of keypoints are saved for debugging.

2. **Feature Matching**
   - BFMatcher with k-nearest neighbors is used to find corresponding keypoints.
   - Lowe's ratio test filters good matches.
   - Matched keypoints are visualized and stored.

3. **Homography Estimation**
   - RANSAC selects the best transformation matrix to align images.
   - Outliers are filtered to improve accuracy.

4. **Image Blending**
   - The transformed images are blended using intensity blending.
   - Blended images are visualized and stored. 

4. **Image Stitching**
   - Images are warped using the homography matrix.
   - A final panorama is created and saved.

## Results & Observations
- The SIFT algorithm effectively detects stable keypoints, even in varying lighting conditions.
- RANSAC improves the accuracy of transformation by removing outlier matches.
- The final stitched image quality depends on the accuracy of keypoint detection and homography estimation.
- Overlapping areas between images help in better alignment; larger overlaps produce smoother results.
- Some distortions may occur when aligning images with perspective differences.




