# Image Processing Scripts

This repository contains two Python scripts for image processing using OpenCV and NumPy. The scripts perform operations like grayscale conversion, Gaussian smoothing, morphological transformations, contour detection, and edge detection.

## Prerequisites

Ensure you have the following dependencies installed:

```sh
pip install opencv-python numpy matplotlib
```

## Running the Scripts

### 1. Coin Detection Script

This script detects and counts coins in an image.

#### Steps:
- Reads an image (`six.jpg`)
- Applies Gaussian blur and morphological operations
- Uses contour detection to identify and count coins
- Displays the final processed image

#### Run the script:
```sh
python coin_detection.py
```

### 2. Edge Detection Script

This script processes an image to detect edges.

#### Steps:
- Converts the image (`one.jpg`) to grayscale
- Applies Gaussian blur to reduce noise
- Uses morphological opening to remove small noise
- Performs Canny edge detection
- Displays results at each step

#### Run the script:
```sh
python edge_detection.py
```

## Notes
- Ensure that images (`six.jpg` and `one.jpg`) are placed in the `images/` directory.
- The scripts display images using `cv2.imshow` and `matplotlib.pyplot`.
- Press any key in the OpenCV window to close it when running `coin_detection.py`.

Happy Coding!

