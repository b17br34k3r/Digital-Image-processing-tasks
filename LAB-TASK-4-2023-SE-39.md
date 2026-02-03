# LAB-TASK-4-2023-SE-39: Image Enhancement and Histogram Processing

## Overview

This lab task focuses on image enhancement techniques using histogram processing and various transformation methods. Students will learn how to improve image contrast, visibility, and quality through histogram equalization, contrast stretching, and non-linear transformations.

## Learning Objectives

- Understand histogram concepts and its importance in image processing
- Implement histogram equalization using both built-in and manual methods
- Apply contrast stretching to enhance image contrast
- Perform non-linear transformations (log and gamma correction)
- Visualize histograms before and after enhancement

## Concepts Covered

### 1. Histogram Basics

A histogram represents the distribution of pixel intensities in an image. It shows the frequency of each intensity value from 0-255.

### 2. Histogram Equalization

**Purpose:** Redistributes pixel intensities to improve image contrast by making the histogram more uniform.

**Methods:**

- **Built-in Method:** Uses OpenCV's `equalizeHist()` function with YCrCb color space for better results
- **Manual Method:** Implements histogram equalization from scratch using CDF (Cumulative Distribution Function)

**Process:**

1. Compute histogram of pixel intensities
2. Calculate Cumulative Distribution Function (CDF)
3. Normalize CDF to range [0, 255]
4. Map old pixel values to new values using the normalized CDF

### 3. Contrast Stretching

A linear transformation that expands the range of pixel intensities to use the full dynamic range [0, 255].

**Formula:**

```
stretched_pixel = ((original_pixel - min) / (max - min)) * 255
```

**Advantages:**

- Simple and fast
- Preserves pixel relationships
- Useful for low-contrast images

### 4. Log Transformation

A non-linear transformation that compresses the dynamic range of high-intensity images.

**Formula:**

```
output = c × log(1 + input)
```

**Applications:**

- Processing images with large dynamic range variations
- Enhancing details in brighter regions

### 5. Gamma Correction

A non-linear transformation used to correct or emphasize pixel intensities.

**Formula:**

```
output = input^(1/gamma)
```

**Effects:**

- **Gamma > 1:** Darkens the image
- **Gamma < 1:** Brightens the image

## Task Implementation

### Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Read and Display Original Image

- Load image using `cv2.imread()`
- Convert BGR to RGB color space for proper display
- Display using Matplotlib

### Step 3: Built-in Histogram Equalization

- Convert image to YCrCb color space
- Apply histogram equalization to Y channel only
- Convert back to RGB for display
- Preserve color information by processing only luminance

### Step 4: Manual Histogram Equalization

- Convert to grayscale
- Calculate histogram using `np.histogram()`
- Compute CDF (Cumulative Distribution Function)
- Normalize CDF values
- Map pixel values using normalized CDF

### Step 5: Histogram Comparison

- Plot histograms before and after equalization
- Compare distributions to visualize improvement
- 1x2 subplot arrangement for side-by-side comparison

### Step 6: Contrast Stretching

- Identify minimum and maximum pixel values
- Apply linear scaling formula
- Enhance visibility of low-contrast images

### Step 7: Advanced Transformations

- **Log Transform:** Compress high dynamic range
- **Gamma Correction:** Apply power-law transformation with adjustable gamma value

## Key Functions Used

| Function             | Purpose                                   |
| -------------------- | ----------------------------------------- |
| `cv2.imread()`       | Read image from file                      |
| `cv2.cvtColor()`     | Convert between color spaces              |
| `cv2.equalizeHist()` | Built-in histogram equalization           |
| `np.histogram()`     | Calculate histogram manually              |
| `plt.imshow()`       | Display images                            |
| `plt.hist()`         | Plot histograms                           |
| `np.log()`           | Logarithmic transformation                |
| `np.power()`         | Power transformation for gamma correction |

## Input Requirements

- **Image Format:** JPEG, PNG, or other standard formats
- **Recommended:** Color image for better demonstration
- **File Path:** Update the path in the code to match your image location

## Expected Output

1. Original image display
2. Histogram-equalized image (built-in method)
3. Histogram-equalized image (manual method)
4. Side-by-side histogram comparison (original vs. equalized)
5. Contrast-stretched image
6. Log-transformed and gamma-corrected variations

## Color Space Explanation

- **BGR/RGB:** Standard color representation (3 channels)
- **YCrCb:** Separates luminance (Y) from chrominance (Cr, Cb)
  - Y: Brightness/luminance
  - Cr, Cb: Color information
  - Advantage: Equalize only Y channel to preserve colors

## Practical Applications

- **Medical Imaging:** Enhance details in X-rays and CT scans
- **Satellite Imagery:** Improve visibility of land features
- **Photography:** Enhance low-light or high-contrast photos
- **Document Scanning:** Improve readability of scanned documents

## Common Issues & Solutions

| Issue                 | Solution                                                |
| --------------------- | ------------------------------------------------------- |
| Image path not found  | Verify file path and use forward slashes or raw strings |
| Color distortion      | Ensure proper color space conversion (BGR↔RGB)          |
| Histogram not showing | Check image data type and range (0-255 for uint8)       |
| Memory issues         | Use smaller images or reduce resolution                 |

## References

- OpenCV Documentation: https://docs.opencv.org/
- Matplotlib Documentation: https://matplotlib.org/
- Digital Image Processing by Gonzalez & Woods
- Image Enhancement Techniques: https://en.wikipedia.org/wiki/Histogram_equalization

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 4
- **Academic Year:** 2023

---

**Note:** Make sure to update the image file path in the code before execution. The code assumes the image is in an accessible location.
