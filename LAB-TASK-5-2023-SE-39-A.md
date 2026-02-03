# LAB-TASK-5-2023-SE-39-A: Image Smoothing Filters

## Overview

This lab task focuses on applying various smoothing (low-pass) filters to images to reduce noise and blur details. Students will learn how different filtering techniques work, their effects on images, and how to compare their results.

## Learning Objectives

- Understand the concept of image filtering and convolution
- Apply mean (box blur) filter for noise reduction
- Implement median filter for salt-and-pepper noise removal
- Use mode filter for local majority filtering
- Apply Gaussian blur for smooth transitions
- Visualize and compare filtering effects on images
- Master PIL (Python Imaging Library) for image processing

## Concepts Covered

### 1. Image Filtering Basics

Image filtering uses a kernel (small matrix) that slides over the image to modify pixel values based on neighboring pixels. This process is called **convolution**.

**Common Use Cases:**

- Noise reduction
- Image smoothing
- Edge detection
- Image sharpening

### 2. Mean Filter (Box Blur)

A linear filter that replaces each pixel with the average of its neighbors.

**Characteristics:**

- **Kernel Size:** Determines neighborhood size
- **Effect:** Reduces noise but blurs edges
- **Radius Parameter:** In PIL, radius=n means (2n+1)×(2n+1) kernel
  - radius=1 → 3×3 kernel
  - radius=2 → 5×5 kernel
  - radius=3 → 7×7 kernel

**Advantages:**

- Simple and fast to compute
- Good for uniform noise

**Disadvantages:**

- Blurs edges significantly
- Not ideal for preserving image details

**Formula (3×3 kernel):**

```
output(i,j) = (1/9) × Σ neighboring_pixels
```

### 3. Median Filter

A non-linear filter that replaces each pixel with the median value of its neighbors.

**Characteristics:**

- **Size Parameter:** Determines neighborhood area
- **Effect:** Excellent for salt-and-pepper noise
- **Preserves:** Edges better than mean filter

**Advantages:**

- Excellent for salt-and-pepper noise removal
- Preserves edges and details better than mean filter
- Non-linear operation

**Disadvantages:**

- Computationally more expensive than mean filter
- May remove small details along with noise

**Best For:**

- Removing impulse noise (salt-and-pepper)
- Preserving edge structures

### 4. Mode Filter

A non-linear filter that replaces each pixel with the most frequently occurring value in its neighborhood.

**Characteristics:**

- **Size Parameter:** Determines neighborhood area
- **Effect:** Finds dominant color/intensity in region
- **Use:** Reduces noise while preserving regions

**Advantages:**

- Preserves uniform regions well
- Good for textured images
- Effective for categorical data

**Disadvantages:**

- Computationally expensive
- May create artificial boundaries
- Less smooth transitions than mean/median

**Best For:**

- Textured and categorical images
- Segmentation and preprocessing

### 5. Gaussian Blur

A weighted blur filter where weights follow a Gaussian (normal) distribution.

**Characteristics:**

- **Radius Parameter:** Controls spread of the Gaussian
- **Effect:** Smooth, natural-looking blur
- **Weight Distribution:** Center pixel weighted heavily, edges less

**Advantages:**

- Produces smooth, natural results
- Maintains edge information better than box blur
- Closely mimics natural image defocusing

**Disadvantages:**

- More computationally expensive than box blur
- Still blurs edges, though less severely

**Formula (Gaussian Distribution):**

```
weight(x,y) = (1/(2πσ²)) × e^(-(x²+y²)/(2σ²))
```

**Comparison with Mean Filter:**

- Gaussian weights decrease from center outward
- Mean filter uses uniform weights

## Task Implementation

### Step 1: Import Required Libraries

```python
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
```

**Libraries:**

- **PIL (Pillow):** For image processing
- **matplotlib:** For visualization

### Step 2: Load and Display Original Image

- Use `Image.open()` to load the image
- Handle file not found errors gracefully
- Display with `plt.imshow()` and `plt.axis('off')`

### Step 3: Apply Mean Filter (Box Blur)

```python
filtered_img = img.filter(ImageFilter.BoxBlur(3))
```

- **Parameter:** radius=3 (7×7 kernel)
- Effect: Noticeable blur with edge smoothing

### Step 4: Apply Median Filter

```python
filtered_img_median = img.filter(ImageFilter.MedianFilter(size=3))
```

- **Parameter:** size=3 (3×3 neighborhood)
- Effect: Noise reduction with better edge preservation

### Step 5: Apply Mode Filter

```python
filtered_img_mode = img.filter(ImageFilter.ModeFilter(size=3))
```

- **Parameter:** size=3 (3×3 neighborhood)
- Effect: Region smoothing with detail preservation

### Step 6: Apply Gaussian Blur

```python
filtered_img_gaussian = img.filter(ImageFilter.GaussianBlur(radius=2))
```

- **Parameter:** radius=2
- Effect: Smooth, natural-looking blur

### Step 7: Compare All Filters

Create a 2×3 grid (or 1×5 for single row):

- Original image + 4 filtered versions
- Use `plt.subplot()` for layout
- Display side-by-side with titles

## Key Functions Used

| Function                     | Purpose               | Parameters          |
| ---------------------------- | --------------------- | ------------------- |
| `Image.open()`               | Load image from file  | filepath (str)      |
| `ImageFilter.BoxBlur()`      | Mean filter           | radius (int)        |
| `ImageFilter.MedianFilter()` | Median filter         | size (int)          |
| `ImageFilter.ModeFilter()`   | Mode filter           | size (int)          |
| `ImageFilter.GaussianBlur()` | Gaussian blur         | radius (float)      |
| `img.filter()`               | Apply filter to image | filter object       |
| `plt.imshow()`               | Display image         | image object        |
| `plt.subplot()`              | Create subplot grid   | (rows, cols, index) |
| `plt.tight_layout()`         | Adjust spacing        | none                |

## Execution Flow

```
Load Image
    ↓
Display Original
    ↓
Apply Individual Filters:
    ├── Mean Filter (BoxBlur)
    ├── Median Filter
    ├── Mode Filter
    └── Gaussian Blur
    ↓
Compare Results in Grid Layout
    ↓
Analyze Differences
```

## Input Requirements

- **Image Format:** JPG, PNG, BMP, or other PIL-supported formats
- **Location:** Specify path in `image_path` variable
- **File Path Examples:**
  - `/content/img5.jpg` (Colab)
  - `./image.jpg` (Local directory)
  - `C:/Users/path/to/image.jpg` (Windows absolute path)

## Expected Output

1. Original image display
2. Mean filtered image (3×3 → 7×7 region)
3. Median filtered image (noise removal)
4. Mode filtered image (region smoothing)
5. Gaussian blur (smooth transitions)
6. Combined comparison grid with all 5 images

## Filter Comparison Table

| Filter       | Kernel Type      | Edge Preservation | Noise Removal | Speed  | Best For        |
| ------------ | ---------------- | ----------------- | ------------- | ------ | --------------- |
| **Mean**     | Linear, Uniform  | Poor              | Moderate      | Fast   | Uniform noise   |
| **Median**   | Non-linear       | Good              | Excellent     | Medium | Salt-&-pepper   |
| **Mode**     | Non-linear       | Good              | Good          | Slow   | Textured images |
| **Gaussian** | Linear, Weighted | Better            | Moderate      | Medium | Natural blur    |

## Visual Effects Summary

### Mean Filter Effect:

- Strongest blur effect
- All neighbors equally weighted
- Creates halo artifacts around edges

### Median Filter Effect:

- Moderate blur
- Excellent noise removal
- Preserves edges reasonably well
- Best for binary/impulse noise

### Mode Filter Effect:

- Preserves regions
- May create posterization effect
- Good for segmentation
- Less blur than mean/median

### Gaussian Filter Effect:

- Smooth, natural blur
- Weighted by distance
- Better edge preservation than mean
- Most aesthetically pleasing

## Practical Applications

- **Medical Imaging:** Noise reduction while preserving diagnostically important details
- **Photography:** Subtle smoothing and noise removal
- **Satellite Imagery:** Pre-processing for feature extraction
- **Video Processing:** Temporal smoothing for noise reduction
- **Image Restoration:** Recovering damaged or noisy images

## Error Handling

| Error                        | Cause                | Solution                               |
| ---------------------------- | -------------------- | -------------------------------------- |
| FileNotFoundError            | Image path incorrect | Verify path and file existence         |
| Image.UnidentifiedImageError | Unsupported format   | Check file format, convert if needed   |
| MemoryError                  | Image too large      | Resize image or use smaller resolution |
| AttributeError               | 'img' not loaded     | Run previous cells to load image       |

## Troubleshooting Tips

1. **Image Not Loading:**
   - Check file path (use forward slashes)
   - Verify file exists and is readable
   - Try absolute path instead of relative

2. **Filters Not Working:**
   - Ensure previous cells executed successfully
   - Check 'img' variable exists (use `print(img)`)
   - Verify image is PIL Image object

3. **Display Issues:**
   - Ensure matplotlib is imported
   - Use `plt.show()` after plotting
   - Check subplot parameters are correct

4. **Kernel Size Effects:**
   - Larger radius/size = stronger effect
   - Smaller radius/size = subtle effect
   - Experiment with different values

## Extension Activities

1. **Compare Multiple Kernel Sizes:**
   - Test radius = 1, 2, 3, 4 for same filter
   - Observe effect of kernel size

2. **Create Custom Kernels:**
   - Define custom convolution kernels
   - Implement custom filter logic

3. **Combine Filters:**
   - Apply multiple filters sequentially
   - Compare cascade effects

4. **Quantitative Analysis:**
   - Calculate image statistics (mean, variance)
   - Measure blur using edge detection
   - Compare signal-to-noise ratios

5. **Apply to Different Image Types:**
   - Natural images
   - Noisy images
   - Document images
   - Medical images

## References

- PIL Documentation: https://pillow.readthedocs.io/
- Matplotlib Documentation: https://matplotlib.org/
- Digital Image Processing by Gonzalez & Woods
- Image Filtering: https://en.wikipedia.org/wiki/Image_filter
- Convolution: https://en.wikipedia.org/wiki/Convolution
- Gaussian Blur: https://en.wikipedia.org/wiki/Gaussian_blur

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 5-A (Smoothing Filters)
- **Academic Year:** 2023

---

**Note:** Ensure the image file path is correctly specified before executing the code. All filter operations are non-destructive and do not modify the original image object.
