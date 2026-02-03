# LAB-TASK-5-2023-SE-39-B: Custom Image Filtering with SciPy

## Overview

This lab task focuses on implementing custom image filtering operations from scratch using NumPy and SciPy. Students will understand the mathematical foundations of convolution, implement filters manually, and compare results using scientific computing libraries instead of built-in PIL filters.

## Learning Objectives

- Understand convolution and its application to image processing
- Implement custom kernels for image filtering
- Use SciPy's convolution functions for advanced filtering
- Apply mean, median, mode, and Gaussian filters using manual methods
- Compare results between different filtering approaches
- Master NumPy arrays and SciPy signal processing functions

## Concepts Covered

### 1. Convolution Fundamentals

Convolution is the mathematical operation that underlies all image filtering. It combines two functions (image and kernel) to produce a third function (filtered image).

**Mathematical Definition:**

```
output(i,j) = Σ Σ kernel(m,n) × image(i+m, j+n)
```

**Steps in Convolution:**

1. Position kernel over image region
2. Multiply corresponding values
3. Sum all products
4. Place result at kernel center position
5. Repeat for all image positions

**Boundary Handling Modes:**

- **'reflect':** Mirror pixels at boundaries
- **'constant':** Pad with constant values
- **'nearest':** Repeat edge pixels
- **'wrap':** Wrap around to opposite edge

### 2. Kernel/Filter Concepts

A kernel is a small matrix of weights used in convolution.

**Common Kernel Properties:**

- **Size:** Typically 3×3, 5×5, or 7×7 (odd dimensions)
- **Normalization:** Sum of weights often equals 1 for stable filtering
- **Symmetry:** Most kernels are symmetric for better results

### 3. Mean Filter Implementation

Manually create and apply a uniform-weight averaging kernel.

**Manual Kernel Creation:**

```python
kernel_size = 3
mean_filter = np.ones((kernel_size, kernel_size), np.float32)
mean_filter = mean_filter / (kernel_size * kernel_size)
```

**3×3 Mean Filter Kernel:**

```
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

**Effect:** Average of all 9 pixels becomes the output pixel

**Advantages:**

- Simple linear operation
- Predictable results
- Fast computation with convolve

**Disadvantages:**

- Creates blur with significant edge distortion
- Not effective for salt-and-pepper noise

### 4. Median Filter (SciPy Implementation)

Non-linear filter using `scipy.ndimage.median_filter`.

**How It Works:**

1. For each pixel, extract neighborhood values
2. Sort all values
3. Return the middle value (median)

**Key Parameters:**

- **size:** Determines neighborhood dimensions
- **mode:** Boundary handling method

**Code:**

```python
from scipy.ndimage import median_filter
filtered_image = median_filter(image_array, size=3)
```

**Advantages:**

- Excellent for salt-and-pepper noise
- Preserves edges better than mean
- Non-linear filtering

**Disadvantages:**

- Computationally more expensive
- May remove fine details
- Signature ringing effects at edges

### 5. Mode Filter (Custom Implementation)

Non-linear filter that finds the most frequent pixel value in a neighborhood.

**Custom Implementation:**

```python
from scipy.ndimage import generic_filter

def mode(arr):
    """Return the mode of the array."""
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

filtered_image = generic_filter(image_array, mode, size=3)
```

**Process:**

1. Extract neighborhood pixels
2. Count frequency of each value
3. Return value with highest frequency
4. `generic_filter` applies custom function to each neighborhood

**Advantages:**

- Good for textured images
- Preserves regions well
- Effective for categorical data

**Disadvantages:**

- Slow computation (custom Python function)
- Creates posterization effects
- May create artificial boundaries

### 6. Gaussian Filter (SciPy Implementation)

Weighted blur using Gaussian (normal) distribution weights.

**Implementation:**

```python
from scipy.ndimage import gaussian_filter
filtered_image = gaussian_filter(image_array, sigma=1)
```

**Gaussian Kernel Concept:**
Weights decrease from center following Gaussian distribution:

```
weight(x,y) = (1/(2πσ²)) × e^(-(x²+y²)/(2σ²))
```

**Parameters:**

- **sigma:** Standard deviation (spread of Gaussian)
  - sigma=1: Narrow, subtle blur
  - sigma=2: Wider, more pronounced blur
  - Higher sigma = stronger blur effect

**Advantages:**

- Natural-looking blur
- Smooth gradients
- Better edge preservation than box blur

**Disadvantages:**

- Still blurs edges
- Computationally expensive
- Requires tuning sigma parameter

### 7. Image Preparation

Converting images to NumPy arrays for mathematical operations.

**Conversion Process:**

```python
# Load image
img = Image.open(image_path)

# Convert to grayscale for single-channel processing
img = img.convert('L')

# Convert to NumPy array for computation
img_array = np.array(img)

# Result: 2D array of pixel intensities (0-255)
```

## Task Implementation Steps

### Step 1: Import Libraries

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, median_filter, generic_filter, gaussian_filter
```

### Step 2: Load and Display Original Image

```python
image_path = '/content/image5(a).png'
img = Image.open(image_path).convert('L')  # Grayscale
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Image')
plt.show()
```

### Step 3: Image to NumPy Array Conversion

```python
img_array = np.array(img)
# Result: 2D array shape (height, width), values 0-255
```

### Step 4: Apply Mean Filter with Convolution

```python
from scipy.ndimage import convolve

kernel_size = 3
mean_filter = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
filtered_image = convolve(img_array, mean_filter, mode='reflect')
```

**What happens:**

1. Creates 3×3 kernel with all values = 1/9
2. Convolves kernel with image
3. Returns filtered image same size as original

### Step 5: Apply Median Filter

```python
from scipy.ndimage import median_filter

filtered_image = median_filter(img_array, size=3)
```

### Step 6: Apply Mode Filter

```python
from scipy.ndimage import generic_filter

def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

filtered_image = generic_filter(img_array, mode, size=3)
```

### Step 7: Apply Gaussian Filter

```python
from scipy.ndimage import gaussian_filter

filtered_image = gaussian_filter(img_array, sigma=1)
```

### Step 8: Compare All Filters

```python
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Repeat for each filtered version...

plt.tight_layout()
plt.show()
```

## Key Libraries and Functions

### NumPy Functions

| Function      | Purpose                           |
| ------------- | --------------------------------- |
| `np.ones()`   | Create array of ones (for kernel) |
| `np.array()`  | Convert image to NumPy array      |
| `np.unique()` | Find unique values and counts     |
| `np.argmax()` | Find index of maximum value       |

### SciPy Signal Processing Functions

| Function            | Purpose                                |
| ------------------- | -------------------------------------- |
| `convolve()`        | Apply custom convolution               |
| `median_filter()`   | Median filtering                       |
| `generic_filter()`  | Apply custom function to neighborhoods |
| `gaussian_filter()` | Gaussian blur                          |

### Convolution Modes

| Mode       | Behavior                     |
| ---------- | ---------------------------- |
| 'reflect'  | Mirror pixels at edge        |
| 'constant' | Pad with constant value      |
| 'nearest'  | Repeat edge pixels           |
| 'wrap'     | Wrap around to opposite edge |

## Execution Flow

```
Load Image
    ↓
Convert to Grayscale
    ↓
Convert to NumPy Array
    ↓
Apply Filters:
    ├── Mean Filter (convolve)
    ├── Median Filter (median_filter)
    ├── Mode Filter (generic_filter)
    └── Gaussian Filter (gaussian_filter)
    ↓
Compare Results in 2×2 Grid
    ↓
Analyze Differences
```

## Advanced Concepts

### 1. Convolution vs. Correlation

- **Convolution:** Kernel is flipped (both horizontally and vertically)
- **Correlation:** Kernel is not flipped
- For symmetric kernels, they're equivalent

### 2. Custom Kernels

Create specialized kernels for specific effects:

**Sharpening Kernel (Laplacian):**

```python
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)
```

**Edge Detection Kernel (Sobel-X):**

```python
edge_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
```

### 3. Filter Cascading

Apply multiple filters sequentially:

```python
# First: Mean filter
temp = convolve(img_array, mean_kernel, mode='reflect')

# Then: Gaussian filter
result = gaussian_filter(temp, sigma=1)
```

### 4. Kernel Size Effects

Experiment with different sizes:

```python
# Subtle blur
gaussian_filter(img, sigma=0.5)

# Moderate blur
gaussian_filter(img, sigma=1.0)

# Strong blur
gaussian_filter(img, sigma=2.0)
```

## Input Requirements

- **Format:** JPG, PNG, or other PIL-supported formats
- **Color:** Can be color or grayscale (converted to grayscale internally)
- **Path:** Update `image_path` variable
- **Location:** Upload to Colab or provide absolute path

## Expected Output

1. Original grayscale image
2. Mean filtered version (using convolve)
3. Median filtered version
4. Mode filtered version
5. Gaussian filtered version
6. Combined 2×2 comparison grid

## Filter Comparison Summary

| Filter       | Method          | Speed  | Edge Preservation | Noise Removal | Best For        |
| ------------ | --------------- | ------ | ----------------- | ------------- | --------------- |
| **Mean**     | convolve        | Fast   | Poor              | Moderate      | Uniform noise   |
| **Median**   | median_filter   | Medium | Good              | Excellent     | Salt-&-pepper   |
| **Mode**     | generic_filter  | Slow   | Good              | Good          | Textured images |
| **Gaussian** | gaussian_filter | Fast   | Better            | Moderate      | Natural blur    |

## Performance Considerations

### Speed Comparison (for 512×512 image):

- **Mean filter (convolve):** ~10-50ms
- **Gaussian filter:** ~20-100ms
- **Median filter:** ~100-500ms
- **Mode filter:** ~500ms-2s (slowest - Python function overhead)

### Optimization Tips:

1. Use compiled functions (median_filter, gaussian_filter) over custom Python
2. Convert to grayscale before filtering (faster than color)
3. For large images, consider downsampling first
4. Use appropriate boundary mode ('reflect' is often fastest)

## Practical Applications

- **Medical Imaging:** Preprocessing for feature extraction
- **Satellite Imagery:** Noise reduction and smoothing
- **Document Scanning:** Cleanup and denoising
- **Photography:** Artistic effects and noise reduction
- **Computer Vision:** Preprocessing for detection algorithms

## Error Handling

| Error                  | Cause                | Solution                                        |
| ---------------------- | -------------------- | ----------------------------------------------- |
| FileNotFoundError      | Incorrect image path | Check path syntax, use forward slashes          |
| ValueError (size)      | Invalid kernel size  | Use odd numbers (3, 5, 7)                       |
| TypeError (mode param) | Wrong boundary mode  | Use 'reflect', 'constant', 'nearest', or 'wrap' |
| MemoryError            | Image too large      | Reduce image size or resolution                 |
| AttributeError         | Missing SciPy        | Install: `pip install scipy`                    |

## Extension Activities

1. **Custom Kernel Design:**
   - Create sharpening kernel
   - Implement edge detection
   - Design specialized kernels

2. **Parameter Tuning:**
   - Test different kernel sizes
   - Experiment with sigma values
   - Compare boundary modes

3. **Performance Analysis:**
   - Measure execution time for each filter
   - Profile memory usage
   - Optimize for speed

4. **Quantitative Evaluation:**
   - Calculate PSNR (Peak Signal-to-Noise Ratio)
   - Measure edge preservation
   - Analyze noise reduction effectiveness

5. **Advanced Techniques:**
   - Bilateral filtering (preserves edges better)
   - Anisotropic diffusion
   - Multi-scale filtering (pyramid)
   - Morphological operations

## References

- NumPy Documentation: https://numpy.org/doc/
- SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/ndimage.html
- PIL/Pillow Documentation: https://pillow.readthedocs.io/
- Digital Image Processing by Gonzalez & Woods
- Convolution Tutorial: https://en.wikipedia.org/wiki/Convolution_(image_processing)

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 5-B (Custom Filtering with SciPy)
- **Academic Year:** 2023

---

**Note:** This task emphasizes understanding the mathematical foundations of filtering through manual implementation and SciPy functions rather than high-level PIL filters. Ensure all required packages (NumPy, SciPy, Pillow, Matplotlib) are installed before execution.
