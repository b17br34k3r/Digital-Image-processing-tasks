# LAB-TASK-8-2023-SE-39: Motion Blur Restoration and Image Deblurring

## Overview

This lab task focuses on restoring motion-blurred images using multiple complementary techniques. Students will learn various approaches to image deblurring, compare their effectiveness, and understand the trade-offs between different restoration methods. The task demonstrates practical image restoration using unsharp masking, deconvolution, bilateral filtering, adaptive sharpening, and frequency domain techniques.

## Learning Objectives

- Understand motion blur and its effects on images
- Learn multiple image restoration and deblurring techniques
- Implement unsharp masking for edge enhancement
- Apply deconvolution with custom kernels
- Use bilateral filtering for edge-preserving smoothing
- Implement adaptive sharpening based on local variance
- Apply frequency domain filtering for restoration
- Compare and evaluate restoration methods
- Measure image quality improvement using sharpness metrics

## Concepts Covered

### 1. Motion Blur Fundamentals

Motion blur occurs when an image is captured during camera or object movement.

**Causes:**

- Camera movement during capture
- Subject movement during exposure
- Fast motion in video frames
- Slow shutter speed

**Characteristics:**

- Directional blur along motion path
- Loss of high-frequency details
- Reduced image clarity and sharpness
- Linear or curved blur pattern

**Mathematical Representation:**

```
Blurred Image = Original Image * Motion Blur Kernel
```

Where kernel represents the motion path (typically linear or diagonal).

### 2. Image Restoration vs. Enhancement

Important distinction in image processing:

**Restoration:**

- Attempts to recover original content
- Inverse modeling of degradation process
- Tries to undo known degradation
- Example: Motion deblurring, denoising

**Enhancement:**

- Improves visual appearance
- May not recover original content
- Subjective improvement
- Example: Contrast stretching, sharpening

### 3. Method 1: Unsharp Masking

Creates a sharp image by subtracting a blurred version from the original.

**Mathematical Formula:**

```
Sharpened = Original + Amount × (Original - Blurred)
```

Or using weights:

```
Sharpened = (1 + Amount) × Original - Amount × Blurred
```

**Implementation:**

```python
gaussian_blur = cv2.GaussianBlur(img_gray, (0, 0), 2.0)
unsharp_amount = 1.5
restored = cv2.addWeighted(img_gray, 1.0 + unsharp_amount,
                           gaussian_blur, -unsharp_amount, 0)
restored = np.clip(restored, 0, 255).astype(np.uint8)
```

**Parameters:**

- **Blur kernel:** Controls what's considered "blurred"
- **Amount:** Controls sharpening intensity
  - 0.5: Subtle sharpening
  - 1.0-2.0: Strong sharpening
  - 3.0+: Very aggressive

**Advantages:**

- Simple and fast
- No artifacts (minimal)
- Predictable results
- Works on various blur types

**Disadvantages:**

- Doesn't truly recover lost information
- Can enhance noise
- Limited by amount of blur information preserved

**Best For:**

- General-purpose sharpening
- Mild to moderate motion blur
- Quick enhancement needs

### 4. Method 2: Deconvolution with Custom Kernel

Attempts to reverse the blurring process by applying an inverse kernel.

**Concept:**
If image was blurred with kernel K:

```
Blurred = Original * K
```

Then to restore:

```
Restored = Blurred * K_inverse
```

**Kernel Design:**

```python
kernel_size = 9
kernel = np.zeros((kernel_size, kernel_size))
kernel[kernel_size//2, :] = 1  # Horizontal line
kernel = kernel / kernel.sum()

# Create inverse kernel
kernel_inverse = np.zeros_like(kernel)
kernel_inverse[kernel_size//2, kernel_size//2] = 2
kernel_inverse = kernel_inverse - kernel
```

**Application:**

```python
restored_deconv = cv2.filter2D(img_gray, -1, kernel_inverse)
```

**Advantages:**

- Targets specific blur type
- Theoretically sound
- Can be very effective

**Disadvantages:**

- Requires knowing blur kernel
- Noise amplification
- Can produce artifacts
- Sensitive to kernel accuracy

**Best For:**

- Known motion direction/distance
- Controlled blur conditions
- Research applications

### 5. Method 3: Bilateral Filter + Sharpening

Combines edge-preserving smoothing with selective sharpening.

**Bilateral Filter:**

```python
bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
```

**Parameters:**

- **Diameter (9):** Neighborhood size
- **Sigma color (75):** Color/intensity difference threshold
- **Sigma space (75):** Distance threshold

**Behavior:**

- Smooths similar regions
- Preserves edges (important pixel transitions)
- Non-linear filtering

**Sharpening Kernel (Laplacian-based):**

```python
sharpen_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])
```

**Formula:**

- Center pixel boosted by 9
- Neighbors subtracted
- Emphasizes local differences

**Application:**

```python
restored_bilateral = cv2.filter2D(bilateral, -1, sharpen_kernel)
```

**Advantages:**

- Edge preservation
- Good visual results
- Reduces noise before sharpening
- Balances smoothing and enhancement

**Disadvantages:**

- Requires parameter tuning
- Slower than simple methods
- May over-sharpen edges

**Best For:**

- Images with significant noise
- Selective enhancement needed
- Professional results desired

### 6. Method 4: Adaptive Sharpening

Applies sharpening more aggressively to blurred regions, preserving sharp areas.

**Variance Calculation:**

```python
mean = cv2.blur(img_gray.astype(float), (5, 5))
sqr_mean = cv2.blur((img_gray.astype(float))**2, (5, 5))
variance = sqr_mean - mean**2
```

**Interpretation:**

- High variance: Sharp, detailed regions
- Low variance: Flat, blurred regions

**Normalization:**

```python
variance_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-7)
```

**Adaptive Application:**

```python
sharpened = cv2.filter2D(img_gray, -1, sharpen_kernel)
restored_adaptive = (variance_norm * sharpened + (1 - variance_norm) * img_gray)
```

**Logic:**

- Where variance is high: Use original (already sharp)
- Where variance is low: Use sharpened (needs enhancement)

**Advantages:**

- Intelligent region-based processing
- Avoids over-sharpening sharp areas
- Targets problem regions
- Preserves fine details

**Disadvantages:**

- More complex implementation
- Parameter tuning needed
- Slower computation
- May miss some artifacts

**Best For:**

- Variable blur across image
- Mixed sharp and blurred regions
- Quality-conscious applications

### 7. Method 5: Wiener-Inspired Frequency Domain Filtering

Uses frequency domain analysis to enhance high frequencies selectively.

**FFT Transformation:**

```python
f_transform = np.fft.fft2(img_float)
f_shift = np.fft.fftshift(f_transform)
```

**High-Pass Mask Design:**

```python
mask = np.ones((rows, cols), np.float32)
r = 30
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0.3  # Reduce low frequencies
```

**Application:**

```python
f_shift_filtered = f_shift * mask
f_ishift = np.fft.ifftshift(f_shift_filtered)
img_back = np.fft.ifft2(f_ishift)
restored_fft = np.abs(img_back)
```

**Concept:**

- Low frequencies: General intensity structure
- High frequencies: Edges and details
- Boosting high frequencies: Sharpening effect

**Advantages:**

- Frequency-aware processing
- Selective enhancement
- Theoretical foundation
- Can target specific frequencies

**Disadvantages:**

- Requires FFT computation
- Complex parameter tuning
- Edge artifacts possible
- Slower than spatial methods

**Best For:**

- Specific frequency content issues
- Research and analysis
- When frequency domain insight matters

## Task Implementation Steps

### Step 1: Import Libraries and Setup

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Load and Prepare Image

```python
# Upload or load image
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display original
plt.imshow(img_rgb)
plt.title('Original Motion-Blurred Image')
plt.show()
```

### Step 3: Apply All Five Methods

Each method follows similar pattern:

1. Initialize parameters
2. Apply processing
3. Clip values to 0-255 range
4. Convert to uint8

### Step 4: Compare Results

```python
plt.figure(figsize=(20, 12))

# Original
plt.subplot(2, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Blurred')
plt.axis('off')

# Each method
plt.subplot(2, 3, 2)
plt.imshow(restored_unsharp, cmap='gray')
plt.title('Method 1: Unsharp Masking')
plt.axis('off')

# ... repeat for other methods

plt.tight_layout()
plt.show()
```

### Step 5: Measure Quality Improvement

```python
def calculate_sharpness(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

original_sharpness = calculate_sharpness(img_gray)
restored_sharpness = calculate_sharpness(final_result)
improvement = ((restored_sharpness - original_sharpness) / original_sharpness) * 100
```

### Step 6: Save Results

```python
cv2.imwrite('restored_unsharp.jpg', restored_unsharp)
cv2.imwrite('comparison_before_after.jpg', comparison)
```

## Sharpness Metric Explanation

**Laplacian Variance:**

```
Sharpness = Variance of Laplacian filtered image
```

**Why it works:**

- Laplacian detects edges (2nd derivative)
- Blurred images have low edge contrast
- Sharp images have high edge variance
- Higher variance = Sharper image

**Limitations:**

- Sensitive to noise
- Doesn't account for visual perception
- Can be fooled by high-frequency noise

## Key Functions Used

### OpenCV Functions

| Function                | Purpose                  |
| ----------------------- | ------------------------ |
| `cv2.imread()`          | Read image from file     |
| `cv2.cvtColor()`        | Convert color space      |
| `cv2.GaussianBlur()`    | Gaussian blurring        |
| `cv2.bilateralFilter()` | Edge-preserving filter   |
| `cv2.filter2D()`        | 2D convolution           |
| `cv2.Laplacian()`       | Laplacian edge detection |
| `cv2.addWeighted()`     | Weighted image addition  |

### NumPy Functions

| Function             | Purpose                      |
| -------------------- | ---------------------------- |
| `np.fft.fft2()`      | 2D Fast Fourier Transform    |
| `np.fft.ifft2()`     | Inverse FFT                  |
| `np.fft.fftshift()`  | Shift FFT for visualization  |
| `np.fft.ifftshift()` | Reverse fftshift             |
| `np.abs()`           | Magnitude of complex numbers |
| `np.clip()`          | Bound values to range        |

## Method Selection Guide

### Choose Unsharp Masking When:

- Quick results needed
- Moderate blur present
- Simplicity preferred
- Consistent quality important

### Choose Deconvolution When:

- Blur type is known
- Specific motion direction identified
- Research application
- Precision is critical

### Choose Bilateral + Sharpening When:

- Noise present with blur
- Edge preservation important
- Higher quality needed
- Processing time acceptable

### Choose Adaptive Sharpening When:

- Blur varies across image
- Some regions already sharp
- Intelligent processing desired
- Fine tuning possible

### Choose Frequency Domain When:

- Specific frequency bands need adjustment
- Theoretical analysis important
- Advanced filtering needed
- Complex restoration required

## Comparison Table

| Method              | Speed     | Artifact Risk | Noise Sensitivity | Parameter Tuning | Visual Quality       |
| ------------------- | --------- | ------------- | ----------------- | ---------------- | -------------------- |
| **Unsharp**         | Very Fast | Low           | High              | Easy             | Good                 |
| **Deconv**          | Fast      | High          | Very High         | Moderate         | Excellent (if tuned) |
| **Bilateral+Sharp** | Moderate  | Low           | Very Low          | Moderate         | Excellent            |
| **Adaptive**        | Moderate  | Low           | Low               | Complex          | Very Good            |
| **Frequency**       | Moderate  | Medium        | Medium            | Complex          | Good                 |

## Input Requirements

- **Format:** JPG, PNG, or standard image formats
- **Color:** Converted to grayscale internally
- **Motion blur:** Linear or somewhat uniform
- **Artifacts:** Minimal heavy JPEG artifacts preferred

## Expected Output

1. Original blurred image
2. Five different restoration methods
3. Before/after comparison
4. Quality metrics (sharpness improvement %)
5. Saved restoration results

## Parameters and Tuning

### Unsharp Masking

- **Gaussian blur sigma:** 1.0-3.0 (controls what's blurred)
- **Amount:** 0.5-3.0 (sharpening intensity)
- **Higher sigma:** Targets larger blurs
- **Higher amount:** Stronger enhancement

### Bilateral Filter

- **Diameter:** 5-15 (neighborhood size)
- **Sigma color:** 30-150 (intensity difference tolerance)
- **Sigma space:** 30-150 (spatial distance tolerance)
- **Larger values:** More smoothing

### Adaptive Sharpening

- **Variance window:** 3×3 to 7×7 (region analysis)
- **Sharpening kernel:** Laplacian or custom
- **Blend factor:** 0.0-1.0 (adaptation strength)

### Frequency Domain

- **Mask radius:** 10-100 (center area to reduce)
- **Mask value:** 0.1-0.5 (attenuation strength)
- **Larger radius:** More low-frequency reduction

## Artifacts and Solutions

### Ringing Artifacts

- **Cause:** Over-sharpening or oscillations
- **Solution:** Reduce amount or use bilateral first
- **Prevention:** Use Gaussian bilateral filter

### Noise Amplification

- **Cause:** Sharpening high frequencies
- **Solution:** Use bilateral filter first
- **Alternative:** Apply light Gaussian blur before

### Loss of Detail

- **Cause:** Over-smoothing in bilateral
- **Solution:** Adjust diameter and sigma values
- **Alternative:** Reduce bilateral before sharpening

### Halos Around Edges

- **Cause:** Unsharp masking too strong
- **Solution:** Reduce amount parameter
- **Prevention:** Use adaptive method

## Quality Assessment

### Sharpness Metric Issues

- Noise increases apparent sharpness
- Doesn't correlate perfectly with perception
- Useful for relative comparison

### Visual Inspection Checklist

- [ ] Edges appear crisp
- [ ] No obvious halos around objects
- [ ] Noise not excessive
- [ ] Overall contrast reasonable
- [ ] Colors/tones preserved
- [ ] No visible artifacts

### When Methods Fail

- **Extreme motion blur:** Loss of information irreversible
- **Very noisy image:** Noise amplified
- **JPEG artifacts:** Compression artifacts enhanced
- **Unknown blur type:** Difficult to correct

## Advanced Concepts

### Richardson-Lucy Deconvolution

More sophisticated deconvolution method:

- Iterative process
- Uses likelihood maximization
- Better results than simple deconvolution
- Much slower (computationally expensive)
- Can diverge if not carefully tuned
- Often produces ringing artifacts

### Wiener Filtering

Optimal linear restoration:

- Requires noise power spectrum
- Requires blur kernel
- Minimizes mean square error
- Practical implementation complex

### Total Variation Methods

Preserves edges while smoothing:

- Solves optimization problem
- Edge-aware smoothing
- Computationally intensive
- Advanced technique

## Practical Applications

### Photography

- Rescue slightly motion-blurred photos
- Enhance camera shake images
- Improve low-light photo clarity

### Video Processing

- Stabilize shaky video frames
- Enhance low-resolution video
- Motion artifact reduction

### Document Scanning

- Improve scanned document clarity
- Handle hand-shake during scan
- Enhance text readability

### Medical Imaging

- Motion artifact correction
- Image deblurring
- Enhancement for diagnosis

### Satellite/Aerial

- Fix satellite image blur
- Enhance aerial photography
- Restore motion-affected images

## Performance Considerations

### Computation Time (for 512×512 image)

- Unsharp masking: ~10-20ms
- Bilateral filter: ~50-100ms
- Adaptive sharpening: ~30-50ms
- Frequency domain: ~100-200ms (includes FFT)
- Deconvolution: ~20-30ms

### Memory Usage

- All methods: ~3-4× image size
- FFT methods: ~5-6× image size (complex numbers)
- Temporary arrays: ~2× image size

## Troubleshooting

| Issue             | Cause                    | Solution                      |
| ----------------- | ------------------------ | ----------------------------- |
| Output too bright | np.clip not applied      | Use np.clip(result, 0, 255)   |
| Extreme ringing   | Amount too high          | Reduce unsharp_amount         |
| Noise enhanced    | Sharpening too strong    | Use bilateral first or reduce |
| Black edges       | FFT not properly shifted | Check fftshift/ifftshift      |
| Results too dark  | Normalization issue      | Check data type conversion    |
| Artifacts visible | Over-processing          | Reduce parameter values       |

## Extension Activities

1. **Compare with Ground Truth:**
   - Create synthetic blurred image
   - Apply restoration methods
   - Compare with original

2. **Motion Direction Detection:**
   - Analyze blur direction
   - Create oriented kernel
   - Optimize deconvolution

3. **Blind Deconvolution:**
   - Estimate blur kernel from image
   - Apply deconvolution
   - Iterate for improvement

4. **Real-time Processing:**
   - Implement GPU acceleration
   - Process video streams
   - Optimize for speed

5. **Hybrid Methods:**
   - Combine multiple methods
   - Weighted averaging
   - Adaptive selection

## Visual Quality Comparison Guide

### Unsharp Masking

- ✓ Natural appearance
- ✓ No artifacts
- ✗ Doesn't recover lost detail
- ✗ May enhance noise

### Bilateral + Sharpening

- ✓ Edge preserved
- ✓ Noise handled well
- ✓ Good visual quality
- ✗ More complex
- ✗ Slower

### Adaptive Sharpening

- ✓ Intelligent processing
- ✓ Targets problem areas
- ✓ Preserves sharp regions
- ✗ Parameter tuning complex

### Deconvolution

- ✓ Theoretically sound
- ✓ Can be very effective
- ✗ Noise amplification
- ✗ Requires kernel knowledge

## References

- Digital Image Processing by Gonzalez & Woods
- Image Restoration Theory: https://en.wikipedia.org/wiki/Image_restoration
- Deconvolution: https://en.wikipedia.org/wiki/Deconvolution
- Unsharp Masking: https://en.wikipedia.org/wiki/Unsharp_masking
- OpenCV Documentation: https://docs.opencv.org/
- NumPy FFT: https://numpy.org/doc/stable/reference/fft.html

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 8
- **Academic Year:** 2023

---

**Note:** Motion blur restoration is an ill-posed problem - information is truly lost. These methods enhance and recover what can be recovered, but cannot perfectly restore severely blurred images. The choice of method depends on specific application requirements, desired quality, and acceptable processing time.
