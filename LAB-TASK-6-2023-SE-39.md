# LAB-TASK-6-2023-SE-39: Frequency Domain Analysis and Filtering

## Overview

This lab task focuses on understanding and working with images in the frequency domain using the Fast Fourier Transform (FFT). Students will learn how to decompose images into frequency components, analyze magnitude and phase spectra, and perform frequency-domain filtering using low-pass and high-pass filters.

## Learning Objectives

- Understand the Fourier Transform and its application to image processing
- Compute 2D FFT of images and interpret frequency domain representation
- Analyze magnitude spectrum and phase spectrum separately
- Implement low-pass filtering in the frequency domain
- Implement high-pass filtering in the frequency domain
- Compare frequency domain filtering with spatial domain filtering
- Understand the relationship between spatial and frequency domains

## Concepts Covered

### 1. Frequency Domain Fundamentals

The frequency domain represents an image as a sum of sinusoidal components at different frequencies.

**Key Concepts:**

- **Spatial Domain:** Image pixels (intensity values)
- **Frequency Domain:** Sinusoidal components (amplitudes at different frequencies)
- **Transformation:** Fourier Transform converts between these domains
- **Inverse Transform:** Converts back from frequency to spatial domain

**Why Frequency Domain?**

- Easier to implement certain filters
- More efficient computation for large kernels
- Better understanding of image characteristics
- Separation of low and high frequency information

### 2. Fast Fourier Transform (FFT)

Efficient algorithm to compute the Discrete Fourier Transform (DFT).

**Mathematical Definition (2D FFT):**

```
F(u,v) = Σ Σ f(x,y) × e^(-j2π(ux+vy)/N)
```

Where:

- f(x,y) = Pixel intensity at position (x,y)
- F(u,v) = Frequency component
- u,v = Frequency indices
- j = Imaginary unit

**Complexity:**

- Direct DFT: O(N⁴) for N×N image
- FFT: O(N² log N) - much faster

**In NumPy:**

```python
fft_image = np.fft.fft2(image)  # 2D FFT
fft_shifted = np.fft.fftshift(fft_image)  # Center zero frequency
```

### 3. FFT Shift Operation

Centers the zero-frequency component for easier visualization and filtering.

**Why Shift?**

- Zero frequency (DC component) is at image corners
- Shifting brings it to center for easier visualization
- Simplifies mask creation for filtering

**Operations:**

```python
# Forward shift: moves DC to center
fft_shifted = np.fft.fftshift(fft_image)

# Inverse shift: moves DC back to corner
fft_original = np.fft.ifftshift(fft_shifted)
```

**Visual Effect:**

- Before shift: DC component in corners, frequencies around edges
- After shift: DC component at center, higher frequencies at edges

### 4. Magnitude Spectrum

Represents the amplitude of each frequency component.

**Calculation:**

```python
magnitude_spectrum = np.abs(fft_shifted)
```

**Visualization Enhancement:**

```python
# Use logarithmic scale for better visualization
magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
```

**Why Log Scale?**

- Raw magnitude values have huge dynamic range
- Log compression makes details visible
- Prevents visualization issues due to extreme values
- Adding 1 avoids log(0) = undefined

**Interpretation:**

- Bright areas = High frequency content (details)
- Dark areas = Low frequency content (smooth regions)
- Center brightness = Overall image intensity

### 5. Phase Spectrum

Contains phase information (angle) of each frequency component.

**Calculation:**

```python
phase_spectrum = np.angle(fft_shifted)
```

**Range:** -π to π radians

**Importance:**

- Preserves spatial relationships
- Often underestimated in image processing
- Crucial for image reconstruction

**Interesting Property:**
Swapping magnitude of one image with phase of another creates visually similar image to phase source.

### 6. Low-Pass Filtering

Keeps low frequencies, removes high frequencies.

**Creating Circular Low-Pass Mask:**

```python
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)

# Create circular mask
r = 30  # Radius parameter
cv2.circle(mask, (ccol, crow), r, 1, thickness=-1)
```

**Application:**

```python
low_freq_fft = fft_shifted * mask
```

**Effect:**

- Smooths image
- Removes sharp transitions
- Reduces noise
- Blurs details

**Parameters:**

- **Radius:** Controls cutoff frequency
  - Small radius: More smoothing, more blur
  - Large radius: Less smoothing, preserves details

### 7. High-Pass Filtering

Keeps high frequencies, removes low frequencies.

**Creating Circular High-Pass Mask:**

```python
high_mask = 1 - mask  # Inverse of low-pass mask
```

**Application:**

```python
high_freq_fft = fft_shifted * high_mask
```

**Effect:**

- Enhances edges and details
- Removes smooth backgrounds
- Highlights transitions
- Produces edge-like appearance

**Properties:**

- Complementary to low-pass
- High-pass + Low-pass = Original

### 8. Inverse FFT

Converts filtered frequency domain back to spatial domain.

**Process:**

```python
# Remove shift
fft_unshifted = np.fft.ifftshift(filtered_fft)

# Apply inverse FFT
spatial_image = np.fft.ifft2(fft_unshifted)

# Take magnitude (remove complex part)
result = np.abs(spatial_image)
```

**Important:**

- `ifftshift` must reverse the earlier `fftshift`
- Use `np.abs()` to get real values
- Small imaginary components due to numerical errors

## Task Implementation Steps

### Step 1: Import Libraries

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

### Step 2: Load Image

```python
# Load in grayscale for simpler processing
image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

# Display original
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()
```

### Step 3: Compute 2D FFT

```python
# Compute FFT
fft_image = np.fft.fft2(image)

# Shift zero frequency to center
fft_shifted = np.fft.fftshift(fft_image)
```

### Step 4: Compute and Visualize Spectra

```python
# Magnitude spectrum with log scaling
magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)

# Phase spectrum
phase_spectrum = np.angle(fft_shifted)

# Display both
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')

plt.subplot(1, 2, 2)
plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum')
plt.show()
```

### Step 5: Create Filter Masks

```python
# Get image dimensions
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create circular mask
mask = np.zeros((rows, cols), np.uint8)
r = 30  # Adjust radius for different cutoff frequencies
cv2.circle(mask, (ccol, crow), r, 1, thickness=-1)

# Low-pass filtered FFT
low_freq_fft = fft_shifted * mask

# High-pass filtered FFT
high_mask = 1 - mask
high_freq_fft = fft_shifted * high_mask
```

### Step 6: Apply Inverse FFT

```python
# Low-frequency image
low_freq_image = np.fft.ifft2(np.fft.ifftshift(low_freq_fft))
low_freq_image = np.abs(low_freq_image)

# High-frequency image
high_freq_image = np.fft.ifft2(np.fft.ifftshift(high_freq_fft))
high_freq_image = np.abs(high_freq_image)
```

### Step 7: Display Results

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(low_freq_image, cmap='gray')
plt.title('Low-Frequency Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(high_freq_image, cmap='gray')
plt.title('High-Frequency Image')
plt.axis('off')
plt.show()
```

### Step 8: Process Multiple Images

Repeat steps 2-7 for additional images (e.g., 'building.jpg')

## Key Functions Used

### NumPy FFT Functions

| Function             | Purpose                                         |
| -------------------- | ----------------------------------------------- |
| `np.fft.fft2()`      | Compute 2D Fast Fourier Transform               |
| `np.fft.ifft2()`     | Compute inverse 2D FFT                          |
| `np.fft.fftshift()`  | Shift zero-frequency to center                  |
| `np.fft.ifftshift()` | Reverse the fftshift operation                  |
| `np.abs()`           | Compute magnitude of complex number             |
| `np.angle()`         | Compute phase of complex number                 |
| `np.log()`           | Natural logarithm (for magnitude visualization) |

### OpenCV Functions

| Function       | Purpose                         |
| -------------- | ------------------------------- |
| `cv2.imread()` | Read image from file            |
| `cv2.circle()` | Draw circle (for mask creation) |

### Matplotlib Functions

| Function        | Purpose             |
| --------------- | ------------------- |
| `plt.imshow()`  | Display image       |
| `plt.subplot()` | Create subplot grid |
| `plt.figure()`  | Create new figure   |

## Execution Flow

```
Load Image
    ↓
Compute 2D FFT
    ↓
Shift FFT (center DC component)
    ↓
Compute & Display Magnitude and Phase Spectra
    ↓
Create Low-Pass Mask (circular)
    ↓
Create High-Pass Mask (inverted)
    ↓
Apply Masks to FFT
    ↓
Compute Inverse FFT
    ↓
Display Filtered Results
    ↓
Repeat for Additional Images
```

## Filter Characteristics

### Low-Pass Filter

- **Input:** All frequencies
- **Operation:** Keeps DC and low frequencies
- **Output:** Smooth, blurred image
- **Effect:** Noise reduction, smoothing
- **Application:** Preprocessing, noise removal

### High-Pass Filter

- **Input:** All frequencies
- **Operation:** Removes DC and low frequencies
- **Output:** Edge-like, detail-enhanced image
- **Effect:** Edge detection, detail enhancement
- **Application:** Edge detection, sharpening

## Mask Design Parameters

### Radius Effect (Circular Mask)

| Radius         | Cutoff Behavior                | Low-Pass Result             | High-Pass Result               |
| -------------- | ------------------------------ | --------------------------- | ------------------------------ |
| Small (5-10)   | Sharp cutoff at small radius   | More blurred, fewer details | More edges, cleaner background |
| Medium (20-40) | Moderate cutoff                | Balanced smoothing          | Balanced edge content          |
| Large (50+)    | Gradual cutoff at large radius | Less blurred, more details  | Fewer edges, more noise        |

## Input Requirements

- **Format:** JPG, PNG, or other standard image formats
- **Color Mode:** Can be color or grayscale (converted internally)
- **File Path:** Update path in `cv2.imread()` call
- **Size:** Smaller images process faster; typical: 256×256 to 512×512

## Expected Output

1. Original image
2. Magnitude spectrum (log-scaled)
3. Phase spectrum
4. Low-frequency filtered image
5. High-frequency filtered image
6. Repeated for each additional image

## Frequency Domain Concepts

### Nyquist Frequency

Maximum frequency that can be represented in digital image:

```
f_nyquist = 1 / (2 × pixel_size)
```

### Zero-Frequency Component (DC)

- Represents average intensity of image
- Located at center after fftshift
- Largest value in magnitude spectrum
- Removed in high-pass filtering

### Frequency Bands

- **Ultra-low (0-2):** Image gradient and overall structure
- **Low (3-10):** Smooth regions and general shapes
- **Mid (11-30):** Normal details and textures
- **High (31+):** Sharp edges and fine details
- **Ultra-high (near Nyquist):** Noise and artifacts

## Advanced Topics

### 1. Alternative Mask Shapes

**Gaussian Mask:**

```python
# Smooth transition instead of sharp cutoff
xx, yy = np.meshgrid(np.arange(-ccol, cols-ccol),
                     np.arange(-crow, rows-crow))
gaussian_mask = np.exp(-(xx**2 + yy**2) / (2 * (r**2)))
```

**Square Mask:**

```python
# Square region instead of circle
cv2.rectangle(mask, (ccol-r, crow-r), (ccol+r, crow+r), 1, -1)
```

### 2. Butterworth Filters

Smooth transition filters:

```python
# Butterworth low-pass filter
n = 2  # Order
d = np.sqrt(xx**2 + yy**2)
butterworth = 1 / (1 + (d / r) ** (2 * n))
```

### 3. Notch Filters

Remove specific frequencies (useful for removing periodic noise):

```python
# Remove frequency at (u0, v0)
notch = 1 - cv2.circle(...)
```

## Practical Applications

- **Image Denoising:** Low-pass filtering removes high-frequency noise
- **Edge Detection:** High-pass filtering enhances edges
- **Pattern Recognition:** Analyze frequency characteristics
- **Image Restoration:** Selective frequency recovery
- **Compression:** Discard high-frequency components
- **Medical Imaging:** Enhance specific frequency bands
- **Astronomy:** Enhance celestial object detection

## Performance Considerations

### Computational Complexity

- FFT: O(N² log N) vs Spatial convolution: O(N² × K²)
- FFT faster for large kernels (K > 10)
- FFT requires transfer to/from frequency domain

### Memory Usage

- Complex numbers: 2× real representation
- Consider for large images

### Optimization Tips

1. Ensure image dimensions are powers of 2 for optimal FFT speed
2. Pad image with zeros if needed
3. Reuse masks for multiple images
4. Use grayscale instead of color when possible

## Error Handling

| Error                | Cause                  | Solution                           |
| -------------------- | ---------------------- | ---------------------------------- |
| FileNotFoundError    | Image path incorrect   | Verify path, use forward slashes   |
| ValueError           | Image dimensions issue | Ensure image loads correctly       |
| Complex result       | Forgetting np.abs()    | Always use np.abs() before display |
| Visualization issues | Incorrect cmap         | Use cmap='gray' for grayscale      |
| Memory error         | Image too large        | Resize or reduce image dimensions  |

## Extension Activities

1. **Experiment with Different Radii:**
   - Test r = 10, 20, 30, 40, 50
   - Observe transition between low and high pass effects

2. **Create Custom Masks:**
   - Implement Gaussian masks
   - Create Butterworth filters
   - Design notch filters

3. **Frequency Analysis:**
   - Identify dominant frequencies
   - Analyze different image types
   - Compare frequency content

4. **Hybrid Images:**
   - Combine low-frequency of image A with high-frequency of image B
   - Create images that look different at different scales

5. **Filter Combinations:**
   - Band-pass filtering (keep middle frequencies)
   - Cascade multiple filters
   - Compare results with spatial domain equivalents

## Comparison with Spatial Domain Filtering

| Aspect             | Spatial Domain          | Frequency Domain           |
| ------------------ | ----------------------- | -------------------------- |
| **Method**         | Convolution with kernel | Multiplication with mask   |
| **Computation**    | O(N² × K²)              | O(N² log N)                |
| **Kernel Size**    | Limited (typically 3-9) | Unlimited                  |
| **Visualization**  | Not straightforward     | Clear magnitude/phase      |
| **Intuition**      | Easy to understand      | Requires Fourier knowledge |
| **Implementation** | Simple                  | More complex               |
| **Best For**       | Small kernels           | Large kernels              |

## References

- NumPy FFT Documentation: https://numpy.org/doc/stable/reference/fft.html
- OpenCV Documentation: https://docs.opencv.org/
- Digital Image Processing by Gonzalez & Woods
- Fourier Transform: https://en.wikipedia.org/wiki/Fourier_transform
- FFT Tutorial: https://en.wikipedia.org/wiki/Fast_Fourier_transform

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 6
- **Academic Year:** 2023

---

**Note:** Understanding frequency domain processing requires knowledge of Fourier mathematics. Practice with different radii and mask shapes to build intuition about frequency content effects.
