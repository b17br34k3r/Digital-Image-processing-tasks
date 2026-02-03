# LAB-TASK-7-2023-SE-39: Frequency Domain Filtering - Ideal, Butterworth, and Gaussian Filters

## Overview

This lab task focuses on implementing and comparing three types of frequency domain filters: Ideal, Butterworth, and Gaussian filters. Students will learn how to design different low-pass and high-pass filters, understand their frequency responses, and compare their effects on images. This task builds on frequency domain concepts from LAB-TASK-6.

## Learning Objectives

- Understand the characteristics of Ideal, Butterworth, and Gaussian filters
- Implement low-pass and high-pass versions of each filter type
- Compare filter performance and visual artifacts
- Analyze frequency response characteristics
- Compare frequency domain filtering with spatial domain filtering
- Master filter design in the frequency domain

## Concepts Covered

### 1. Filter Classification

Filters can be classified by their frequency response characteristics.

**Classification by Transition:**

- **Ideal Filter:** Sharp, abrupt cutoff
- **Butterworth Filter:** Smooth transition
- **Gaussian Filter:** Smooth Gaussian transition

**Classification by Function:**

- **Low-Pass Filter (LPF):** Keeps low frequencies
- **High-Pass Filter (HPF):** Keeps high frequencies

### 2. Ideal Low-Pass Filter (ILPF)

Sharp cutoff at frequency boundary - mathematically ideal but has practical limitations.

**Mathematical Definition:**

```
H(u,v) = 1  if D(u,v) ≤ D0
H(u,v) = 0  if D(u,v) > D0
```

Where:

- D(u,v) = Distance from center: √((u - u₀)² + (v - v₀)²)
- D₀ = Cutoff frequency (radius)

**Implementation:**

```python
def create_lpf_ideal(shape, cutoff):
    """Create an Ideal Low Pass Filter mask."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 1, thickness=-1)
    return mask
```

**Characteristics:**

- **Transition:** Abrupt (sharp edge)
- **Ringing:** Significant - produces halo artifacts
- **Cutoff:** Sharp at frequency boundary
- **Computation:** Very fast

**Advantages:**

- Simple to understand and implement
- Computationally efficient
- Mathematically well-defined

**Disadvantages:**

- Ringing artifacts (Gibbs phenomenon)
- Spatial domain representation is infinite (sinc function)
- Practical limitations in real applications

**Ringing Artifact Explanation:**
The sharp cutoff in frequency domain produces sinc-function ripples in spatial domain, causing visible halos around edges.

### 3. Ideal High-Pass Filter (IHPF)

Complementary to low-pass - removes low frequencies.

**Mathematical Definition:**

```
H(u,v) = 0  if D(u,v) ≤ D0
H(u,v) = 1  if D(u,v) > D0
```

**Implementation:**

```python
def create_hpf_ideal(shape, cutoff):
    """Create an Ideal High Pass Filter mask."""
    return 1 - create_lpf_ideal(shape, cutoff)
```

**Effects:**

- Removes smooth regions
- Enhances edges
- Produces high-contrast output
- Still exhibits ringing artifacts

### 4. Butterworth Low-Pass Filter (BLPF)

Smooth transition without sharp cutoff - balances between ideal and practical filtering.

**Mathematical Definition:**

```
H(u,v) = 1 / (1 + (D(u,v)/D₀)^(2n))
```

Where:

- n = Order (determines transition steepness)
- D₀ = Cutoff frequency

**Implementation:**

```python
def create_lpf_butterworth(shape, cutoff, order=2):
    """Create a Butterworth Low Pass Filter mask."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(0, rows).reshape(rows, 1)
    y = np.arange(0, cols)
    d = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
    mask = 1 / (1 + (d / cutoff) ** (2 * order))
    return mask
```

**Order Effects:**

- **order=1:** Gentle transition, less ringing
- **order=2:** Moderate transition (default)
- **order=3+:** Steeper transition, approaches ideal

**Transition Behavior:**

- At D = D₀: H = 0.5 (-3dB point)
- Smooth rolloff above and below cutoff
- No sharp discontinuities

**Characteristics:**

- **Transition:** Smooth, adjustable
- **Ringing:** Minimal
- **Cutoff:** Gradual at -3dB point
- **Computation:** Moderate

**Advantages:**

- No ringing artifacts
- Smooth frequency response
- Adjustable via order parameter
- Good practical performance

**Disadvantages:**

- More complex to implement than ideal
- Slower computation than ideal
- -3dB point may not match desired cutoff exactly

### 5. Butterworth High-Pass Filter (BHPF)

High-pass version of Butterworth filter.

**Mathematical Definition:**

```
H(u,v) = 1 / (1 + (D₀/D(u,v))^(2n))
```

**Implementation:**

```python
def create_hpf_butterworth(shape, cutoff, order=2):
    """Create a Butterworth High Pass Filter mask."""
    return 1 - create_lpf_butterworth(shape, cutoff, order)
```

**Effects:**

- Smooth edge enhancement
- Minimal ringing
- Natural-looking results
- Good contrast preservation

### 6. Gaussian Low-Pass Filter (GLPF)

Smooth Gaussian bell-curve transition - no ringing, natural appearance.

**Mathematical Definition:**

```
H(u,v) = exp(-(D(u,v)² / (2σ²)))
```

Where:

- σ = Standard deviation (controls width)
- Cutoff parameter = σ for Gaussian

**Implementation:**

```python
def create_lpf_gaussian(shape, cutoff):
    """Create a Gaussian Low Pass Filter mask."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(0, cols)
    y = np.arange(0, rows).reshape(rows, 1)
    d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    mask = np.exp(-(d ** 2) / (2 * (cutoff ** 2)))
    return mask
```

**Sigma Effects:**

- **σ small:** Sharp transition, subtle effects
- **σ medium:** Balanced smoothing
- **σ large:** Gentle transition, strong smoothing

**Characteristics:**

- **Transition:** Smooth Gaussian curve
- **Ringing:** None (Gaussian has no sharp edges)
- **Cutoff:** 0.6065 (-3dB) at D = σ
- **Computation:** Fast (same as other approaches)

**Advantages:**

- No ringing artifacts whatsoever
- Natural-looking results
- Smooth frequency response
- Matches natural image statistics

**Disadvantages:**

- Softer transition than Butterworth
- May over-blur if σ too large
- Requires tuning σ parameter

### 7. Gaussian High-Pass Filter (GHPF)

High-pass version of Gaussian filter.

**Mathematical Definition:**

```
H(u,v) = 1 - exp(-(D(u,v)² / (2σ²)))
```

**Implementation:**

```python
def create_hpf_gaussian(shape, cutoff):
    """Create a Gaussian High Pass Filter mask."""
    return 1 - create_lpf_gaussian(shape, cutoff)
```

**Effects:**

- Smooth edge enhancement
- No ringing
- Natural edge appearance
- Better contrast than Gaussian low-pass alone

## Distance Function in Frequency Domain

The distance function D(u,v) determines which frequencies are filtered.

**Euclidean Distance:**

```python
d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
```

Where:

- (x, y) = Pixel coordinates in frequency domain
- (ccol, crow) = Center (after FFT shift)
- Creates circular filters

**Interpretation:**

- d = 0: Center (DC component, overall intensity)
- d = small: Low frequencies (smooth regions)
- d = large: High frequencies (edges, details)

## Task Implementation

### Step 1: Import Libraries

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

### Step 2: Define Helper Functions

**Load Image:**

```python
def load_image(image_path):
    """Load the image and convert it to grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image
```

**Compute FFT:**

```python
def compute_fft(image):
    """Compute the 2D Fourier Transform of the image."""
    return np.fft.fftshift(np.fft.fft2(image))
```

**Apply Filter:**

```python
def apply_filter(image, filter_mask):
    """Apply the filter in frequency domain and perform Inverse FFT."""
    fft_image = compute_fft(image)
    filtered_fft = fft_image * filter_mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)
```

### Step 3: Create Filter Masks

All three types for both low-pass and high-pass with helper functions provided in code.

### Step 4: Apply and Display Filters

```python
image_path = '/content/cat.jpg'
image = load_image(image_path)

cutoff_value = 60

# Apply filters
lpf_ideal_mask = create_lpf_ideal(image.shape, cutoff_value)
lpf_ideal_result = apply_filter(image, lpf_ideal_mask)

lpf_butterworth_mask = create_lpf_butterworth(image.shape, cutoff_value)
lpf_butterworth_result = apply_filter(image, lpf_butterworth_mask)

lpf_gaussian_mask = create_lpf_gaussian(image.shape, cutoff_value)
lpf_gaussian_result = apply_filter(image, lpf_gaussian_mask)

# Display results
plt.imshow(lpf_ideal_result, cmap='gray')
plt.title('Ideal Low Pass Filter')
plt.show()
```

## Filter Comparison Summary

### Low-Pass Filters

| Characteristic        | Ideal         | Butterworth   | Gaussian          |
| --------------------- | ------------- | ------------- | ----------------- |
| **Transition**        | Abrupt        | Smooth        | Very Smooth       |
| **Ringing**           | Severe        | None          | None              |
| **Formula**           | Sharp cutoff  | Smooth curve  | Exponential decay |
| **Order Parameter**   | N/A           | Adjustable    | N/A               |
| **Cutoff Definition** | At D₀         | At -3dB (0.5) | At σ (0.6065)     |
| **Spatial Domain**    | Sinc function | Complex       | Gaussian          |
| **Speed**             | Fastest       | Medium        | Fast              |
| **Practical Use**     | Poor          | Good          | Excellent         |
| **Visual Quality**    | Poor (halos)  | Good          | Excellent         |

### High-Pass Filters

| Characteristic        | Ideal        | Butterworth | Gaussian    |
| --------------------- | ------------ | ----------- | ----------- |
| **Transition**        | Abrupt       | Smooth      | Very Smooth |
| **Ringing**           | Severe       | None        | None        |
| **Edge Enhancement**  | Sharp        | Smooth      | Smooth      |
| **Noise Enhancement** | High         | Medium      | Low         |
| **Practical Use**     | Poor         | Good        | Excellent   |
| **Visual Quality**    | Poor (halos) | Good        | Excellent   |

## Key Concepts

### -3dB Point

Standard frequency where filter response drops to 70.7% (1/√2) of maximum.

**Ideal Filter:** Infinite rolloff at cutoff
**Butterworth:** -3dB point at D = D₀
**Gaussian:** -3dB point at D = σ√(2ln2) ≈ 1.177σ

### Order Parameter (Butterworth)

Controls steepness of frequency response:

- **n=1:** Gentlest, least ringing
- **n=2:** Balanced (commonly used)
- **n=3+:** Steeper, approaches ideal

### Cutoff Frequency

- **Ideal:** Sharp boundary at D₀
- **Butterworth:** At -3dB point (0.5 magnitude)
- **Gaussian:** At σ (0.6065 magnitude)

## Execution Flow

```
Load Image (grayscale)
    ↓
Set Cutoff Frequency/Radius
    ↓
Create Filter Masks:
    ├── Ideal LPF/HPF
    ├── Butterworth LPF/HPF
    └── Gaussian LPF/HPF
    ↓
Apply Each Filter:
    ├── Compute FFT
    ├── Multiply by mask
    └── Inverse FFT
    ↓
Display Results for Comparison
    ↓
Compare with Spatial Domain Filter
```

## Spatial Domain Comparison

**Gaussian Blur (Spatial):**

```python
spatial_filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
```

**Comparison:**

- Frequency domain: Exact frequency control
- Spatial domain: Kernel size determines frequency response
- Frequency faster for large kernels
- Results similar but not identical

## Input Requirements

- **Format:** JPG, PNG, or standard image formats
- **Color:** Converted to grayscale internally
- **File Path:** Update in `load_image()` call
- **Typical Size:** 256×256 to 512×512

## Expected Output

For each image:

1. Original grayscale image
2. Ideal LPF (notice ringing)
3. Butterworth LPF (smooth, no ringing)
4. Gaussian LPF (smoothest)
5. Ideal HPF (notice ringing)
6. Butterworth HPF (smooth edge enhancement)
7. Gaussian HPF (smoothest edge enhancement)
8. Spatial Gaussian Blur (for comparison)

## Parameter Tuning

### Cutoff Frequency Effects

**Lower cutoff (e.g., 30):**

- LPF: More blur, fewer details
- HPF: Fewer edges, cleaner background

**Higher cutoff (e.g., 100):**

- LPF: Less blur, more details preserved
- HPF: More edges, more noise

### Butterworth Order Effects

**Lower order (n=1):**

- Gentler transition
- Less ringing
- Smoother frequency response

**Higher order (n=3+):**

- Steeper transition
- Approaches ideal (more ringing)
- Better frequency separation

## Artifacts and Solutions

### Ringing Artifact (Ideal Filters)

- **Cause:** Sharp frequency cutoff
- **Appearance:** Halo around edges
- **Solution:** Use Butterworth or Gaussian
- **Why others don't have it:** Smooth transitions in frequency domain

### Over-Smoothing

- **Cause:** Cutoff too low
- **Solution:** Increase cutoff frequency
- **Alternative:** Reduce Gaussian σ

### Insufficient Smoothing

- **Cause:** Cutoff too high
- **Solution:** Decrease cutoff frequency
- **Alternative:** Increase Gaussian σ or Butterworth order

## Advanced Concepts

### Frequency Response Visualization

Plot magnitude response:

```python
# For Gaussian
d = np.linspace(0, 200, 1000)
response = np.exp(-(d**2) / (2 * (60**2)))
plt.plot(d, response)
plt.xlabel('Frequency D(u,v)')
plt.ylabel('H(u,v)')
plt.title('Gaussian LPF Frequency Response')
plt.show()
```

### Composite Filters

Combine multiple filters:

```python
# Band-pass: Keep only middle frequencies
bandpass_mask = create_lpf_gaussian(shape, 100) - create_lpf_gaussian(shape, 30)
```

### 2D vs 1D Frequency Response

Filters extend in 2D circles - all directions affected equally.

## Practical Applications

### Ideal Filter

- **Use:** Educational purposes
- **Avoid:** Production work
- **When:** When sharp cutoff needed despite artifacts

### Butterworth Filter

- **Use:** Good balance of smoothness and transition
- **Applications:** General-purpose filtering
- **Best:** When artifacts matter but transition speed important

### Gaussian Filter

- **Use:** Best visual results
- **Applications:** Professional image processing
- **Best:** When quality most important
- **Matches:** Natural image statistics

## Performance Considerations

### Computation Time (for 512×512 image)

- All filters similar: ~50-200ms total
- FFT computation: ~30-100ms
- Mask creation: ~1-10ms
- Inverse FFT: ~30-100ms

### Memory Usage

- Complex FFT array: 2× real image size
- Masks: Same size as image
- Total: ~3-4× image size in memory

## Troubleshooting

| Issue               | Cause               | Solution                       |
| ------------------- | ------------------- | ------------------------------ |
| Output all dark     | np.abs not applied  | Use np.abs() before display    |
| Incorrect magnitude | FFT not shifted     | Use fftshift before display    |
| Filter not applied  | Mask not multiplied | Check: `filtered = fft * mask` |
| Memory error        | Image too large     | Resize or reduce dimensions    |
| Ringing visible     | Using ideal filter  | Switch to Butterworth/Gaussian |

## Extension Activities

1. **Compare Frequency Responses:**
   - Plot magnitude response for all three filters
   - Visualize -3dB points
   - Compare transition characteristics

2. **Optimize Parameters:**
   - Find best cutoff for different images
   - Test different Butterworth orders
   - Adjust Gaussian σ values

3. **Create Composite Filters:**
   - Design band-pass filters
   - Implement notch filters
   - Create custom filter combinations

4. **Quantitative Analysis:**
   - Measure ringing artifact severity
   - Calculate smoothing effectiveness
   - Compare with spatial domain results

5. **Filter Design:**
   - Design filters for specific applications
   - Optimize for particular image characteristics
   - Create hybrid filters

## Filter Selection Guide

**Use Ideal Filter when:**

- Teaching frequency domain concepts
- Need maximum frequency separation
- Accepting visual artifacts

**Use Butterworth Filter when:**

- Need good balance
- Want smooth transition
- Don't need excessive smoothness

**Use Gaussian Filter when:**

- Want best visual quality
- Can't afford ringing artifacts
- Processing critical images

## References

- Digital Image Processing by Gonzalez & Woods
- Filter Design Theory: https://en.wikipedia.org/wiki/Filter_design
- Butterworth Filter: https://en.wikipedia.org/wiki/Butterworth_filter
- Frequency Domain Processing: https://en.wikipedia.org/wiki/Frequency_domain
- NumPy FFT: https://numpy.org/doc/stable/reference/fft.html
- OpenCV Documentation: https://docs.opencv.org/

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 7
- **Academic Year:** 2023

---

**Note:** Understanding the differences between Ideal, Butterworth, and Gaussian filters is crucial for effective frequency domain image processing. Practice with different cutoff values and parameters to develop intuition about filter behavior.
