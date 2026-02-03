# LAB-TASK-12-2023-SE-39: Image Segmentation Techniques

## Overview

This lab task focuses on implementing and comparing various image segmentation techniques. Students will learn how to partition images into meaningful regions using global thresholding, local/adaptive thresholding, K-Means clustering, and Mean Shift algorithms. These techniques are fundamental for object detection, image analysis, and computer vision applications.

## Learning Objectives

- Understand image segmentation principles and applications
- Implement global thresholding using Otsu's method
- Apply local and adaptive thresholding for varying illumination
- Use K-Means clustering with different cluster numbers
- Implement Mean Shift segmentation
- Compare segmentation techniques and evaluate results
- Choose appropriate segmentation method for different scenarios

## Concepts Covered

### 1. Image Segmentation Fundamentals

**Definition:**
Image segmentation is the process of partitioning an image into multiple regions or objects based on certain criteria.

**Purpose:**

- Separate foreground from background
- Identify objects of interest
- Group similar pixels
- Prepare for higher-level processing

**Types:**

- **Threshold-based:** Binary classification (foreground/background)
- **Clustering-based:** Multiple groups based on similarity
- **Region-based:** Connected components with similar properties
- **Edge-based:** Boundaries between regions

**Challenges:**

- Varying illumination
- Similar colors/intensities
- Noise
- Object overlap
- Computational efficiency

**Applications:**

- Medical image analysis
- Object detection
- Face recognition
- Document scanning
- Autonomous driving
- Satellite imagery

### 2. Global Thresholding

**Definition:**
Convert grayscale image to binary using a single global threshold value.

**Basic Process:**

```
Output[x,y] = 1 if Input[x,y] > T
Output[x,y] = 0 if Input[x,y] ≤ T
```

**Simple Thresholding:**

```python
T = 127  # Fixed threshold (middle of 0-255 range)
BW = I > T
```

**Advantages:**

- Fast and simple
- Easy to understand
- Minimal computation
- Works well for high-contrast images

**Disadvantages:**

- Manual threshold selection
- Fails with varying lighting
- No spatial information
- Assumes bimodal histogram

**When to Use:**

- High-contrast images
- Known foreground/background characteristics
- Real-time processing required

### 3. Otsu's Method (Automatic Thresholding)

**Definition:**
Automatic method for determining optimal global threshold by maximizing between-class variance.

**Key Idea:**
Find threshold that separates histogram into two classes with maximum variance between them.

**Mathematical Basis:**

```
Maximize: σ²_B = w₀w₁(μ₀ - μ₁)²

Where:
w₀, w₁ = Class weights
μ₀, μ₁ = Class means
```

**Implementation:**

```python
from skimage.filters import threshold_otsu

T = threshold_otsu(I)
BW_global = I > T
```

**How It Works:**

1. Calculate histogram of image
2. For each possible threshold:
   - Calculate foreground mean
   - Calculate background mean
   - Calculate between-class variance
3. Select threshold with maximum variance

**Advantages:**

- Automatic (no manual selection)
- Theoretically optimal for bimodal distributions
- Works well for most images
- No parameters to tune

**Disadvantages:**

- Assumes bimodal histogram (two classes)
- Fails with skewed histograms
- Can't handle varying illumination
- Doesn't use spatial information

**When to Use:**

- Unknown threshold value
- Roughly equal-sized classes
- Decent contrast between foreground/background

### 4. Local Thresholding

**Definition:**
Calculate threshold for each pixel based on local neighborhood statistics.

**Process:**

```
For each pixel [x,y]:
  T[x,y] = mean(neighborhood) + offset
  Output[x,y] = 1 if Input[x,y] > T[x,y]
```

**Implementation:**

```python
from skimage.filters import threshold_local

block_size = 51
local_threshold = threshold_local(I, block_size, offset=0.4)
BW_local = I > local_threshold
```

**Parameters:**

- **block_size:** Size of neighborhood (must be odd)
  - Small (11-21): Detailed features
  - Medium (31-51): Balanced
  - Large (71+): Smooth, coarse

- **offset:** Constant subtracted from mean
  - 0: Use mean value
  - Positive: More lenient (more foreground)
  - Negative: More strict (less foreground)

**Advantages:**

- Handles varying illumination
- Adapts to local conditions
- Better for uneven lighting
- Preserves local details

**Disadvantages:**

- Slower than global
- Parameter tuning needed
- Can create artifacts
- Sensitivity to noise

**When to Use:**

- Varying illumination across image
- Uneven lighting conditions
- Documents with shading
- Images with gradients

### 5. Adaptive Thresholding

**Definition:**
Similar to local thresholding but optimized for specific characteristics.

**Alternative Implementations:**

```python
# Using Gaussian-weighted neighborhood
block_size = 51
adaptive_threshold = threshold_local(I, block_size, offset=0, method='gaussian')
BW_adaptive = I > adaptive_threshold

# Using Niblack method
from skimage.filters import threshold_niblack
adaptive_niblack = threshold_niblack(I, window_size=25, k=0.8)
BW_niblack = I > adaptive_niblack
```

**Key Difference from Local:**

- Local uses mean
- Adaptive may use:
  - Gaussian-weighted mean
  - Standard deviation
  - Other statistics

**Advantages:**

- More sophisticated than local
- Better results for specific images
- Handles complex lighting
- More robust

**Disadvantages:**

- More complex
- Slower computation
- More parameters
- Harder to tune

**When to Use:**

- Complex lighting conditions
- Document scanning with shadows
- Medical images
- When local thresholding insufficient

### 6. K-Means Clustering Segmentation

**Definition:**
Partition pixels into k clusters based on intensity similarity.

**Algorithm:**

1. Initialize k random cluster centers
2. Assign each pixel to nearest center
3. Recalculate centers as cluster means
4. Repeat until convergence

**Implementation:**

```python
from sklearn.cluster import KMeans

img = img_as_float(I)  # Normalize to [0, 1]
pixel = img.flatten()  # Flatten to 1D array

k = 2  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
kmeans.fit(pixel.reshape(-1, 1))
idx = kmeans.labels_

seg = idx.reshape(img.shape)
```

**Parameters:**

- **n_clusters (k):** Number of segments
  - k=2: Foreground/background
  - k=3-4: Basic objects
  - k=5+: Detailed segmentation

**Visual Quality vs k:**
| k | Result | Use Case |
|---|--------|----------|
| 2 | Simple foreground/background | High contrast, simple objects |
| 3 | Three main intensity levels | Multiple objects, basic |
| 4 | Four segments | More detail |
| 5+ | Fine segmentation | Detailed analysis |

**Advantages:**

- Simple and intuitive
- Produces k segments
- Fast convergence
- Works with color images

**Disadvantages:**

- Must specify k in advance
- Sensitive to initialization
- Assumes roughly equal cluster sizes
- May converge to local minima
- Doesn't use spatial information

**When to Use:**

- Known number of objects
- Roughly equal-sized objects
- Color-based clustering
- Quick segmentation needed

### 7. Mean Shift Clustering Segmentation

**Definition:**
Data-driven clustering that finds modes (peaks) of pixel density distribution.

**Key Concept:**
Move each point toward region of highest pixel density.

**Algorithm:**

1. For each pixel:
   - Calculate mean of neighboring pixels (within bandwidth)
   - Move pixel toward that mean
2. Repeat until convergence
3. Group converged pixels into clusters

**Implementation:**

```python
from sklearn.cluster import MeanShift, estimate_bandwidth

# Reshape image to pixel list (3D for color)
data = img_color.reshape((-1, 3)) / 255.0

# Estimate bandwidth automatically
bandwidth = estimate_bandwidth(data, quantile=0.2,
                              n_samples=500, random_state=0)

# Perform Mean Shift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_

seg_ms = labels.reshape(img_color.shape[:2])
```

**Key Parameters:**

- **bandwidth:** Search radius
  - Small (0.1-0.3): Fine segmentation, many clusters
  - Medium (0.3-0.5): Balanced
  - Large (0.5+): Coarse, fewer clusters

- **quantile:** For automatic bandwidth estimation
  - 0.1-0.2: Finer segmentation
  - 0.2-0.3: Balanced
  - 0.3+: Coarser

**Advantages:**

- Automatic cluster detection (no need to specify k)
- Finds natural clusters
- Non-parametric
- Works with any number of clusters
- Good for arbitrary cluster shapes

**Disadvantages:**

- Slower than K-Means
- Bandwidth selection critical
- More memory required
- Can produce too many or too few clusters
- Sensitive to data density variations

**When to Use:**

- Unknown number of clusters
- Want automatic segmentation
- Irregular cluster shapes
- Real-world complex images

## Task Implementation

### Task 1: Thresholding Techniques

#### (a) Global Thresholding (Otsu)

```python
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.io import imread

# Load and prepare image
I_color = imread('image22.jpeg')
I = rgb2gray(I_color)

# Apply Otsu's method
T = threshold_otsu(I)
BW_global = I > T

# Display
plt.imshow(BW_global, cmap='gray')
plt.title('Global Thresholding (Otsu)')
plt.show()
```

#### (b) Local Thresholding

```python
from skimage.filters import threshold_local

block_size = 51
local_threshold = threshold_local(I, block_size, offset=0.4)
BW_local = I > local_threshold

plt.imshow(BW_local, cmap='gray')
plt.title('Local Thresholding')
plt.show()
```

#### (c) Adaptive Thresholding

```python
from skimage.filters import threshold_local

block_size = 51
adaptive_threshold = threshold_local(I, block_size, offset=0)
BW_adaptive = I > adaptive_threshold

plt.imshow(BW_adaptive, cmap='gray')
plt.title('Adaptive Thresholding')
plt.show()
```

### Task 2: K-Means Segmentation

#### K=2

```python
from sklearn.cluster import KMeans
from skimage.util import img_as_float

img = img_as_float(I)
pixel = img.flatten()

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(pixel.reshape(-1, 1))
idx = kmeans.labels_
seg2 = idx.reshape(img.shape)

plt.imshow(seg2, cmap='viridis')
plt.title('K-Means (k=2)')
plt.show()
```

#### K=3 and K=4

Similar implementation with `n_clusters=3` and `n_clusters=4`

### Task 3: Mean Shift Segmentation

```python
from sklearn.cluster import MeanShift, estimate_bandwidth

# Prepare color data
data = img_color.reshape((-1, 3)) / 255.0

# Estimate bandwidth
bandwidth = estimate_bandwidth(data, quantile=0.2,
                              n_samples=500, random_state=0)

# Perform Mean Shift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_
seg_ms = labels.reshape(img_color.shape[:2])

plt.imshow(seg_ms, cmap='viridis')
plt.title('Mean Shift Segmentation')
plt.show()
```

## Key Functions Used

### scikit-image (skimage) Functions

| Function              | Purpose                     |
| --------------------- | --------------------------- |
| `rgb2gray()`          | Convert RGB to grayscale    |
| `threshold_otsu()`    | Automatic optimal threshold |
| `threshold_local()`   | Local adaptive threshold    |
| `threshold_niblack()` | Niblack adaptive method     |
| `imread()`            | Load image                  |
| `img_as_float()`      | Normalize image to [0,1]    |

### scikit-learn (sklearn) Functions

| Function               | Purpose                    |
| ---------------------- | -------------------------- |
| `KMeans()`             | K-Means clustering         |
| `MeanShift()`          | Mean Shift clustering      |
| `estimate_bandwidth()` | Auto bandwidth calculation |

### Parameters Explanation

**KMeans Parameters:**

- `n_clusters`: Number of clusters (k)
- `random_state`: Seed for reproducibility
- `n_init`: Number of initializations (10 is default)
- `max_iter`: Maximum iterations (default 300)

**MeanShift Parameters:**

- `bandwidth`: Search radius
- `bin_seeding`: Use mode seeding (faster)
- `n_jobs`: Parallel jobs (-1 = all cores)

**estimate_bandwidth Parameters:**

- `quantile`: Quantile of distances (0.1-0.3)
- `n_samples`: Random samples for estimation
- `random_state`: Seed for reproducibility

## Segmentation Technique Comparison

| Technique         | Speed     | Parameters         | Clusters | Best For              |
| ----------------- | --------- | ------------------ | -------- | --------------------- |
| **Global (Otsu)** | Very Fast | None               | 2        | High contrast, binary |
| **Local**         | Fast      | block_size, offset | 2        | Varying illumination  |
| **Adaptive**      | Fast      | block_size, method | 2        | Complex lighting      |
| **K-Means**       | Moderate  | k                  | Fixed k  | Known object count    |
| **Mean Shift**    | Slow      | bandwidth          | Auto     | Unknown clusters      |

## Execution Flow

```
Load Image
    ↓
Convert to Grayscale (if needed)
    ↓
TASK 1 - THRESHOLDING:
├── Global Thresholding (Otsu)
├── Local Thresholding
└── Adaptive Thresholding
    ↓
TASK 2 - K-MEANS:
├── K-Means with k=2
├── K-Means with k=3
└── K-Means with k=4
    ↓
TASK 3 - MEAN SHIFT:
└── Mean Shift Segmentation
    ↓
COMPARISON AND ANALYSIS
```

## Input Requirements

- **Format:** JPG, PNG, or standard image formats
- **Color:** Can be color or grayscale
- **Size:** Any size (larger images slower for clustering)
- **Content:** Any scene with multiple regions/objects

## Expected Output

1. Global thresholding result (binary)
2. Local thresholding result (binary)
3. Adaptive thresholding result (binary)
4. K-Means segmentation (k=2)
5. K-Means segmentation (k=3)
6. K-Means segmentation (k=4)
7. Mean Shift segmentation

## Parameter Tuning Guidelines

### Thresholding Parameters

```python
# Local/Adaptive block_size
11 or 21    # Small, detailed
31 or 51    # Medium, balanced
71 or 101   # Large, smooth

# Offset
-0.1 to 0   # Stricter
0           # Mean value
0.1 to 0.5  # More lenient
```

### K-Means Parameters

```python
# For choosing k:
# Start with k=2, gradually increase
# Look for elbow point in distortion
# Use Silhouette score for evaluation

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixel.reshape(-1, 1))
    print(f"k={k}, Inertia={kmeans.inertia_}")
```

### Mean Shift Parameters

```python
# Bandwidth estimation
bandwidth = estimate_bandwidth(data, quantile=q, n_samples=n)

# Test different quantiles:
for q in [0.1, 0.2, 0.3, 0.4]:
    bandwidth = estimate_bandwidth(data, quantile=q)
    ms = MeanShift(bandwidth=bandwidth)
    labels = ms.fit_predict(data)
    print(f"quantile={q}, clusters={len(np.unique(labels))}")
```

## Common Issues and Solutions

| Issue              | Cause                | Solution                         |
| ------------------ | -------------------- | -------------------------------- |
| All black output   | Threshold too high   | Lower threshold or check image   |
| All white output   | Threshold too low    | Increase threshold               |
| Over-segmentation  | k too high           | Reduce k or increase bandwidth   |
| Under-segmentation | k too low            | Increase k or decrease bandwidth |
| Artifacts visible  | Block size too small | Increase block_size              |
| Loss of detail     | Block size too large | Decrease block_size              |
| Memory error       | Image too large      | Resize or use smaller sample     |

## Quality Assessment

### Visual Inspection Checklist

- [ ] Objects clearly separated
- [ ] Noise minimal
- [ ] Boundaries well-defined
- [ ] No over-segmentation
- [ ] Appropriate level of detail
- [ ] No artificial artifacts

### Quantitative Metrics

**Silhouette Score (for clustering):**

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(data, labels)
# Range: -1 to 1
# Higher is better (0.5+ is good)
```

**Davies-Bouldin Index:**

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(data, labels)
# Lower is better (0.5-2.0 is good range)
```

## Advanced Concepts

### Combination Strategies

```python
# Combine threshold with morphology
binary = I > T
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
```

### Multi-scale Segmentation

```python
# Segment at multiple scales
from skimage.transform import pyramid_gaussian

for scale, image in enumerate(pyramid_gaussian(I, max_layer=3)):
    # Segment at each scale
    pass
```

### Watershed Algorithm

```python
from scipy import ndimage
from skimage.morphology import watershed

# Find sure background and foreground
# Use watershed for boundary detection
```

## Practical Applications

### Medical Imaging

- Tumor segmentation
- Organ boundary extraction
- Tissue classification
- Lesion detection

### Document Processing

- Text extraction
- Table detection
- Handwriting segmentation
- Document cleanup

### Object Detection

- Multiple objects in scene
- Background/foreground separation
- Instance segmentation
- Contour extraction

### Satellite/Aerial

- Land use classification
- Water body detection
- Building extraction
- Change detection

## Performance Considerations

### Computation Time (512×512 color image)

- Global Otsu: ~5-10ms
- Local threshold: ~50-100ms
- Adaptive threshold: ~50-150ms
- K-Means (k=2): ~100-300ms
- Mean Shift: ~500-2000ms

### Memory Usage

- Thresholding: ~1MB
- K-Means: ~10-50MB (depends on k)
- Mean Shift: ~50-200MB

## Extension Activities

1. **Threshold Optimization:**
   - Try multiple threshold values
   - Plot results vs threshold
   - Find optimal value for dataset

2. **K-Means Analysis:**
   - Calculate inertia for k=2 to k=10
   - Create elbow curve
   - Determine optimal k

3. **Mean Shift Tuning:**
   - Test different bandwidth values
   - Analyze cluster count changes
   - Find best bandwidth

4. **Combination Methods:**
   - Apply preprocessing (blur, denoising)
   - Combine threshold + morphology
   - Compare results

5. **Evaluation Metrics:**
   - Calculate Silhouette scores
   - Compute Davies-Bouldin indices
   - Compare techniques quantitatively

6. **Real-time Application:**
   - Process video stream
   - Track segmented objects
   - Measure performance

## Segmentation Quality Evaluation

### For Thresholding

- Binary contrast (foreground vs background)
- Edge sharpness
- Noise presence
- Small object preservation

### For Clustering

- Homogeneity (pixels in cluster similar)
- Separation (different clusters distinct)
- Cluster count appropriateness
- Computational efficiency

## References

- scikit-image Documentation: https://scikit-image.org/
- scikit-learn Clustering: https://scikit-learn.org/stable/modules/clustering.html
- Image Segmentation by Gonzalez & Woods
- Otsu Thresholding: https://en.wikipedia.org/wiki/Otsu%27s_method
- K-Means Clustering: https://en.wikipedia.org/wiki/K-means_clustering
- Mean Shift: https://en.wikipedia.org/wiki/Mean_shift

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 12
- **Academic Year:** 2023

---

**Note:** Image segmentation is fundamental to computer vision. Different techniques suit different scenarios - use global thresholding for high-contrast images, local/adaptive for varying illumination, K-Means when you know object count, and Mean Shift for unknown, complex scenes. Combining multiple techniques often yields best results.
