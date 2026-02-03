# LAB-TASK-9-2023-SE-39: Color Space Analysis and Color Image Processing

## Overview

This lab task focuses on understanding and working with different color spaces in digital image processing. Students will learn how to extract individual color channels, convert between color spaces (RGB, HSV, YCbCr, Lab), perform white balance correction, apply color masking, and compare color space effectiveness for image segmentation.

## Learning Objectives

- Understand color space fundamentals and representations
- Extract individual color channels from images
- Convert images between different color spaces
- Apply white balance correction using the Gray World algorithm
- Implement color masking for selective color detection
- Compare segmentation effectiveness across color spaces
- Choose appropriate color space for specific tasks

## Concepts Covered

### 1. Color Space Fundamentals

A color space is a mathematical model for representing colors using tuples of numbers (typically 3 values for RGB-based systems).

**Key Concepts:**

- **Color Model:** Mathematical representation (e.g., RGB, HSV, YCbCr)
- **Color Space:** Specific instance of a color model with defined mapping
- **Channels:** Individual components of a color representation
- **Color Gamut:** Range of colors representable in a space

**Why Multiple Color Spaces?**

- Different spaces suited for different applications
- Some spaces match human perception better
- Some spaces separate color information from intensity
- Some spaces simplify specific processing tasks

### 2. RGB Color Space

Most common and intuitive color representation.

**Structure:**

- Three channels: Red (R), Green (G), Blue (B)
- Each channel: 0-255 (8-bit) or 0.0-1.0 (normalized)
- Additive color model

**Channel Extraction:**

```python
R = img_rgb[:,:,0]  # Red channel
G = img_rgb[:,:,1]  # Green channel
B = img_rgb[:,:,2]  # Blue channel
```

**Characteristics:**

- **Intuitive:** Direct mapping to light sensors
- **Standard:** Used everywhere (displays, cameras)
- **Additive:** Colors combine by addition
  - Black: (0, 0, 0)
  - White: (255, 255, 255)
  - Red: (255, 0, 0)

**Advantages:**

- Familiar and intuitive
- Direct hardware representation
- Suitable for display

**Disadvantages:**

- Channels highly correlated
- Intensity mixed with color
- Not ideal for color processing

**Applications:**

- Display and rendering
- Cameras and sensors
- Standard image storage

### 3. HSV Color Space

Represents colors as perceived by humans.

**Structure:**

- **Hue (H):** Color type (0-360° or 0-180 in OpenCV)
- **Saturation (S):** Color intensity (0-255)
- **Value (V):** Brightness/luminance (0-255)

**Conversion from RGB:**

```python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
H = hsv[:,:,0]  # Hue channel
S = hsv[:,:,1]  # Saturation channel
V = hsv[:,:,2]  # Value channel
```

**Hue Values (OpenCV):**

- Red: 0, 180 (circular)
- Orange: 15
- Yellow: 30
- Green: 60
- Cyan: 90
- Blue: 120
- Magenta: 150

**Saturation:**

- 0: Gray (no color)
- 255: Pure color

**Value:**

- 0: Black
- 255: Maximum brightness

**Advantages:**

- Matches human color perception
- Separates color from intensity
- Easy color range selection
- Robust to lighting changes

**Disadvantages:**

- Hue undefined for low saturation
- More complex computation
- Non-linear relationship with RGB

**Applications:**

- Color-based segmentation
- Object detection by color
- Color correction
- Skin tone detection
- Color range selection

### 4. YCbCr Color Space

Separates luminance (brightness) from chrominance (color).

**Structure:**

- **Y:** Luminance/Brightness (intensity information)
- **Cb:** Blue chrominance (color difference from gray)
- **Cr:** Red chrominance (color difference from gray)

**Conversion:**

```python
ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
# OpenCV uses YCrCb (Cr and Cb swapped)
Y = ycbcr[:,:,0]   # Luminance
Cr = ycbcr[:,:,1]  # Red chrominance
Cb = ycbcr[:,:,2]  # Blue chrominance
```

**Characteristics:**

- Y contains most image information (like grayscale)
- Cb and Cr contain color information
- Can compress Cb, Cr more than Y

**Advantages:**

- Separates brightness from color
- Efficient for compression (JPEG uses this)
- Can improve contrast without affecting color
- Better for video processing

**Disadvantages:**

- Less intuitive than HSV
- Chrominance components uncorrelated with perception
- Requires linear RGB conversion

**Applications:**

- Image and video compression
- Histogram equalization (Y channel only)
- Skin detection
- Chroma subsampling
- Video transmission

### 5. Lab Color Space

Perceptually uniform color space based on human vision.

**Structure:**

- **L:** Lightness (0-100)
- **a:** Green-Red axis (-127 to 127)
- **b:** Blue-Yellow axis (-127 to 127)

**Conversion:**

```python
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
L = lab[:,:,0]  # Lightness
a = lab[:,:,1]  # Green-Red
b = lab[:,:,2]  # Blue-Yellow
```

**Perceptual Uniformity:**

- Distance in Lab space approximates perceived color difference
- Equal distances ≈ Equal perceived differences

**Advantages:**

- Perceptually uniform
- Device-independent
- Good for color difference calculations
- Excellent for color-based processing
- Separates luminance from chrominance

**Disadvantages:**

- Non-linear from RGB
- More complex computation
- Requires calibrated color space
- Less intuitive than HSV

**Applications:**

- Color difference measurement
- Image editing and correction
- Image clustering
- Color matching
- Professional image processing

## Task Implementation

### Task 1: Extract RGB Channels

**Objective:** Separate image into individual color channels

**Code:**

```python
R = img_rgb[:,:,0]  # Red channel
G = img_rgb[:,:,1]  # Green channel
B = img_rgb[:,:,2]  # Blue channel

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(R, cmap='gray')
plt.title("Red Channel")

plt.subplot(1,3,2)
plt.imshow(G, cmap='gray')
plt.title("Green Channel")

plt.subplot(1,3,3)
plt.imshow(B, cmap='gray')
plt.title("Blue Channel")
plt.show()
```

**What's Happening:**

- Slicing image array to extract each channel
- Each channel is 2D grayscale image
- Values show intensity of that color

**Visual Interpretation:**

- Red channel: Bright where red colors appear
- Green channel: Bright where green colors appear
- Blue channel: Bright where blue colors appear

### Task 2: Color Space Conversion

**Objective:** Convert between RGB, HSV, YCbCr, and Lab

**Code:**

```python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

# Display all three
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(hsv)
plt.title("HSV")

plt.subplot(1,3,2)
plt.imshow(ycbcr)
plt.title("YCbCr")

plt.subplot(1,3,3)
plt.imshow(lab)
plt.title("Lab")
plt.show()
```

**Important Notes:**

- OpenCV uses BGR internally (not RGB)
- YCbCr in OpenCV is actually YCrCb (channels swapped)
- Lab values may appear unusual due to scaling

**Conversion Functions:**
| Conversion | OpenCV Function |
|---|---|
| RGB to HSV | `cv2.cvtColor(img, cv2.COLOR_RGB2HSV)` |
| RGB to YCbCr | `cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)` |
| RGB to Lab | `cv2.cvtColor(img, cv2.COLOR_RGB2LAB)` |
| BGR to HSV | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` |
| Any to Any | Various combinations available |

### Task 3: White Balance Correction

**Objective:** Correct color cast using Gray World algorithm

**Gray World Hypothesis:**
"Average color in image should be gray"

**Algorithm:**

1. Calculate average of each channel
2. Calculate average of all averages
3. Scale each channel: channel \* (average_of_all / channel_average)

**Code:**

```python
def white_balance(img):
    result = img.copy().astype(np.float32)

    # Calculate channel averages
    avgB = np.mean(result[:,:,0])
    avgG = np.mean(result[:,:,1])
    avgR = np.mean(result[:,:,2])
    avgGray = (avgB + avgG + avgR) / 3

    # Scale channels
    result[:,:,0] *= avgGray / avgB
    result[:,:,1] *= avgGray / avgG
    result[:,:,2] *= avgGray / avgR

    return np.clip(result, 0, 255).astype(np.uint8)

wb_img = white_balance(img_rgb)
plt.imshow(wb_img)
plt.title("White Balanced Image")
plt.show()
```

**Example:**

- Original averages: R=100, G=150, B=120
- Average of all: (100+150+120)/3 = 123.3
- Scale R: 100 → 100 × (123.3/100) = 123.3
- Scale G: 150 → 150 × (123.3/150) = 123.3
- Scale B: 120 → 120 × (123.3/120) = 123.3
- Result: Balanced (123.3, 123.3, 123.3) → Gray

**Effects:**

- Removes color cast from lighting
- Makes whites appear truly white
- Balances warm/cool tones

**Limitations:**

- Assumes average should be gray
- May fail with predominantly colored images
- Can clip values if ranges too extreme

### Task 4: Color Masking

**Objective:** Extract specific colors using HSV thresholds

**HSV Color Range Selection:**

```python
hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# Define red color range
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# Create mask
mask = cv2.inRange(hsv_img, lower_red, upper_red)

# Apply mask
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.imshow(mask, cmap='gray')
plt.title("Red Mask")

plt.subplot(1,2,2)
plt.imshow(result)
plt.title("Masked Image")
plt.show()
```

**How It Works:**

1. `cv2.inRange()`: Creates binary mask where pixels fall within range
2. `cv2.bitwise_and()`: Applies mask to original image
   - Mask 255 (white): Keep pixel
   - Mask 0 (black): Remove pixel

**Color Ranges for HSV:**

```python
# Red (two ranges due to circular hue)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Green
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Blue
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Yellow
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Orange
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([20, 255, 255])

# Purple
lower_purple = np.array([125, 30, 30])
upper_purple = np.array([155, 255, 255])
```

**Why HSV for Color Masking?**

- Hue is independent of lighting
- Single range selects similar colors
- Intuitive color parameter

**Applications:**

- Object detection by color
- Skin detection
- Traffic light detection
- Specific color extraction

### Task 5: Segmentation in Different Color Spaces

**Objective:** Compare segmentation performance across color spaces

**Code:**

```python
# Thresholding in different color spaces

# RGB (Red channel)
_, rgb_seg = cv2.threshold(R, 120, 255, cv2.THRESH_BINARY)

# HSV (Hue channel)
_, hsv_seg = cv2.threshold(hsv[:,:,0], 50, 255, cv2.THRESH_BINARY)

# YCbCr (Cb channel)
_, ycbcr_seg = cv2.threshold(ycbcr[:,:,2], 120, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.imshow(rgb_seg, cmap='gray')
plt.title("RGB Segmentation")

plt.subplot(1,3,2)
plt.imshow(hsv_seg, cmap='gray')
plt.title("HSV Segmentation")

plt.subplot(1,3,3)
plt.imshow(ycbcr_seg, cmap='gray')
plt.title("YCbCr Segmentation")
plt.show()
```

**Analysis:**
Compare characteristics of segmentations:

- Clarity of object boundaries
- Noise in background
- Completeness of object detection
- Separation from background

**Why Different Results?**

- RGB: Highly correlated channels, lighting-dependent
- HSV: Color information isolated, lighting-robust
- YCbCr: Luminance separated, specific to color difference

## Key Functions Used

### OpenCV Color Space Functions

| Function            | Purpose                       |
| ------------------- | ----------------------------- |
| `cv2.cvtColor()`    | Convert between color spaces  |
| `cv2.inRange()`     | Create mask from value ranges |
| `cv2.bitwise_and()` | Apply mask to image           |
| `cv2.threshold()`   | Binary thresholding           |

### NumPy Array Operations

| Function       | Purpose               |
| -------------- | --------------------- |
| `array[:,:,i]` | Extract channel       |
| `np.mean()`    | Calculate average     |
| `np.clip()`    | Bound values to range |

## Color Space Comparison

### When to Use Each

| Color Space | Best For                   | Advantages                     | Disadvantages                           |
| ----------- | -------------------------- | ------------------------------ | --------------------------------------- |
| **RGB**     | Display, storage           | Intuitive, standard            | Correlated channels, lighting-dependent |
| **HSV**     | Color selection, detection | Intuitive hue, lighting-robust | Hue undefined at low saturation         |
| **YCbCr**   | Compression, processing    | Separates luminance            | Less intuitive, complex conversion      |
| **Lab**     | Perceptual tasks, analysis | Perceptually uniform           | More computation, device-dependent      |

## Segmentation Effectiveness

**Segmentation Quality Factors:**

- Separation of foreground and background
- Minimal noise in background
- Complete object detection
- Clean object boundaries

**Color Space Impact:**

- **RGB:** Often poor due to channel correlation
- **HSV:** Excellent for single-color objects
- **YCbCr:** Good for specific chrominance ranges
- **Lab:** Excellent for perceptually similar colors

## Execution Flow

```
Load Image
    ↓
Convert BGR → RGB
    ↓
Extract RGB Channels
    ↓
Convert to HSV, YCbCr, Lab
    ↓
Display Channel Separations
    ↓
Apply White Balance Correction
    ↓
Color Masking (HSV)
    ↓
Compare Segmentation in Different Spaces
```

## Input Requirements

- **Format:** JPG, PNG, or standard image formats
- **Color:** RGB color image preferred
- **Size:** Any size (typical 256×256 to 1024×1024)

## Expected Output

1. Original color image
2. Individual R, G, B channels
3. HSV, YCbCr, Lab color spaces
4. White-balanced version
5. Color-masked image
6. Segmentation comparisons

## Parameter Guidelines

### White Balance

- Generally no parameters (automatic)
- Alternative: Manual channel correction

### Color Masking (HSV Ranges)

- Hue: 0-180 in OpenCV
- Saturation: 0-255
- Value: 0-255
- Fine-tune ranges for specific colors

### Thresholding

- Threshold value: 0-255
- Adjust based on channel characteristics

## Common Issues and Solutions

| Issue                     | Cause                        | Solution                             |
| ------------------------- | ---------------------------- | ------------------------------------ |
| Red hue wraps around      | Circular hue (0-180)         | Use two ranges: [0-10] and [170-180] |
| Over-saturated colors     | Saturation too high          | Lower saturation threshold           |
| Dim colors missed         | Value threshold too high     | Lower value threshold                |
| Noise in mask             | Threshold too inclusive      | Tighten range or increase threshold  |
| Black image after masking | Mask inverted                | Check mask creation logic            |
| Extreme brightness change | White balance too aggressive | Check channel values, may clip       |

## Advanced Color Concepts

### Hue Circularity

Hue is circular: Red appears at both 0 and 180:

```python
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)
```

### Color Space Conversion Matrices

Mathematical transformations underlying conversions:

- RGB to HSV: Non-linear (involves min/max/hue calculation)
- RGB to YCbCr: Linear transformation matrix
- RGB to Lab: Involves gamma correction and matrix multiplication

### Saturation and Value in HSV

- **Saturation = 0:** All RGB channels equal (gray)
- **Value = 0:** RGB channels all zero (black)
- **Saturation = 255, Value = 255:** Pure color

## Practical Applications

### Object Detection

- Detect red traffic lights: HSV color range
- Detect skin tone: Specific Cr/Cb ranges in YCbCr
- Detect green vegetation: HSV hue range

### Image Enhancement

- Adjust specific color range
- Preserve other colors
- Apply selective correction

### Medical Imaging

- Detect lesions by color
- Enhance specific tissues
- Color-based diagnosis support

### Video Processing

- Extract moving colored objects
- Track specific colors over frames
- Color-based segmentation

## Performance Considerations

### Computation Time (for 512×512 image)

- RGB channel extraction: < 1ms
- Color space conversion: 1-5ms
- White balance: 2-3ms
- Color masking: 1-3ms
- Thresholding: < 1ms

### Memory Usage

- RGB image: ~750KB (512×512×3 bytes)
- All conversions: ~3MB (multiple copies)

## Extension Activities

1. **Multi-Color Detection:**
   - Detect multiple colors in one image
   - Create composite mask
   - Extract multiple objects

2. **Advanced White Balance:**
   - Implement other algorithms (max-white, perfect reflector)
   - Compare results
   - Evaluate on different lighting

3. **Color Space Analysis:**
   - Analyze which space best separates colors
   - Create statistical analysis
   - Visualize distributions

4. **Real-time Color Tracking:**
   - Track colored object in video
   - Adjust ranges dynamically
   - Implement smoothing

5. **Color Palette Analysis:**
   - Extract dominant colors
   - Create color histogram
   - Analyze color distribution

## Color Space Quick Reference

| Space         | Best For   | Key Insight                     |
| ------------- | ---------- | ------------------------------- |
| **RGB**       | Display    | Matches display technology      |
| **HSV**       | Selection  | Intuitive hue-saturation-value  |
| **YCbCr**     | Video      | Separates brightness from color |
| **Lab**       | Science    | Perceptually uniform            |
| **CMYK**      | Print      | Subtractive (ink) model         |
| **Grayscale** | Processing | Single channel, reduces data    |

## References

- OpenCV Color Conversion: https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html
- HSV Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV
- YCbCr Color Model: https://en.wikipedia.org/wiki/YCbCr
- CIELAB Color Space: https://en.wikipedia.org/wiki/CIELAB_color_space
- Digital Image Processing by Gonzalez & Woods
- Color Image Processing: https://en.wikipedia.org/wiki/Color_image_processing

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 9
- **Academic Year:** 2023

---

**Note:** Understanding color spaces is fundamental to color image processing. Different applications require different spaces - RGB for display, HSV for color selection, YCbCr for compression, and Lab for perceptual analysis. Choose the right space for your application!
