# LAB-TASK-10-2023-SE-39: Image Compression - Huffman Coding and JPEG-like DCT Compression

## Overview

This lab task focuses on implementing and comparing two fundamental image compression techniques: Huffman Coding (lossless) and JPEG-like DCT Compression (lossy). Students will learn how to compress images using entropy-based methods, discrete cosine transforms, quantization, and evaluate compression performance using quality metrics.

## Learning Objectives

- Understand lossless vs. lossy compression principles
- Implement Huffman coding for entropy encoding
- Build Huffman trees and generate optimal codes
- Implement JPEG-like compression using DCT
- Apply quantization matrices for compression
- Calculate compression metrics (ratio, PSNR, MSE, rate-distortion)
- Compare compression techniques
- Evaluate trade-offs between compression and quality

## Concepts Covered

### 1. Image Compression Fundamentals

**Why Compress Images?**

- Storage space reduction
- Faster transmission
- Bandwidth conservation
- Archive efficiency

**Compression Types:**

- **Lossless:** No information lost, original recoverable
- **Lossy:** Information lost, original not recoverable

**Lossless Compression Use Cases:**

- Medical images
- Archival
- Transmission where errors unacceptable

**Lossy Compression Use Cases:**

- Photography
- Video
- Streaming
- When minor quality loss acceptable

### 2. Huffman Coding (Lossless)

**Fundamental Concept:**
Assign shorter bit codes to more frequent symbols and longer codes to less frequent symbols.

**Example:**

```
Original: A=100 times, B=50 times, C=30 times
Naive: Each symbol = 2 bits → 180 symbols × 2 = 360 bits
Huffman: A=1 bit, B=2 bits, C=3 bits
        = 100×1 + 50×2 + 30×3 = 100+100+90 = 290 bits
Compression: 360 → 290 bits (19% reduction)
```

**How Huffman Coding Works:**

1. **Frequency Analysis:**
   Count how often each symbol appears

   ```
   Image pixels: {0: 1000, 1: 500, 2: 300, 3: 200}
   ```

2. **Build Tree (Bottom-Up):**
   - Start with leaf nodes (symbols + frequencies)
   - Repeatedly combine two lowest-frequency nodes
   - New node frequency = sum of children
   - Continue until single root node

   **Example Tree:**

   ```
        (2000)
        /     \
      (1500)   (500):3
      /    \
    (1000)  500:2
    /   \
   1000:0 500:1

   Code: 0 → 00, 1 → 01, 2 → 10, 3 → 11
   ```

3. **Generate Codes:**
   - Traverse tree: left = 0, right = 1
   - Path from root to leaf = code for that symbol

4. **Encode:**
   Replace each symbol with its code

**Implementation Details:**

```python
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol      # Pixel value
        self.freq = freq           # Frequency count
        self.left = None           # Left child (0)
        self.right = None          # Right child (1)

    def __lt__(self, other):
        # For heap ordering by frequency
        return self.freq < other.freq
```

**Tree Building:**

```python
def build_huffman_tree(image):
    # Count pixel frequencies
    pixels = image.flatten()
    freq = Counter(pixels)

    # Create heap of leaf nodes
    heap = [HuffmanNode(sym, fr) for sym, fr in freq.items()]
    heapq.heapify(heap)

    # Build tree bottom-up
    while len(heap) > 1:
        n1 = heapq.heappop(heap)  # Smallest
        n2 = heapq.heappop(heap)  # Second smallest

        # Create parent node
        merged = HuffmanNode(freq=n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)

    return heap[0], freq
```

**Code Generation:**

```python
def generate_codes(node, code="", codes={}):
    if node:
        if node.symbol is not None:
            # Leaf node: store code
            codes[node.symbol] = code
        generate_codes(node.left, code + "0", codes)
        generate_codes(node.right, code + "1", codes)
    return codes
```

**Advantages:**

- Optimal lossless compression
- No information lost
- Mathematically proven optimal
- Can be very efficient

**Disadvantages:**

- Must store code table
- Sequential decoding (not random access)
- Different table needed for different data

**Applications:**

- ZIP, gzip, PNG compression
- Fax transmission
- Text compression
- Prefix for other codecs

### 3. DCT (Discrete Cosine Transform)

**Purpose:**
Convert image from spatial domain to frequency domain for compression.

**Mathematical Basis:**
Any signal can be represented as sum of cosine functions at different frequencies.

```
Pixel values = Sum of DC component + Low-frequency cosines + High-frequency cosines
```

**Why DCT for JPEG?**

- Energy concentration: Most image energy in low frequencies
- Separates important from unimportant information
- Human eye less sensitive to high frequencies
- Efficient quantization possible

**2D DCT Formula:**

```
DCT[u][v] = C(u)C(v) × Σ Σ pixel[x][y] × cos((2x+1)πu/16) × cos((2y+1)πv/16)

Where:
C(u) = 1/√2 if u=0, else 1
x,y = pixel coordinates (0-7 for 8×8 block)
u,v = frequency indices (0-7)
```

**In OpenCV:**

```python
dct_block = cv2.dct(np.float32(block))
```

**DCT Characteristics:**

- **Block-based:** Operates on 8×8 blocks
- **Energy concentration:** DC (u=0, v=0) contains average
- **Frequency separation:** Higher u,v = higher frequencies
- **Separable:** 2D DCT = 1D DCT on rows then columns

### 4. Quantization

**Purpose:**
Reduce precision of DCT coefficients to achieve compression.

**JPEG Quantization Matrix:**

```
Q = [
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
]
```

**Process:**

```
Quantized[u,v] = Round(DCT[u,v] / Q[u,v])
Dequantized[u,v] = Quantized[u,v] × Q[u,v]
```

**Effect:**

- Large values divided by large numbers → become 0 or small
- High frequencies (large Q values) heavily reduced
- Low frequencies (small Q values) preserved

**Quality Factor:**

```python
Q = Q * (100 / quality)  # quality: 1-100
```

- quality = 100: No quantization (best quality)
- quality = 50: Medium compression
- quality = 1: Heavy compression

### 5. JPEG-Like Compression Process

**Complete Pipeline:**

1. **Pad Image:**
   Ensure dimensions are multiples of 8

   ```python
   pad_h = (8 - original_h % 8) % 8
   pad_w = (8 - original_w % 8) % 8
   padded = np.pad(image, ((0, pad_h), (0, pad_w)), 'edge')
   ```

2. **Center Shift:**
   Shift pixel values from [0,255] to [-128,127]

   ```python
   padded = padded - 128
   ```

3. **Process 8×8 Blocks:**

   ```python
   for i in range(0, h, 8):
       for j in range(0, w, 8):
           # Extract 8×8 block
           block = padded[i:i+8, j:j+8]

           # Apply DCT
           dct_block = dct2(block)

           # Quantize
           quant = np.round(dct_block / Q)

           # Dequantize (simulating decompression)
           dequant = quant * Q

           # Inverse DCT
           reconstructed = idct2(dequant)
           compressed[i:i+8, j:j+8] = reconstructed
   ```

4. **Inverse Transform:**
   Apply inverse DCT to reconstruct

   ```python
   reconstructed = idct2(dequant)
   ```

5. **Reshift and Clip:**

   ```python
   result = reconstructed + 128
   result = np.clip(result, 0, 255)
   ```

6. **Remove Padding:**
   Return to original size

**Advantages:**

- Significant compression (up to 50:1)
- Quality scalable via quality factor
- Human-eye tuned (perceptual)
- Industry standard (JPEG)

**Disadvantages:**

- Information lost (lossy)
- Blocking artifacts at low quality
- Not suitable for graphics/text
- Requires full block processing

## Compression Metrics

### 1. Mean Squared Error (MSE)

**Measures:** Average squared difference between original and compressed

```
MSE = (1/N) × Σ(original[i] - compressed[i])²
```

**Interpretation:**

- 0: Perfect reconstruction
- Larger values: More distortion
- Typically 1-100 range for visible quality

**Code:**

```python
def mse(original, compressed):
    return np.mean((original - compressed) ** 2)
```

### 2. PSNR (Peak Signal-to-Noise Ratio)

**Measures:** Quality in logarithmic scale (dB)

```
PSNR = 10 × log₁₀(MAX² / MSE)

Where MAX = 255 for 8-bit images
```

**Interpretation:**

- 50+ dB: Excellent (imperceptible difference)
- 40-50 dB: Good (minor artifacts)
- 30-40 dB: Acceptable (visible artifacts)
- 20-30 dB: Poor (significant distortion)
- <20 dB: Very poor (heavy compression)

**Code:**

```python
def psnr(mse_val):
    if mse_val == 0:
        return float('inf')
    return 10 * math.log10((255 ** 2) / mse_val)
```

### 3. Compression Ratio (CR)

**Measures:** How much compression achieved

```
CR = Original Size / Compressed Size

Examples:
CR = 2: Compression by 50%, 1:2 ratio
CR = 10: Compression by 90%, 1:10 ratio
```

**Code:**

```python
def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size
```

### 4. Rate-Distortion (RD)

**Measures:** Bits per pixel (bpp)

```
RD = Compressed Size (bits) / Number of Pixels
```

**Interpretation:**

- Grayscale: 8 bpp (uncompressed)
- Lossy compression: 0.5-2 bpp typically
- Lossless compression: 4-8 bpp typically

**Code:**

```python
RD = compressed_size / image.size
```

## Task Implementation

### Step 1: Load Image

```python
image = cv2.imread("image11.jpeg", cv2.IMREAD_GRAYSCALE)
```

### Step 2: Huffman Compression

```python
# Build tree
tree, freq = build_huffman_tree(image)

# Generate codes
codes = generate_codes(tree)

# Calculate compressed bits
encoded_bits = sum(freq[p] * len(codes[p]) for p in freq)
```

### Step 3: JPEG Compression

```python
compressed_img = jpeg_compress(image, q=50)
```

### Step 4: Calculate Metrics

```python
mse_val = mse(image, compressed_img)
psnr_val = psnr(mse_val)

original_size = image.size * 8  # bits
compressed_size = encoded_bits  # bits

CR = compression_ratio(original_size, compressed_size)
RD = compressed_size / image.size
```

### Step 5: Display Results

```python
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(compressed_img, cmap='gray')
plt.title("Compressed")
plt.show()

print(f"Compression Ratio: {CR:.2f}")
print(f"PSNR: {psnr_val:.2f} dB")
print(f"MSE: {mse_val:.2f}")
print(f"Rate: {RD:.2f} bpp")
```

## Key Functions Used

### Custom Functions

| Function               | Purpose                        |
| ---------------------- | ------------------------------ |
| `build_huffman_tree()` | Build Huffman tree from image  |
| `generate_codes()`     | Generate variable-length codes |
| `dct2()`               | Apply 2D DCT                   |
| `idct2()`              | Apply inverse 2D DCT           |
| `jpeg_compress()`      | Apply JPEG-like compression    |
| `mse()`                | Calculate Mean Squared Error   |
| `psnr()`               | Calculate PSNR                 |
| `compression_ratio()`  | Calculate compression ratio    |

### OpenCV Functions

| Function       | Purpose                   |
| -------------- | ------------------------- |
| `cv2.dct()`    | Discrete Cosine Transform |
| `cv2.idct()`   | Inverse DCT               |
| `cv2.imread()` | Read image                |

### NumPy Functions

| Function     | Purpose        |
| ------------ | -------------- |
| `np.pad()`   | Pad array      |
| `np.round()` | Round values   |
| `np.clip()`  | Bound values   |
| `np.mean()`  | Calculate mean |

## Execution Flow

```
Load Image (Grayscale)
    ↓
HUFFMAN CODING:
├── Count pixel frequencies
├── Build Huffman tree
├── Generate variable-length codes
└── Calculate compressed bits
    ↓
JPEG-LIKE COMPRESSION:
├── Pad to 8×8 blocks
├── Subtract 128 (center shift)
├── For each 8×8 block:
│   ├── Apply DCT
│   ├── Quantize with Q matrix
│   ├── Dequantize
│   └── Apply inverse DCT
├── Add 128 back
├── Remove padding
└── Clip to [0,255]
    ↓
CALCULATE METRICS:
├── MSE (original vs compressed)
├── PSNR from MSE
├── Compression Ratio
└── Rate-Distortion
    ↓
DISPLAY COMPARISON
```

## Input Requirements

- **Format:** JPG, PNG, or standard image formats
- **Color:** Grayscale (or converted to grayscale)
- **Size:** Any size (padded to 8×8 blocks)

## Expected Output

1. Original image display
2. Compressed image display
3. Huffman tree structure (optional visualization)
4. Compression metrics:
   - Compression Ratio
   - MSE
   - PSNR (dB)
   - Rate-Distortion (bpp)

## Quality Parameter Effects

### Quality Factor (1-100)

```python
def jpeg_compress(image, q=50):
    Q = Q * (100 / q)
```

| Quality | Effect                                | PSNR   |
| ------- | ------------------------------------- | ------ |
| 1       | Maximum compression, heavy artifacts  | ~20 dB |
| 10      | Strong compression, visible blocks    | ~25 dB |
| 25      | Moderate compression, minor artifacts | ~32 dB |
| 50      | Balanced, barely visible artifacts    | ~38 dB |
| 75      | Light compression, minimal artifacts  | ~44 dB |
| 90      | Very light compression, near original | ~48 dB |
| 100     | No compression (lossless DCT)         | ∞ dB   |

## Huffman vs JPEG Comparison

| Aspect                | Huffman      | JPEG-DCT          |
| --------------------- | ------------ | ----------------- |
| **Compression Type**  | Lossless     | Lossy             |
| **Quality Loss**      | None         | Perceptual        |
| **Compression Ratio** | 1.5-3:1      | 10-50:1           |
| **Speed**             | Fast         | Moderate          |
| **Artifacts**         | None         | Blocking at low Q |
| **Best For**          | Generic data | Photography       |
| **Reversible**        | Yes          | No                |

## Artifacts and Issues

### Blocking Artifacts (JPEG)

- **Cause:** 8×8 block boundaries visible at low quality
- **Appearance:** Grid pattern, blocky edges
- **Solution:** Increase quality factor
- **Prevention:** Use higher bitrate (better quality)

### Banding Artifacts (JPEG)

- **Cause:** Coarse quantization of gradients
- **Appearance:** Visible bands/stripes in smooth transitions
- **Solution:** Increase quality
- **Prevention:** Pre-filtering or better quantization

### Loss of Detail (JPEG)

- **Cause:** High-frequency information discarded
- **Appearance:** Blurred edges, loss of texture
- **Solution:** Increase quality
- **Trade-off:** Reduces compression

## Advanced Concepts

### Progressive JPEG

- Encodes multiple passes
- Low quality first, progressively refines
- Better visual perception during download

### Entropy Coding Variants

- **Arithmetic Coding:** More efficient than Huffman
- **Range Coding:** Similar to arithmetic
- **LZ77/DEFLATE:** Dictionary-based (ZIP)

### Transform Alternatives to DCT

- **Wavelet Transform:** Better edge preservation
- **KL Transform:** Optimal for specific images
- **Haar Transform:** Simpler but less efficient

### Rate-Distortion Optimization

- **Lagrangian Optimization:** Balance rate and distortion
- **R-D Curve:** Shows trade-off between compression and quality
- **Optimal Quantization:** Choose Q values for best R-D

## Practical Applications

### Photography

- JPEG standard compression
- Adjustable quality-storage trade-off
- Widely supported format

### Medical Imaging

- Lossless compression (Huffman + LOCO-I)
- DICOM standard
- Critical information preservation

### Video Compression

- DCT-based: MPEG-1/2, H.261
- Temporal + spatial compression
- Motion compensation

### Document Scanning

- Mixed lossless/lossy approaches
- Preserve text (lossless), smooth images (lossy)
- JBIG2 for binary images

## Performance Considerations

### Computation Time (512×512 image)

- Huffman tree building: 1-5ms
- Huffman encoding: 10-20ms
- JPEG DCT compression: 50-150ms
- Metric calculation: <1ms

### Memory Usage

- Huffman codes: ~1KB (max 256 codes)
- DCT workspace: ~1MB (block buffers)
- Overall: Proportional to image size

## Troubleshooting

| Issue                    | Cause                        | Solution                                           |
| ------------------------ | ---------------------------- | -------------------------------------------------- |
| Invalid MSE              | MSE calculation error        | Check formula, ensure same size                    |
| Infinity PSNR            | MSE = 0 (identical images)   | Handle as special case                             |
| Padding visible          | Incorrect removal            | Ensure correct slice: `[:original_h, :original_w]` |
| Compression ratio < 1    | Overhead larger than savings | Add compressed size overhead                       |
| Blocking visible         | Too low quality              | Increase quality factor                            |
| No compression (Huffman) | High entropy image           | Compare against theoretical limit                  |

## Extension Activities

1. **Quality Comparison:**
   - Test multiple quality values (10, 25, 50, 75, 90)
   - Plot PSNR vs compression ratio
   - Create R-D curves

2. **Different Images:**
   - Natural images vs synthetic
   - High texture vs smooth
   - Analyze compression differences

3. **Huffman Optimization:**
   - Implement adaptive Huffman
   - Compare with arithmetic coding
   - Measure bit efficiency

4. **Hybrid Compression:**
   - Apply Huffman after JPEG quantization
   - Implement progressive JPEG
   - Add entropy coding

5. **Block Size Effects:**
   - Compare different DCT block sizes
   - Analyze edge effects
   - Optimize for specific images

6. **Perceptual Metrics:**
   - Implement SSIM (Structural Similarity)
   - Compare with PSNR
   - User perception studies

## Rate-Distortion Analysis

**Creating R-D Curve:**

```python
qualities = range(1, 101, 5)
rates = []
distortions = []

for q in qualities:
    compressed = jpeg_compress(image, q)
    mse_val = mse(image, compressed)
    rate = calculate_rate(compressed)

    rates.append(rate)
    distortions.append(mse_val)

plt.plot(rates, distortions)
plt.xlabel("Rate (bpp)")
plt.ylabel("Distortion (MSE)")
plt.title("Rate-Distortion Curve")
```

## References

- JPEG Standard: https://en.wikipedia.org/wiki/JPEG
- Huffman Coding: https://en.wikipedia.org/wiki/Huffman_coding
- DCT: https://en.wikipedia.org/wiki/Discrete_cosine_transform
- Information Theory: https://en.wikipedia.org/wiki/Information_theory
- Image Compression by Gonzalez & Woods
- H.264/AVC Video Compression
- Rate-Distortion Theory (Shannon)

## Author Information

- **Course:** Digital Image Processing
- **Roll Number:** 2023-SE-39
- **Lab Task:** 10
- **Academic Year:** 2023

---

**Note:** Image compression represents the balance between storage/transmission efficiency and quality preservation. Understanding these trade-offs is essential for modern multimedia applications. Lossless compression (Huffman) preserves all information but achieves less compression, while lossy compression (JPEG-DCT) achieves high compression at the cost of imperceptible or acceptable information loss.
