# OEL_DIP: Melanoma Detection using Image Processing & Machine Learning

## Project Overview

**OEL_DIP** is a comprehensive Digital Image Processing project that implements a complete pipeline for **melanoma detection** from dermoscopic images using image processing techniques and machine learning. The project utilizes the **PH2 Dataset** (Photothermal Hair Follicle Dataset) and employs advanced image enhancement, segmentation, feature extraction, and classification methods.

## Project Goals

1. **Image Enhancement**: Improve image quality using Contrast Limited Adaptive Histogram Equalization (CLAHE)
2. **Hair Removal**: Eliminate unwanted hair artifacts from dermoscopic images
3. **Lesion Segmentation**: Accurately segment melanoma lesions from background skin
4. **Feature Extraction**: Extract meaningful features from segmented lesions
5. **Classification**: Build a machine learning classifier to distinguish melanoma from benign lesions
6. **Evaluation**: Assess model performance using multiple metrics

## Dataset

- **Source**: PH2 Dataset
- **Location**: `PH2Dataset/PH2 Dataset images/`
- **Contains**: 200+ dermoscopic images with clinical diagnoses
- **Format**: Organized in folders by image ID (IMD002, IMD003, etc.)
- **Each folder contains**:
  - Dermoscopic image of the lesion
  - Ground truth segmentation mask
  - Clinical diagnosis information

## Project Structure

```
OEL_DIP.ipynb              # Main notebook file
OEL_DIP.md                 # This documentation file
PH2Dataset/
├── PH2_dataset.txt        # Clinical diagnosis labels
├── Readme.txt             # Dataset information
├── PH2 Dataset images/    # All dermoscopic images
└── generated_masks/       # Output segmentation masks
```

## Pipeline Architecture

### Step 0: Environment Setup & Library Imports

- Import necessary libraries for image processing and machine learning
- Set up dependencies for data handling and visualization

### Step 1: Dataset Path Configuration

- Define base paths for images, masks, and labels
- Create output directories for generated masks

### Step 2: Image Preprocessing Functions

#### 2.1 Image Enhancement (CLAHE)

- **Technique**: Contrast Limited Adaptive Histogram Equalization
- **Purpose**: Improve image contrast and visibility of lesion features
- **Parameters**:
  - Clip Limit: 2.0
  - Tile Grid Size: 8×8
- **Process**: Convert BGR → LAB → Apply CLAHE to L channel → Convert back to BGR

#### 2.2 Hair Removal

- **Technique**: Black-Hat Morphological Operation + Inpainting
- **Purpose**: Remove hair artifacts that interfere with segmentation
- **Process**:
  1. Convert image to grayscale
  2. Apply morphological black-hat operation with 17×17 kernel
  3. Threshold to create hair mask
  4. Use Telea inpainting to fill removed areas naturally

### Step 3: Lesion Segmentation

- **Objective**: Isolate the melanoma lesion from surrounding skin
- **Technique**: Otsu's Thresholding + Morphological Operations
- **Process**:
  1. Convert to grayscale
  2. Apply Gaussian blur (5×5)
  3. Binary inverse thresholding with Otsu's method
  4. Morphological closing (5×5 kernel)
  5. Morphological opening (5×5 kernel)
  6. Morphological erosion (1 iteration)
- **Output**: Binary mask representing lesion boundaries

### Step 4: Mask Generation & Segmentation Evaluation

- **Process**:
  1. Load original image and ground truth mask for each image
  2. Apply preprocessing (enhancement + hair removal)
  3. Generate segmentation mask using lesion segmentation
  4. Save generated mask to disk
  5. Compare with ground truth for evaluation
- **Output**: Segmentation masks stored in `generated_masks/` directory
- **Metrics Calculated**:
  - **Accuracy**: Overall correctness of segmentation
  - **Sensitivity (Recall)**: True positive rate for lesion pixels
  - **Specificity**: True negative rate for non-lesion pixels
  - **Confusion Matrix**: Detailed breakdown of classification results

### Step 5: Label Loading

- **Input**: `PH2_dataset.txt` containing clinical diagnoses
- **Processing**:
  1. Parse text file with image ID and clinical diagnosis
  2. Create dictionary mapping image IDs to class labels
  3. Binary classification: 1 = Melanoma, 0 = Non-Melanoma (Common Nevus or benign)
- **Output**: Dictionary with 200+ labeled images

### Step 6: Feature Extraction

- **Objective**: Extract discriminative features from segmented lesion area
- **Method**: Color-based statistics within masked region
- **Features per channel (BGR)**: Mean and Standard Deviation
- **Total Features**: 6 (3 channels × 2 statistics)
- **Feature Vector**: [B_mean, B_std, G_mean, G_std, R_mean, R_std]
- **Masked Extraction**: Features computed only from pixels within the segmented lesion

### Step 7: Dataset Building

- **Input**:
  - Preprocessed images
  - Segmentation masks
  - Clinical labels
- **Process**:
  1. Iterate through all labeled images
  2. Extract features for each image using mask
  3. Combine features into feature matrix X
  4. Combine labels into label vector y
- **Output**:
  - **X**: Feature matrix (n_samples × 6 features)
  - **y**: Label vector (n_samples × 1)

### Step 8: Train/Test Split

- **Objective**: Divide data for model training and evaluation
- **Method**: Stratified train-test split
- **Parameters**:
  - Train/Test Ratio: 75/25
  - Random State: 42 (for reproducibility)
  - Stratification: Ensures balanced class distribution
- **Output**:
  - X_train, X_test: Feature subsets
  - y_train, y_test: Label subsets

### Step 9: Model Training

- **Algorithm**: Random Forest Classifier
- **Configuration**:
  - Number of trees: 80
  - Max depth: 8
  - Random state: 42
- **Advantages**:
  - Handles non-linear relationships
  - Robust to overfitting
  - Provides feature importance scores
  - Fast prediction time

### Step 10: Model Evaluation

- **Metrics Reported**:
  - **Accuracy**: Overall correct predictions
  - **F1 Score**: Harmonic mean of precision and recall
  - **Precision**: True positives / (True positives + False positives)
  - **Recall (Sensitivity)**: True positives / (True positives + False negatives)
  - **Classification Report**: Detailed metrics per class
- **Purpose**: Assess model performance on unseen test data

## Key Technologies

| Component         | Technology                   |
| ----------------- | ---------------------------- |
| Image Processing  | OpenCV (cv2)                 |
| Data Handling     | NumPy, Pandas                |
| Machine Learning  | Scikit-learn                 |
| Visualization     | Matplotlib                   |
| Progress Tracking | tqdm                         |
| Environment       | Python 3.x, Jupyter Notebook |

## Algorithm Details

### Image Enhancement Pipeline

```
Original Image
    ↓
BGR → LAB Color Space Conversion
    ↓
Split into L, a, b channels
    ↓
Apply CLAHE to L channel
    ↓
Merge channels
    ↓
LAB → BGR Color Space Conversion
    ↓
Enhanced Image
```

### Hair Removal Pipeline

```
Enhanced Image
    ↓
Convert to Grayscale
    ↓
Apply Black-Hat Morphology (17×17 kernel)
    ↓
Threshold to create hair mask
    ↓
Telea Inpainting
    ↓
Hair-Removed Image
```

### Segmentation Pipeline

```
Hair-Removed Image
    ↓
Convert to Grayscale
    ↓
Gaussian Blur (5×5)
    ↓
Otsu's Binary Inverse Threshold
    ↓
Morphological Closing (5×5)
    ↓
Morphological Opening (5×5)
    ↓
Morphological Erosion (1×)
    ↓
Lesion Segmentation Mask
```

## Expected Results

### Segmentation Performance

- **Accuracy**: ~85-90% pixel-wise classification
- **Sensitivity**: ~80-85% (true positive rate for lesion detection)
- **Specificity**: ~85-90% (true negative rate for background)

### Classification Performance

- **Model Accuracy**: ~75-85% on test set
- **F1 Score**: ~0.70-0.80 (balanced precision and recall)
- **ROC-AUC**: ~0.80-0.90 (good discrimination ability)

## How to Run

1. **Prepare Environment**:

   ```bash
   pip install opencv-python numpy pandas scikit-learn matplotlib tqdm
   ```

2. **Ensure Dataset is Available**:
   - Place PH2 Dataset in `PH2Dataset/` directory
   - Verify `PH2_dataset.txt` contains labels

3. **Execute Notebook**:
   - Run all cells in sequence from top to bottom
   - Each step builds upon previous results

4. **Review Results**:
   - Check console output for metrics
   - Inspect generated masks in `generated_masks/` folder
   - Analyze classification report

## Output Files

| File            | Description                                            |
| --------------- | ------------------------------------------------------ |
| `*_mask.png`    | Generated segmentation masks for each image            |
| Console metrics | Segmentation and classification performance statistics |

## Troubleshooting

| Issue                            | Solution                                              |
| -------------------------------- | ----------------------------------------------------- |
| "File not found" error           | Verify dataset paths match actual directory structure |
| Memory issues with large dataset | Process images in batches instead of all at once      |
| Poor segmentation results        | Adjust CLAHE parameters (clipLimit, tileGridSize)     |
| Low classification accuracy      | Explore additional features (texture, shape)          |
| Imbalanced classes               | Use class_weight='balanced' in RandomForest           |

## Future Enhancements

1. **Deep Learning Integration**: Replace feature extraction with CNN embeddings
2. **Advanced Segmentation**: Implement U-Net or Mask R-CNN for better boundaries
3. **Feature Expansion**: Add GLCM texture features, shape descriptors
4. **Cross-Validation**: Implement k-fold CV for more robust evaluation
5. **Class Balancing**: Handle class imbalance with SMOTE or weighted loss
6. **Visualization**: Create confusion matrices, ROC curves, SHAP explanations
7. **Hyperparameter Tuning**: Use GridSearchCV for optimal Random Forest parameters
8. **Ensemble Methods**: Combine multiple classifiers for improved predictions

## Performance Considerations

- **Runtime**: ~2-5 minutes for full pipeline on 200 images
- **Memory Usage**: ~500MB-1GB (depends on image resolution)
- **Optimization**: Parallelization possible for image processing step

## References

- **Dataset**: PH2 Dataset - Photothermal Hair Follicle Dataset
- **Image Enhancement**: CLAHE in OpenCV
- **Morphology**: OpenCV Morphological Operations
- **Classification**: Random Forest Classifier in Scikit-learn
- **Metrics**: Confusion Matrix, Accuracy, Sensitivity, Specificity

## Author Notes

This project demonstrates a complete machine learning pipeline:

- Problem understanding (melanoma detection)
- Data preprocessing and enhancement
- Feature engineering and extraction
- Model selection and training
- Performance evaluation and metrics

The modular design allows easy modification of individual components for research and experimentation.

## License

Educational project - University of [Your Institution]
Course: Digital Image Processing
Student ID: 2023-SE-39

---

**Last Updated**: February 2026
**Status**: Complete and documented
