# Object-detection
# Gradient Analysis & Image Processing

This repository contains Jupyter Notebooks implementing gradient-based image processing techniques using Python, OpenCV, and NumPy. Each notebook corresponds to a different task focused on image gradients, edge detection, and visualization.

---

## 📁 Contents

### `CannyEdge.ipynb` – 📷 Grayscale Image Display

- **Concepts**:
  - Image reading using OpenCV
  - Displaying images with `matplotlib`
- **Implementation Details**:
  - The image `s25.jpeg` is read in grayscale mode.
  - The image is displayed using `plt.imshow()` with proper color conversion (`cv2.COLOR_BGR2RGB`).
  - Title and axis are configured for clean presentation.

![s25](https://github.com/user-attachments/assets/30ecf8d6-de3d-4e2d-80ba-7c5acb484766)\\

![edge_map1](https://github.com/user-attachments/assets/4dd0f0e3-348f-49be-ba63-db32e21887bf)\\

![gradient_image1](https://github.com/user-attachments/assets/efb0fa4f-cc7b-4289-b134-22e9eb97fcf4)

---

### `Gaussian Blur.ipynb` – 📐 Sobel Filter & Gradient Magnitude

- **Concepts**:
  - Convolution with Sobel filters
  - Computing gradient magnitude
  - Manual implementation of convolution
- **Implementation Details**:
  - Functions:
    - `convolve()`: Applies a kernel to an image manually (zero padding + dot product).
    - `gradient_mag()`: Applies Sobel X and Y filters to compute the gradient magnitude.
    - `visualize_images()`: Displays the original and gradient images side by side.
  - Two grayscale images are processed and visualized to compare gradient magnitudes.

---

### `Gaussian Filter.ipynb` – 🔧 Image Processing Setup

- **Concepts**:
  - Imports and setup for further image processing
- **Implementation Details**:
  - Sets up basic libraries: NumPy, OpenCV, Matplotlib
  - Appears to be a base notebook or part of a multi-step pipeline

---

## 🛠 Dependencies

To run these notebooks, install the following Python packages:




```bash
pip install numpy opencv-python matplotlib scikit-image


Focuses on training and evaluating neural network models for dog breed classification using the Stanford Dogs Dataset (SDD) and Dog Breed Images (DBI), as well as analyzing dataset bias and generalization.

---
```

## 📁 Contents & Task Mapping

| Notebook | Task | Description |
|----------|------|-------------|
| `trainCNN.ipynb` | Training a custom CNN from scratch on DBI |
| `restNet18-34.ipynb` |Training ResNet18 from scratch on DBI and evaluating on SDD |
| `Fine-Tuning.ipynb` | Fine-tuning pre-trained ResNet18, ResNet34, and ResNeXt on DBI and testing on both datasets |
| `classifier_train.ipynb` |  Training a classifier to distinguish whether an image comes from DBI or SDD |

---

- Convolutional Neural Networks (CNNs)
- Data augmentation and regularization (dropout)
- Transfer learning & fine-tuning with PyTorch
- Domain generalization and cross-dataset evaluation
- Dataset bias detection

---

## 🧠 Implementation Highlights

### `trainCNN.ipynb` – Custom CNN on DBI

- A lightweight CNN was trained from scratch on the DBI dataset.
- Techniques used:
  - Batch normalization
  - Dropout layers for regularization
  - Cross-entropy loss
  - Random cropping, flipping, and rotation for data augmentation
- **Goal**: Observe how dropout impacts training/test accuracy and generalization.

---

### `restNet18-34.ipynb` – ResNet18 from Scratch

- A modified ResNet18 model was trained on the DBI dataset **without pre-trained weights**.
- Evaluation:
  - Accuracy was plotted on training, validation, and test sets.
  - The trained model was then tested on SDD for cross-dataset comparison.
- **Key Concept**: Domain generalization — observing how a model trained on one dataset performs on another.

---

### `Fine-Tuning.ipynb` – Fine-Tuning Pre-trained Models

- Models used:
  - `ResNet18`, `ResNet34`, and `ResNeXt50_32x4d` from PyTorch Hub
- Strategy:
  - Fine-tune the last layers while retaining learned representations
- Evaluation:
  - Performance tested on DBI and full SDD
- **Goal**: Compare generalization and overfitting across A!architectures

---

### `classifier_train.ipynb` – Dataset Classifier

- Objective: Classify whether an image belongs to DBI or SDD
- Approach:
  - Used a pre-trained model (e.g. ResNet18) as base
  - Trained a binary classifier on dataset identity
- Evaluation: 
  - Model accuracy on test set
  - Confusion matrix and classification report
- **Goal**: Quantify and detect dataset-specific features that affect generalization

---

## 🛠 Setup

Make sure the following libraries are installed:

Gradient-based image analysis, with a focus on Histogram of Oriented Gradients (HOG), Laplacian of Gaussian, and Harris Corner Detection techniques.

---

## 📁 Notebook Breakdown & Task Mapping

| Notebook         | Task | Description |
|------------------|------|-------------|
| `laplcian.ipynb`  | Laplacian of Gaussian scale analysis + gradient visualizations |
| `local_HOG_descriptor.ipynb` | Full implementation of Local HOG descriptor: extraction, normalization, and comparison |
| `cornerEdgeDetection.ipynb` | Harris corner detection via Second Moment Matrix eigenvalue analysis |

---

## 🔍 Concepts Covered

- Laplacian of Gaussian (LoG) for blob detection
- Gradient magnitude and orientation computation
- Histogram of Oriented Gradients (HOG) for local feature description
- Illumination-invariant HOG normalization using L2 norm
- Eigenvalue-based corner detection (Harris detector)
- Comparative visualizations of gradient- and corner-based feature maps

---

## 📓 Notebook Details

### 📘 `laplacian.ipynb` – LoG Analysis & Gradient Visualization

- **Implementation**: Computes LoG response for a black square to find the optimal σ that maximizes response magnitude.
- Uses:
  - Manual generation of black-square-on-white images
  - Laplacian of Gaussian kernel formulas
- Also includes gradient visualization on sample images (likely used to validate orientation detection for HOG setup).

---

### 📘 `local_HOG_descriptor.ipynb` – Local HOG Feature Descriptor

Implements the full pipeline for extracting a **HOG descriptor** from images:

- **Gradient Calculation**:
  - Computes gradient magnitudes and angles using `np.gradient` or Sobel filters.
  - Applies thresholding to ignore low-strength gradients.

- **Cell Binning**:
  - Divides the image into fixed-size cells (e.g., 8x8).
  - Quantizes gradient orientations into 6 bins.
  - Two approaches implemented:
    - **Weighted HOG**: Accumulate magnitudes per bin
    - **Unweighted HOG**: Count pixel votes per bin

- **Quiver Visualization**:
  - Uses matplotlib’s `quiver` to draw orientations for each cell.

- **Normalization**:
  - Combines adjacent 2x2 cells into blocks
  - L2-normalizes the resulting 24-element block vector
  - Saves normalized descriptors as `.txt` for input images

- **Illumination Robustness**:
  - Evaluates HOG robustness to lighting by comparing with-flash vs no-flash grayscale images.
  - Files stored: `image_with_flash.txt`, `image_without_flash.txt`

---

### 📘 `cornerEdgeDetection.ipynb` – Corner Detection with Eigenvalues

- Implements eigenvalue-based corner detection using the **Second Moment Matrix**.
- Steps:
  - Computes image gradients (`Ix`, `Iy`)
  - Constructs the structure tensor (M) for each pixel
  - Calculates eigenvalues λ₁ and λ₂ using closed-form expressions
  - Scatter plots of eigenvalues are generated
  - Corners are identified where `min(λ₁, λ₂)` exceeds a chosen threshold
  - Experiments repeated with varying Gaussian σ values to evaluate smoothing effects

---

## 🛠 Setup Instructions

Install dependencies:

```bash
pip install numpy opencv-python matplotlib


```bash
pip install torch torchvision matplotlib scikit-learn

