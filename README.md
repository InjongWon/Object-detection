# Object-detection
# Computer Vision Assignments – Gradient Analysis & Image Processing

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

## 📁 Contents & Task Mapping

| Notebook | Task | Description |
|----------|------|-------------|
| `trainCNN.ipynb` | Task II | Training a custom CNN from scratch on DBI |
| `restNet18-34.ipynb` | Task III (a & b) | Training ResNet18 from scratch on DBI and evaluating on SDD |
| `Fine-Tuning.ipynb` | Task IV | Fine-tuning pre-trained ResNet18, ResNet34, and ResNeXt on DBI and testing on both datasets |
| `classifier_train.ipynb` | Task V | Training a classifier to distinguish whether an image comes from DBI or SDD |

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
- **Goal**: Compare generalization and overfitting across architectures

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

```bash
pip install torch torchvision matplotlib scikit-learn

