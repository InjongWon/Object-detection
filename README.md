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



![s25](https://github.com/user-attachments/assets/30ecf8d6-de3d-4e2d-80ba-7c5acb484766)\\

![edge_map1](https://github.com/user-attachments/assets/4dd0f0e3-348f-49be-ba63-db32e21887bf)\\

![gradient_image1](https://github.com/user-attachments/assets/efb0fa4f-cc7b-4289-b134-22e9eb97fcf4)
