# EEP 596A Computer Vision – Homework 3

### Overview
This project focuses on practical image processing and tensor manipulation using **Python**, **NumPy**, **OpenCV**, and **PyTorch**.
The assignment deepens understanding of tensor arithmetic, normalization, noise modeling, and stride-based convolution through hands-on implementation.
All required functions are implemented manually to demonstrate low-level image and tensor operations without relying on prebuilt PyTorch or OpenCV shortcuts.

The assignment is implemented in a single Python file:`assignment3.py`

---

### Contents

- `assignment3.py` – Main implementation script
- `cat_eye.jpg` – Sample image for testing convolution and tensor transformations
- `original_image.png` – Sample image for testing normalization and stride convolution

---

### Implemented Tasks
| #	| Task	| Description |
|---| ------| ------------|
| 1	| **Torch Image Conversion** |	Converts a BGR OpenCV image into an RGB PyTorch tensor (float32), ensuring channel consistency for later tensor operations. |
| 2	| **Brighten** |	Adds a constant intensity (100) to all pixels to simulate image brightening, returning a torch.float32 tensor. |
| 3	| **Saturation Arithmetic** |	Demonstrates integer overflow behavior when performing arithmetic on uint8 tensors (values wrap around instead of clamping). |
| 4	| **Add Noise** |	Adds Gaussian noise (mean = 0, σ = 100 gray levels) to an image tensor, then normalizes pixel values to [0, 1]. |
| 5	| **Normalization (Per-Image)** |	Converts the image to float64 and normalizes each RGB channel by subtracting its mean and dividing by its standard deviation. |
| 6	| **ImageNet Normalization** |	Normalizes RGB images using the standard ImageNet mean and standard deviation ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), then clamps the output to [0, 1]. |
| 7	| **Dimension Rearrangement** |	Reorders image dimensions from (H × W × C) to (1 × C × H × W), matching PyTorch batch-channel input convention. |
| 8	| **Stride Convolution** |	Implements stride = 2 convolution using a manually defined Scharr X filter. The operation uses zero padding and nested loops to compute each output element. |
| 9	| **(Optional) Chain Rule / ReLU** |	Function templates for gradient computation and ReLU derivative analysis are left for later extension. |

---

### Key Features

- All operations implemented manually without high-level APIs such as torchvision.transforms or nn.Conv2d.

- Includes both arithmetic-based and statistical normalization methods.

- Demonstrates the relationship between pixel-level math and tensor operations.

- Explicit use of permute, unsqueeze, and pad to illustrate shape transformations and stride handling.

- Every function can be tested independently in the __main__ section for modular debugging and validation.

---

### Output Files

During testing, the following intermediate results can be displayed or saved:

- Brightened image (bright_img)
- Saturated image (saturated_img)
- Noisy image (noisy_img)
- Normalized image (image_norm)
- ImageNet normalized tensor (ImageNet_norm)
- Rearranged tensor (shape = [1, 3, H, W])
- Stride convolution output (stride) computed on grayscale version of cat_eye.jpg