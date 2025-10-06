# EEP 596A Computer Vision – Homework 1

### Overview
This project introduces fundamental image processing and computer vision techniques using **Python**, **NumPy**, and **OpenCV**.  
The assignment implements core image manipulation, transformation, and analysis functions through a single Python script (`Assignment1.py`).

### Contents
- `Assignment1.py` – Main implementation  
- `original_image.png` – Input color image  
- `binary_image.png` / `glasses_outline.png` – Binary images for later tasks  
- `HW1.docx` / `Report.pdf` – Assignment instructions and report  

### Implemented Tasks
| # | Task | Description |
|---|------|--------------|
| 0 | **Setup** | Verify OpenCV, NumPy, Matplotlib installation and versions. |
| 1 | **Load and Analyze Image** | Load an image, print data types, and dimensions. |
| 2 | **Color Channels (Red Image)** | Keep red channel only, zero out green and blue. |
| 3 | **Photographic Negative** | Compute image negative via pixel inversion (`255 - pixel`). |
| 4 | **Swap Color Channels** | Exchange red and blue channels. |
| 5 | **Foliage Detection** | Detect green areas using thresholding (`G ≥ 50`, others `< 50`). |
| 6 | **Shift** | Translate image right by 200 px and down by 100 px, fill with black. |
| 7 | **Rotate** | Rotate the image clockwise by 90°. |
| 8 | **Similarity Transform** | Apply combined scaling, rotation, and translation (scale=2.0, θ=45°, shift=[100,100]) using inverse mapping and nearest-neighbor interpolation. |
| 9 | **Grayscale Conversion** | Convert color image to grayscale using `(3R + 6G + 1B) / 10`. |
| 10 | **Moments** | Compute first- and second-order raw and central moments for `glasses_outline.png`. |
| 11 | **Orientation & Eccentricity** | Derive object orientation (clockwise degrees), eccentricity, and draw a best-fit red ellipse on the glasses outline. |

### Dependencies
```bash
pip install numpy opencv-python matplotlib
