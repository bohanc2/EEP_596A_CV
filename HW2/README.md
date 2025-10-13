# EEP 596A Computer Vision – Homework 2

### Overview
This project implements core **image processing and convolution-based filtering** techniques using **Python**, **NumPy**, **SciPy**, and **OpenCV**.  
All required functions are developed manually (using nested loops for convolution) to deepen understanding of Gaussian smoothing, derivative filtering, and edge detection.  
The assignment is implemented in a single Python file: `assignment2.py`.

---

### Contents
- `assignment2.py` – Main implementation script  
- `cat_eye.jpg` – Input grayscale image  
- `ant_outline.png` – Input binary image  
- `EEP 596 HW #2.pdf` – Assignment instructions  

---

### Implemented Tasks

| # | Task | Description |
|---|------|--------------|
| 1 | **Flood Fill** | Implemented region filling algorithm using a stack-based approach to fill connected pixels in a binary image (`ant_outline.png`). |
| 2 | **Gaussian Blur** | Applied separable 1D Gaussian kernel convolution (`0.25 * [1, 2, 1]`) horizontally and vertically, repeating 5 times to produce increasingly blurred results. |
| 3 | **Vertical Derivative** | Computed the 2D Gaussian vertical derivative using the smoothing kernel along the horizontal direction and a differentiating kernel (`0.5 * [1, 0, -1]`) along the vertical direction. |
| 4 | **Horizontal Derivative** | Similar to Task 3 but transposed the derivative direction — differentiation along the horizontal direction and smoothing along the vertical direction. |
| 5 | **Gradient Magnitude** | Calculated image gradient magnitude using **Manhattan norm** (`|gx| + |gy|`), combining horizontal and vertical derivatives to highlight image edges. |
| 6 | **Built-in Convolution (SciPy)** | Repeated Task 3 using `scipy.signal.convolve2d` with zero padding to verify equivalence with the manually implemented convolution. |
| 7 | **Repeated Box Filter** | Implemented 1D box filter `[1, 1, 1]` repeatedly convolved with itself to demonstrate convergence toward a Gaussian distribution. The filter results for 0–5 convolutions are plotted using Matplotlib. |

---

### Key Features
- All manual convolution operations are implemented using **nested for-loops** and **basic arithmetic** (no `np.convolve` or other shortcuts).
- Each stage saves its output images (e.g., `gaussian blur 0.jpg`, `vertical_0.jpg`, `gradient_0.jpg`) for verification.
- Functions ensure:
  - Zero padding at borders  
  - Clamping to `[0, 255]` to prevent overflow  
  - Conversion back to `uint8` type for image compatibility  
- Gradient scaling and offset (`2 * pin + 127`) preserve visibility of both positive and negative derivative responses.

---

### Output Files
Generated images include:
- `floodfill.jpg`  
- `gaussian blur 0–4.jpg`  
- `vertical_0–4.jpg`, `horizontal_0–4.jpg`  
- `gradient_0–4.jpg`  
- `scipy smooth 0–4.jpg`  
- Box filter visualization plot (`box_filter_plot.png`, if saved)

