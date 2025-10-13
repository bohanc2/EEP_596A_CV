# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
      self.ant_img = cv2.imread('ant_outline.png')
      self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

      # Define the fill color (e.g., bright green)
      fill_color = (0, 0, 255)  # (B, G, R)

      # Create a copy of the input image to keep the original image unchanged
      output_image = self.ant_img.copy()
      h, w = output_image.shape[:2]

      # single chanel
      ant_img_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY).astype(np.int32)
      
      # Define a stack for floodfill
      start_color = ant_img_gray[seed[1], seed[0]]
      stack = [seed]
      while stack:
          x, y = stack.pop()

          # check boundries
          if x < 0 or x >= w or y < 0 or y >= h:
              continue
          
          if ant_img_gray[y, x] == start_color:
              output_image[y, x] = fill_color

              # mark if already filled color
              ant_img_gray[y, x] = -1

              # add up, down, left, right to stack
              stack.append((x + 1, y))
              stack.append((x - 1, y))
              stack.append((x, y + 1))
              stack.append((x, y - 1))

      cv2.imwrite('floodfill.jpg', output_image)
      
      return output_image

  def gaussian_blur(self):
      """
      Apply Gaussian blur to the image iteratively.
      """
      kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32) # 1D Gaussian kernel: 0.25 * [1, 2, 1]
      image = self.cat_eye.astype(np.float32)
      h, w = image.shape[:2]
      self.blurred_images = []

      for i in range(5):
          # Apply convolution horizontal
          padded_h = np.pad(image, ((0, 0), (1, 1)), mode='constant', constant_values=0)
          temp_h = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  temp_h[y, x] = (padded_h[y, x] * kernel[0] + padded_h[y, x+1] * kernel[1] + padded_h[y, x+2] * kernel[2])
          
          # Apply convolution vertical
          padded_v = np.pad(temp_h, ((1, 1), (0, 0)), mode='constant', constant_values=0)
          temp_v = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  temp_v[y, x] = (padded_v[y, x] * kernel[0] + padded_v[y+1, x] * kernel[1] + padded_v[y+2, x] * kernel[2])

          # Store the blurred image
          out_image = np.clip(np.round(temp_v), 0, 255).astype(np.uint8)
          self.blurred_images.append(out_image)
          cv2.imwrite(f'gaussian_blur_{i}.jpg', out_image)
          image = temp_v
          
      return self.blurred_images

  def gaussian_derivative_vertical(self):
      # Define kernels
      kernel_h = [0.25, 0.5, 0.25]
      kernel_v = [0.5, 0, -0.5] # This is the Sobel kernel

      # Store images
      self.vDerive_images = []

      for i in range(5):
          image = self.blurred_images[i].astype(np.float32)
          h, w = image.shape[:2]

          # Apply convolution horizontal
          padded_h = np.pad(image, ((0, 0), (1, 1)), mode='constant', constant_values=0)
          temp_h = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  temp_h[y, x] = (padded_h[y, x] * kernel_h[0] + padded_h[y, x+1] * kernel_h[1] + padded_h[y, x+2] * kernel_h[2])

          # Apply convolution vertical
          padded_v = np.pad(temp_h, ((1, 1), (0, 0)), mode='constant', constant_values=0)
          temp_v = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                    temp_v[y, x] = (padded_v[y, x] * kernel_v[0] + padded_v[y+1, x] * kernel_v[1] + padded_v[y+2, x] * kernel_v[2])
          
          # the scale factor of 2 increases visibility, adding an offset of 127 preserves negative values after conversion
          image = 2 * temp_v + 127
          # clamping avoids wraparound
          image = np.clip(image, 0, 255).astype(np.uint8)

          self.vDerive_images.append(image)
          cv2.imwrite(f'vertical_{i}.jpg', image)
          
      return self.vDerive_images

  def gaussian_derivative_horizontal(self):
      # Define kernels(switch h & v)
      kernel_v = [0.25, 0.5, 0.25]
      kernel_h = [0.5, 0, -0.5] # This is the Sobel kernel

      # Store images after computing horizontal derivative
      self.hDerive_images = []

      for i in range(5):
          image = self.blurred_images[i].astype(np.float32)
          h, w = image.shape[:2]

          # Apply convolution vertical
          padded_v = np.pad(image, ((1, 1), (0, 0)), mode='constant', constant_values=0)
          temp_v = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  temp_v[y, x] = (padded_v[y, x] * kernel_v[0] + padded_v[y+1, x] * kernel_v[1] + padded_v[y+2, x] * kernel_v[2])

          # Apply convolution horizontal
          padded_h = np.pad(temp_v, ((0, 0), (1, 1)), mode='constant', constant_values=0)
          temp_h = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                    temp_h[y, x] = (padded_h[y, x] * kernel_h[0] + padded_h[y, x+1] * kernel_h[1] + padded_h[y, x+2] * kernel_h[2])
          
          # the scale factor of 2 increases visibility, adding an offset of 127 preserves negative values after conversion
          image = 2 * temp_h + 127
          # clamping avoids wraparound
          image = np.clip(image, 0, 255).astype(np.uint8)

          self.hDerive_images.append(image)
          cv2.imwrite(f'horizontal_{i}.jpg', image)
      return self.hDerive_images

  def gradient_magnitute(self):
      # Define kernels
      kernel_smooth = np.array([0.25, 0.5, 0.25])
      kernel_diff = np.array([0.5, 0, -0.5])
      
      self.gdMagnitute_images = []

      for i, img in enumerate(self.blurred_images):
          image = img.copy().astype(np.float32)
          h, w = image.shape[:2]
      
          # Vertical smoothing
          padded_v = np.pad(image, ((1, 1), (0, 0)), mode='constant', constant_values=0)
          temp_v = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  temp_v[y, x] = (padded_v[y, x] * kernel_smooth[0] + padded_v[y+1, x] * kernel_smooth[1] + padded_v[y+2, x] * kernel_smooth[2])

          # Horizontal derivative
          padded_h = np.pad(temp_v, ((0, 0), (1, 1)), mode='constant', constant_values=0)
          gx = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  gx[y, x] = (padded_h[y, x]   * kernel_diff[0] + padded_h[y, x+1] * kernel_diff[1] + padded_h[y, x+2] * kernel_diff[2])

          # horizontal smoothing
          padded_h = np.pad(image, ((0, 0), (1, 1)), mode='constant', constant_values=0)
          temp_h = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  temp_h[y, x] = (padded_h[y, x] * kernel_smooth[0] + padded_h[y, x+1] * kernel_smooth[1] + padded_h[y, x+2] * kernel_smooth[2])

          # vertical derivative
          padded_v = np.pad(temp_h, ((1, 1), (0, 0)), mode='constant', constant_values=0)
          gy = np.zeros_like(image)
          for y in range(h):
              for x in range(w):
                  gy[y, x] = (padded_v[y, x] * kernel_diff[0] + padded_v[y+1, x] * kernel_diff[1] + padded_v[y+2, x] * kernel_diff[2])
          
          # Store the computed gradient magnitute
          # Manhattan Gradient Magnitude
          pin = np.abs(gx) + np.abs(gy)
          pout = 4.0 * pin # scale for visibility
          pout = np.clip(np.round(pout), 0, 255).astype(np.uint8)

          self.gdMagnitute_images.append(pout)
          cv2.imwrite(f'gradient_{i}.jpg', pout)

      return self.gdMagnitute_images

  def scipy_convolve(self):
      # Define the 2D smoothing kernel
      kernel_h = np.array([0.25, 0.5, 0.25], dtype=np.float32)
      kernel_v = np.array([0.5, 0, -0.5], dtype=np.float32)
      # kernel_2d = np.outer(kernel_v, kernel_h) # we cant just use outer product to generate 2d kernel
      
      # Store outputs
      self.scipy_smooth = []

      for i in range(5):
          image = self.blurred_images[i].astype(np.float32)
          # horizontal
          temp_h = scipy.signal.convolve2d(
              image,
              kernel_h[np.newaxis, :],  # shape (1, 3)
              mode='same',
              boundary='fill',
              fillvalue=0
          )

          # vertical
          temp_v = scipy.signal.convolve2d(
              temp_h,
              kernel_v[:, np.newaxis],  # shape (3, 1)
              mode='same',
              boundary='fill',
              fillvalue=0
          )

          # pout = clamp(2 * pin + 127)
          out_img = 2 * temp_v + 127
          out_img = np.clip(out_img, 0, 255).astype(np.uint8)
          
          self.scipy_smooth.append(out_img)
          cv2.imwrite(f'scipy_smooth_{i}.jpg', out_img)
          image = out_img.astype(np.float32)

      return self.scipy_smooth
  
  def box_filter(self, num_repetitions):
      # Define box filter
      box_filter = [1, 1, 1]
      out = [1, 1, 1]

      for _ in range(num_repetitions):
          # Perform 1D conlve
          out_new = [0] * (len(out) + len(box_filter) - 1)
          for i in range(len(out)):
              for j in range(len(box_filter)):
                  out_new[i + j] += out[i] * box_filter[j]
          out = out_new
      
      return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    floodfill_img = ass.floodfill(seed=(100, 100))
    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()
    
    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
    
    filters = [ass.box_filter(i) for i in range(6)]
    for i, f in enumerate(filters):
        plt.plot(f, label=f"{i} convs")
    plt.legend()
    plt.title("Repeated Box Filter Approximating Gaussian")
    plt.show()
    