#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 19:41:56 2025

@author: mateoeljatib
"""
import imagecodecs
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, color, transform
from skimage.transform import resize
from skimage.draw import circle_perimeter
from skimage.draw import disk

#%%

# leer the imagen .tif 
base_dir = os.path.dirname(__file__)
image_path = os.path.join(base_dir, "202406_1445000611.bmp")  # Replace with your image path
image = io.imread(image_path)

# Step 2: Convert to grayscale if the image is not already
if len(image.shape) == 3:
    image_gray = color.rgb2gray(image)
else:
    image_gray = image
    
#%%

# Apply Gaussian blur
blurred_image = filters.gaussian(image_gray, sigma=4)
#%%
plt.imshow(image_gray, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.title('gray Image')
plt.axis('off')  # Hide axis
plt.show()


#%%
plt.imshow(blurred_image, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.title('Blurred Image')
plt.axis('off')  # Hide axis
plt.show()

#%%
# Detecccion de edges con Canny
edges = feature.canny(blurred_image)
# edges = feature.canny(blurred_image, sigma= 10, low_threshold=0.0001, high_threshold= 10)  # Adjust thresholds
#%%

plt.imshow(edges, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.title('edges Image')
plt.axis('off')  # Hide axis
plt.show()


#%%
# Perform Hough Transform to find circles
hough_radii = np.arange(1, 100)  # Use a wide range for dynamic radius
hough_res = transform.hough_circle(edges, hough_radii)

# Step 6: Detect peaks in the Hough accumulator
accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)
# Step 7: Visualize the results
# Create a color version of the original image for visualization
image_color = color.gray2rgb(image_gray)

# Draw detected circles
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image_color.shape)
    image_color[circy, circx] = (220, 20, 20)  # Color the circle

# Display the original, edges, and detected circles
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
ax[0].imshow(image_gray, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Edges Detected
ax[1].imshow(edges, cmap='gray')
ax[1].set_title('Canny Edges')
ax[1].axis('off')

# Detected Circles
ax[2].imshow(image_color)
ax[2].set_title('Detected Circles')
ax[2].axis('off')

plt.tight_layout()
plt.show()


#%%

# Extract the ROI based on the detected circle
# Assuming you want to extract the first detected circle
if len(cx) > 0:
    center_x = cx[0]
    center_y = cy[0]
    radius = radii[0]

    # Create a mask for the circular region
    mask = np.zeros_like(image_gray, dtype=bool)
    rr, cc = disk((center_y, center_x), radius, shape=image_gray.shape)
    mask[rr, cc] = True

    # Extract the circular region from the original image
    roi = np.zeros_like(image)  # Create an empty array for the ROI
    roi[mask] = image[mask]  # Apply the mask to the original image

#%%
'''
    # Visualize the ROI
    plt.figure(figsize=(8, 8))
    plt.imshow(roi, cmap='gray')
    plt.title('Region of Interest (ROI)')
    plt.axis('off')
    plt.show()
else:
    print("No circles detected.")
'''
#%%
plt.tight_layout()
plt.show()

#%%

# Visualize all images in one figure
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# Original Image
ax[0, 0].imshow(image_gray, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

# Blurred Image
ax[0, 1].imshow(blurred_image, cmap='gray')
ax[0, 1].set_title('Blurred Image')
ax[0, 1].axis('off')

# Detected Circles
ax[1, 0].imshow(image_color)
ax[1, 0].set_title('Detected Circles')
ax[1, 0].axis('off')

# Region of Interest (ROI)
ax[1, 1].imshow(roi, cmap='gray')
ax[1, 1].set_title('Region of Interest (ROI)')
ax[1, 1].axis('off')






'''
HASTA ACA ESTAMOS OK PERO DESPUES QUISE RESIZEAR
'''
#%%
import numpy as np

def crop_circle_roi(roi, center, radius):
    """Crops a square region around a detected circle in the ROI matrix.
    
    Parameters:
        roi (numpy.ndarray): The original image/matrix containing the circle.
        center (tuple): The (y, x) coordinates of the circle center.
        radius (int): The detected circle's radius.

    Returns:
        numpy.ndarray: The cropped square matrix containing the full circle.
    """
    center_y, center_x = center  # Extract coordinates
    D = 2 * radius  # Diameter (new cropped size)

    # Define the bounding box limits
    min_y, max_y = max(0, center_y - radius - np.int64(5)), min(roi.shape[0], center_y + radius + np.int64(5))
    min_x, max_x = max(0, center_x - radius - np.int64(5)), min(roi.shape[1], center_x + radius + np.int64(5))

    # Crop the matrix
    cropped_roi = roi[min_y:max_y, min_x:max_x]

    return cropped_roi

# Example usage with detected circle   
cropped_roi = crop_circle_roi(roi, (center_y, center_x), radius)

print(f"Original ROI size: {roi.shape}, Cropped ROI size: {cropped_roi.shape}")

#%%

plt.imshow(cropped_roi, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.title('Cropped ROI')
plt.axis('off')  # Hide axis
plt.show()

#%%

# Visualize all images in one figure
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# Original Image
ax[0, 0].imshow(image_gray, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

# Blurred Image
ax[0, 1].imshow(blurred_image, cmap='gray')
ax[0, 1].set_title('Blurred Image')
ax[0, 1].axis('off')

# Detected Circles
ax[1, 0].imshow(image_color)
ax[1, 0].set_title('Detected Circles')
ax[1, 0].axis('off')

# Region of Interest (ROI)
ax[1, 1].imshow(cropped_roi, cmap='gray')
ax[1, 1].set_title('Region of Interest (ROI)')
ax[1, 1].axis('off')
#%%
#  Extract the ROI based on the detected circle
if len(cx) > 0:
    center_x = cx[0]
    center_y = cy[0]
    radius = radii[0]

    # Create a mask for the circular region
    mask = np.zeros_like(image_gray, dtype=bool)
    rr, cc = disk((center_y, center_x), radius, shape=image_gray.shape)
    mask[rr, cc] = True

    # Extract the circular region from the original image
    roi = np.zeros_like(image)  # Create an empty array for the ROI
    roi[mask] = image[mask]  # Apply the mask to the original image

    # Step 8: Create a new square matrix of size D x D
    diameter = radius * 2
    new_roi = np.zeros((diameter, diameter, 3), dtype=image.dtype)  # Create a new empty square matrix

    # Calculate the position to place the circular ROI in the new matrix
    start_x = radius  # Center the circle in the new matrix
    start_y = radius

    # Create a circular mask for the new ROI
    new_mask = np.zeros((diameter, diameter), dtype=bool)
    rr_new, cc_new = disk((radius, radius), radius, shape=new_mask.shape)
    new_mask[rr_new, cc_new] = True

    # Place the circular region in the new matrix
    # Extract the circular region from roi and reshape it
    circular_region = roi[mask]  # This will be a 1D array
    circular_region = circular_region.reshape(-1, 3)  # Reshape to (N, 3) for RGB

    # Assign the circular region to the new ROI
    new_roi[new_mask] = circular_region[:new_mask.sum()]  # Ensure we only assign the correct number of pixels

    # Visualize the new ROI
    plt.figure(figsize=(8, 8))
    plt.imshow(new_roi)
    plt.title('New Circular Region of Interest (ROI)')
    plt.axis('off')
    plt.show()
else:
    print("No circles detected.")

#%%
'''
Bueno hasta ahora ya tengo el roi y a eso le quiero poder hacer un fcd
necesito darle una imagen de referencia
'''














