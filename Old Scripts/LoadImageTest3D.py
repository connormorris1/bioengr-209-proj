import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Uses pydicom to load a single image as a test, convert to tensor, interpolate to given size, and display it
# https://pydicom.github.io/pydicom/stable/auto_examples/image_processing/reslice.html
img_path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train/7/study/sax_45/'
final_dim = 64 # Final image interpolated to be final_dim x final_dim x final_dim

# Load all dicom files into a list given the img_path
dicoms = [pydicom.dcmread(img_path + file) for file in os.listdir(img_path) if os.path.isfile(img_path + file)]


# skip files with no SliceLocation (eg scout views)
# NOTE: This code is taken from the PyDicom tutorial (ultimately not used)
slices = []
skipcount = 0
for f in dicoms:
    if hasattr(f, "SliceLocation"):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print(f"skipped, no SliceLocation: {skipcount}")

# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

# Code from PyDicom ends here

# Now staple these into a 3D image (assuming slices are all equal dimension)
z_dim = len(slices)
x_dim, y_dim = np.shape(slices[0].pixel_array)
final_image = np.zeros((x_dim, y_dim, z_dim))

for i in range(0, len(slices)):
    slice_arr = slices[i].pixel_array
    final_image[:, :, i] = slice_arr[:, :]

# Convert image to tensor object
final_image = torch.tensor(final_image)

# Resize image to 256x256x256 using torch.nn.functional.interpolate
# This part of the code was done with help from ChatGPT
# To do this we need to add a batch dimension and a channel dimension of size 1 (so our image becomes 1 x 1 x X x Y x Z)

# Adds our two extra dimensions
final_image = final_image.unsqueeze(0)
final_image = final_image.unsqueeze(0)

# Perform the interpolation
final_image_resized = F.interpolate(final_image, size=(final_dim, final_dim, final_dim), mode='trilinear', align_corners=False)

# Remove the two extra dimensions
final_image_resized = final_image_resized.squeeze(0)
final_image_resized = final_image_resized.squeeze(0)

print(final_image_resized.shape)



plt.imshow(final_image_resized[:, :,  int(final_dim / 2)])
plt.show()