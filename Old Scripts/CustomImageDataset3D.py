import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

# Based on tutorial from PyTorch website https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    # annotations_file is a path to a csv file with the DICOM folder name, label
    # img_dir is path to folder containing all image folders, each with a collection of dicom images
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

        # Dimension that we interpolate our MRI images to (NxNxN)
        self.target_dimensions = 64

    def __len__(self):
        return len(self.img_labels)

    # Loads the dicoms and label of a given MRI, converts DICOMS into a 3Dntensor,
    # interpolates to a given size, and returns image tensor and label
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = self.dicom_folder_to_tensor(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


    # Uses pydicom to load a folder of dicoms (given by img_path), convert to tensor, then interpolate to given size
    def dicom_folder_to_tensor(self, img_path):
        # Load all dicom files into a list given the img_path
        dicoms = [pydicom.dcmread(img_path + file) for file in os.listdir(img_path) if os.path.isfile(img_path + file)]

        # skip files with no SliceLocation (eg scout views)
        # NOTE: THIS CODE IS TAKEN FROM THE PYDICOM TUTORIAL
        # TODO Replace by sorting by PatientPosition if this doesn't work for some of our data
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

        # STOLEN CODE ENDS HERE

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
        final_image_resized = F.interpolate(final_image, size=(self.target_dimensions, self.target_dimensions, self.target_dimensions), mode='trilinear',
                                            align_corners=False)

        # Remove the two extra dimensions
        final_image_resized = final_image_resized.squeeze(0)
        final_image_resized = final_image_resized.squeeze(0)

        return final_image_resized

