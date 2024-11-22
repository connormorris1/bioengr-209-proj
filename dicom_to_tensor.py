#import os
import pydicom
import torch
from torchvision.transforms import Resize
import numpy as np

# Takes a path to a 2D dicom file and converts it to a tensor (with dimensions interp_resolution x interp_resolution)
def dicom_path_to_tensor(img_path, interp_resolution):
    # Load all dicom files into a list given the img_path
    #dicoms = [pydicom.dcmread(img_path + file) for file in os.listdir(img_path) if os.path.isfile(img_path + file)]

    # Load image dicom and conver to tensor
    dicom = pydicom.dcmread(img_path)
    dicom_tensor = torch.tensor(dicom.pixel_array)

    # Add an extra dimension for color channel because it is required for interpolation
    dicom_tensor = dicom_tensor.unsqueeze(0)
    dicom_tensor_resized = Resize(size=(interp_resolution, interp_resolution))(dicom_tensor)

    # Turn from a 1 color channel image into 3 channel
    # TODO Make this better since model expects 3 color channel so I'm just copying the intensity into two other channels
    dicom_tensor_resized = np.repeat(dicom_tensor_resized, 3, axis=0)

    # Model expects a float instead of int
    dicom_tensor_resized = dicom_tensor_resized.to(torch.float32)

    return dicom_tensor_resized

# Test code:
'''
import matplotlib.pyplot as plt
img_path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train/10/study/sax_5/IM-13299-0001.dcm'
img_path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train/10/study/sax_5/IM-13299-0015.dcm'
img_path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train/10/study/sax_11/IM-13305-0001.dcm'
img_path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train/10/study/sax_11/IM-13305-0015.dcm'
#img_path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train/10/study/sax_8/IM-13302-0001.dcm'
#img_path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train/10/study/sax_6/IM-13300-0001.dcm'
tens = dicom_path_to_tensor(img_path, 128)
plt.imshow(tens[2, :, :],cmap='gray')
plt.show()
'''
