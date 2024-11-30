#import os
import pydicom
import torch
from skimage.transform import rescale
import numpy as np

# Takes a path to a 2D dicom file and converts it to a tensor (with dimensions interp_resolution x interp_resolution)
def dicom_path_to_tensor(img_path, interp_resolution):
    # Load all dicom files into a list given the img_path
    #dicoms = [pydicom.dcmread(img_path + file) for file in os.listdir(img_path) if os.path.isfile(img_path + file)]

    # Load image dicom and conver to tensor
    dicom = pydicom.dcmread(img_path)
    array = dicom.pixel_array

    #rescale to 1mmx1mm pixel size
    array = rescale(array, dicom.PixelSpacing,anti_aliasing=dicom.PixelSpacing[0] < 1)
    #center crop to 224x224, with 0 padding if less
    target_dim = 224
    if array.shape[0] > target_dim:
        center_point = array.shape[0]//2
        array = array[center_point - 112:center_point + 112,:]
    elif array.shape[0] < target_dim:
        if array.shape[1] % 2 == 0:
            correction_factor = 0
        else:
            correction_factor = 1
        pad_size = (target_dim - array.shape[0])//2
        array = np.pad(array,((pad_size + correction_factor,pad_size),(0,0)),constant_values=0)
    if array.shape[1] > target_dim:
        center_point = array.shape[1]//2
        array = array[:,center_point - 112:center_point + 112]
    elif array.shape[1] < target_dim:
        if array.shape[1] % 2 == 0:
            correction_factor = 0
        else:
            correction_factor = 1
        pad_size = (target_dim - array.shape[1])//2
        array = np.pad(array,((0,0),(pad_size + correction_factor,pad_size)),constant_values=0)
    
    mean = np.mean(array)
    std = np.std(array)
    normalized_array = (array - mean)/std
    dicom_tensor = torch.from_numpy(normalized_array)
    # Add an extra dimension for color channel because it is required for interpolation
    dicom_tensor = dicom_tensor.unsqueeze(0)

    # Turn from a 1 color channel image into 3 channel
    # TODO Make this better since model expects 3 color channel so I'm just copying the intensity into two other channels
    # ***Medical images are grayscale. Maybe an architecture w/out extra color channels is better?***
    dicom_tensor_resized = np.repeat(dicom_tensor, 3, axis=0)

    # Model expects a float instead of int
    # ***may need to normalize pixel values to mean of 0, std of 1***
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
