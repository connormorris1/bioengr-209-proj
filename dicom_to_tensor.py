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
        if array.shape[0] % 2 == 0:
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
    dicom_tensor_resized = np.repeat(dicom_tensor, 3, axis=0)

    # Model expects a float instead of int
    dicom_tensor_resized = dicom_tensor_resized.to(torch.float32)
    shape = dicom_tensor_resized.shape
    if shape[0] != 3 or shape[1] != 224 or shape[2] != 224:
        print(f"Improper shape {shape}")
        print(img_path)

    return dicom_tensor_resized
