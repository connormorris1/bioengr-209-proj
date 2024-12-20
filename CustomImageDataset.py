import pandas as pd
import torch
from torch.utils.data import Dataset
from dicom_to_tensor import dicom_path_to_tensor
from balance_labels import balance_labels

# Based on tutorial from PyTorch website https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    # annotations_file is a path to a csv file with the DICOM folder name, label
    # img_dir is path to folder containing all image folders, each with a collection of dicom images
    # set balance to True if you want to balance the positive and negative labels
    def __init__(self, annotations_file, interp_resolution, balance=False, transform=None, target_transform=None,file_is_df=False):

        # Note here we need to balance the positive and negative labels if train data
        if balance:
            self.img_labels = balance_labels(pd.read_csv(annotations_file, header=None))
        elif file_is_df:
            self.img_labels = annotations_file
        else:
            self.img_labels = pd.read_csv(annotations_file, header=None)

        self.transform = transform
        self.target_transform = target_transform

        # Dimension that we interpolate our MRI images to (NxN)
        # Note we should keep this at 224x224 since that is what ResNet is built for/trained on
        self.target_dimensions = interp_resolution

    def __len__(self):
        return len(self.img_labels)

    # Loads the dicoms and label of a given MRI, converts DICOMS into a 3Dntensor,
    # interpolates to a given size, and returns image tensor and label
    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = self.img_labels.iloc[idx, 0]
        image = dicom_path_to_tensor(img_path, self.target_dimensions)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.tensor([label],dtype=torch.float32)

