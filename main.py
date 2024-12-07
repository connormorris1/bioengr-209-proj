import torch
import torch.nn as nn
from torchvision import models
from CustomImageDataset import CustomImageDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from train_and_test_loop import train_loop, test_loop
import numpy as np
import pandas as pd
import time
import wandb
from FoundationModel import initialize_radimagenet_resnet
from ContrastiveLearning import SupervisedContrastiveLoss

############################################################################

# Path to directory containing dicom files
# Expected format of these files: csv files where each line is path_to_dicom, label
labels_train = '/home/cjmorris/repos/bioengr-209-proj/data_paths/all_train.csv'
labels_test = '/home/cjmorris/repos/bioengr-209-proj/data_paths/all_test.csv'
#labels_train = 'train.txt'
#labels_test = 'test.txt'
interp_resolution = 224 # Resnets expect a 224x224 image

batch_num = 100 # Batch size
learning_rate = 0.001
num_epochs = 50
save_model_path = 'resnet_weights_longrun2.pth'

pretrained = False # Set this to True if you want to use the pretrained version

foundation = False # Set this to True if you want to use the pretrained foundation model (RadImageNet Reset50)
freeze_encoder_foundation = True # Set this to True if you want to freeze the encoder of the foundation model

train_contrastive_encoder = False # Set this to true if you want to train the contrastive encoder (will save to save_model_path above)
use_contrastive_encoder = False # Set this to true if you want to use contrastive encoder in a classifier
contrastive_encoder_path = '' # Path of encoder for contrastive learning model, note encoder complexity tag below must match structure of model

# There are three different encoder models: Resnet18, Resnet34, and Resnet50
# Set this to 0 for Resnet18, 1 for Resnet34, and 2 for Resnet50
# The higher the complexity, the longer the training time but the better the performance on complex tasks
# Suggestion: Use Resnet18 to prototype and Resnet 34 or 50 for final model
encoder_complexity = 0

############################################################################

# Produces DataLoader from our data
# This Custom Data Loader first handles conversion of .dicom to tensor objects
# Balance training data (so positive and negative labels equal) but don't do so for test data
#try using WeightedRandomSampler to simplify balancing dataset
labels = list(pd.read_csv(labels_train,header=None)[1])
'''
class_counts = np.bincount(labels)
weights = 1/class_counts
sample_weights = weights[labels]
sampler = WeightedRandomSampler(weights=sample_weights,num_samples=int(class_counts[1]*2),replacement=False) #weighted random sampler that chooses a random sample of 2x the # of positive examples each epoch w/out replacement
#train_dataloader = DataLoader(CustomImageDataset(labels_train, interp_resolution, False), batch_size=batch_num,sampler=sampler)
'''
train_dataloader = DataLoader(CustomImageDataset(labels_train, interp_resolution, True), batch_size=batch_num)
test_dataloader = DataLoader(CustomImageDataset(labels_test, interp_resolution, False), batch_size=batch_num)

# This tests the MRI Data Loader
'''
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
'''

# Uses GPU or Mac backend if available, otherwise use CPU
# This code obtained from official pytorch docs
device = (
    "cuda:2"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Following code from ChatGPT

# Use a Resnet model (pretrained or not depending on input parameter above)
model = models.resnet18(pretrained=pretrained)
# If we set advanced_encoder to True then use Resnet50 as encoder instead (which is more accurate for complex tasks but takes longer to train)
if encoder_complexity == 1:
    model = models.resnet34(pretrained=pretrained)
if encoder_complexity == 2:
    model = models.resnet50(pretrained=pretrained)

# If we are using the contrastive learning encoder then freeze the encoder
if use_contrastive_encoder:
    model.load_state_dict(torch.load(contrastive_encoder_path), strict=False)
    for parameter in model.parameters():
        parameter.requires_grad = False

# Modify the final fully connected layer for 2 classes (single ventricle or not) ***only need 1 prediction - 0 is not, 1 is single ventricle***
num_classes = 1
model.fc = nn.Linear(model.fc.in_features, num_classes,bias=True) #***this model isn't built properly***
if train_contrastive_encoder:
    # Remove the final layer so we just get the encoder in order to perform contrastive learning
    # Then once encoder is trained we will freeze the weights and add a classifier layer
    # Basically this step uses the contrastive loss to adjust the weights of the encoder such that it maximizes the distance b/w the two groups, after which if we train/add a classifier we get much easier classification due to better separation
    # See details under Experiment 2 in https://keras.io/examples/vision/supervised-contrastive-learning/
    # Note this line specifically to remove the fc layer is from ChatGPT
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

if foundation == True and train_contrastive_encoder == False:
    model = initialize_radimagenet_resnet('RadImageNet_ResNet50.pt', 1, freeze_encoder_foundation)

# Move resnet to the device we stated earlier (GPU, mps, or CPU)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
if train_contrastive_encoder:
    criterion = SupervisedContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

wandb.login(key="3ecaafdc4daf42051628a9bdebf0debb6eb8a1c5")
run = wandb.init(project='bioengr-209-project',
                 config={
                     "model": model,
                     "save_model_path": save_model_path,
                     "epochs": num_epochs,
                     "batch_size": batch_num,
                     "input_dim": interp_resolution,
                     "model_complexity": encoder_complexity
                 },
                 name='test_run_complex1_2'
                 )

# Timer
start_time = time.time()

# Training loop
num_batches = len(train_dataloader) # Get total number of batches so we can split the training loop
for i in range(0, num_epochs):
    epoch_start_time = time.time()

    train_loop(train_dataloader, model, criterion, device, batch_num, optimizer, start_batch=0, end_batch=num_batches // 2)
    test_loop(test_dataloader, model, criterion, device)

    train_loop(train_dataloader, model, criterion, device, batch_num, optimizer, start_batch=num_batches // 2, end_batch=num_batches)
    test_loop(test_dataloader, model, criterion, device)

    elapsed_time = time.time() - epoch_start_time
    print("Epoch " + str(i + 1) + " complete at " + str(elapsed_time) + " seconds")

elapsed_time = time.time() - start_time
print('Total time: ' + str(elapsed_time) + ' seconds')

# Save our model
# See this tutorial for how to load our model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
torch.save(model.state_dict(), save_model_path)