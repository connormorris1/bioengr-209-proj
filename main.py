import torch
import torch.nn as nn
from torchvision import models
from CustomImageDataset import CustomImageDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_and_test_loop import train_loop, test_loop
import time

############################################################################

# Path to directory containing dicom files
# Expected format of these files: csv files where each line is path_to_dicom, label
labels_train = '/home/cjmorris/repos/bioengr-209-proj/data_paths/all_train.csv'
labels_test = '/home/cjmorris/repos/bioengr-209-proj/data_paths/all_test.csv'
interp_resolution = 224 # Resnets expect a 224x224 image

batch_num = 100 # Batch size
learning_rate = 0.001
num_epochs = 3

pretrained = False # Set this to True if you want to use the pretrained version

# There are three different encoder models: Resnet18, Resnet34, and Resnet50
# Set this to 0 for Resnet18, 1 for Resnet34, and 2 for Resnet50
# The higher the complexity, the longer the training time but the better the performance on complex tasks
# Suggestion: Use Resnet18 to prototype and Resnet 34 or 50 for final model
encoder_complexity = 0

############################################################################

# Produces DataLoader from our data
# This Custom Data Loader first handles conversion of .dicom to tensor objects
# Balance training data (so positive and negative labels equal) but don't do so for test data
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
    "cuda:0"
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

# Modify the final fully connected layer for 2 classes (single ventricle or not) ***only need 1 prediction - 0 is not, 1 is single ventricle***
num_classes = 1
model.fc = nn.Linear(model.fc.in_features, num_classes,bias=True) #***this model isn't built properly***
# Move resnet to the device we stated earlier (GPU, mps, or CPU)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Timer
start_time = time.time()

# Training loop
for i in range(0, num_epochs):
    epoch_start_time = time.time()

    train_loop(train_dataloader, model, criterion, device, batch_num, optimizer) #***will want to structure our training loop to see equal numbers of SVP and normal examples each epoch (oversample SVP cases)***
    test_loop(test_dataloader, model, criterion, device)

    elapsed_time = time.time() - epoch_start_time
    print("Epoch " + str(i + 1) + " complete at " + str(elapsed_time) + " seconds")

elapsed_time = time.time() - start_time
print('Total time: ' + str(elapsed_time) + ' seconds')

# Save our model
# See this tutorial for how to load our model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
torch.save(model.state_dict(), 'resnet_weights.pth')