import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from torchvision.transforms import v2
from pytorch_metric_learning import losses

# All this file is based on Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
'''
# Data augmentation for contrastive learning
# From https://pytorch.org/vision/stable/transforms.html
data_augmentation = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_encoder():
    resnet = models.resnet18(pretrained=False)

'''
# Obtained from https://www.kaggle.com/code/debarshichanda/pytorch-supervised-contrastive-learning
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        feature_vectors_sq = torch.squeeze(feature_vectors)

        # Normalize feature vectors
        feature_vectors_normalized_sq = F.normalize(feature_vectors_sq, p=2, dim=1)

        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized_sq, torch.transpose(feature_vectors_normalized_sq, 0, 1)
            ),
            self.temperature,
        )

        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))