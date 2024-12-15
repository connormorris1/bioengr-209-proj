import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

# All this file is based on Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362

# Loss function code from https://www.kaggle.com/code/debarshichanda/pytorch-supervised-contrastive-learning
# Ultimately not used in final paper
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