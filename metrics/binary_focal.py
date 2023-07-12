from metrics.metric import Metric
from typing import Dict, Union
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.BCE=nn.BCELoss(reduction='none')

    def forward(self, input, target):
        x=(input+1)/2
        ce_loss = self.BCE(x, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)
        return focal_loss

class Binary_focal_loss(Metric):
    """
    binary focal loss for occlusion match
    """
    def __init__(self, args: Dict):
        self.name = 'Binary_focal_loss_' + args['target']
        self.alpha = args['alpha']
        self.gamma = args['gamma']
        self.reduction = 'mean'
        self.target=args['target']
        self.BCE=nn.BCELoss(reduction='none')

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute binary focal loss for occlusion match
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        if self.target=='motion':
            x=predictions['scores']
        elif self.target=='map':
            x=predictions['map_scores']
        mask=(~(predictions['masks'][:,:,:,0]).bool()).any(dim=-1)
        target=ground_truth
        
        ce_loss = self.BCE(x, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).squeeze(-1)
        
        focal_loss=torch.sum(focal_loss[mask])

        return focal_loss
