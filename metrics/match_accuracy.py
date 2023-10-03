from metrics.metric import Metric
from typing import Dict, Union
import torch
import torch.nn as nn



class Match_accuracy(Metric):
    """
    binary focal loss for occlusion match
    """
    def __init__(self, args: Dict):
        self.name = 'Accuracy_' + args['target']
        self.target=args['target']


    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute binary focal loss for occlusion match
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        if self.target=='motion':
            x=predictions['scores'].squeeze(-1).clone()
        elif self.target=='map':
            x=predictions['map_scores'].squeeze(-1).clone()
        elif self.target=='baseline':
            x=predictions['baseline'].squeeze(-1).clone()
        elif self.target=='sum':
            x=(predictions['map_scores'].squeeze(-1).clone()+predictions['scores'].squeeze(-1).clone())/2
        mask=predictions['masks'][:,:,:,0]
        selections=torch.argmax(x,dim=-1)
        acc=(selections==0).sum()/x.shape[0]

        return acc
