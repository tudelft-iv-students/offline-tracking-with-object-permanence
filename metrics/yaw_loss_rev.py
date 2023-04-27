from metrics.metric import Metric
from typing import Dict, Union
import torch



class Yaw_loss_REV(Metric):
    """
    Minimum average displacement error for the top K trajectories.
    """
    def __init__(self, args: Dict):
        self.name = 'yaw_loss_rev' 

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute Yaw loss
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        yaw = predictions['yaw_rev'].squeeze(-1)
        yaw_gt = ground_truth['traj_rev'][:,:,-1] if type(ground_truth) == dict else ground_truth[:,:,-1]

        # Useful params
        batch_size = yaw.shape[0]
        sequence_length = yaw.shape[1]

        # Masks for variable length ground truth trajectories
        masks = predictions['mask'] if type(predictions) == dict and 'mask' in predictions.keys() \
            else torch.zeros(batch_size, sequence_length).to(yaw.device)
        indices=torch.abs(yaw-yaw_gt)>(3.14159/2)
        # multipliers=torch.abs(yaw-yaw_gt)//(3.14159/2)
        larger_inds=(yaw_gt>yaw)*indices
        smaller_inds=(yaw_gt<yaw)*indices
        yaw_gt[larger_inds]-=(3.14159/2)
        yaw_gt[smaller_inds]+=(3.14159/2)

        errs=torch.sum(torch.abs(yaw-yaw_gt)*(1-masks),dim=1)/torch.sum((1-masks),dim=1)


        return torch.mean(errs)
