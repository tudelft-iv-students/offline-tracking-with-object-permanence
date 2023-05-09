from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import min_ade


class ADE(Metric):
    """
    Minimum average displacement error for the top K trajectories.
    """
    def __init__(self, args: Dict):
        self.name = 'ade_' +args['target']
        self.target = args['target']

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute ADE for occlusion recovery
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        if self.target=='initial':
            traj = predictions['traj']
        elif self.target=='refine':
            traj = predictions['refined_traj']
        elif self.target=='endpoints':
            traj = predictions['endpoint_traj']
        else:
            raise Exception('Target needs to be one of {initial,endpoints or refine}')
        traj_gt = ground_truth['traj'][:,:,:-1] if type(ground_truth) == dict else ground_truth[:,:,:-1]

        # Useful params
        batch_size = traj.shape[0]
        num_pred_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        # Masks for variable length ground truth trajectories
        if self.target == 'endpoints':
            masks = torch.zeros(batch_size, sequence_length).to(traj.device)
            traj_gt = ground_truth['endpoints'][:,:,:-1]
        else:
            masks = predictions['mask'] if type(predictions) == dict and 'mask' in predictions.keys() \
                else torch.zeros(batch_size, sequence_length).to(traj.device)


        errs, _ = min_ade(traj, traj_gt, masks)


        return torch.mean(errs)
