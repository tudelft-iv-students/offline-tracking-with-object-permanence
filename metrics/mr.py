from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import miss_rate


class MissRate(Metric):
    """
    Miss rate for the top K trajectories.
    """

    def __init__(self, args: Dict):
        self.dist_thresh = args['dist_thresh']
        self.target=args['target']
        self.name = 'MR_'+args['target']

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute miss rate
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        traj_gt = ground_truth['traj'][:,:,:-1] if type(ground_truth) == dict else ground_truth[:,:,:-1]
        
        if self.target=='initial':
            traj = predictions['traj']
        elif self.target=='refine':
            traj = predictions['refined_traj']
        else:
            raise Exception('Target needs to be one of {initial, or refine}')

        # Useful params
        batch_size = traj.shape[0]
        num_pred_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        # Masks for variable length ground truth trajectories
        masks = predictions['mask'] if type(predictions) == dict and 'mask' in predictions.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)


        return miss_rate(traj, traj_gt, masks, dist_thresh=self.dist_thresh)
