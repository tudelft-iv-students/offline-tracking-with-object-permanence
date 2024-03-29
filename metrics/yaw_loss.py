from metrics.metric import Metric
from typing import Dict, Union
import torch



class Yaw_loss(Metric):
    """
    Minimum average displacement error for the top K trajectories.
    """
    def __init__(self, args: Dict):
        self.name = 'yaw_' + args['target']
        self.target=args['target']
        if 'add_quadratic' in args:
            self.add_quadratic=True
        else:
            self.add_quadratic=False

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute Yaw loss
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        if self.target=='initial':
            yaw = predictions['yaw'].squeeze(-1)
        elif self.target=='refine':
            yaw = predictions['refined_yaw'].squeeze(-1)
        elif self.target=='endpoints':
            yaw = predictions['endpoint_yaw'].squeeze(-1)
        elif self.target=='pre-refine':
            yaw = predictions['init_yaw'].squeeze(-1)
        else:
            raise Exception('Target needs to be one of {initial, endpoints, pre_refine or refine}')
        
        yaw_gt = ground_truth['traj'][:,:,-1] if type(ground_truth) == dict else ground_truth[:,:,-1]

        # Useful params
        batch_size = yaw.shape[0]
        sequence_length = yaw.shape[1]

        # Masks for variable length ground truth trajectories
        masks = predictions['mask'] if type(predictions) == dict and 'mask' in predictions.keys() \
            else torch.zeros(batch_size, sequence_length).to(yaw.device)
        if self.target=='endpoints':
            yaw_gt = ground_truth['endpoints'][:,:,-1]
            masks = torch.zeros(batch_size, sequence_length).to(yaw.device)
        indices=torch.abs(yaw-yaw_gt)>(3.14159/2)
        # multipliers=torch.abs(yaw-yaw_gt)//(3.14159/2)
        larger_inds=(yaw_gt>yaw)*indices
        smaller_inds=(yaw_gt<yaw)*indices
        yaw_gt[larger_inds]-=(3.14159/2)
        yaw_gt[smaller_inds]+=(3.14159/2)
        
        if self.add_quadratic:
            errs=torch.sum((torch.abs(yaw-yaw_gt)+torch.pow(yaw-yaw_gt, exponent=2))*(1-masks),dim=1)/torch.sum((1-masks),dim=1)
        else:
            errs=torch.sum(torch.abs(yaw-yaw_gt)*(1-masks),dim=1)/torch.sum((1-masks),dim=1)


        return torch.mean(errs)
