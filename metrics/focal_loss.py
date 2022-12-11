from metrics.metric import Metric
from typing import Dict, Union
import torch
import torch.nn.functional as F

class FocalLoss(Metric):
    """
    Purely computes the classification component of the MTP loss.
    """

    def __init__(self):
        """
        Initialize CoverNetLoss
        """
        self.name = 'focal_loss'

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute focal loss
        :param predictions: Dictionary with 'pred': predicted heatmap
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """

        # Unpack arguments
        self.resolution=predictions['resolution']
        self.compensation=predictions['offset']
        pred = predictions['pred']
        mask = predictions['mask']
        
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth
        gt_map = self.generate_gtmap(traj_gt,pred.shape)
        # # Useful variables
        # batch_size = traj.shape[0]
        # sequence_length = traj.shape[2]

        # # Masks for variable length ground truth trajectories
        # masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
        #     else torch.zeros(batch_size, sequence_length).to(traj.device)

        # # Obtain mode with minimum MSE with respect to ground truth:
        # errs, inds = min_mse(traj, traj_gt, masks)

        # # Calculate NLL loss for trajectories corresponding to selected outputs (assuming model uses log_softmax):
        # loss = - torch.squeeze(probs.gather(1, inds.unsqueeze(1)))
        # loss = torch.mean(loss)

        return 

    def generate_gtmap(self, traj_gt: torch.Tensor,shape) -> torch.Tensor:
        swapped=torch.zeros_like(traj_gt)
        swapped[:,:,0],swapped[:,:,1]=-traj_gt[:,:,1],traj_gt[:,:,0]
        coord=torch.round(swapped/self.resolution+self.compensation).int()
        coord=torch.clamp(coord,0,shape[-1])
        gt_map=torch.zeros(shape)
        for batch in range(shape[0]):
            for t in range(shape[1]):
                x,y=coord[batch,t]
                gt_map[batch,t,x,y]=1
        return gt_map

    # def forward(self, pred_heatmap: torch.Tensor, true_heatmap: torch.Tensor, da_area: torch.Tensor) -> torch.Tensor:
    #     # noinspection PyUnresolvedReferences
    #     mask = (true_heatmap == 1).float()
    #     pred_heatmap = torch.clamp(pred_heatmap, min=1e-3, max=1-1e-3)

    #     return -torch.mean(
    #         da_area * torch.pow(pred_heatmap - true_heatmap, 2) * (
    #             mask * torch.log(pred_heatmap)
    #             +
    #             (1-mask) * (torch.pow(1 - true_heatmap, 4) * torch.log(1 - pred_heatmap))
    #         )
    #     )
    # def gaussian(self):
    #     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    #     return gauss/gauss.sum()
