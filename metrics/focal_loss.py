from metrics.metric import Metric
from typing import Dict, Union
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def gaussian(window_size, sigma=1.5):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss#/gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window.to(device)

class FocalLoss(Metric):
    """
    Pixel-wise focal loss on heatmap
    """

    def __init__(self, args: Dict):
        """
        Initialize focal loss
        """
        self.name = 'focal_loss'
        self.window_size=args['window_size']
        self.resolution=args['resolution']
        self.compensation=(torch.Tensor([args['map_extent'][3],-args['map_extent'][0]]).to(device))/self.resolution
        self.gassian_blur=args['gauss_blur']
        self.horizon=args['horizon']
        self.window=create_window(self.window_size, self.horizon)

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute focal loss
        :param predictions: Dictionary with 'pred': predicted heatmap
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """

        # Unpack arguments
        pred = predictions['pred']
        mask_da = predictions['mask'].view(-1,pred.shape[-2],pred.shape[-1]).unsqueeze(1)
        
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth
        true_heatmap,gs_map = self.generate_gtmap(traj_gt,pred.shape)
        gs_map=gs_map*mask_da
        mask = (true_heatmap == 1).float()
        pred_heatmap = torch.clamp(pred, min=1e-4)

        return -torch.sum(
                torch.pow(pred_heatmap - gs_map, 2) * (
                mask * torch.log(pred_heatmap)
                +
                (1-mask) * (torch.pow(1 - gs_map, 4) * torch.log(1 - pred_heatmap))
            )
        )



    def generate_gtmap(self, traj_gt: torch.Tensor,shape) -> torch.Tensor:
        swapped=torch.zeros_like(traj_gt).to(device)
        swapped[:,:,0],swapped[:,:,1]=-traj_gt[:,:,1],traj_gt[:,:,0]
        coord=torch.round(swapped/self.resolution+self.compensation).int()
        coord=torch.clamp(coord,0,shape[-1])
        gt_map=torch.zeros(shape,device=device)
        for batch in range(shape[0]):
            for t in range(shape[1]):
                x,y=coord[batch,t]
                gt_map[batch,t,x,y]=1##Only one ground truth in each heatmap layer
        gs_map=F.conv2d(gt_map, self.window, padding = self.window_size//2, groups = self.horizon)##Gaussian smoothed heatmap
        return gt_map,gs_map

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
