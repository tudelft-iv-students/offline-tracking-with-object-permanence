from metrics.metric import Metric
from typing import Dict, Union
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
from train_eval.utils import return_device
device = return_device()

def gaussian(window_size, sigma=4.0):
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
        # self.compensation=torch.tensor([98,59]).to(device)
        self.gassian_blur=args['gauss_blur']
        self.horizon=args['horizon']
        
        self.type=args['type']
        self.W = int((args['map_extent'][1] - args['map_extent'][0])/self.resolution)
        self.H = int((args['map_extent'][3] - args['map_extent'][2])/self.resolution)
        self.reduce_map = args['reduce_map']
        self.normalize=args['normalize']
        self.only_last=args['only_last']
        if self.only_last:
            self.window=create_window(self.window_size, 1)
        else:
            self.window=create_window(self.window_size, self.horizon)

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute focal loss
        :param predictions: Dictionary with 'pred': predicted heatmap
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """

        # Unpack arguments
        if self.normalize:
            pred = predictions['pred']/(torch.max(predictions['pred'].detach(),dim=-1,keepdim=True)[0])
        else:
            pred = predictions['pred']
        # mask_da = predictions['mask'].view(-1,pred.shape[-2],pred.shape[-1]).unsqueeze(1)
        mask_da = predictions['mask']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth
        # true_heatmap,gs_map = self.generate_gtmap(traj_gt,pred.shape)
        true_heatmap,gs_map = self.generate_gtmap(traj_gt,mask_da)
        # gs_map=gs_map*mask_da
        # mask = (gs_map > 0.25).float()
        mask = (true_heatmap == 1.0).float()
        pred_heatmap = torch.clamp(pred, min=1e-6,max=(1-(1e-6)))
        if self.type=='mean':
            return -torch.mean(
                    torch.pow(pred_heatmap - gs_map, 2) * (
                    mask * torch.log(pred_heatmap)
                    +
                    (1-mask) * (torch.pow(1 - gs_map, 4) * torch.log(1 - pred_heatmap))
                )
            )
        else:
            return -torch.sum(
                    torch.pow(pred_heatmap - gs_map, 2) * (
                    mask * torch.log(pred_heatmap)
                    +
                    (1-mask) * (torch.pow(1 - gs_map, 4) * torch.log(1 - pred_heatmap))
                )
            )/max(1,mask_da.shape[0])



    def generate_gtmap(self, traj_gt: torch.Tensor,mask=None,visualize=False) -> torch.Tensor:
        if self.only_last:
            shape=[traj_gt.shape[0],1,self.H,self.W]
        else:
            shape=[traj_gt.shape[0],self.horizon,self.H,self.W]
        swapped=torch.zeros_like(traj_gt).to(device)
        swapped[:,:,0],swapped[:,:,1]=-traj_gt[:,:,1],traj_gt[:,:,0]
        coord=torch.round(swapped/self.resolution+self.compensation).int()
        coord=torch.clamp(coord,0,shape[-1])
        gt_map=torch.zeros(shape,device=device)
        if visualize:
            gt_map=torch.zeros([traj_gt.shape[0],traj_gt.shape[1],self.H,self.W],device=device)
            for batch in range(shape[0]):
                for t in range(traj_gt.shape[1]):
                    x,y=coord[batch,t]
                    try:
                        gt_map[batch,t,x-1:x+1,y-1:y+1]=1##Only one ground truth in each heatmap layer
                    except:
                        continue
            # gs_map=F.conv2d(gt_map, self.window, padding = self.window_size//2, groups = self.horizon)
            return gt_map
        if self.only_last:
            for batch in range(shape[0]):
                x,y=coord[batch,-1]
                gt_map[batch,0,x,y]=1
            gs_map=F.conv2d(gt_map, self.window, padding = self.window_size//2, groups = 1)
        else:
            for batch in range(shape[0]):
                for t in range(shape[1]):
                    x,y=coord[batch,t]
                    gt_map[batch,t,x,y]=1##Only one ground truth in each heatmap layer
            gs_map=F.conv2d(gt_map, self.window, padding = self.window_size//2, groups = self.horizon)##Gaussian smoothed heatmap
        if not self.reduce_map:
            return gt_map,gs_map
        gs_map=gs_map.view([gs_map.shape[0],gs_map.shape[1],-1])
        gt_map=gt_map.view([gs_map.shape[0],gs_map.shape[1],-1])
        max_num=max(mask.sum(dim=1))
        reduced_maps=[]
        reduced_gts=[]
        for i,batch in enumerate(gs_map):
            reduced_map=batch[mask[i].repeat(self.horizon,1)].view(self.horizon,-1)
            reduced_gt=gt_map[i][mask[i].repeat(self.horizon,1)].view(self.horizon,-1)
            aug_map=torch.cat((reduced_map, torch.zeros(self.horizon,max_num - reduced_map.size(1),device=device)), -1)
            aug_gt=torch.cat((reduced_gt, torch.zeros(self.horizon,max_num - reduced_map.size(1),device=device)), -1)
            reduced_maps.append(aug_map.unsqueeze(0))
            reduced_gts.append(aug_gt.unsqueeze(0))
        reduced_maps=torch.cat(reduced_maps,dim=0).to(device)
        reduced_gts=torch.cat(reduced_gts,dim=0).to(device)
        return reduced_gts,reduced_maps

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
