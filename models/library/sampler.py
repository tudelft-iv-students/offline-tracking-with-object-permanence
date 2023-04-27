"""
Algorithms for trajectory end point sampling from estimated heatmap
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class TorchModalitySampler(nn.Module):
    def __init__(self, n_targets: int, radius: float, upscale: int = 1, swap_rc: bool = False):
        """
        Greedy algorithm to sample N targets from heatmap with the highest area probability
        (Optimized version of ModalitySampler - up to 50 times faster)

        Args:
            n_targets: Number of targets to sample
            radius: Target radius to sum probality of heatmap
            swap_rc: Swap coordinates axis in output
        """
        super(TorchModalitySampler, self).__init__()
        self._n_targets = n_targets
        self._upscale = upscale
        self._radius = round(radius*upscale)
        self._reclen = 2*self._radius+1
        self._swap_rc = swap_rc
        self._square = radius*radius

        # components
        self._avgpool = nn.AvgPool2d(kernel_size=self._reclen, stride=1)

    @torch.no_grad()
    def forward(self, heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Input shape [B,C,H,W]
        batch_size = heatmap.shape[0]
        hm = torch.clone(heatmap)
        hm = TF.resize(hm, size=[self._upscale*hm.shape[-1], self._upscale*hm.shape[-1]])

        batch_end_points, batch_confidences = [], []
        for batch_index in range(batch_size):
            end_points, confidences = [], []
            for _ in range(self._n_targets):
                agg = self._avgpool(hm[batch_index])[0]
                agg_max = torch.max(agg)
                # print(agg_max)
                coords = torch.nonzero((agg == agg_max))[0]
                max_val=torch.max(hm[batch_index, 0, coords[0]:coords[0]+self._reclen, coords[1]:coords[1]+self._reclen])
                coords_hm  = torch.nonzero((hm[batch_index, 0] == max_val))[0]
                confidences.append(hm[batch_index, 0, coords[0]:coords[0]+self._reclen, coords[1]:coords[1]+self._reclen].sum().detach().item())
                hm[batch_index, 0, coords[0]:coords[0]+self._reclen, coords[1]:coords[1]+self._reclen] = 0.0
                
                end_points.append((coords_hm) / self._upscale)
                # confidences.append(agg_max.detach().item()*self._square)

            end_points = torch.stack(end_points)
            confidences = torch.tensor(confidences, dtype=torch.float32)
            batch_end_points.append(end_points)
            batch_confidences.append(confidences)

        final_end_points = torch.stack(batch_end_points)
        final_confidences = torch.stack(batch_confidences)

        if self._swap_rc:
            final_end_points = final_end_points[:, :, [1, 0]]
        return final_end_points, final_confidences


class FDESampler(nn.Module):
    def __init__(self, n_targets: int, n_iterations: int, sample_radius: float, resolution=0.25):
        """

        Args:
            n_targets:
            n_iterations:
        """
        super(FDESampler, self).__init__()
        self._n_targets = n_targets
        self.num_iters = n_iterations
        self.radius=sample_radius
        self.MR_sampler=TorchModalitySampler(n_targets,sample_radius)
        self.resolution=resolution
        
    @torch.no_grad()
    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        assert(heatmap.shape[1]==1)
        hm = heatmap.clone()
        B = hm.shape[0]
        clusters,pre_confidences = self.MR_sampler(heatmap)
        x_coord,y_coord=torch.meshgrid(torch.arange(488), 
                                        torch.arange(488))
        indices=torch.cat([x_coord.unsqueeze(-1),y_coord.unsqueeze(-1)],dim=-1).to(heatmap.device)
        indices=indices.flatten(0,1).unsqueeze(0)
        clusters=clusters.flatten(0,1).unsqueeze(1)

        for _ in range(self.num_iters):
            torch.cuda.empty_cache()
            d=torch.norm(indices-clusters,dim=-1).view(B,self._n_targets,-1)
            d=torch.clamp(d,min=1e-3)
            m,_ = torch.min(d,dim=1)
            mask = (d<(3/self.resolution)).float()
            new_clusters=torch.zeros_like(clusters).view(B,self._n_targets,-1)
            
            for k in range(self._n_targets):
                N=torch.sum(mask[:,k]*(hm.flatten(1)/d[:,k])*(m/d[:,k]),-1)
                new_clusters[:,k]=torch.sum(indices.repeat(B,1,1)*(mask[:,k]*(hm.flatten(1)/d[:,k])*(m/d[:,k])).unsqueeze(-1),-2)/N.unsqueeze(-1)
                
            clusters=new_clusters.flatten(0,1).unsqueeze(1)
        torch.cuda.empty_cache()
        new_clusters=torch.round(new_clusters).long()
        confidences=torch.zeros_like(pre_confidences)
        for b in range(B):
            for i in range(self._n_targets):
                confidences[b,i]=(hm[b,0,new_clusters[b,i,0]-self.radius:new_clusters[b,i,0]+self.radius,new_clusters[b,i,1]-self.radius:new_clusters[b,i,1]+self.radius]).sum()
        return new_clusters,confidences