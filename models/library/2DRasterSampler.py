from skimage.transform import resize
from typing import Dict
import torch
import numpy as np
class Sampler():
    """
    Sample from a goals from a binary raster mask.
    """

    def __init__(self, args: Dict, resolution=None):
        """
        Initialize predict helper, agent and scene representations
        :param args: Dataset arguments
        should include 'img_size','map_extent'
        """

        # Raster parameters
        self.img_size = args['img_size']
        self.map_extent = args['map_extent']

        # Raster map with agent boxes
        if resolution == None:
            self.resolution = (self.map_extent[1] - self.map_extent[0]) / self. img_size[1]
            self.resize = False
        else:
            self.resolution = resolution ## Unit m/pixel
            self.resize = True
    
    def sample(self,mask):
        """
        :param mask: mask raster with shape [B,H,W]
        :output nodes_2D: A list of 2D coordinates of sampled goals  
        """
        B=mask.shape[0]
        device=mask.device
        if self.resize:
            np_mask=np.array(mask.permute(1,2,0))
            W = int((self.map_extent[1] - self.map_extent[0])/self.resolution)
            H = int((self.map_extent[3] - self.map_extent[2])/self.resolution)
            np_mask=resize(np_mask,[H,W]).transpose(2,0,1)
            mask=torch.tensor(np_mask).type(torch.bool).to(device)
        else:
            mask=mask.type(torch.bool)
        x_coord,y_coord=torch.meshgrid(torch.arange(self.map_extent[3],self.map_extent[2],-self.resolution),
                                       torch.arange(self.map_extent[0],self.map_extent[1],self.resolution))
        nodes_2D=[]
        for batch in range(B):
            valid_x=x_coord[mask[batch]]
            valid_y=y_coord[mask[batch]]
            nodes=torch.cat((valid_x.unsqueeze(0),valid_y.unsqueeze(0)),dim=0).T
            nodes_2D.append(nodes)
        return nodes_2D