from skimage.transform import resize
from typing import Dict
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sampler():
    """
    Sample from a goals from a binary raster mask.
    """

    def __init__(self, args: Dict, resolution=None, apply_mask=False):
        """
        Initialize predict helper, agent and scene representations
        :param args: Dataset arguments
        should include 'img_size','map_extent'
        """

        # Raster parameters
        self.img_size = args['img_size']
        self.map_extent = args['map_extent']
        self.apply_mask = apply_mask##If apply mask, the returned goals are reduced to a subset located inside drivable area
        # Raster map with agent boxes
        if resolution == None:
            self.resolution = (self.map_extent[1] - self.map_extent[0]) / self. img_size[1]
            self.resize = False
        else:
            self.resolution = resolution ## Unit m/pixel
            self.resize = True
        self.W = int((self.map_extent[1] - self.map_extent[0])/self.resolution)
        self.H = int((self.map_extent[3] - self.map_extent[2])/self.resolution)

    def sample_mask(self,mask):
        """
        :param mask: mask raster with shape [B,H,W]
        :output binary_mask: A binary tensor indicating which goals are non-drivable, which are drivable 
        """

        if self.resize:
            np_mask=np.array(mask.permute(1,2,0))
            np_mask=resize(np_mask,[self.H,self.W]).transpose(2,0,1)
            binary_mask=torch.tensor(np_mask).type(torch.bool)
            binary_mask=binary_mask.view(binary_mask.shape[0],-1)

        
        return binary_mask.to(device)
    
    def sample_goals(self,mask=None):
        """
        :param mask: mask raster with shape [B,H,W]
        :output nodes_2D: A list of 2D coordinates of sampled goals  
        """
        x_coord,y_coord=torch.meshgrid(torch.arange(self.map_extent[3],self.map_extent[2],-self.resolution),
                                       torch.arange(self.map_extent[0],self.map_extent[1],self.resolution))
        if self.apply_mask and (mask != None):
            B=mask.shape[0]
            if self.resize:
                np_mask=np.array(mask.permute(1,2,0))
                np_mask=resize(np_mask,[self.H,self.W]).transpose(2,0,1)
                mask=torch.tensor(np_mask).type(torch.bool)
            else:
                mask=mask.type(torch.bool)
        
        
            nodes_2D=[]
            for batch in range(B):
                valid_x=x_coord[mask[batch]]
                valid_y=y_coord[mask[batch]]
                nodes=torch.cat((valid_x.unsqueeze(0),valid_y.unsqueeze(0)),dim=0).T
                nodes_2D.append(nodes)
        else:
            nodes_2D=torch.cat((x_coord.unsqueeze(0),y_coord.unsqueeze(0)),dim=0).view([2,-1]).T
        return nodes_2D.to(device)