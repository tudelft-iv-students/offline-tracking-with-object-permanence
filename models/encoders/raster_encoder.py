from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
import numpy as np
# raise NotImplementedError()
from torchvision.models import resnet18, resnet34, resnet50
# raise NotImplementedError()
# from positional_encodings import PositionalEncodingPermute2D
from return_device import return_device
device = return_device()

from typing import Dict
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels
class RasterEncoder(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        CNN encoder for raster representation of HD maps and surrounding agent trajectories.

        args to include
            'backbone': str CNN backbone to use (resnet18, resnet34 or resnet50)
            'input_channels': int Size of scene features at each grid cell
            'use_positional_encoding: bool Whether or not to add positional encodings to final set of features
            'target_agent_feat_size': int Size of target agent state
        """

        super().__init__()

        # Anything more seems like overkill
        resnet_backbones = {'resnet18': resnet18,
                            'resnet34': resnet34,
                            'resnet50': resnet50}

        # Initialize backbone:
        resnet_model = resnet_backbones[args['backbone']](pretrained=False)
        conv1_new = nn.Conv2d(args['input_channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modules = list(resnet_model.children())[:-2]
        modules[0] = conv1_new
        self.backbone = nn.Sequential(*modules)

        # Positional encodings:
        num_channels = 2048 if self.backbone == 'resnet50' else 512
        self.use_pos_enc = args['use_positional_encoding']
        if self.use_pos_enc:
            self.pos_enc = PositionalEncodingPermute2D(num_channels)

        # Linear layer to embed target agent representation.
        self.target_agent_encoder = nn.Linear(args['target_agent_feat_size'], args['target_agent_enc_size'])
        self.relu = nn.ReLU()

    def forward(self, inputs: Dict) -> Dict:

        """
        Forward pass for raster encoder
        :param inputs: Dictionary with
            target_agent_representation: torch.Tensor with target agent state, shape[batch_size, target_agent_feat_size]
            surrounding_agent_representation: Rasterized BEV representation, shape [batch_size, 3, H, W]
            map_representation: Rasterized BEV representation, shape [batch_size, 3, H, W]
        :return encodings: Dictionary with
            'target_agent_encoding': torch.Tensor of shape [batch_size, 3],
            'context_encoding': torch.Tensor of shape [batch_size, N, backbone_feat_dim]
        """

        # Unpack inputs:
        target_agent_representation = (inputs['target_agent_representation']).type(torch.float32).to(device)
        surrounding_agent_representation = inputs['surrounding_agent_representation'].to(device)
        map_representation = inputs['map_representation'][0].to(device)
        
        
        # Apply Conv layers
        rasterized_input = torch.cat((map_representation, surrounding_agent_representation), dim=1).type(torch.float32)
        context_encoding = self.backbone(rasterized_input)

        # Add positional encoding
        if self.use_pos_enc:
            context_encoding = context_encoding + self.pos_enc(context_encoding)

        # Reshape to form a set of features
        # context_encoding = context_encoding.view(context_encoding.shape[0], context_encoding.shape[1], -1)## [Batch number, channel, H*W]
        # context_encoding = context_encoding.permute(0, 2, 1)

        # Target agent encoding
        
        target_agent_enc = self.relu(self.target_agent_encoder(target_agent_representation))

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc,
                     'context_encoding': {'combined': context_encoding,
                                          'combined_masks': None,
                                          'map': rasterized_input,
                                          'vehicles': None,
                                          'pedestrians': None,
                                          'map_masks': inputs['map_representation'][1].type(torch.bool),
                                          'vehicle_masks': None,
                                          'pedestrian_masks': None
                                          },
                     }
        if inputs['gt_traj'] is  not None:
            encodings['gt_traj']= inputs['gt_traj']
        else:
            encodings['gt_traj']= None
        return encodings
