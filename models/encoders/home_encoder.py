from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, resnet34, resnet50
from models.library.blocks import *
from train_eval.utils import return_device
import itertools
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
class HomeEncoder(PredictionEncoder):

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
        if args['backbone']=='home':
            self.backbone = Home_conv(args['input_channels'])
        else:
            resnet_model = resnet_backbones[args['backbone']](pretrained=False)
            conv1_new = nn.Conv2d(args['input_channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            modules = list(resnet_model.children())[:-2]
            modules[0] = conv1_new
            self.backbone = nn.Sequential(*modules)

        self.hidden_dim=args['hidden_dim']
        self.emb_size=args['emb_size']
        self.num_heads=args['num_heads']
        self.traj_forecaster=args['traj_forecaster']
        self.target_agent_encoder=Home_temp_enocder(conv1d_n_in=7, conv1d_n_out=64, gru_in_hidden_feat=self.hidden_dim, gru_in_out_feat=self.emb_size)
        self.agent_encoder=Home_temp_enocder(conv1d_n_in=7, conv1d_n_out=64, gru_in_hidden_feat=self.hidden_dim, gru_in_out_feat=self.emb_size)
        self.social_encoder=Home_social_enocder(self.emb_size,self.num_heads)
        if self.traj_forecaster:
            self.target_traj_encoder=leaky_MLP(args['target_agent_feat_size'],args['target_agent_enc_size'])
        # Positional encodings:
        num_channels = 2048 if self.backbone == 'resnet50' else 512
        self.use_pos_enc = args['use_positional_encoding']
        if self.use_pos_enc:
            self.pos_enc = PositionalEncodingPermute2D(num_channels)

        

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
        target_agent_representation = (inputs['target_agent_representation']).type(torch.float32)
        surrounding_agent_representation = inputs['surrounding_agent_representation']['image']
        agent_representation = torch.cat((inputs['surrounding_agent_representation']['vehicles'],inputs['surrounding_agent_representation']['pedestrians']),dim=1)
        map_representation = inputs['map_representation']['image']
        
        # Apply Conv layers
        # rasterized_input = torch.cat((map_representation, surrounding_agent_representation), dim=1).type(torch.float32)
        rasterized_input = torch.cat((map_representation, surrounding_agent_representation,inputs['map_representation']['mask'].unsqueeze(1)), dim=1).type(torch.float32)
        context_encoding = self.backbone(rasterized_input)

        # Add positional encoding
        if self.use_pos_enc:
            context_encoding = context_encoding + self.pos_enc(context_encoding)

        # Reshape to form a set of features
        # context_encoding = context_encoding.view(context_encoding.shape[0], context_encoding.shape[1], -1)## [Batch number, channel, H*W]
        # context_encoding = context_encoding.permute(0, 2, 1)

        # Target agent encoding
        if self.traj_forecaster:
            traj_feat=self.target_traj_encoder(target_agent_representation[:,-1,2:-2])
        else:
            traj_feat=target_agent_representation
        agent_tracks_full = agent_representation.view(-1,agent_representation.shape[-2],agent_representation.shape[-1]).transpose(-1,-2) 
        agent_masks = get_track_mask(agent_representation)
        target_track= target_agent_representation.transpose(-1,-2)
        target_emb = self.target_agent_encoder(target_track)
        if agent_masks.sum() !=0:
            mask_index = list(itertools.compress(range(len(agent_masks)), agent_masks))
            agent_tracks= torch.stack(list(itertools.compress(agent_tracks_full, agent_masks)), dim=0)
            agent_emb = self.agent_encoder(agent_tracks,mask_index,max_num=agent_tracks_full.shape[0]).view(agent_representation.shape[0],-1,self.emb_size)
            target_agent_enc=self.social_encoder(target_emb,agent_emb,agent_masks.view(target_agent_representation.shape[0],-1))
        else:
            target_agent_enc=target_emb
            

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc,
                     'context_encoding': {'combined': context_encoding,
                                          'combined_masks': None,
                                          'map': None,
                                          'vehicles': None,
                                          'pedestrians': None,
                                          'map_masks': inputs['map_representation']['mask'].type(torch.bool),
                                          'vehicle_masks': None,
                                          'pedestrian_masks': None,
                                          'traj_feature':traj_feat
                                          },
                     }
        if inputs['gt_traj'] is  not None:
            encodings['gt_traj']= inputs['gt_traj']
        else:
            encodings['gt_traj']= None
        return encodings

class Home_social_enocder(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(Home_social_enocder, self).__init__()
        self.attention=nn.MultiheadAttention(emb_size, num_heads)
        self.normalizer=LayerNorm(emb_size)
        self.num_heads=num_heads
    def forward(self,target_emb,agent_emb,agent_mask):
        indices=torch.nonzero(agent_mask.sum(-1)).squeeze(1)
        filtered_agent_emb=agent_emb[indices]
        filtered_agent_mask=agent_mask[indices]
        filtered_target_emb=target_emb[indices]
        attn_mask=~(filtered_agent_mask.repeat(1,self.num_heads).view(filtered_agent_mask.shape[0]*self.num_heads,-1).unsqueeze(1))
        target_query=filtered_target_emb.unsqueeze(0)
        agnet_key_value=filtered_agent_emb.transpose(0,1)
        attn_output,_=self.attention(query=target_query, key=agnet_key_value, value=agnet_key_value, need_weights=False, attn_mask=attn_mask)
        # attn_output[torch.isnan(attn_output)]=0
        attn_output=attn_output.squeeze(0)
        target_agent_enc=target_emb.clone()
        for idx, enc in zip(indices, attn_output):
            target_agent_enc[idx] = target_agent_enc[idx]+ enc
        return self.normalizer(target_agent_enc)

class Home_conv(nn.Module):
    def __init__(self, input_channels=7):
        super(Home_conv, self).__init__()
        self.conv_layers=nn.ModuleList(
            [CNNBlock(in_channels=input_channels, out_channels=32,kernel_size=3,stride=1,padding=0),
            CNNBlock(in_channels=32, out_channels=64,kernel_size=3,stride=1,padding=1),
            CNNBlock(in_channels=64, out_channels=128,kernel_size=3,stride=1,padding=0),
            CNNBlock(in_channels=128, out_channels=256,kernel_size=3,stride=1,padding=1),
            CNNBlock(in_channels=256, out_channels=512,kernel_size=3,stride=1,padding=1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,raster_img):
        x=raster_img
        for layer in self.conv_layers[:-1]:
            x=self.pool(layer(x))
        x=self.conv_layers[-1](x)
        return x

# class Home_conv(nn.Module):
#     def __init__(self, input_channels=6):
#         super(Home_conv, self).__init__()
#         self.conv_layers=nn.Sequential(
#             CNNBlock(in_channels=input_channels, out_channels=32,kernel_size=3,stride=2,padding=1),
#             CNNBlock(in_channels=32, out_channels=64,kernel_size=3,stride=2,padding=1),
#             CNNBlock(in_channels=64, out_channels=128,kernel_size=3,stride=1,padding=1),
#             CNNBlock(in_channels=128, out_channels=256,kernel_size=3,stride=2,padding=1),
#             CNNBlock(in_channels=256, out_channels=512,kernel_size=3,stride=2,padding=1)
#         )

#     def forward(self,raster_img):
        
#         return self.conv_layers(raster_img)