"""
Basic building blocks
"""
from typing import Tuple, Union
import torch.nn as nn
import torch
from math import gcd


IntTuple = Union[Tuple[int], int]
def get_track_mask(agent_representation):
    tracks=agent_representation.view(-1,agent_representation.shape[-2],agent_representation.shape[-1])[:,:,-1]
    return (tracks.sum(dim=-1)>0)

class Home_temp_enocder(nn.Module):
    def __init__(self, conv1d_n_in, conv1d_n_out, gru_in_hidden_feat, gru_in_out_feat):
        super(Home_temp_enocder, self).__init__()
        self.conv1d=Res1d(conv1d_n_in, conv1d_n_out, kernel_size=3, stride=1)
        self.ugru=U_GRU(conv1d_n_out, gru_in_hidden_feat, gru_in_out_feat)
        self.out_dim=gru_in_out_feat

    def forward(self,tracks,mask_index=None,max_num=None):
        x=self.conv1d(tracks).permute(2,0,1)
        agent_embeddings=self.ugru(x)
        if mask_index is not None:
            agent_embedding=[torch.zeros(self.out_dim,device=tracks.device) for _ in range(max_num)]
            for i, h in zip(mask_index, agent_embeddings):
                agent_embedding[i] = h
            agent_embeddings=torch.stack(agent_embedding)
        return agent_embeddings

class U_GRU(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, horizon=5, compress=True):
        super(U_GRU, self).__init__()
        self.encoder1=nn.GRUCell(in_feat , hidden_feat)
        self.encoder2=nn.GRUCell(in_feat+hidden_feat , hidden_feat)
        self.compress=compress
        self.hidden_dim=hidden_feat
        self.out_dim=out_feat
        self.flatten=nn.Flatten()
        self.horizon=horizon
        if self.compress:
            self.decoder=leaky_MLP(horizon*hidden_feat,out_feat)
    def forward(self,temp_feature):
        assert(self.horizon==temp_feature.shape[0])
        num_tracks=temp_feature.shape[1]
        hx=torch.zeros([num_tracks,self.hidden_dim],device=temp_feature.device)
        inter_feat=[]
        x_agent_inv=torch.flip(temp_feature,(0,))
        for i in range(x_agent_inv.shape[0]):
            hx =  self.encoder1(x_agent_inv[i] , hx)
            inter_feat.append(hx)
        hidden_inv=torch.stack(inter_feat)
        hidden_states=torch.flip(hidden_inv,(0,))
        aug_x=torch.cat((temp_feature,hidden_states),-1)
        hx=torch.zeros([num_tracks,self.hidden_dim],device=temp_feature.device)
        feat_list=[]
        for i in range(aug_x.shape[0]):
            hx =  self.encoder2(aug_x[i] , hx)
            feat_list.append(hx)
        output_tensor=torch.stack(feat_list).transpose(0,1) ## [num_tracks,horizon,feat_dim]
        if self.compress:
            output_tensor=self.decoder(self.flatten(output_tensor))
        return output_tensor

class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='BN', ng=8, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU()

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            # self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            # self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        if self.act:
            out = self.relu(out)
        return out


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntTuple,
        stride: IntTuple = 1,
        padding: Union[int, str] = 'same',
        activate_relu: bool = True,
        bias=False,
        add_coord=True,
        **kwargs
    ):
        """
        Convolutional 2D layer combiner with 2d batch norm

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            activate_relu: Use ReLU as activation function
            **kwargs:
        """
        super(CNNBlock, self).__init__()
        if add_coord:
            self.addcoords = AddCoords(with_r=False)
            self.in_channels=in_channels+2
        self._conv = nn.Conv2d(self.in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs)
        self._batchnorm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()
        self._activate_relu = activate_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._batchnorm(self._conv(self.addcoords(x)))
        if self._activate_relu:
            x = self._relu(x)
        return x


class TransposeCNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntTuple,
        stride: IntTuple = 2,
        padding: int = 0,
        output_padding: int = 0,
        activate_relu: bool = True,
        **kwargs
    ):
        """
        Transpose convolutional 2D layer combined with 2D batch norm

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            output_padding: Output padding
            activate_relu: Use ReLU as activation function
        """
        super(TransposeCNNBlock, self).__init__()
        self._conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False, **kwargs),
                                    CNNBlock(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, activate_relu=activate_relu,bias=True))
        self._batchnorm = nn.BatchNorm2d(out_channels)
        self._lrelu = nn.LeakyReLU(0.1)
        self._activate_relu = activate_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._batchnorm(self._conv(x))
        if self._activate_relu:
            x = self._lrelu(x)
        return x

class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        return hidden_states
class leaky_MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None, norm='LN',ng=16):
        super(leaky_MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        if norm=='GN':
            self.normalizer=nn.GroupNorm(gcd(ng, out_features), out_features)
        else:
            self.normalizer = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.normalizer(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, negative_slope=0.05)
        return hidden_states

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret