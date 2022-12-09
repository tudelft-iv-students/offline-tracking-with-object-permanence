"""
Basic 2D CNN building blocks
"""
from typing import Tuple, Union
import torch.nn as nn
import torch


IntTuple = Union[Tuple[int], int]


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntTuple,
        stride: IntTuple = 1,
        padding: Union[int, str] = 'same',
        activate_relu: bool = True,
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
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs)
        self._batchnorm = nn.BatchNorm2d(out_channels)
        self._lrelu = nn.LeakyReLU(0.1)
        self._activate_relu = activate_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._batchnorm(self._conv(x))
        if self._activate_relu:
            x = self._lrelu(x)
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
        self._conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False, **kwargs)
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