import torch
import torch.nn as nn
import abc
from typing import Union, Dict
from return_device import get_freer_gpu

# Initialize device:
device = torch.device(get_freer_gpu() if torch.cuda.is_available() else "cpu")



class PredictionDecoder(nn.Module):
    """
    Base class for decoders for single agent prediction.
    Outputs K trajectories and/or their probabilities
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, agg_encoding: Union[torch.Tensor, Dict]) -> Union[torch.Tensor, Dict]:
        """
        Forward pass for prediction decoder
        :param agg_encoding: Aggregated context encoding
        :return outputs: K Predicted trajectories and/or their probabilities/scores
        """
        raise NotImplementedError()
