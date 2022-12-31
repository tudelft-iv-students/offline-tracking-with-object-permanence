from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.utils import cluster_traj
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class HTMAP(PredictionDecoder):

    def __init__(self, args):
        """
        Naive decoding network which uses 1*1 conv to compress the dimension and directly output
        heatmaps for all time steps

        args to include:
        agg_type: 'home_agg' just to make sure the aggregator is the one implemeneted by myself
        resolution: float Grid size
        agg_channel: int The number of channels of input tensor
        lv_dim: int Dimension of latent variable
        encoding_size: int Dimension of encoded scene + agent context
        hidden_size: int Size of output mlp hidden layer
        num_clusters: int Number of final clustered trajectories to output

        """
        super().__init__()
        self.agg_type = args['agg_type']
        assert (args['agg_type'] == '2D_sample')
        self.resolution=args['resolution']
        # self.local_conc_range = int(args['conc_range']/self.resolution)
        self.agg_channel = args['agg_channel']
        self.layers=[]
        for i in range(args['decoder_depth']-1):
            self.layers.append(nn.Conv2d(self.agg_channel//(2**i), self.agg_channel//(2**(i+1)), kernel_size=1, stride=1, bias=False))
            self.layers.append(nn.LeakyReLU())
        assert (self.agg_channel//(2**(args['decoder_depth']-1))> args['heatmap_channel']-0.01)##Make sure the last layer has the least number of channels
        self.layers.append(nn.Conv2d(self.agg_channel//(2**(args['decoder_depth']-1)), args['heatmap_channel'], kernel_size=1, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        self.decoding_net=nn.ModuleList(self.layers)
        self.softmax=nn.Softmax(dim=-1)
        

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        """
        Forward pass for latent variable model.

        :param inputs: aggregated context encoding,
         shape for combined encoding: [batch_size, channel, H, W]
        :return: predictions
        """

        if type(inputs) is torch.Tensor:
            agg_encoding = inputs
        else:
            agg_encoding = inputs['agg_encoding']
            if 'under_sampled_mask' in inputs:
                mask=inputs['under_sampled_mask']
            else:
                mask=None

        x=agg_encoding
        for i,layer in enumerate(self.decoding_net):
            x=layer(x)
        predictions=x.view(x.shape[0],x.shape[1],-1)
        predictions=self.softmax(predictions).view(x.shape)
        return {'pred': predictions,'mask': mask}
