from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.utils import cluster_traj
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RamDecoder(PredictionDecoder):

    def __init__(self, args):
        """
        Modified random walk model. Instead of using correlation to model edge weight, this method uses MLP to model the eedge weights.

        args to include:
        agg_type: 'combined' or 'sample_specific'. Whether we have a single aggregated context vector or sample-specific
        num_samples: int Number of trajectories to sample
        op_len: int Length of predicted trajectories
        lv_dim: int Dimension of latent variable
        encoding_size: int Dimension of encoded scene + agent context
        hidden_size: int Size of output mlp hidden layer
        num_clusters: int Number of final clustered trajectories to output

        """
        super().__init__()
        self.agg_type = args['agg_type']
        assert (args['agg_type'] == 'ram')
        self.horizon=args['heatmap_channel']
        # self.local_conc_range = int(args['conc_range']/args['resolution'])
        # self.agg_channel = args['agg_channel']
        # self.layers=[]
        # for i in range(args['decoder_depth']-1):
        #     self.layers.append(nn.Conv2d(self.agg_channel//(2**i), self.agg_channel//(2**(i+1)), kernel_size=1, stride=1, bias=False))
        #     self.layers.append(nn.LeakyReLU())
        # assert (self.agg_channel//(2**(args['decoder_depth']-1))> 0.99)##Make sure the last layer has the least number of channels
        # self.layers.append(nn.Conv2d(self.agg_channel//(2**(args['decoder_depth']-1)), 1, kernel_size=1, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        # self.decoding_net=nn.ModuleList(self.layers)

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
            attn_output_weights = inputs['node_connectivity']
            init_states=inputs['initial_states']
            if 'under_sampled_mask' in inputs:
                mask=inputs['under_sampled_mask']
            else:
                mask=None


        predictions=torch.zeros_like(init_states.unsqueeze(1)).repeat(1,self.horizon,1)
        prev_states=init_states.unsqueeze(1)

        for step in range(self.horizon):
            predictions[:,step]=torch.bmm(prev_states,attn_output_weights).squeeze(1)
            prev_states=predictions[:,step].unsqueeze(1)
        
        return {'pred': predictions,'mask': mask}
