from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.utils import cluster_traj
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RAM(PredictionDecoder):

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
        assert (args['agg_type'] == 'home_agg')
        self.local_conc_range = int(args['conc_range']/args['resolution'])
        self.agg_channel = args['agg_channel']
        self.layers=[]
        for i in range(args['decoder_depth']):
            self.layers.append(nn.Conv2d(self.agg_channel//(2**i), self.agg_channel//(2**(i+1)), kernel_size=1, stride=1, bias=False))
            self.layers.append(nn.LeakyReLU())
        self.decoding_net=nn.ModuleList([self.layers])

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        """
        Forward pass for latent variable model.

        :param inputs: aggregated context encoding,
         shape for combined encoding: [batch_size, encoding_size]
         shape if sample specific encoding: [batch_size, num_samples, encoding_size]
        :return: predictions
        """

        if type(inputs) is torch.Tensor:
            agg_encoding = inputs
        else:
            agg_encoding = inputs['agg_encoding']

        if self.agg_type == 'combined':
            agg_encoding = agg_encoding.unsqueeze(1).repeat(1, self.num_samples, 1)
        else:
            if len(agg_encoding.shape) != 3 or agg_encoding.shape[1] != self.num_samples:
                raise Exception('Expected ' + str(self.num_samples) + 'encodings for each train/val data')

        # Sample latent variable and concatenate with aggregated encoding
        batch_size = agg_encoding.shape[0]
        z = torch.randn(batch_size, self.num_samples, self.lv_dim, device=device)
        agg_encoding = torch.cat((agg_encoding, z), dim=2)
        h = self.leaky_relu(self.hidden(agg_encoding))

        # Output trajectories
        traj = self.op_traj(h)
        traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # Cluster
        traj_clustered, probs = cluster_traj(self.num_clusters, traj)

        predictions = {'traj': traj_clustered, 'probs': probs}

        if type(inputs) is dict:
            for key, val in inputs.items():
                if key != 'agg_encoding':
                    predictions[key] = val

        return predictions
