from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.ram_decoder import get_dense,get_index
from train_eval.utils import return_device
from models.library.sampler import TorchModalitySampler
from models.library.blocks import LayerNorm
device = return_device()


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

        self.endpoint_sampler=TorchModalitySampler(args['num_target'],args['sample_radius'])
        self.layers=[]
        for i in range(args['decoder_depth']-1):
            self.layers.append(nn.Conv2d(self.agg_channel//(2**i), self.agg_channel//(2**(i+1)), kernel_size=1, stride=1, bias=False))
            self.layers.append(nn.LeakyReLU())
        assert (self.agg_channel//(2**(args['decoder_depth']-1))> args['heatmap_channel']-0.01)##Make sure the last layer has the least number of channels
        self.layers.append(nn.Conv2d(self.agg_channel//(2**(args['decoder_depth']-1)), args['heatmap_channel'], kernel_size=1, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        self.decoding_net=nn.ModuleList(self.layers)
        self.softmax=nn.Softmax(dim=-1)
        self.map_extent = args['map_extent']
        self.horizon=args['heatmap_channel']
        self.resolution=args['resolution']
        self.W = int((self.map_extent[1] - self.map_extent[0])/self.resolution)
        self.H = int((self.map_extent[3] - self.map_extent[2])/self.resolution)
        self.compensation=torch.round((torch.Tensor([args['map_extent'][3],-args['map_extent'][0]]))/self.resolution).int()
        self.output_traj = args['output_traj']
        self.pretrain_mlp = args['pretrain_mlp']
        if self.output_traj:
            self.diff_emb=args['diff_emb']
            self.feature_dim=args['target_agent_enc_size']+args['agg_channel']+self.diff_emb
            self.num_target=args['num_target']
            self.endpoint_sampler=TorchModalitySampler(args['num_target'],args['sample_radius'])
            self.diff_encoder=nn.Linear(2,self.diff_emb)
            self.decoder = nn.Sequential(nn.Linear(self.feature_dim,self.feature_dim//2),
                                        LayerNorm(self.feature_dim//2),
                                        nn.ReLU(),
                                        nn.Linear(self.feature_dim//2,self.horizon*2))
        

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
        if self.output_traj:
                target_encodings = inputs['target_encodings']

        
        if self.pretrain_mlp:
            map_feature=agg_encoding.permute(0,2,3,1)
            gt_traj=inputs['gt_traj']
            final_points=gt_traj[:,-1].unsqueeze(1)
            swapped=torch.zeros_like(final_points,device=gt_traj.device)
            swapped[:,:,0],swapped[:,:,1]=-final_points[:,:,1],final_points[:,:,0]
            endpoints=torch.round(swapped/self.resolution+self.compensation.to(swapped.device)).long()
            concat_feature=torch.empty([0,1,self.feature_dim],device=map_feature.device)
            for batch_idx in range(len(gt_traj)):
                map_feat = (map_feature[batch_idx])[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]
                diff = final_points[batch_idx]
                # diff = (endpoints[batch_idx]-self.compensation.to(device))*self.resolution
                # diff[:.0] = -diff[:.0]
                feature=torch.cat([map_feat,self.diff_encoder(diff),target_encodings[batch_idx].unsqueeze(0)],dim=-1).unsqueeze(0)
                concat_feature=torch.cat([concat_feature,feature],dim=0)
            trajectories=self.decoder(concat_feature).view(gt_traj.shape[0],1,self.horizon,2)
            torch.cuda.empty_cache()
            return {'pred': None,'mask': mask,'traj': trajectories,'probs':torch.ones([trajectories.shape[0],1],device=device),'endpoints':endpoints}
        
        
        x=agg_encoding
        for i,layer in enumerate(self.decoding_net):
            x=layer(x)
        predictions=x.view(x.shape[0],x.shape[1],-1)
        predictions=self.softmax(predictions).view(x.shape)
        
        if self.output_traj:
            map_feature=agg_encoding.permute(0,2,3,1)
            # nodes_2D=get_index(predictions[:,-1].unsqueeze(1),mask,self.H,self.W)
            dense_pred=predictions[:,-1].unsqueeze(1)
            endpoints,confidences = self.endpoint_sampler(dense_pred)
            endpoints=endpoints.long()
            concat_feature=torch.empty([0,self.num_target,self.feature_dim],device=predictions.device)
            x_coord,y_coord=torch.meshgrid(torch.arange(self.map_extent[-1],self.map_extent[-2],-self.resolution), ##### SHould be changed when image size changes
                                            torch.arange(self.map_extent[0],self.map_extent[1],self.resolution))
            indices=torch.cat([x_coord.unsqueeze(-1),y_coord.unsqueeze(-1)],dim=-1).to(predictions.device)
            for batch_idx in range(len(dense_pred)):
                map_feat = (map_feature[batch_idx])[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]
                diff = (indices[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]).float()
                # diff = (endpoints[batch_idx]-self.compensation.to(device))*self.resolution
                # diff[:.0] = -diff[:.0]
                feature=torch.cat([map_feat,self.diff_encoder(diff),target_encodings[batch_idx].repeat(self.endpoint_sampler._n_targets,1)],dim=-1).unsqueeze(0)
                concat_feature=torch.cat([concat_feature,feature],dim=0)
            trajectories=self.decoder(concat_feature).view(dense_pred.shape[0],self.num_target,self.horizon,2)
            torch.cuda.empty_cache()
            return {'pred': predictions,'mask': mask,'traj': trajectories,'probs':confidences,'endpoints':endpoints}
        return {'pred': predictions,'mask': mask}
