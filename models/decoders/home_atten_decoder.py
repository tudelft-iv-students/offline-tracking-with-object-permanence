from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
# from models.decoders.ram_decoder import get_dense,get_index
from return_device import return_device
from models.library.sampler import TorchModalitySampler
from models.library.blocks import LayerNorm,MLP
# from models.decoders.utils import get_probs
from models.encoders.raster_encoder import PositionalEncodingPermute2D
device = return_device()


class HomeDecoder_attn(PredictionDecoder):

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
        assert (args['agg_type'] == 'home_agg')
        self.resolution=args['resolution']
        # self.local_conc_range = int(args['conc_range']/self.resolution)
        self.agg_channel = args['agg_channel']
        self.step=0.5
        self.sigmoid=nn.Sigmoid()
        self.map_extent = args['map_extent']
        self.horizon=args['heatmap_channel']
        self.resolution=args['resolution']
        self.W = int((self.map_extent[1] - self.map_extent[0])/self.resolution)
        self.H = int((self.map_extent[3] - self.map_extent[2])/self.resolution)
        self.compensation=torch.round((torch.Tensor([args['map_extent'][3],-args['map_extent'][0]]))/self.resolution).int()
        self.output_traj = args['output_traj']
        self.pretrain_mlp = args['pretrain_mlp']
        self.combine_method = 'add'
        
        if self.pretrain_mlp:
            self.horizon=1
            self.end_time=args['heatmap_channel']*self.step
            self.traj_length=args['heatmap_channel']

        self.querys_embedder=nn.Sequential(
            nn.Linear(1,args['time_emb_size']),
            nn.LeakyReLU(),
            MLP(args['time_emb_size'],args['time_emb_size'])
        )

        if self.combine_method == 'add':
            assert(args['time_emb_size']==self.agg_channel-1)

        self.use_pos_enc=args['use_positional_encoding']
        if self.use_pos_enc:
            self.pos_enc=PositionalEncodingPermute2D(self.agg_channel)
        # self.decoding_net=nn.Sequential(
        #     nn.Linear((self.agg_channel+args['time_emb_size']),(self.agg_channel+args['time_emb_size'])//2),
        #     nn.LeakyReLU(),
        #     MLP((self.agg_channel+args['time_emb_size'])//2,(self.agg_channel+args['time_emb_size'])//4),
        #     nn.Linear((self.agg_channel+args['time_emb_size'])//4,1)
        # )

        self.decoding_net=nn.Sequential(
            nn.Linear((self.agg_channel),(self.agg_channel)),
            nn.LeakyReLU(),
            MLP((self.agg_channel),(self.agg_channel)//2),
            MLP((self.agg_channel)//2,(self.agg_channel)//4),
            nn.Linear((self.agg_channel)//4,1)
        )

        if self.output_traj:
            self.diff_emb=args['diff_emb']
            self.feature_dim=args['target_agent_enc_size']+args['agg_channel']+self.diff_emb
            self.num_target=args['num_target']
            self.endpoint_sampler=TorchModalitySampler(args['num_target'],args['sample_radius'])
            self.diff_encoder=nn.Linear(2,self.diff_emb)
            self.decoder = nn.Sequential(nn.Linear(self.feature_dim,self.feature_dim//2),
                                        LayerNorm(self.feature_dim//2),
                                        nn.ReLU(),
                                        nn.Linear(self.feature_dim//2,args['heatmap_channel']*2))

        

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
                mask=inputs['under_sampled_mask'].to(device)
            else:
                mask=None
        if self.output_traj:
            target_encodings = inputs['target_encodings']
            
        if self.pretrain_mlp:
            time_steps=(torch.arange(self.horizon,self.horizon+1,device=agg_encoding.device)*self.end_time).unsqueeze(-1)
        else:
            time_steps=(torch.arange(1,self.horizon+1,device=agg_encoding.device)*self.step).unsqueeze(-1)
        time_querys=torch.cat((self.querys_embedder(time_steps),time_steps),dim=-1).repeat(agg_encoding.shape[0],1,1).unsqueeze(2)## [Batch number, horizon, 1, channel]
        if self.use_pos_enc:
            context_encoding = agg_encoding + self.pos_enc(agg_encoding)
        else:
            context_encoding = agg_encoding
        context_encoding = context_encoding.view(context_encoding.shape[0], context_encoding.shape[1], -1)## [Batch number, channel, H*W]
        torch.cuda.empty_cache()
        context_encoding = context_encoding.permute(0, 2, 1).unsqueeze(1).repeat(1,self.horizon,1,1)## [Batch number, horizon, H*W, channel]
        if self.combine_method== 'concat':
            feature=torch.cat((context_encoding,time_querys.repeat(1,1,context_encoding.shape[2],1)),dim=-1)## [Batch number, horizon, H*W, channel]   
        elif self.combine_method== 'add':  
            feature=context_encoding+time_querys

        if self.pretrain_mlp:
            map_feature=feature.squeeze(1).view(feature.shape[0],self.H,self.W,-1)
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
            trajectories=self.decoder(concat_feature).view(gt_traj.shape[0],1,self.traj_length,2)
            torch.cuda.empty_cache()
            return {'pred': None,'mask': mask,'traj': trajectories,'probs':torch.ones([trajectories.shape[0],1],device=device),'endpoints':endpoints}
        
        x=self.decoding_net(feature).view(feature.shape[0],self.horizon,self.H,self.W)
        torch.cuda.empty_cache()
        predictions=self.sigmoid(x)

        if self.teacher_force:
            map_feature=feature[:,-1].view(feature.shape[0],self.H,self.W,-1)
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
            return {'pred': predictions,'mask': mask,'traj': trajectories,'probs':torch.ones([trajectories.shape[0],1],device=device),'endpoints':endpoints}

        
        if self.output_traj:
            map_feature=feature[:,-1].view(feature.shape[0],self.H,self.W,-1)
            # nodes_2D=get_index(predictions[:,-1].unsqueeze(1),mask,self.H,self.W)
            dense_pred=predictions[:,-1].unsqueeze(1)
            endpoints,confidences = self.endpoint_sampler(dense_pred)
            endpoints=endpoints.long()
            concat_feature=torch.empty([0,self.num_target,self.feature_dim],device=predictions.device)
            x_coord,y_coord=torch.meshgrid(torch.arange(self.map_extent[-1],self.map_extent[-2],-self.resolution)-self.resolution/2, ##### SHould be changed when image size changes
                                            torch.arange(self.map_extent[0],self.map_extent[1],self.resolution)+self.resolution/2)
            indices=torch.cat([y_coord.unsqueeze(-1),x_coord.unsqueeze(-1)],dim=-1).to(predictions.device)
            for batch_idx in range(len(dense_pred)):
                map_feat = (map_feature[batch_idx])[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]
                diff = (indices[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]).float()
                # diff = (endpoints[batch_idx]-self.compensation.to(device))*self.resolution
                # diff[:.0] = -diff[:.0]
                feature=torch.cat([map_feat,self.diff_encoder(diff),target_encodings[batch_idx].repeat(self.endpoint_sampler._n_targets,1)],dim=-1).unsqueeze(0)
                concat_feature=torch.cat([concat_feature,feature],dim=0)
            trajectories=self.decoder(concat_feature).view(dense_pred.shape[0],self.num_target,self.horizon,2)
            torch.cuda.empty_cache()
            # swapped=torch.zeros_like(trajectories)
            # swapped[:,:,:,0],swapped[:,:,:,1]=-trajectories[:,:,:,1],trajectories[:,:,:,0]
            # coord=torch.round(swapped/self.resolution+self.compensation.to(device))
            # coord=torch.clamp(coord,0,self.W).long()
            # confidences=get_probs(coord,predictions,mask,self.H,self.W)
            return {'pred': predictions,'mask': mask,'traj': trajectories,'probs':confidences,'endpoints':endpoints}
        return {'pred': predictions,'mask': mask}
