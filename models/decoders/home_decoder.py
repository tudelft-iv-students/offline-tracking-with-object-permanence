from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.ram_decoder import get_dense,get_index
from return_device import return_device
from models.library.sampler import *
from models.library.blocks import *
device = return_device()


class HomeDecoder(PredictionDecoder):

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
        # self.layers=[]
        # for i in range(args['decoder_depth']-1):
        #     self.layers.append(nn.Conv2d(self.agg_channel//(2**i), self.agg_channel//(2**(i+1)), kernel_size=1, stride=1, bias=False))
        #     self.layers.append(nn.LeakyReLU())
        # assert (self.agg_channel//(2**(args['decoder_depth']-1))> args['heatmap_channel']-0.01)##Make sure the last layer has the least number of channels
        # self.layers.append(nn.Conv2d(self.agg_channel//(2**(args['decoder_depth']-1)), args['heatmap_channel'], kernel_size=1, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        self.teacher_force=False
        
        self.map_extent = args['map_extent']
        self.horizon=args['heatmap_channel']
        self.resolution=args['resolution']
        self.W = int((self.map_extent[1] - self.map_extent[0])/self.resolution)
        self.H = int((self.map_extent[3] - self.map_extent[2])/self.resolution)
        self.compensation=torch.round((torch.Tensor([args['map_extent'][3],-args['map_extent'][0]]))/self.resolution).int()
        self.output_traj = args['output_traj']
        self.pretrain_mlp = args['pretrain_mlp']
        self.decoding_net=nn.Sequential(
            TransposeCNNBlock(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=0),
            TransposeCNNBlock(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            TransposeCNNBlock(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=0),
            TransposeCNNBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, output_padding=1),
            AddCoords(with_r=False),
            nn.Conv2d(in_channels=34, out_channels=1, kernel_size=3, stride=1, padding=1,padding_mode='replicate'),
            nn.Sigmoid()
        )

        if self.output_traj:
            self.diff_emb=args['diff_emb']
            self.feature_dim=args['target_agent_enc_size']+self.diff_emb
            self.num_target=args['num_target']
            self.sampler_type=args['sampler']
            if self.sampler_type=='FDE':
                self.endpoint_sampler=FDESampler(args['num_target'],args['num_iter'],args['sample_radius'],self.resolution)
            else:
                self.endpoint_sampler=TorchModalitySampler(args['num_target'],args['sample_radius'])
            self.diff_encoder=nn.Linear(2,self.diff_emb)
            self.use_attn=args['use_attn']
            if self.use_attn:
                self.decoder = Home_attn_deocder(args['emb_size'],args['num_heads'],input_dim_context=args['map_dim'],
                                                input_dim_traj=self.feature_dim,horizon=self.horizon)
            else:
                self.decoder = Home_MLP_deocder(in_features=7,trajectory_hist_length=5)
        

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
        if 'traj_feature' in inputs:
            target_encodings = inputs['traj_feature']


        if self.pretrain_mlp:
            map_feature=agg_encoding.flatten(2).permute(2,0,1)
            gt_traj=inputs['gt_traj']
            final_points=gt_traj[:,-1]
            swapped=torch.zeros_like(final_points,device=device)
            swapped[:,0],swapped[:,1]=-final_points[:,1],final_points[:,0]
            endpoints=torch.round(swapped/self.resolution+self.compensation.to(device)).long()
            if self.use_attn:
                concat_feature=torch.cat([self.diff_encoder(final_points),target_encodings],dim=-1).unsqueeze(1)
                trajectories=self.decoder(concat_feature,map_feature).view(gt_traj.shape[0],1,self.horizon,2)
            else:
                trajectories=self.decoder(target_encodings,final_points.unsqueeze(1)).view(gt_traj.shape[0],1,self.horizon,2)
            torch.cuda.empty_cache()
            return {'pred': None,'mask': mask,'traj': trajectories,'probs':torch.ones([trajectories.shape[0],1],device=device),'endpoints':endpoints}
        
        predictions=self.decoding_net(agg_encoding)
        
        if self.teacher_force:
            map_feature=agg_encoding.flatten(2).permute(2,0,1)
            gt_traj=inputs['gt_traj']
            final_points=gt_traj[:,-1]
            swapped=torch.zeros_like(final_points,device=device)
            swapped[:,0],swapped[:,1]=-final_points[:,1],final_points[:,0]
            endpoints=torch.round(swapped/self.resolution+self.compensation.to(device)).long()
            if self.use_attn:
                concat_feature=torch.cat([self.diff_encoder(final_points),target_encodings],dim=-1).unsqueeze(1)
                trajectories=self.decoder(concat_feature,map_feature).view(gt_traj.shape[0],1,self.horizon,2)
            else:
                trajectories=self.decoder(target_encodings,final_points.unsqueeze(1)).view(gt_traj.shape[0],1,self.horizon,2)
            torch.cuda.empty_cache()
            return {'pred': predictions,'mask': mask,'traj': trajectories,'probs':torch.ones([trajectories.shape[0],1],device=device),'endpoints':endpoints}
        if self.output_traj:
            map_feature=agg_encoding.flatten(2).permute(2,0,1)
            # nodes_2D=get_index(predictions[:,-1].unsqueeze(1),mask,self.H,self.W)
            dense_pred=predictions[:,-1].unsqueeze(1)
            endpoints,confidences = self.endpoint_sampler(dense_pred)
            endpoints=endpoints.long()
            concat_feature=torch.empty([0,self.num_target,self.feature_dim],device=predictions.device)
            x_coord,y_coord=torch.meshgrid(torch.arange(self.map_extent[-1],self.map_extent[-2],-self.resolution)-self.resolution/2, ##### SHould be changed when image size changes
                                            torch.arange(self.map_extent[0],self.map_extent[1],self.resolution)+self.resolution/2)
            indices=torch.cat([y_coord.unsqueeze(-1),x_coord.unsqueeze(-1)],dim=-1).to(predictions.device)
            if self.use_attn:
                for batch_idx in range(len(dense_pred)):
                    # map_feat = (map_feature[batch_idx])[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]
                    diff = (indices[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]).float()
                    feature=torch.cat([self.diff_encoder(diff),target_encodings[batch_idx].repeat(self.endpoint_sampler._n_targets,1)],dim=-1).unsqueeze(0)
                    concat_feature=torch.cat([concat_feature,feature],dim=0)
                trajectories=self.decoder(concat_feature,map_feature).view(dense_pred.shape[0],self.num_target,self.horizon,2)
            else:
                final_points=[]
                for batch_idx in range(len(dense_pred)):
                    # map_feat = (map_feature[batch_idx])[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]
                    diff = (indices[endpoints[batch_idx,:,0],endpoints[batch_idx,:,1]]).float()
                    final_points.append(diff)
                trajectories=self.decoder(target_encodings,torch.stack(final_points)).view(dense_pred.shape[0],self.num_target,self.horizon,2)
            torch.cuda.empty_cache()
            return {'pred': predictions,'mask': mask,'traj': trajectories,'probs':confidences,'endpoints':endpoints}
        return {'pred': predictions,'mask': mask}

class Home_attn_deocder(nn.Module):
    def __init__(self, emb_size, num_heads,input_dim_context,input_dim_traj,horizon):
        super(Home_attn_deocder, self).__init__()
        self.attention=nn.MultiheadAttention(emb_size, num_heads)
        self.query_embedder=leaky_MLP(input_dim_traj,emb_size)
        self.key_embedder=leaky_MLP(input_dim_context,emb_size)
        self.value_embedder=leaky_MLP(input_dim_context,emb_size)
        self.decode_head=nn.Sequential(leaky_MLP(emb_size+input_dim_traj,(emb_size+input_dim_traj)//2),
                                        nn.Linear((emb_size+input_dim_traj)//2,horizon*2))

    def forward(self,traj_emb,context_emb):
        traj_query=self.query_embedder(traj_emb.transpose(0,1))
        context_key=self.key_embedder(context_emb)
        context_value=self.value_embedder(context_emb)
        attn_output,_=self.attention(query=traj_query, key=context_key, value=context_value, need_weights=False)
        traj_enc = torch.cat((attn_output.transpose(0,1), traj_emb),dim=-1)
        trajs=self.decode_head(traj_enc)
        return trajs

class Home_MLP_deocder(nn.Module):
    def __init__(self, in_features,trajectory_hist_length,emb_size=32,horizon=12):
        super(Home_MLP_deocder, self).__init__()
        self.traj_encoder=nn.Sequential(
            nn.Flatten(1),
            leaky_MLP(in_features*trajectory_hist_length, emb_size)
        )
        self.traj_decoder=nn.Sequential(
            leaky_MLP(emb_size+2,64),
            nn.Linear(64,horizon*2)
        )

    def forward(self,traj_hist,diff):
        traj_emb=self.traj_encoder(traj_hist)
        traj_emb=traj_emb.unsqueeze(1).repeat(1,diff.shape[1],1)
        x=torch.cat((traj_emb,diff),-1)
        trajs=self.traj_decoder(x)
        return trajs
