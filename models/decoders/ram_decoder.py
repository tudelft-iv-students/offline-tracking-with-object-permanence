from models.decoders.decoder import PredictionDecoder
from models.library.blocks import LayerNorm
import torch
import torch.nn as nn
from typing import Dict, Union
from models.library.sampler import TorchModalitySampler
from return_device import return_device
device = return_device()


def get_index(pred,mask,H=122,W=122):
    x_coord,y_coord=torch.meshgrid(torch.arange(0,H,1), ##### SHould be changed when image size changes
                                torch.arange(0,W,1))
    nodes_candidates=torch.cat((x_coord.unsqueeze(0),y_coord.unsqueeze(0)),dim=0).view(2,-1).T
    nodes_2D=torch.zeros([mask.shape[0],pred.shape[-1],2])
    for i in range(mask.shape[0]):
        nodes_batch=nodes_candidates[mask[i]]
        nodes_2D[i,:nodes_batch.shape[0]]=nodes_batch
    return nodes_2D.int().permute(0,2,1).to(pred.device)

def get_dense(pred,nodes_2D,H,W):
    dense_rep=torch.empty(0,pred.shape[1],H,W,device=pred.device)
    for batch in range(pred.shape[0]):
        batch_heatmap=torch.empty(0,H,W,device=pred.device)
        for step in range(pred.shape[1]):
            heatmap=torch.sparse_coo_tensor(nodes_2D[batch],pred[batch,step],(H,W))
            batch_heatmap=torch.cat((batch_heatmap,heatmap.to_dense().unsqueeze(0)),dim=0)
        dense_rep=torch.cat((dense_rep,batch_heatmap.unsqueeze(0)),dim=0)
    return dense_rep


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
        assert (args['agg_type'] == 'attention' or args['agg_type'] == 'concat')
        self.map_extent = args['map_extent']
        self.horizon=args['heatmap_channel']
        self.resolution=args['resolution']
        self.W = int((self.map_extent[1] - self.map_extent[0])/self.resolution)
        self.H = int((self.map_extent[3] - self.map_extent[2])/self.resolution)
        self.compensation=torch.round((torch.Tensor([args['map_extent'][3],-args['map_extent'][0]]))/self.resolution).int()
        # self.compensation=torch.tensor([98,59])
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
            if self.output_traj:
                target_encodings = inputs['target_encodings']
                map_feature=inputs['feature'].permute(0,2,3,1)
            

        torch.cuda.empty_cache()
        if self.pretrain_mlp:
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
            return {'pred': None,'mask': mask,'traj': trajectories,'probs':None,'endpoints':endpoints}
        
        predictions=torch.empty([init_states.shape[0],0,init_states.shape[-1]],device=attn_output_weights.device)
        prev_states=init_states.unsqueeze(1).to_dense()

        for step in range(self.horizon):
            predictions=torch.cat((predictions,torch.bmm(prev_states,attn_output_weights)),dim=1)
            prev_states=predictions[:,step].unsqueeze(1)

        
        if self.output_traj:
            nodes_2D=get_index(predictions[:,-1].unsqueeze(1),mask,self.H,self.W)
            dense_pred=get_dense(predictions[:,-1].unsqueeze(1),nodes_2D,self.H,self.W)
            endpoints,confidences = self.endpoint_sampler(dense_pred)
            endpoints=endpoints.long()
            concat_feature=torch.empty([0,self.num_target,self.feature_dim],device=attn_output_weights.device)
            x_coord,y_coord=torch.meshgrid(torch.arange(self.map_extent[-1],self.map_extent[-2],-self.resolution), ##### SHould be changed when image size changes
                                            torch.arange(self.map_extent[0],self.map_extent[1],self.resolution))
            indices=torch.cat([x_coord.unsqueeze(-1),y_coord.unsqueeze(-1)],dim=-1).to(attn_output_weights.device)
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

        torch.cuda.empty_cache()
        # print('################ Memory usage after forward pass ####################')
        # print(torch.cuda.memory_summary(device=predictions.device, abbreviated=False))
        # nodes_2D=get_index(predictions,mask)
        # pred=get_dense(predictions,nodes_2D,self.H,self.W)
        # return {'pred': pred,'mask': mask}
        return {'pred': predictions,'mask': mask}
