from models.decoders.decoder import PredictionDecoder
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch
import torch.nn as nn
from typing import Dict
from models.library.blocks import *


class Match_decoder(PredictionDecoder):

    def __init__(self, args):
        """
        Simple MLP decoder used on occlusion recovery
        """

        super().__init__()
        self.fuse_map=args['fuse_map']
        if self.fuse_map:
            self.hist_aggtor=Att(args['hist_enc_size'],args['node_enc_size'])
            self.future_aggtor=Att(args['future_enc_size'],args['node_enc_size'])
            input_size=args['hist_enc_size']+args['future_enc_size']
            self.map_score_decoder=nn.Sequential(leaky_MLP(input_size,input_size//2),
                                         leaky_MLP(input_size//2,input_size//4),
                                        nn.Linear(input_size//4,1),
                                        nn.Sigmoid())
        self.motion_score_decoder=nn.Sequential(leaky_MLP(args['input_size'],args['input_size']//2),
                                         leaky_MLP(args['input_size']//2,args['input_size']//4),
                                        nn.Linear(args['input_size']//4,1),
                                        nn.Sigmoid())
                
        
                
        

    def forward(self, agg_encoding: Dict) -> Dict:
        """
        Decode yaw and trajectory coordinates
        """
        
        
        motion_scores =  self.motion_score_decoder(agg_encoding['target_agent_encodings'])
        
        predictions={'scores':motion_scores,
                     'masks':agg_encoding['masks']}
        if self.fuse_map:
            future_encodings=agg_encoding['future_encodings']
            hist_encodings=agg_encoding['hist_encodings']
            future_ctrs=agg_encoding['future_ctrs']
            hist_ctrs=agg_encoding['hist_ctrs']
            map_enc=agg_encoding['context_encoding']
            # lane_node_masks=map_enc['lane_mask']
            lane_node_encodings=map_enc['lane_enc']
            lane_ctrs=map_enc['lane_ctrs']
            future_masks=agg_encoding['masks']
            agt_mask=(~(future_masks[:,:,:,0].bool())).any(dim=-1)
            agt_mask=list((~agt_mask).float())
            hist_encodings = get_attention(  hist_ctrs, hist_encodings, lane_ctrs, lane_node_encodings, self.hist_aggtor)
            future_encodings = get_attention(  future_ctrs, future_encodings, lane_ctrs, lane_node_encodings, self.future_aggtor,query_mask=agt_mask)
            hist_encodings_concat=hist_encodings.repeat(1,future_encodings.shape[1],1)
            concat_encodings=torch.cat((hist_encodings_concat,future_encodings),dim=-1)
            map_scores=self.map_score_decoder(concat_encodings)
            predictions['map_scores']=map_scores
        return predictions
    @staticmethod
    def get_attention(traj_mask, traj_ctrs, query, lane_ctrs, lane_enc, aggtor, dist_th=8):
        agt_idcs=[]
        agt_ctrs=[]
        ctx_idcs=[]
        ctx_ctrs=[]
        agt_mask=[]
        for batch_id in range(len(traj_mask)):
            agt_idcs.append(torch.arange(traj_ctrs.shape[1],device=traj_ctrs.device).long())
            agt_ctrs.append(traj_ctrs[batch_id])
            ctx_idcs.append(torch.arange(lane_enc.shape[1],device=lane_enc.device).long())
            ctx_ctrs.append(lane_ctrs[batch_id])
            agt_mask.append(traj_mask[batch_id])
        map_enc=aggtor(agts=query.flatten(0,1), agt_idcs=agt_idcs, agt_ctrs=agt_ctrs, ctx=lane_enc.flatten(0,1), 
                                ctx_idcs=ctx_idcs, ctx_ctrs=ctx_ctrs, dist_th=dist_th,agt_mask=agt_mask)
        map_enc=map_enc.view(query.shape)
        return map_enc
    @staticmethod
    def get_bigru_enc(concat_mask,target_concat_embedding,bi_gru):
        concat_seq_lens = torch.sum(1 - concat_mask, dim=-1)
        seq_lens_batched = concat_seq_lens[concat_seq_lens != 0].cpu()
        concat_embedding_packed = pack_padded_sequence(target_concat_embedding , seq_lens_batched,batch_first=True, enforce_sorted=False)
        hidden_batched, _ = bi_gru(concat_embedding_packed)
        hidden_unpacked, _ = pad_packed_sequence(hidden_batched, batch_first=True,total_length=concat_mask.shape[1])#B,L,2*C
        return hidden_unpacked
    
