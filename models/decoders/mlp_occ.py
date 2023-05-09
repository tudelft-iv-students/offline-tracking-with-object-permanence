from models.decoders.decoder import PredictionDecoder
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch
import torch.nn as nn
from typing import Dict
from models.library.blocks import *


class MLP_occ(PredictionDecoder):

    def __init__(self, args):
        """
        Simple MLP decoder used on occlusion recovery
        """

        super().__init__()
        
        self.add_attention=args['add_attention']
        self.refine=args['refinement']
        self.traj_decoder=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                nn.Linear(args['time_emb_size'],2))
        self.yaw_decoder=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                        nn.Linear(args['time_emb_size'],1))
        self.leaky_relu = nn.LeakyReLU()
        if self.refine:
            self.use_gru=args['use_gru']
            if self.use_gru:
                self.traj_encoder=leaky_MLP(4+args['motion_emb_size'],args['traj_emb_size'])
                self.map_aggtor = Att(n_agt=args['traj_emb_size'],n_ctx=args['node_enc_size'])
                self.bi_gru = nn.GRU(args['traj_emb_size']*2, args['time_emb_size'], batch_first=True, bidirectional= True)
                self.init_attn = nn.MultiheadAttention(args['motion_emb_size'], 1)
                self.query_emb = leaky_MLP(4, args['motion_emb_size'])
                self.key_emb = leaky_MLP(4, args['motion_emb_size'])
                self.val_emb = leaky_MLP(4, args['motion_emb_size'])
                self.traj_ref_head=nn.Sequential(leaky_MLP(args['time_emb_size']*2,args['time_emb_size']),
                                                    nn.Linear(args['time_emb_size'],2))
                self.yaw_ref_head=nn.Sequential(leaky_MLP(args['time_emb_size']*2,args['time_emb_size']),
                                                    nn.Linear(args['time_emb_size'],1))
            else:
                self.map_aggtor = Att(n_agt=args['time_emb_size'],n_ctx=args['node_enc_size'])
                if self.add_attention:
                    self.sep_motion_enc=args['sep_motion_emb']
                    if self.sep_motion_enc:
                        self.motion_emb=leaky_MLP(3, args['motion_emb_size'])
                        self.tq_self_attn = nn.MultiheadAttention(args['time_emb_size'], 1)
                        self.query_emb = leaky_MLP(args['time_emb_size']+args['motion_emb_size'], args['time_emb_size'])
                        self.key_emb = leaky_MLP(args['time_emb_size']+args['motion_emb_size'], args['time_emb_size'])
                        self.val_emb = leaky_MLP(args['time_emb_size']+args['motion_emb_size'], args['time_emb_size'])
                        
                        self.traj_ref_head=nn.Sequential(leaky_MLP(3*args['time_emb_size']+args['motion_emb_size'],2*args['time_emb_size']),
                                                        leaky_MLP(2*args['time_emb_size'],args['time_emb_size']),
                                                        nn.Linear(args['time_emb_size'],2))
                        self.yaw_ref_head=nn.Sequential(leaky_MLP(3*args['time_emb_size']+args['motion_emb_size'],2*args['time_emb_size']),
                                                        leaky_MLP(2*args['time_emb_size'],args['time_emb_size']),
                                                        nn.Linear(args['time_emb_size'],1))
                        
                    else:
                        self.tq_self_attn = nn.MultiheadAttention(args['time_emb_size'], 1)
                        self.query_emb = leaky_MLP(args['time_emb_size']+3, args['time_emb_size'])
                        self.key_emb = leaky_MLP(args['time_emb_size']+3, args['time_emb_size'])
                        self.val_emb = leaky_MLP(args['time_emb_size']+3, args['time_emb_size'])
                        
                        self.traj_ref_head=nn.Sequential(leaky_MLP(3*args['time_emb_size']+3,args['time_emb_size']),
                                                        nn.Linear(args['time_emb_size'],2))
                        self.yaw_ref_head=nn.Sequential(leaky_MLP(3*args['time_emb_size']+3,args['time_emb_size']),
                                                        nn.Linear(args['time_emb_size'],1))
                
        
                
        

    def forward(self, agg_encoding: Dict) -> Dict:
        """
        Decode yaw and trajectory coordinates
        """
        
        query=agg_encoding['query']
        traj=self.traj_decoder(query).unsqueeze(1)
        yaw=self.yaw_decoder(query)
        mask=agg_encoding['mask'][:,:,0]
        predictions = {'traj': traj, 'yaw': yaw, 'mask': mask}
        if 'endpoints' in agg_encoding:
            endpoints_query=agg_encoding['endpoints']
            endpoints_coords=self.traj_decoder(endpoints_query).unsqueeze(1)
            endpoints_yaw=self.yaw_decoder(endpoints_query)
            predictions['endpoint_traj']=endpoints_coords
            predictions['endpoint_yaw']=endpoints_yaw
        if self.refine:
            lane_enc=agg_encoding['map_info']['lane_encodings'] #32
            lane_ctrs=agg_encoding['map_info']['lane_ctrs']
            if 'refine_input' in agg_encoding:
                base_info=agg_encoding['refine_input']['traj'].clone()
                refine_mask=agg_encoding['refine_input']['mask']
                fill_in_mask=torch.isinf(base_info)
                fill_in_vals=torch.cat((traj.clone().squeeze(1),yaw.clone()),-1)
                extract_mask=(1-(mask.clone().unsqueeze(-1).repeat(1,1,3))).bool()
                base_info[fill_in_mask]=fill_in_vals[extract_mask]
                traj_ctrs=base_info[:,:,:2]
                traj_mask=refine_mask[:,:,0]

                attn_mask=traj_mask.unsqueeze(1).repeat(1,agg_encoding['refine_input']['mask'].shape[1],1).bool()
                query_embd=self.query_emb(base_info).transpose(0,1)
                key_embd=self.key_emb(base_info).transpose(0,1)
                val_embd=self.val_emb(base_info).transpose(0,1)
                att_op, _ = self.init_attn(query_embd, key_embd, val_embd, attn_mask=attn_mask)
                att_op=att_op.transpose(0,1)
                concat_base=self.leaky_relu(self.traj_encoder(torch.cat((att_op,base_info),-1)))
                map_enc=self.get_attention( traj_mask, traj_ctrs, concat_base, lane_ctrs, lane_enc)
                concat_enc=torch.cat((map_enc,concat_base),-1)
                query_refined=self.get_bigru_enc(traj_mask,concat_enc)
                traj_offset=self.traj_ref_head(query_refined)
                yaw_offset=self.yaw_ref_head(query_refined)
                refined_traj=traj.clone().squeeze(1)
                refined_yaw=yaw.clone().squeeze(1)
                refined_traj[(1-agg_encoding['mask'][:,:,:2]).bool()]+=(traj_offset[fill_in_mask[:,:,:2]])
                refined_yaw[(1-agg_encoding['mask'][:,:,:1]).bool()]+=(yaw_offset[fill_in_mask[:,:,:1]])
                refined_traj=refined_traj.unsqueeze(1)
                refined_yaw=refined_yaw
            else:
                traj_ctrs=traj.clone().squeeze(1)
                traj_mask=mask.clone()
                map_enc=self.get_attention( traj_mask, traj_ctrs, query, lane_ctrs, lane_enc)
                if not self.sep_motion_enc:
                    feature=torch.cat((map_enc,traj_ctrs,yaw.clone()),-1)
                else:
                    original_motion=torch.cat((traj_ctrs,yaw.clone()),-1)
                    feature=torch.cat((map_enc,self.motion_emb(original_motion)),-1)
                if self.add_attention:
                    attn_mask=mask.unsqueeze(1).repeat(1,agg_encoding['mask'].shape[1],1).bool()
                    query_embd=self.query_emb(feature).transpose(0,1)
                    key_embd=self.key_emb(feature).transpose(0,1)
                    val_embd=self.val_emb(feature).transpose(0,1)
                    att_op, _ = self.tq_self_attn(query_embd, key_embd, val_embd, attn_mask=attn_mask)
                    att_op=att_op.transpose(0,1)

                    query_refined=torch.cat((feature,att_op,query),dim=-1)
                else:
                    query_refined=torch.cat((feature,query),dim=-1)
                traj_offset=self.traj_ref_head(query_refined).unsqueeze(1)
                yaw_offset=self.yaw_ref_head(query_refined)
                refined_traj=traj_offset+traj
                refined_yaw=yaw_offset+yaw
            predictions['refined_traj']=refined_traj
            predictions['refined_yaw']=refined_yaw
        return predictions
    
    def get_attention(self, traj_mask, traj_ctrs, query, lane_ctrs, lane_enc):
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
        map_enc=self.map_aggtor.forward(agts=query.flatten(0,1), agt_idcs=agt_idcs, agt_ctrs=agt_ctrs, ctx=lane_enc.flatten(0,1), 
                                ctx_idcs=ctx_idcs, ctx_ctrs=ctx_ctrs, dist_th=20,agt_mask=agt_mask)
        map_enc=map_enc.view(query.shape)
        return map_enc
    def get_bigru_enc(self,concat_mask,target_concat_embedding):
        concat_seq_lens = torch.sum(1 - concat_mask, dim=-1)
        seq_lens_batched = concat_seq_lens[concat_seq_lens != 0].cpu()
        concat_embedding_packed = pack_padded_sequence(target_concat_embedding , seq_lens_batched,batch_first=True, enforce_sorted=False)
        hidden_batched, _ = self.bi_gru(concat_embedding_packed)
        hidden_unpacked, _ = pad_packed_sequence(hidden_batched, batch_first=True,total_length=concat_mask.shape[1])#B,L,2*C
        return hidden_unpacked
    
