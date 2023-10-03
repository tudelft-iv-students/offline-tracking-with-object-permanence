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
        self.init_traj_decoder=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                nn.Linear(args['time_emb_size'],2))
        self.init_yaw_decoder=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                        nn.Linear(args['time_emb_size'],1))
        self.leaky_relu = nn.LeakyReLU()
        self.traj_emb=leaky_MLP(6, args['traj_feat_size'])
        self.bi_gru_refiner=nn.GRU((6+args['traj_feat_size']), args['traj_feat_size'], batch_first=True, bidirectional= True)
        self.init_traj_refiner=nn.Sequential(leaky_MLP(args['traj_feat_size']*2,args['traj_feat_size']),
                                                    nn.Linear(args['traj_feat_size'],2))
        self.init_yaw_refiner=nn.Sequential(leaky_MLP(args['traj_feat_size']*2,args['traj_feat_size']),
                                                    nn.Linear(args['traj_feat_size'],1))
        if self.refine:
            self.use_gru=args['use_gru']
            self.add_nbr = args['add_nbr']
            if self.use_gru:
                self.traj_encoder=leaky_MLP(6,args['traj_emb_size'])
                self.lane_aggtor = Att(n_agt=args['traj_emb_size'],n_ctx=args['node_enc_size'])
                self.att_radius = args['att_radius']
                if self.add_nbr:
                    self.nbr_aggtor = Att(n_agt=args['traj_emb_size'],n_ctx=args['nbr_enc_size'])
                    self.mixer=leaky_MLP(args['traj_emb_size']*3+6,args['traj_emb_size']*2)
                    self.query_emb = leaky_MLP(args['traj_emb_size']*2, args['traj_emb_size'])
                    self.key_emb = leaky_MLP(args['traj_emb_size']*2, args['traj_emb_size'])
                    self.val_emb = leaky_MLP(args['traj_emb_size']*2, args['traj_emb_size'])
                else:
                    self.mixer=leaky_MLP(args['traj_emb_size']*2+6,args['traj_emb_size']*2)
                    self.query_emb = leaky_MLP(args['traj_emb_size'], args['traj_emb_size'])
                    self.key_emb = leaky_MLP(args['traj_emb_size'], args['traj_emb_size'])
                    self.val_emb = leaky_MLP(args['traj_emb_size'], args['traj_emb_size'])
                self.bi_gru_decoder = nn.GRU(args['traj_emb_size']*2, args['time_emb_size'], batch_first=True, bidirectional= True)
                self.traj_attn = nn.MultiheadAttention(args['traj_emb_size'], 1)
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
        initial_traj=self.init_traj_decoder(query).unsqueeze(1)
        initial_yaw=self.init_yaw_decoder(query)
        mask=agg_encoding['mask'][:,:,0]
        if 'refine_input' in agg_encoding:
            base_info=agg_encoding['refine_input']['traj'].clone()
            refine_mask=agg_encoding['refine_input']['mask'].clone()
            fill_in_mask=torch.isinf(base_info)
            fill_in_vals=torch.cat((initial_traj.clone().squeeze(1),initial_yaw.clone(),torch.cos(initial_yaw),torch.sin(initial_yaw)),-1)
            extract_mask=(1-(mask.clone().unsqueeze(-1).repeat(1,1,5))).bool()
            base_info[fill_in_mask]=fill_in_vals[extract_mask]
            init_traj_feature=self.traj_emb(base_info)
            traj_mask=refine_mask[:,:,0]
            query_refined=self.get_bigru_enc(traj_mask,torch.cat((init_traj_feature,base_info),-1),self.bi_gru_refiner)
            traj_offset=self.init_traj_refiner(query_refined)
            yaw_offset=self.init_yaw_refiner(query_refined)
            traj=initial_traj.clone().squeeze(1)
            yaw=initial_yaw.clone()
            traj[(1-agg_encoding['mask'][:,:,:2]).bool()]+=(traj_offset[fill_in_mask[:,:,:2]])
            yaw[(1-agg_encoding['mask'][:,:,:1]).bool()]+=(yaw_offset[fill_in_mask[:,:,:1]])
            traj=traj.unsqueeze(1)

        predictions = {'init_traj': initial_traj, 'init_yaw': initial_yaw,'traj': traj, 'yaw': yaw, 'mask': mask}
        if 'endpoints' in agg_encoding:
            endpoints_query=agg_encoding['endpoints']
            endpoints_coords=self.init_traj_decoder(endpoints_query).unsqueeze(1)
            endpoints_yaw=self.init_yaw_decoder(endpoints_query)
            predictions['endpoint_traj']=endpoints_coords
            predictions['endpoint_yaw']=endpoints_yaw
        if self.refine:
            lane_enc=agg_encoding['lane_info']['lane_enc'] 
            lane_ctrs=agg_encoding['lane_info']['lane_ctrs']
            if self.add_nbr:
                nbr_enc=agg_encoding['nbr_info']['nbr_enc'] 
                nbr_ctrs=agg_encoding['nbr_info']['nbr_ctrs']
            if 'refine_input' in agg_encoding:
                base_info=agg_encoding['refine_input']['traj'].clone()
                refine_mask=agg_encoding['refine_input']['mask'].clone()
                fill_in_mask=torch.isinf(base_info)
                fill_in_vals=torch.cat((traj.clone().squeeze(1),yaw.clone(),torch.cos(yaw),torch.sin(yaw)),-1)
                extract_mask=(1-(mask.clone().unsqueeze(-1).repeat(1,1,5))).bool()
                base_info[fill_in_mask]=fill_in_vals[extract_mask]
                traj_ctrs=base_info[:,:,:2]
                traj_mask=refine_mask[:,:,0]

                concat_base=self.traj_encoder(base_info)
                map_enc=self.get_attention( traj_mask, traj_ctrs, concat_base, lane_ctrs, lane_enc, self.lane_aggtor,self.att_radius)
                if self.add_nbr:
                    interact_enc=self.get_attention( traj_mask, traj_ctrs, concat_base, nbr_ctrs, nbr_enc, self.nbr_aggtor)
                    concat_enc=torch.cat((map_enc,interact_enc),-1) 
                else:
                    concat_enc=map_enc
                
                attn_mask=traj_mask.unsqueeze(1).repeat(1,agg_encoding['refine_input']['mask'].shape[1],1).bool()
                query_embd=self.query_emb(concat_enc).transpose(0,1)
                key_embd=self.key_emb(concat_enc).transpose(0,1)
                val_embd=self.val_emb(concat_enc).transpose(0,1)
                att_op, _ = self.traj_attn(query_embd, key_embd, val_embd, attn_mask=attn_mask)
                att_op=att_op.transpose(0,1)
                
                mixed_enc=self.mixer(torch.cat((concat_enc,att_op,base_info),-1))
                query_refined=self.get_bigru_enc(traj_mask,mixed_enc,self.bi_gru_decoder)
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
    @staticmethod
    def get_attention(traj_mask, traj_ctrs, query, lane_ctrs, lane_enc, aggtor, dist_th=8.0):
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
    
