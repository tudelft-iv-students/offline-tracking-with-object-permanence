from models.decoders.decoder import PredictionDecoder
from models.decoders.utils import bivariate_gaussian_activation
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
        self.add_reverse=args['add_reverse']
        if self.add_attention:
            self.tq_self_attn = nn.MultiheadAttention(args['time_emb_size'], 1)
            self.query_emb = nn.Sequential(nn.Linear(args['time_emb_size'], args['time_emb_size']),nn.LeakyReLU())
            self.key_emb = nn.Sequential(nn.Linear(args['time_emb_size'], args['time_emb_size']),nn.LeakyReLU())
            self.val_emb = nn.Sequential(nn.Linear(args['time_emb_size'], args['time_emb_size']),nn.LeakyReLU())
            
            self.traj_decoder=nn.Sequential(leaky_MLP(2*args['time_emb_size'],args['time_emb_size']),
                                            nn.Linear(args['time_emb_size'],2))
            self.yaw_decoder=nn.Sequential(leaky_MLP(2*args['time_emb_size'],args['time_emb_size']),
                                            nn.Linear(args['time_emb_size'],1))
        
        else:

            self.traj_decoder=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                            nn.Linear(args['time_emb_size'],2))
            self.yaw_decoder=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                            nn.Linear(args['time_emb_size'],1))
        if self.add_reverse:
            self.traj_decoder_rev=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                            nn.Linear(args['time_emb_size'],2))
            self.yaw_decoder_rev=nn.Sequential(leaky_MLP(args['time_emb_size'],args['time_emb_size']),
                                            nn.Linear(args['time_emb_size'],1))
        

    def forward(self, agg_encoding: Dict) -> Dict:
        """
        Decode yaw and trajectory coordinates
        """
        if self.add_attention:
            attn_mask=agg_encoding['mask'][:,:,0].unsqueeze(-1).transpose(1,2).repeat(1,agg_encoding['mask'].shape[1],1).bool()
            query_embd=self.query_emb(agg_encoding['query']).transpose(0,1)
            key_embd=self.key_emb(agg_encoding['query']).transpose(0,1)
            val_embd=self.val_emb(agg_encoding['query']).transpose(0,1)
            att_op, _ = self.tq_self_attn(query_embd, key_embd, val_embd, attn_mask=attn_mask)
            att_op=att_op.transpose(0,1)

            query=torch.cat((agg_encoding['query'],att_op),dim=-1)
        else:
            query=agg_encoding['query']
        traj=self.traj_decoder(query).unsqueeze(1)
        yaw=self.yaw_decoder(query)
        mask=agg_encoding['mask'][:,:,0]
        if self.add_reverse:
            traj_rev=self.traj_decoder_rev(query).unsqueeze(1)
            yaw_rev=self.yaw_decoder_rev(query)

        predictions = {'traj': traj, 'yaw': yaw, 'mask': mask,'traj_rev':traj_rev,'yaw_rev':yaw_rev}

        return predictions
