from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from typing import Dict
from models.library.blocks import *
from train_eval.utils import return_device
import math

device = return_device()


class MatchEncoder(PredictionEncoder):

    def __init__(self, args: Dict):

        super().__init__()
        
        self.fuse_map=args['fuse_map']
        
        if self.fuse_map:
            self.subnode_emb = nn.Linear(args['node_feat_size'], args['node_emb_size'])
            self.lane_bi_gru = nn.GRU(args['node_emb_size'], args['node_emb_size'], batch_first=True, bidirectional= True)
            self.future_map_aggtor = Att(n_agt=args['node_emb_size'],n_ctx=args['future_enc_size'])
            self.past_map_aggtor = Att(n_agt=args['node_emb_size'],n_ctx=args['past_enc_size'])
            self.lane_node_aggtor = nn.GRU(args['node_emb_size']*2, args['lane_enc_size'], batch_first=True,bidirectional= True)


        # Target agent encoder
        self.target_past_emb = nn.Linear(args['target_feat_size'], args['target_emb_size'])
        self.target_fut_emb = nn.Linear(args['target_feat_size'], args['target_emb_size'])

        self.bi_gru = nn.GRU(args['target_emb_size'], args['past_enc_size'], batch_first=True, bidirectional= True)
        self.target_past_enc = nn.GRU(args['target_emb_size'], args['past_enc_size'], batch_first=True)
        self.target_future_enc = nn.GRU(args['target_emb_size']+args['past_enc_size'], args['future_enc_size'], batch_first=True)

        # Non-linearities
        self.leaky_relu = nn.LeakyReLU(0.1)

        

    def forward(self, inputs: Dict) -> Dict:
       

        # Encode target agent
        history = inputs['history']
        history_mask=inputs['history_mask']
        future = inputs['future']
        future_mask=inputs['future_mask']
        
        # concat_indicator = torch.cat((torch.zeros([concat.shape[0],hist.shape[1],1]),torch.ones([concat.shape[0],future.shape[1],1])),1).cuda()

        history_emb = self.leaky_relu(self.target_past_emb(history))
        future_emb = self.leaky_relu(self.target_fut_emb(future))
        
        hist_seq_lens = torch.sum(1 - history_mask[:,  :, 0], dim=-1)
        seq_lens_batched = hist_seq_lens[hist_seq_lens != 0].cpu()
        hist_embedding_packed = pack_padded_sequence(history_emb, seq_lens_batched,batch_first=True, enforce_sorted=False)
        _, h0 = self.target_past_enc(hist_embedding_packed)
        hidden_init=h0.transpose(0,1).repeat(1,future.shape[1],1)
        rev_encoding=rev_gru_encode(future_emb,future_mask,self.bi_gru,hidden_init)
        target_enc=ugru_encode(rev_encoding,future_emb,future_mask, self.target_future_enc)
        concat_enc=torch.cat((target_enc,hidden_init),dim=-1)
        encodings = {'target_agent_encodings': concat_enc,
                    'masks':future_mask
                    }
        if self.fuse_map:
            lane_node_feats=inputs['map_representation']['lane_node_feats']
            lane_node_masks=inputs['map_representation']['lane_node_masks']
            lane_subnode_embedding = self.leaky_relu(self.subnode_emb(lane_node_feats))
            lane_node_ctrs = lane_node_feats.clone().detach().flatten(1,2)
            lane_node_ctrs[lane_node_masks.flatten(1,2).bool()] = math.inf
            lane_node_ctrs=lane_node_ctrs[:,:,:2]
            # gru_lane_feats = node_gru_enc(lane_node_masks, lane_subnode_embedding, self.sub_node_encoder)
            last_inds=torch.argmax(future_mask[:,:,:,0],dim=-1).cpu()-1
            inds1,inds2=torch.meshgrid(torch.arange(last_inds.shape[0]), 
                                        torch.arange(last_inds.shape[1]))
            inds=(torch.cat((inds1.unsqueeze(-1),inds2.unsqueeze(-1),last_inds.unsqueeze(-1)),dim=-1).flatten(0,1)).T
            future_ctrs=future[:,:,:,:2]
            last_ctrs=future_ctrs[inds[0],inds[1],inds[2]].view(future_ctrs.shape[0],future_ctrs.shape[1],2)
            ctx_mask=(~(future_mask[:,:,:,0].bool())).any(dim=-1)
            ctx_mask=list((~ctx_mask).float())
            lane_subnode_embedding = get_attention( lane_node_ctrs, lane_subnode_embedding.flatten(1,2), last_ctrs, target_enc, self.future_map_aggtor,ctx_mask=ctx_mask)
            
            
            hist_last_inds=torch.argmax(history_mask[:,:,0],dim=-1).cpu()-1
            hist_inds1=torch.arange(hist_last_inds.shape[0])
            hist_inds=torch.cat((hist_inds1.unsqueeze(-1),hist_last_inds.unsqueeze(-1)),dim=-1).T
            hist_ctrs=history[:,:,:2]
            hist_last_ctrs=hist_ctrs[hist_inds[0],hist_inds[1]].view(hist_ctrs.shape[0],2)
            
            lane_subnode_embedding = get_attention( lane_node_ctrs, lane_subnode_embedding, hist_last_ctrs.unsqueeze(1), h0.transpose(0,1), self.past_map_aggtor)
            lane_subnode_embedding=lane_subnode_embedding.view(lane_subnode_embedding.shape[0],lane_node_feats.shape[1],
                                                               lane_node_feats.shape[2],lane_subnode_embedding.shape[-1])
            bigru_lane_feats = node_gru_enc(lane_node_masks, lane_subnode_embedding, self.lane_bi_gru)
            lane_node_encodings = self.variable_size_gru_encode(bigru_lane_feats, lane_node_masks, self.lane_node_aggtor)
            # Return encodings
            if 's_next' in inputs['map_representation'].keys():
                s_next=inputs['map_representation']['s_next']
                edge_type=inputs['map_representation']['edge_type']
            else:
                s_next=None
                edge_type=None
            encodings = {'target_agent_encodings': concat_enc,
                        'future_encodings':target_enc,
                        'hist_encodings':h0.transpose(0,1),
                        'masks':future_mask,
                        'future_ctrs':last_ctrs,
                        'hist_ctrs':hist_last_ctrs.unsqueeze(1),
                        'context_encoding': {'lane_enc': lane_node_encodings,
                                            'lane_mask': lane_node_masks,
                                            'lane_ctrs':inputs['map_representation']['lane_ctrs'],
                                            's_next':s_next,
                                            'edge_type':edge_type,
                                            },
                        }

      

        return encodings
    
    @staticmethod
    def variable_size_gru_encode(feat_embedding: torch.Tensor, masks: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        """
        Returns GRU encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)# select the feature from actual lane nodes and get rid of padded placeholder
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])

        # Pack padded sequences
        seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
        seq_lens_batched = seq_lens[seq_lens != 0].cpu()
        if len(seq_lens_batched) != 0:
            feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
                                                         batch_first=True, enforce_sorted=False)

            # Encode
            _, encoding_batched = gru(feat_embedding_packed)
            encoding_batched = encoding_batched.squeeze(0)

            # Scatter back to appropriate batch index
            masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
            encoding = torch.zeros(masks_for_scattering.shape, device=device)
            encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            hidden_state_size = gru.hidden_size
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=feat_embedding.device)

        return encoding
    
def rev_gru_encode(feat_embedding: torch.Tensor, masks: torch.Tensor, bi_gru: nn.GRU, h0: torch.Tensor) -> torch.Tensor:
    """
    Returns GRU encoding for a batch of inputs where each sample in the batch is a set of a variable number
    of sequences, of variable lengths.
    """

    # Form a large batch of all sequences in the batch
    masks_for_batching = ~masks[:, :, :, 0].bool()
    masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
    feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)# select the feature from actual lane nodes and get rid of padded placeholder
    h0_batched = torch.masked_select(h0.unsqueeze(2), masks_for_batching).view(-1, 1, h0.shape[-1])
    feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])

    # Pack padded sequences
    seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
    seq_lens_batched = seq_lens[seq_lens != 0].cpu()
    if len(seq_lens_batched) != 0:
        feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
                                                        batch_first=True, enforce_sorted=False)

        # Encode
        # output, hn = gru(feat_embedding_packed,h0_batched)
        output, hn = bi_gru(feat_embedding_packed,h0_batched.transpose(0,1).repeat(2,1,1))
        encoding_batched,_=pad_packed_sequence(output,batch_first=True,total_length=feat_embedding.shape[2])
        encoding_batched=encoding_batched.reshape(encoding_batched.shape[0],encoding_batched.shape[1],2,-1)
        encoding_batched=encoding_batched[:,:,1]
        
        masks_for_scattering = masks_for_batching.repeat(1, 1, encoding_batched.shape[-2],encoding_batched.shape[-1])
        encoding = torch.zeros(masks_for_scattering.shape,device=feat_embedding.device)
        encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

    else:
        batch_size = feat_embedding.shape[0]
        max_num = feat_embedding.shape[1]
        hidden_state_size = bi_gru.hidden_size
        encoding = torch.zeros((batch_size, max_num, hidden_state_size),device=feat_embedding.device)

    return encoding

def ugru_encode(rev_encoding: torch.Tensor,feat_embedding: torch.Tensor, masks: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
    feature=torch.cat((rev_encoding,feat_embedding),dim=-1)
    masks_for_batching = ~masks[:, :, :, 0].bool()
    masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
    feat_embedding_batched = torch.masked_select(feature, masks_for_batching)# select the feature from actual lane nodes and get rid of padded placeholder
    feat_embedding_batched = feat_embedding_batched.view(-1, feature.shape[2], feature.shape[3])

    # Pack padded sequences
    seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
    seq_lens_batched = seq_lens[seq_lens != 0].cpu()
    if len(seq_lens_batched) != 0:
        feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
                                                        batch_first=True, enforce_sorted=False)

        # Encode
        _, encoding_batched = gru(feat_embedding_packed)
        encoding_batched = encoding_batched.squeeze(0)

        # Scatter back to appropriate batch index
        masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
        encoding = torch.zeros(masks_for_scattering.shape,device=feat_embedding.device)
        encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

    else:
        batch_size = feat_embedding.shape[0]
        max_num = feat_embedding.shape[1]
        hidden_state_size = gru.hidden_size
        encoding = torch.zeros((batch_size, max_num, hidden_state_size),device=feat_embedding.device)

    return encoding
