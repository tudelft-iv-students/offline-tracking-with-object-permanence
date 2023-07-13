import torch
import torch.nn as nn
from models.aggregators.aggregator import PredictionAggregator
from typing import Dict, Tuple
from models.library.blocks import *
from models.encoders.pgp_encoder_v1 import GAT,PGPEncoder_occ


class Match_agg(PredictionAggregator):
    """
    Aggregate context encoding using scaled dot product attention. Query obtained using target agent encoding,
    Keys and values obtained using map and surrounding agent encodings.
    """

    def __init__(self, args: Dict):

        """
        args to include

        enc_size: int Dimension of encodings generated by encoder
        emb_size: int Size of embeddings used for queries, keys and values
        num_heads: int Number of attention heads

        """
        super().__init__()
        self.agg_type=args['agg_method']
        if self.agg_type=='fully_connected':
            self.graph1=GlobalGraph(args['node_enc_size'],args['att_hidden_size'])
            self.graph2=GlobalGraph(args['node_enc_size'],args['att_hidden_size'])
        elif self.agg_type=='GAT':
            self.gat=nn.ModuleList([GAT(args['node_enc_size'], args['node_enc_size'])
                                    for _ in range(args['num_gat_layers'])])
        

    def forward(self, encodings: Dict) -> torch.Tensor:
        """
        Forward pass for attention aggregator
        """
        if self.agg_type=='fully_connected':
            map_enc=encodings['context_encoding']
            lane_node_masks=map_enc['lane_mask']
            lane_node_encodings=map_enc['lane_enc']
            mask_base=(~lane_node_masks[:,:,:,0].bool()).any(-1).unsqueeze(-1)
            attn_mask=(mask_base*mask_base.transpose(1,-1)).float()
            lane_encodings= torch.cat([self.graph1(lane_node_encodings, attn_mask),
                                        self.graph2(lane_node_encodings, attn_mask)], dim=-1)
            encodings['context_encoding']['lane_enc']=lane_encodings
        elif self.agg_type=='GAT':
            map_enc=encodings['context_encoding']
            lane_node_enc=map_enc['lane_enc']
            adj_mat = PGPEncoder_occ.build_adj_mat(map_enc['s_next'], map_enc['edge_type'])
            for gat_layer in self.gat:
                lane_node_enc += gat_layer(lane_node_enc, adj_mat)
            encodings['context_encoding']['lane_enc']=lane_node_enc
        return encodings

    @staticmethod
    def get_combined_encodings(context_enc: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a combined set of map and surrounding agent encodings to be aggregated using attention.
        """
        encodings = []
        masks = []
        if 'map' in context_enc:
            encodings.append(context_enc['map'])
            masks.append(context_enc['map_masks'])
        if 'vehicles' in context_enc:
            encodings.append(context_enc['vehicles'])
            masks.append(context_enc['vehicle_masks'])
        if 'pedestrians' in context_enc:
            encodings.append(context_enc['pedestrians'])
            masks.append(context_enc['pedestrian_masks'])
        combined_enc = torch.cat(encodings, dim=1)
        combined_masks = torch.cat(masks, dim=1).bool()
        return combined_enc, combined_masks

