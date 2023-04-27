from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from typing import Dict
from models.library.blocks import leaky_MLP
from return_device import return_device
device = return_device()


class PGPEncoder_occ(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        GRU based encoder from PGP. Lane node features and agent histories encoded using GRUs.
        Additionally, agent-node attention layers infuse each node encoding with nearby agent context.
        Finally GAT layers aggregate local context at each node.

        args to include:

        target_agent_feat_size: int Size of target agent features
        target_agent_emb_size: int Size of target agent embedding
        taret_agent_enc_size: int Size of hidden state of target agent GRU encoder

        node_feat_size: int Size of lane node features
        node_emb_size: int Size of lane node embedding
        node_enc_size: int Size of hidden state of lane node GRU encoder

        nbr_feat_size: int Size of neighboring agent features
        nbr_enb_size: int Size of neighboring agent embeddings
        nbr_enc_size: int Size of hidden state of neighboring agent GRU encoders

        num_gat_layers: int Number of GAT layers to use.
        """

        super().__init__()
        self.motion_enc_type = args['motion_enc_type']
        self.map_aggregation = args['map_aggregation']
        # Target agent encoder
        self.target_past_emb = nn.Linear(args['target_agent_feat_size'], args['target_agent_emb_size'])
        self.target_past_enc = nn.GRU(args['target_agent_emb_size']*2, args['target_agent_enc_size'], batch_first=True)
        # self.target_past_enc = nn.GRU(args['target_agent_emb_size'], args['target_agent_enc_size'], batch_first=True)
        self.target_fut_emb = nn.Linear(args['target_agent_feat_size'], args['target_agent_emb_size'])
        self.target_fut_enc = nn.GRU(args['target_agent_emb_size']*2, args['target_agent_enc_size'], batch_first=True)
        # self.target_fut_enc = nn.GRU(args['target_agent_emb_size'], args['target_agent_enc_size'], batch_first=True)
        self.target_concat_emb = nn.Linear(args['target_agent_feat_size']+1, args['target_agent_emb_size'])
        self.bi_gru = nn.GRU(args['target_agent_emb_size'], args['target_agent_emb_size'], batch_first=True, bidirectional= True)

        # Node encoders
        self.node_emb = nn.Linear(args['node_feat_size'], args['node_emb_size'])
        self.node_encoder = nn.GRU(args['node_emb_size'], args['node_enc_size'], batch_first=True)

        # Surrounding agent encoder
        self.nbr_emb = nn.Linear(args['nbr_feat_size'], args['nbr_emb_size'])
        self.nbr_enc = nn.GRU(args['nbr_emb_size'], args['nbr_enc_size'], batch_first=True)

        # Agent-node attention
        self.query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.key_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        self.val_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        self.a_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        # self.mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])
        
        # Target node attention
        self.nd_tgt_emb = nn.Linear(args['target_agent_enc_size'], args['map_size'])
        self.nd_key_emb = nn.Linear(args['node_enc_size'], args['map_size'])
        self.nd_val_emb = nn.Linear(args['node_enc_size'], args['map_size'])
        self.t_n_att = nn.MultiheadAttention(args['map_size'], num_heads=args['tn_head_num'])
        self.tn_head_num=args['tn_head_num']

        # Non-linearities
        self.leaky_relu = nn.LeakyReLU()

        # GAT layers
        self.gat = nn.ModuleList([GAT(args['node_enc_size'], args['node_enc_size'])
                                  for _ in range(args['num_gat_layers'])])

    def forward(self, inputs: Dict) -> Dict:
        """
        Forward pass for PGP encoder
        :param inputs: Dictionary with
            target_agent_representation: torch.Tensor, shape [batch_size, t_h, target_agent_feat_size]
            map_representation: Dict with
                'lane_node_feats': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]
                'lane_node_masks': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]

                (Optional)
                's_next': Edge look-up table pointing to destination node from source node
                'edge_type': Look-up table with edge type

            surrounding_agent_representation: Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'vehicle_masks': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'pedestrians': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
                'pedestrian_masks': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
            agent_node_masks:  Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_nodes, max_vehicles]
                'pedestrians': torch.Tensor, shape [batch_size, max_nodes, max_pedestrians]

            Optionally may also include the following if edges are defined for graph traversal
            'init_node': Initial node in the lane graph based on track history.
            'node_seq_gt': Ground truth node sequence for pre-training

        :return:
        """

        # Encode target agent
        target_agent_representation=inputs['target_agent_representation']
        hist=target_agent_representation['history']['traj']
        hist_mask=target_agent_representation['history']['mask']
        future=target_agent_representation['future']['traj']
        future_mask=target_agent_representation['future']['mask']
        concat=target_agent_representation['concat_motion']['traj']
        concat_mask=target_agent_representation['concat_motion']['mask']
        # concat_indicator = torch.cat((torch.zeros([concat.shape[0],hist.shape[1],1]),torch.ones([concat.shape[0],future.shape[1],1])),1).cuda()

        target_past_embedding = self.leaky_relu(self.target_past_emb(hist))
        target_future_embedding = self.leaky_relu(self.target_fut_emb(future))
        lane_node_feats = inputs['map_representation']['lane_node_feats']
        lane_node_masks = inputs['map_representation']['lane_node_masks']

        ## U-GRU method
        if self.motion_enc_type == 'ugru':
            target_concat_embedding = self.leaky_relu(self.target_concat_emb(concat))
            hist_unpacked,future_unpacked=self.get_ugru_enc(concat_mask,target_concat_embedding,hist_mask,future_mask)
            # concatenate bi-gru output with original feature 
            past_feature=torch.cat((target_past_embedding,hist_unpacked),dim=-1).unsqueeze(1)
            future_feature=torch.cat((target_future_embedding,future_unpacked),dim=-1).unsqueeze(1)
            target_past_encdoings = self.variable_size_gru_encode(past_feature, hist_mask.unsqueeze(1), self.target_past_enc)
            target_future_encdoings = self.variable_size_gru_encode(future_feature, future_mask.unsqueeze(1), self.target_fut_enc)
        else:
        ## Baseline method
            target_past_encdoings = self.variable_size_gru_encode(target_past_embedding.unsqueeze(1), hist_mask.unsqueeze(1), self.target_past_enc)
            target_future_encdoings = self.variable_size_gru_encode(target_future_embedding.unsqueeze(1), future_mask.unsqueeze(1), self.target_fut_enc)
        lane_node_enc=None
        if self.map_aggregation:
            # Encode lane nodes

            lane_node_embedding = self.leaky_relu(self.node_emb(lane_node_feats))
            lane_node_enc = self.variable_size_gru_encode(lane_node_embedding, lane_node_masks, self.node_encoder)

            # Encode surrounding agents
            # nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
            # nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)
            # nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
            # nbr_vehicle_embedding = self.leaky_relu(self.nbr_emb(nbr_vehicle_feats))
            # nbr_vehicle_enc = self.variable_size_gru_encode(nbr_vehicle_embedding, nbr_vehicle_masks, self.nbr_enc)
            # nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
            # nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
            # nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']
            # nbr_ped_embedding = self.leaky_relu(self.nbr_emb(nbr_ped_feats))
            # nbr_ped_enc = self.variable_size_gru_encode(nbr_ped_embedding, nbr_ped_masks, self.nbr_enc)

            # Agent-node attention
            # nbr_encodings = torch.cat((nbr_vehicle_enc, nbr_ped_enc), dim=1)
            # queries = self.query_emb(lane_node_enc).permute(1, 0, 2)
            # keys = self.key_emb(nbr_encodings).permute(1, 0, 2)
            # vals = self.val_emb(nbr_encodings).permute(1, 0, 2)
            # attn_masks = torch.cat((inputs['agent_node_masks']['agent']['vehicles'],
            #                         inputs['agent_node_masks']['agent']['pedestrians']), dim=2)

            # att_op, _ = self.a_n_att(queries, keys, vals, attn_mask=attn_masks.bool())
            # att_op = att_op.permute(1, 0, 2)
            # indication_mask = (~attn_masks.bool())
            # nbr_attn_enc=torch.empty([0,lane_node_enc.shape[1],lane_node_enc.shape[2]]).cuda()
            # for i,mask in enumerate(indication_mask):
            #     if mask.any():
            #         nbr_attn_enc = torch.cat((nbr_attn_enc,att_op[i].unsqueeze(0)),dim=0)
            #     else:
            #         nbr_attn_enc = torch.cat((nbr_attn_enc,torch.zeros_like(att_op[i]).unsqueeze(0)),dim=0)
            # lane_node_enc=nbr_attn_enc+lane_node_enc
            # att_op[torch.isnan(att_op)]=0
            # Concatenate with original node encodings and 1x1 conv
            # lane_node_enc = self.leaky_relu(self.mix(torch.cat((lane_node_enc, att_op), dim=2)))

            # GAT layers
            adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
            for gat_layer in self.gat:
                lane_node_enc += gat_layer(lane_node_enc, adj_mat)
                
            # Target-node attention
            tgt_queries = self.leaky_relu(self.nd_tgt_emb(torch.cat((target_past_encdoings, target_future_encdoings), dim=1))).permute(1, 0, 2)
            nd_keys = self.leaky_relu(self.nd_key_emb(lane_node_enc)).permute(1, 0, 2)
            nd_vals = self.leaky_relu(self.nd_val_emb(lane_node_enc)).permute(1, 0, 2)
            t_n_attn_masks = inputs['agent_node_masks']['target']
            if self.tn_head_num>1:
                t_n_attn_masks=t_n_attn_masks.unsqueeze(1).repeat(1,self.tn_head_num,1,1).flatten(0,1)
            t_n_att_op, _ = self.t_n_att(tgt_queries, nd_keys, nd_vals, attn_mask=t_n_attn_masks.bool())
            t_n_att_op = t_n_att_op.permute(1, 0, 2)
            # t_n_att_op[torch.isnan(t_n_att_op)]=0
            # aug_past_enc=self.past_mix(torch.cat((target_past_encdoings,t_n_att_op[:,0].unsqueeze(1)),dim=2))
            # aug_future_enc=self.future_mix(torch.cat((target_future_encdoings,t_n_att_op[:,1].unsqueeze(1)),dim=2))
            target_agent_enc={'hist':target_past_encdoings,'future':target_future_encdoings,
                            'time_query':target_agent_representation['time_query'],
                            'map_info': t_n_att_op}
        else: 
            target_agent_enc={'hist':target_past_encdoings,'future':target_future_encdoings,
                            'time_query':target_agent_representation['time_query']}
        # Lane node masks
        lane_node_masks = ~lane_node_masks[:, :, :, 0].bool()
        lane_node_masks = lane_node_masks.any(dim=2)
        lane_node_masks = ~lane_node_masks
        lane_node_masks = lane_node_masks.float()

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc,
                     'context_encoding': {'combined': lane_node_enc,
                                          'combined_masks': lane_node_masks,
                                          'map': None,
                                          'vehicles': None,
                                          'pedestrians': None,
                                          'map_masks': None,
                                          'vehicle_masks': None,
                                          'pedestrian_masks': None
                                          },
                     }

        # Pass on initial nodes and edge structure to aggregator if included in inputs
        if 'init_node' in inputs:
            encodings['init_node'] = inputs['init_node']
            encodings['node_seq_gt'] = inputs['node_seq_gt']
            encodings['s_next'] = inputs['map_representation']['s_next']
            encodings['edge_type'] = inputs['map_representation']['edge_type']

        return encodings
    
    def get_ugru_enc(self,concat_mask,target_concat_embedding,hist_mask,future_mask):
        concat_seq_lens = torch.sum(1 - concat_mask[:,  :, 0], dim=-1)
        seq_lens_batched = concat_seq_lens[concat_seq_lens != 0].cpu()
        concat_embedding_packed = pack_padded_sequence(target_concat_embedding , seq_lens_batched,batch_first=True, enforce_sorted=False)
        hidden_batched, _ = self.bi_gru(concat_embedding_packed)
        hidden_unpacked, _ = pad_packed_sequence(hidden_batched, batch_first=True,total_length=10)
        hidden_unpacked=hidden_unpacked.reshape(hidden_unpacked.shape[0],hidden_unpacked.shape[1],2,-1)#B,L,2,C
        forward_hidden=hidden_unpacked[:,:,0]
        backward_hidden=hidden_unpacked[:,:,1]
        hist_seq_length=torch.sum(1 - hist_mask[:,  :, 0], dim=-1).cpu()
        future_seq_length=torch.sum(1 - future_mask[:,  :, 0], dim=-1).cpu()
        future_unpacked=get_bidirec_data(forward_hidden,hist_seq_length,future_seq_length+hist_seq_length)
        hist_packed = pack_padded_sequence(backward_hidden,hist_seq_length,batch_first=True, enforce_sorted=False)
        hist_unpacked, _ = pad_packed_sequence(hist_packed, batch_first=True)
        if hist_unpacked.shape[1]<hist_mask.shape[1]:
            hist_unpacked=torch.cat((hist_unpacked,torch.zeros([hist_unpacked.shape[0],hist_mask.shape[1]-hist_unpacked.shape[1],hist_unpacked.shape[-1]],device=hist_unpacked.device)),dim=1)
        
        return hist_unpacked,future_unpacked
    
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
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

        return encoding

    @staticmethod
    def build_adj_mat(s_next, edge_type):
        """
        Builds adjacency matrix for GAT layers.
        """
        batch_size = s_next.shape[0]
        max_nodes = s_next.shape[1]
        max_edges = s_next.shape[2]
        adj_mat = torch.diag(torch.ones(max_nodes, device=device)).unsqueeze(0).repeat(batch_size, 1, 1).bool()

        dummy_vals = torch.arange(max_nodes, device=device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        dummy_vals = dummy_vals.float()
        s_next[edge_type == 0] = dummy_vals[edge_type == 0]
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_edges)
        src_indices = torch.arange(max_nodes).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        adj_mat[batch_indices[:, :, :-1], src_indices[:, :, :-1], s_next[:, :, :-1].long()] = True
        adj_mat = adj_mat | torch.transpose(adj_mat, 1, 2)

        return adj_mat


class GAT(nn.Module):
    """
    GAT layer for aggregating local context at each lane node. Uses scaled dot product attention using pytorch's
    multihead attention module.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize GAT layer.
        :param in_channels: size of node encodings
        :param out_channels: size of aggregated node encodings
        """
        super().__init__()
        self.query_emb = nn.Linear(in_channels, out_channels)
        self.key_emb = nn.Linear(in_channels, out_channels)
        self.val_emb = nn.Linear(in_channels, out_channels)
        self.att = nn.MultiheadAttention(out_channels, 1)

    def forward(self, node_encodings, adj_mat):
        """
        Forward pass for GAT layer
        :param node_encodings: Tensor of node encodings, shape [batch_size, max_nodes, node_enc_size]
        :param adj_mat: Bool tensor, adjacency matrix for edges, shape [batch_size, max_nodes, max_nodes]
        :return:
        """
        queries = self.query_emb(node_encodings.permute(1, 0, 2))
        keys = self.key_emb(node_encodings.permute(1, 0, 2))
        vals = self.val_emb(node_encodings.permute(1, 0, 2))
        att_op, _ = self.att(queries, keys, vals, attn_mask=~adj_mat)

        return att_op.permute(1, 0, 2)

def get_bidirec_data(input_tensor, start_indices, end_indices, max_length=5):
    """
    Extracts data from a 3D tensor according to start and end indices.

    :param input_tensor: the input tensor of shape (B, T, C)
    :param start_indices: a tensor of shape (B,) containing the start indices for each mini-batch
    :param end_indices: a tensor of shape (B,) containing the end indices for each mini-batch
    :param max_length: the maximum length of the extracted data
    :return: a tensor of shape (B, Q, C), where Q is the maximum length
    """
    # Create a tensor to hold the extracted data
    B, T, C = input_tensor.shape
    Q = max_length
    extracted_data = torch.zeros((B, Q, C), dtype=input_tensor.dtype, device=input_tensor.device)

    # Iterate over each mini-batch and extract the data
    for b in range(B):
        start_index = int(start_indices[b])
        end_index = int(end_indices[b])
        length = end_index - start_index

        # Ensure the length is not greater than the maximum length
        if length > Q:
            length = Q

        # Extract the data
        if length > 0:
            extracted_data[b, :length] = extracted_data[b, :length]+input_tensor[b, start_index:end_index]

    return extracted_data
