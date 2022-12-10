import torch
import torch.nn as nn
from models.aggregators.aggregator import PredictionAggregator
from models.library.blocks import *
from models.library.RasterSampler import *
from typing import Dict, Tuple
from models.library.blocks import CNNBlock


class HomeAggregator(PredictionAggregator):
    """
    Aggregate context encoding home decoder, composed of multiple transposed convolution blocks.
    """

    def __init__(self, args: Dict):

        """
        args to include

        enc_size: int Dimension of encodings generated by encoder
        emb_size: int Size of embeddings used for queries, keys and values
        num_heads: int Number of attention heads

        """
        super().__init__()
        if args['backbone']=='resnet50':
            input_channel=2048+args['target_agent_enc_size']
        elif (args['backbone']=='resnet18' or args['backbone']=='resnet34'):
            input_channel=512+args['target_agent_enc_size']
        else:
            raise RuntimeError("The encoder should be a raster encoder!")
        interm_channel=int((input_channel+args['context_enc_size'])/2)
        self.dim_reduction_block=nn.Sequential(
            nn.Conv2d(input_channel, interm_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(interm_channel),
            nn.ReLU(),
            nn.Conv2d(interm_channel, args['context_enc_size'], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(args['context_enc_size']),
            nn.ReLU()
        )
        self.query_emb = nn.Linear(2, args['emb_size'])
        self.key_emb = nn.Linear(args['context_enc_size'], args['emb_size'])
        self.val_emb = nn.Linear(args['context_enc_size'], args['emb_size'])
        self.mha = nn.MultiheadAttention(args['emb_size'], args['num_heads'])
        self.sampler = Sampler(args,resolution=args['resolution'])
        self.num_heads = args['num_heads']
        self.conv = args['convolute']
        if self.conv:
            self.conv_kernel=int(args['conv_kernel_ratio']*self.sampler.H)
            if self.conv_kernel % 2 ==0:
                self.conv_kernel+=1
            self.final_convs = nn.Sequential(CNNBlock(in_channels=args['emb_size'], out_channels=3*args['emb_size']//4, kernel_size=self.conv_kernel,padding=self.conv_kernel//2),
                                CNNBlock(in_channels=3*args['emb_size']//4, out_channels=args['emb_size']//2, kernel_size=self.conv_kernel,padding=self.conv_kernel//2))


    def forward(self, encodings: Dict) -> torch.Tensor:
        """
        Forward pass for attention aggregator
        """
        
        target_agent_enc = encodings['target_agent_encoding']
        context_enc = encodings['context_encoding']
        if context_enc['combined'] is not None:
            combined_enc, map_mask = context_enc['combined'], context_enc['map_masks'].bool()
        else:
            combined_enc, _ = self.get_combined_encodings(context_enc)

        augmented_target_agent_enc = target_agent_enc.unsqueeze(2).unsqueeze(3).repeat(1,1,combined_enc.shape[-2],combined_enc.shape[-1])
        concatenated_encodings=torch.cat([combined_enc,augmented_target_agent_enc],dim=1)
        context_encoding = self.dim_reduction_block(concatenated_encodings)##Fuse agent feat with map feat by compressing the dimension (actually is linear layer)
        context_encoding = context_encoding.view(context_encoding.shape[0], context_encoding.shape[1], -1)## [Batch number, channel, H*W]
        context_encoding = context_encoding.permute(0, 2, 1)## [Batch number, H*W, channel]

        nodes_2D=self.sampler.sample_goals().repeat(context_encoding.shape[0],1,1).type(torch.float32)
        mask_under=(self.sampler.sample_mask(map_mask))
        attn_mask=~mask_under.unsqueeze(-1).repeat(self.num_heads,1,context_encoding.shape[1])
        query = self.query_emb(nodes_2D).permute(1,0,2)
        keys = self.key_emb(context_encoding).permute(1, 0, 2)
        vals = self.val_emb(context_encoding).permute(1, 0, 2)
        attn_output, attn_output_weights = self.mha(query, keys, vals, attn_mask=attn_mask)
        op = attn_output.permute(1,0,2)
        op = op.view(op.shape[0],self.sampler.H,self.sampler.W,-1)
        # op = torch.cat((target_agent_enc, op), dim=-1)
        if self.conv:
            op=self.final_convs(op.permute(0,3,1,2))

        return op

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
