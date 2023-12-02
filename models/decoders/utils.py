import torch
from models.decoders.ram_decoder import get_dense,get_index
from train_eval.utils import return_device
device = return_device()


def get_probs(coord,predictions,mask,H=122,W=122):
    horizon=coord.shape[-2]
    B=coord.shape[0]
    x_coord,y_coord=torch.meshgrid(torch.arange(B,device=coord.device), 
                                torch.arange(horizon,device=coord.device))
    indices_front=torch.cat([x_coord.unsqueeze(-1),y_coord.unsqueeze(-1)],dim=-1).long().unsqueeze(1).repeat(1,coord.shape[1],1,1)
    indices=torch.cat((indices_front,coord),dim=-1).long()
    if predictions.dim()==4:
        dense_pred=predictions
    else:
        nodes_2D=get_index(predictions,mask,H,W)
        dense_pred=get_dense(predictions,nodes_2D,H,W)
    ind0=indices.view(-1,4)[:,0]
    ind1=indices.view(-1,4)[:,1]
    ind2=indices.view(-1,4)[:,2]
    ind3=indices.view(-1,4)[:,3]
    probs=dense_pred[ind0,ind1,ind2,ind3].view(B,coord.shape[1],horizon).sum(-1)
    return probs

