import torch

def CVM_pred(inputs):
    future = inputs['future'].clone()
    history = inputs['history'].clone()
    future_mask = inputs['future_mask'].clone()
    future_mask_bool=~(future_mask.bool())
    history_mask = inputs['history_mask'].clone()
    last_inds=torch.argmax(future_mask[:,:,:,0],dim=-1).cpu()-1
    inds1,inds2=torch.meshgrid(torch.arange(last_inds.shape[0]), 
                                torch.arange(last_inds.shape[1]))
    inds=(torch.cat((inds1.unsqueeze(-1),inds2.unsqueeze(-1),last_inds.unsqueeze(-1)),dim=-1).flatten(0,1)).T
    future_ctrs=future[:,:,:,:2]
    future_times=future[:,:,:,5]
    last_ctrs=future_ctrs[inds[0],inds[1],inds[2]].view(future_ctrs.shape[0],future_ctrs.shape[1],2)
    last_times=future_times[inds[0],inds[1],inds[2]].view(future_ctrs.shape[0],future_ctrs.shape[1],1)

    hist_last_inds=torch.argmax(history_mask[:,:,0],dim=-1).cpu()-1
    hist_inds1=torch.arange(hist_last_inds.shape[0])
    hist_inds=torch.cat((hist_inds1.unsqueeze(-1),hist_last_inds.unsqueeze(-1)),dim=-1).T
    hist_ctrs=history[:,:,:2]
    hist_velos=history[:,:,-2:]
    hist_last_velos=hist_velos[hist_inds[0],hist_inds[1]].view(hist_velos.shape[0],2)
    hist_last_ctrs=hist_ctrs[hist_inds[0],hist_inds[1]].view(hist_ctrs.shape[0],2)
    pseudo_scores=torch.zeros([future.shape[0],future.shape[1],1])
    for index,last_ctr in enumerate(hist_last_ctrs):
        sample_mask=future_mask_bool[index,:,:,0].any(-1)#Number of total future tracklets
        velocity=hist_last_velos[index].unsqueeze(0)
        valid_times=last_times[index][sample_mask]
        pred_future_locations=valid_times@velocity+last_ctr
        candidate_ctrs=last_ctrs[index][sample_mask]
        distances=torch.norm(pred_future_locations-candidate_ctrs,p=2,dim=1)
        select_idx=torch.argmin(distances)
        pseudo_scores[index,select_idx]=1
    return pseudo_scores