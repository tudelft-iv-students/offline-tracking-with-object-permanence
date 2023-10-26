# from pcdet.datasets.nuscenes import nuscenes_utils
from nuscenes.utils import splits
import yaml,copy
from nuscenes import NuScenes
import torch
import torch.nn as nn
import os
import numpy as np
from collections import defaultdict
from nuscenes_dataset_match import NuScenesDataset_MATCH_EXT
from easydict import EasyDict

from torch.utils.data import DataLoader
import sys
from typing import Dict, Union
from models.model import PredictionModel
from models.encoders.match_encoder import MatchEncoder
from models.aggregators.match_agg import Match_agg
from models.decoders.match_decoder import Match_decoder
from data_extraction.prediction import PredictHelper_occ
# from torch.utils.data import DistributedSampler as _DistributedSampler
from scipy.optimize import linear_sum_assignment
from train_eval.utils import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import pickle
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["OMP_NUM_THREADS"] = "10"

class_names=[
            'car',
            'bus',
            'truck',
            'trailer'
            ]


def match_collate(batch_list):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)

    ret = {}
    map_representation = {}

    for key, val in data_dict.items():

        if key in ['history', 'history_mask','future', 'future_mask']:
            ret[key] = torch.stack(val,dim=0)
            
        elif key in ['lane_node_feats','lane_node_masks','lane_ctrs','s_next','edge_type']:
            value=torch.stack(val,dim=0)
            map_representation[key]=value
        elif key in ['matrix_indices']:
            ret[key] = torch.cat(val,dim=0)
        else:
            ret[key]=val
        ret['map_representation']=map_representation
    return ret

def convert_double_to_float(data: Union[Dict, torch.Tensor]):
    """
    Utility function to convert double tensors to float tensors in nested dictionary with Tensors
    """
    if type(data) is torch.Tensor and data.dtype == torch.float64:
        return data.float()
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert_double_to_float(v)
        return data
    else:
        return data

def make_scene_datasets(args,dataset_cfg,nusc,class_name,mode='load_data'):
    version = dataset_cfg.VERSION
    if version == 'v1.0-trainval':
        val_scenes = splits.val
    elif version == 'v1.0-test':
        val_scenes = splits.test
    elif version == 'v1.0-mini':
        val_scenes = splits.mini_val
    datasets=[]
    global helper
    helper =PredictHelper_occ(nusc)
    
    for scene_idx in range(len(val_scenes)):
        dataset=NuScenesDataset_MATCH_EXT(scene_idx,
            dataset_cfg=dataset_cfg, class_name=class_name,
            logger=create_logger(),helper=helper,mode=mode)
        if not dataset.skip:
            datasets.append(dataset)
    return datasets

def draw(hist,
         hist_mask,
         future,
         future_mask,
         origin,
         dist,
         final_scores,
         sample_token,
         lane_node_feats,
         lane_node_masks,
         max_ind,
         scene_idx,
         batch_idx,
         img_save_dir):
    layer_names = ['drivable_area', 'ped_crossing']
    maps= load_all_maps(helper)

    colors = [(255, 255, 255), (119, 136, 153)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue","violet","red"])

    image_side_length = 2 * max(25,dist)
    image_side_length_pixels = 640
    resolution=image_side_length_pixels/image_side_length
    patchbox = get_patchbox(origin[0], origin[1], image_side_length)

    angle_in_degrees = angle_of_rotation(origin[2]) * 180 / np.pi
    # sample_annotation = helper.get_sample_annotation(instance_token, sample_token)
    map_name = helper.get_map_name_from_sample_token(sample_token)
    canvas_size = (image_side_length_pixels, image_side_length_pixels)
    masks = maps[map_name].get_map_mask(patchbox, angle_in_degrees, layer_names, canvas_size=canvas_size)
    
    images = []
    for mask, color in zip(masks, colors):
        images.append(change_color_of_binary_mask(np.repeat(mask[::-1, :, np.newaxis], 3, 2), color))
    image = Rasterizer().combine(images)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 
    lanes=lane_node_feats.flatten(0,1).clone()
    lanes_mask=~lane_node_masks.flatten(0,1)[:,0].bool()

    xsl, ysl, dxsl, dysl=local_pose_to_image(lanes,lanes_mask,resolution,canvas_size,2.0)
    xs, ys, dxs, dys = local_pose_to_image(hist,hist_mask,resolution,canvas_size,10.0)

    ax.scatter(xsl, ysl,s=3,color='orange',alpha=0.8)

    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        ax.arrow(x, y, dx, dy, width=5, color=(0,1,0,1))

    valid_length=(future_mask.any(-1)).sum()


    for idx in range(valid_length):
        vehicles=future[idx]
        mask=future_mask[idx]
        if idx==max_ind:
            # print(vehicles)
            xsf, ysf, dxsf, dysf = local_pose_to_image(vehicles,mask,resolution,canvas_size,12.0)
            color=cmap(final_scores[idx].item())
            assert(final_scores[idx].item()>0.9)
            for x, y, dx, dy in zip(xsf, ysf, dxsf, dysf):
                ax.arrow(x, y, dx, dy, width=5, color=color)
        else:
            xsf, ysf, dxsf, dysf = local_pose_to_image(vehicles,mask,resolution,canvas_size,8.0)
            color=cmap(final_scores[idx].item())
            for x, y, dx, dy in zip(xsf, ysf, dxsf, dysf):
                ax.arrow(x, y, dx, dy, width=3, color=color)

    ax.grid(False)
    ax.imshow(image)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    im.imsave(os.path.join(img_save_dir,'scene'+str(scene_idx)+'_batch'+str(batch_idx)+'_max'+str(max_ind.item())), image_from_plot)
    
def vis_associations(vis_idx,mini_batch,final_scores,scene_idx,batch_idx,img_save_dir):
    for idx in vis_idx:
        hist=mini_batch['history'][idx]
        hist_mask=~(mini_batch['history_mask'][idx,:,0].bool())
        future=mini_batch['future'][idx]
        future_mask=~(mini_batch['future_mask'][idx,:,:,0].bool())
        # vis_future=future[future_mask]
        origin=mini_batch['origin'][idx]
        max_ind=torch.max(final_scores[idx].flatten(0,1),-1)[-1]
        # time_ind=torch.max(mini_batch['future_mask'][idx,max_ind,:,0],-1)[-1]-1
        farest_loc=future[max_ind,:,:2]
        dist=(torch.max(torch.norm(farest_loc,p=2,dim=-1))+10).item()
        # print(dist)
        sample_token=mini_batch['sample_token'][idx]
        lane_node_feats=mini_batch['map_representation']['lane_node_feats'][idx]
        lane_node_masks=mini_batch['map_representation']['lane_node_masks'][idx]
        scores=final_scores[idx].squeeze(-1)
        draw(hist,
             hist_mask,
             future,
             future_mask,
             origin,dist,
             scores,
             sample_token,
             lane_node_feats,
             lane_node_masks,
             max_ind,
             scene_idx,
             batch_idx,
             img_save_dir)

def build_and_load_model(ckpt_path,model_cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()
    print("Loading checkpoint from " + ckpt_path + " ...", end=" ")
    model = PredictionModel(
        MatchEncoder(model_cfg['encoder_args']),
        Match_agg(model_cfg['aggregator_args']),
        Match_decoder(model_cfg['decoder_args'])
    ).float().to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path,map_location='cuda:0')
    else:
        checkpoint = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Done')
    return model

def greedy_match(matching_matrix,trk_ids):
    matching_matrix=matching_matrix.clone()
    matches={}
    while True:
        largest_score=torch.max(matching_matrix)
        if largest_score<=0:
            break
        row_idx,col_idx = torch.nonzero(matching_matrix==largest_score)[0,0],torch.nonzero(matching_matrix==largest_score)[0,1]
        matching_matrix[row_idx]=0
        matching_matrix[:,col_idx]=0
        key=trk_ids[col_idx]
        val=trk_ids[row_idx]
        matches[key]=val
    return matches
def get_diff_mat(times1,times2):
    times1=times1.unsqueeze(-1)
    times2=times2.unsqueeze(0)
    return times1-times2
def load_time_diff(scene_token,class_name,data_dir,version='v1.0-test'):
    filepath=Path(data_dir)/version/class_name/scene_token/'time_diff.pickle'
    if not os.path.isfile(filepath):
        raise Exception('Could not find data. Please run the dataset in compute_stats mode')
    with open(filepath, 'rb') as handle:
        time_diff = pickle.load(handle)
            
    return time_diff

def time_diff2decay(time_diff,decay_factor=0.975,buffer_time=1.5,decay_thresh=4.5):
    mask1=(time_diff>=-buffer_time)
    decay_matrix=decay_factor**(-decay_thresh-time_diff)
    decay_matrix[mask1]=1
    # decay_matrix=torch.ones_like(time_diff)
    # decay_matrix[time_diff<-decay_thresh]=0
    return decay_matrix

def send_to_device(data: Union[Dict, torch.Tensor]):
    """
    Utility function to send nested dictionary with Tensors to GPU
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(data) is torch.Tensor:
        return data.to(device).clone().detach()
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v)
        return data
    else:
        return data

def save_decay_matrix(dataset_cfg,args,nusc,match_info_dir,class_name):
    file_name=class_name + '_' + 'match_info.pkl'
    datasets=make_scene_datasets(args,dataset_cfg,nusc,class_name,mode='add_time_decay')
    if ((match_info_dir / file_name).exists()):
        
        with open(match_info_dir / file_name, 'rb') as f:
            match_info=pickle.load(f)
            f.close()
        similairty_matrices,similarity_matrices_with_map,id_lists,s_t_list = match_info
        decay_matrices=[]
        for i,s_t in enumerate(s_t_list):
            assert(datasets[i].scene_token == s_t)
            decay_matrix=datasets[i].decay_matrix
            assert(similairty_matrices[i].shape == decay_matrix.shape)
            decay_matrices.append(decay_matrix)
        match_info=(similairty_matrices,similarity_matrices_with_map,id_lists,s_t_list,decay_matrices)
        with open(match_info_dir / file_name, 'wb') as f:
            pickle.dump(match_info, f)
            f.close()
            
        return similairty_matrices,similarity_matrices_with_map,id_lists,s_t_list,decay_matrices
    else:
        raise Exception("No match info!!!")


def get_similarity_matrix(dataset_cfg,args,nusc,match_info_dir,class_name):
    file_name=class_name + '_' + 'match_info.pkl'
    if ((match_info_dir / file_name).exists()):
    
        with open(match_info_dir / file_name, 'rb') as f:
            match_info=pickle.load(f)
            f.close()
        similairty_matrices,similarity_matrices_with_map,id_lists,s_t_list = match_info
        return similairty_matrices,similarity_matrices_with_map,id_lists,s_t_list
    else:
        datasets=make_scene_datasets(args,dataset_cfg,nusc,class_name)
        dataloaders = [DataLoader(
            nuscenes_dataset, batch_size=dataset_cfg.batch_size, pin_memory=True, num_workers=12,
            shuffle=False, collate_fn=match_collate,
            drop_last=False) for nuscenes_dataset in datasets]
        
        motion_model = build_and_load_model(dataset_cfg.motion_ckpt_path,dataset_cfg.motion_model_cfg)
        map_model = build_and_load_model(dataset_cfg.map_ckpt_path,dataset_cfg.map_model_cfg)
        similairty_matrices=[]
        id_lists=[]
        s_t_list=[]
        similarity_matrices_with_map=[]
        with torch.no_grad():
            motion_model.eval()
            map_model.eval()
            for scene_idx,dl in enumerate(dataloaders):
                sys.stdout.write('processing %d, %d/%d\r' % (scene_idx, scene_idx+1, len(dataloaders)))
                sys.stdout.flush()
                similarity_mask=dl.dataset.mask
                id_list=dl.dataset.tracker_ids
                similarity_matrix=torch.zeros_like(similarity_mask).float()

                similarity_matrix_with_map=torch.zeros_like(similarity_mask).float()
                for mini_batch in dl:
                    mini_batch=send_to_device(convert_double_to_float(mini_batch))
                    predictions1=motion_model(mini_batch)
                    predictions2=map_model(mini_batch)
                    masks=(~(predictions1['masks'][:,:,:,0].bool())).any(dim=-1)
                    scores=predictions1['scores'][masks].cpu()                    
                    similarity_matrix[mini_batch['matrix_indices'][:,0],mini_batch['matrix_indices'][:,1]]=scores.squeeze(-1)

                    map_scores=predictions2['map_scores'][masks].cpu()
                    similarity_matrix_with_map[mini_batch['matrix_indices'][:,0],mini_batch['matrix_indices'][:,1]]=map_scores.squeeze(-1)
                assert(similarity_matrix[~similarity_mask].sum()==0)
                similairty_matrices.append(similarity_matrix)
                id_lists.append(id_list)
                s_t_list.append(dl.dataset.scene_token)

                similarity_matrices_with_map.append(similarity_matrix_with_map)
        match_info=(similairty_matrices,similarity_matrices_with_map,id_lists,s_t_list)
        with open(match_info_dir / file_name, 'wb') as f:
            pickle.dump(match_info, f)
            f.close()
        return similairty_matrices,similarity_matrices_with_map,id_lists,s_t_list

def visualization_function(dataset_cfg,args,nusc,img_save_dir,class_name):
    datasets=make_scene_datasets(args,dataset_cfg,nusc,class_name)
    dataloaders = [DataLoader(
        nuscenes_dataset, batch_size=dataset_cfg.batch_size, pin_memory=True, num_workers=12,
        shuffle=False, collate_fn=match_collate,
        drop_last=False) for nuscenes_dataset in datasets]
    motion_model = build_and_load_model(dataset_cfg.motion_ckpt_path,dataset_cfg.motion_model_cfg)
    map_model = build_and_load_model(dataset_cfg.map_ckpt_path,dataset_cfg.map_model_cfg)
    with torch.no_grad():
        motion_model.eval()
        map_model.eval()
        for scene_idx,dl in enumerate(dataloaders):
            sys.stdout.write('processing %d, %d/%d\r' % (scene_idx, scene_idx+1, len(dataloaders)))
            sys.stdout.flush()
            for batch_idx,mini_batch in enumerate(dl):
                mini_batch=send_to_device(convert_double_to_float(mini_batch))
                predictions1=motion_model(mini_batch)
                predictions2=map_model(mini_batch)
                masks=((~(predictions1['masks'][:,:,:,0].bool())).any(dim=-1)).float().unsqueeze(-1).cpu()
                scores=predictions1['scores'].cpu()*masks                 
                # similarity_matrix[mini_batch['matrix_indices'][:,0],mini_batch['matrix_indices'][:,1]]=scores.squeeze(-1)
                map_scores=predictions2['map_scores'].cpu()*masks
                final_scores=(scores+map_scores)/2
                max_scores_inds=torch.max(final_scores.flatten(1,2),-1)
                bool_inds=max_scores_inds[0]>0.9
                vis_idx=torch.arange(len(bool_inds))[bool_inds]
                if len(vis_idx)>0:
                    vis_associations(vis_idx,mini_batch,final_scores,scene_idx,batch_idx,img_save_dir)


def refine_association(result: dict,
                       nusc: NuScenes,
                       thresh: float,
                       match_type: str,
                       similarity_matrices: dict,
                       id_lists: dict,
                       s_t_lists: dict,
                       data_dir,
                       map_similarity_matrices: dict=None,
                       association_rounds=1,
                       add_map=True,
                       decay_factor=0.975,
                       version="v1.0-test"):
    for class_name in class_names:
        matched_id_dicts=defaultdict(dict)
        class_matrices=similarity_matrices[class_name]
        class_id_lists=id_lists[class_name]
        class_s_t_list=s_t_lists[class_name]
        if add_map:
            class_map_similarity_matrices=map_similarity_matrices[class_name]
        for scene_idx,(base_matrix,id_list) in enumerate(zip(class_matrices,class_id_lists)):
            matched_ids={}
            similarity_matrix=base_matrix.clone()
            time_diff=load_time_diff(class_s_t_list[scene_idx],class_name,data_dir,version)
            class_decay_matrix=time_diff2decay(time_diff,decay_factor=decay_factor)
            if match_type == 'greedy':
                # similarity_matrices[type_key]=similarity_matrix
                similarity_matrix*=(class_decay_matrix).clone()
                score_mask=similarity_matrix<thresh
                
                if add_map:
                    similarity_matrix_with_map=class_map_similarity_matrices[scene_idx].clone()
                    similarity_matrix_with_map*=(class_decay_matrix).clone()
                    map_score_mask=(similarity_matrix_with_map<thresh)
                    final_mask=map_score_mask*score_mask
                    matching_matrix=(similarity_matrix+similarity_matrix_with_map)/2
                    matching_matrix[final_mask]=0
                else:
                    matching_matrix=similarity_matrix.clone()
                    matching_matrix[score_mask]=0
                match_result=greedy_match(matching_matrix.cpu(),id_list)
                print('Total match quries: %d , number of matches: %d' % (len(similarity_matrix),len(match_result)))
            elif match_type == 'hungarian':

                matching_matrix=base_matrix.clone().cpu().numpy()
                matching_matrix*=(class_decay_matrix).clone().cpu().numpy()
                score_mask=matching_matrix<thresh
                if add_map:
                    similarity_matrix_with_map=class_map_similarity_matrices[scene_idx].clone().cpu().numpy()
                    similarity_matrix_with_map*=(class_decay_matrix).clone().cpu().numpy()
                    map_score_mask=similarity_matrix_with_map<thresh
                    final_mask=map_score_mask*score_mask
                    matching_matrix=(matching_matrix+similarity_matrix_with_map.copy())/2
                    matching_matrix[final_mask]=0
                else:
                    matching_matrix[score_mask]=0
                row_ind, col_ind = linear_sum_assignment(matching_matrix,maximize=True)
                match_result={}
                trk_ids=id_list
                for idx,(row,col) in enumerate(zip(row_ind, col_ind)):
                    if matching_matrix[row,col] > thresh:
                        key=int(trk_ids[col])
                        val=int(trk_ids[row])
                        match_result[str(key)]=str(val)
                # print(class_name)
                print('Total match quries: %d , number of matches: %d' % (len(similarity_matrix),len(match_result)))
            else:
                raise Exception('Match type should be either hungarian or greedy')
            matched_ids.update(match_result)
            matched_id_dicts[class_s_t_list[scene_idx]]=matched_ids
        for i in range(association_rounds):
            for scene_token in matched_id_dicts.keys():
                scene_info = nusc.get('scene',scene_token)
                sample_token = scene_info['first_sample_token']
                for sample_idx in range(scene_info['nbr_samples']):
                    predictions=result[sample_token]
                    for idx,box in enumerate(predictions):
                        if box['tracking_id'] in matched_id_dicts[scene_token].keys():
                            assert(box['tracking_name']==class_name)
                            box['tracking_id']=matched_id_dicts[scene_token][box['tracking_id']]
                            # print('Refined!!')
                            # result[sample_token][idx]['tracking_id']=matched_id_dicts[scene_token][box['tracking_id']]
                    sample_token = nusc.get('sample',sample_token)['next']
    return result
    

if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="data_extraction/nuscenes_dataset_occ.yaml", help='specify the config of dataset')
    # parser.add_argument('--version', type=str, default='v1.0-test', help='')
    parser.add_argument('--result_path', type=str, default='mot_results/v1.0-test/tracking_result.json', help='')
    parser.add_argument('--data_path', type=str, default="/home/stanliu/data/mnt/nuScenes/nuscenes", help='')
    parser.add_argument('--ckpt_path', type=str, help='Trained motion matcher', required=False)
    parser.add_argument('--save_dir', type=str, default='mot_results/Re-ID_results/')
    parser.add_argument('--add_map', type=bool, default=True)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    thresh = dataset_cfg.ASSOCIATION_THRESHOLD
    DATA_DIR = dataset_cfg.DATA_DIR
    type_name = dataset_cfg.ASSOCIATION_METHOD
    decay_factor= dataset_cfg.DECAY_FACTOR
    version= dataset_cfg.VERSION
    nusc = NuScenes(version=version, dataroot=args.data_path, verbose=True)
    with open(args.result_path,'rb') as f:
        original_data = json.load(f)
    if args.visualize:
        match_info_dir=Path(os.path.join(args.save_dir,'matching_info',version,'plots'))
        match_info_dir.mkdir(parents=True, exist_ok=True) 
        for class_name in class_names:
            visualization_function(dataset_cfg,args,nusc,match_info_dir,class_name)


    else:
        match_info_dir=Path(os.path.join(args.save_dir,'matching_info',version))
        match_info_dir.mkdir(parents=True, exist_ok=True) 

        similarity_matrices={}
        map_similarity_matrices={}
        id_lists={}
        s_t_lists={}
        for class_name in class_names:
            similarity_matrix, map_similarity_matrix,id_list,s_t_list =get_similarity_matrix(dataset_cfg,args,nusc,match_info_dir,class_name)
            similarity_matrices[class_name] = similarity_matrix
            map_similarity_matrices[class_name] = map_similarity_matrix
            id_lists[class_name] = id_list
            s_t_lists[class_name] = s_t_list

        data=copy.deepcopy(original_data)
        result=data['results']
        refined_result=refine_association(result,
                                        nusc,
                                        thresh,
                                        type_name,
                                        similarity_matrices,
                                        id_lists,
                                        s_t_lists,
                                        DATA_DIR,
                                        map_similarity_matrices,
                                        decay_factor=decay_factor,
                                        version=version)
        data['results']=refined_result
        data['meta']['use_map']=True
        Path(os.path.join(args.save_dir,'motion_refine_with_map',version)).mkdir(parents=True, exist_ok=True) 
        with open(os.path.join(args.save_dir,'motion_refine_with_map',version,type_name+str(thresh)+'_tracking_result.json'), "w") as f:
            json.dump(data, f)