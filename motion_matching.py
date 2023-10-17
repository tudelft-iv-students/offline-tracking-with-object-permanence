# from pcdet.datasets.nuscenes import nuscenes_utils
from nuscenes.utils import splits
import yaml,copy
from nuscenes import NuScenes
import torch
import torch.nn as nn
import os
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
from train_eval.utils import create_logger
import json
import numpy as np
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
    version = args.version
    if version == 'v1.0-trainval':
        val_scenes = splits.val
    elif version == 'v1.0-test':
        val_scenes = splits.test
    elif version == 'v1.0-mini':
        val_scenes = splits.mini_val
    # nusc = NuScenes(version=version, dataroot=args.data_path, verbose=True)
    # available_scenes = nuscenes_utils.get_available_scenes(nusc)
    # available_scene_names = [s['name'] for s in available_scenes]
    # val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])
    datasets=[]
    helper =PredictHelper_occ(nusc)
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    
    for scene_idx in range(len(val_scenes)):
        dataset=NuScenesDataset_MATCH_EXT(scene_idx,
            dataset_cfg=dataset_cfg, class_name=class_name,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=create_logger(),helper=helper,mode=mode)
        if not dataset.skip:
            datasets.append(dataset)
    return datasets

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
    largest_score=100
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
def load_time_diff(scene_token,class_name,data_dir,version='v1.0-trainval'):
    filepath=Path(data_dir)/version/class_name/scene_token/'time_diff.pickle'
    if not os.path.isfile(filepath):
        raise Exception('Could not find data. Please run the dataset in compute_stats mode')
    with open(filepath, 'rb') as handle:
        time_diff = pickle.load(handle)
            
    return time_diff

def time_diff2decay(time_diff,decay_factor=0.95,buffer_time=1.5,decay_thresh=3.0):
    mask1=(time_diff>=-buffer_time)
    decay_matrix=decay_factor**(-decay_thresh-time_diff)
    decay_matrix[mask1]=1
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
                       decay_factor=0.95):
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
            time_diff=load_time_diff(class_s_t_list[scene_idx],class_name,data_dir)
            class_decay_matrix=time_diff2decay(time_diff,decay_factor=decay_factor)
            if match_type == 'greedy':
                # similarity_matrices[type_key]=similarity_matrix
                similarity_matrix*=(class_decay_matrix).clone()
                score_mask=similarity_matrix<thresh
                
                if add_map:
                    similarity_matrix_with_map=class_map_similarity_matrices[scene_idx].clone()
                    similarity_matrix_with_map*=(class_decay_matrix).clone()
                    map_score_mask=similarity_matrix_with_map<thresh
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
    parser.add_argument('--version', type=str, default='v1.0-test', help='')
    parser.add_argument('--result_path', type=str, default='mot_results/v1.0-test/tracking_result.json', help='')
    parser.add_argument('--data_path', type=str, default="/home/stanliu/data/mnt/nuScenes/nuscenes", help='')
    parser.add_argument('--ckpt_path', type=str, help='Trained motion matcher', required=False)
    parser.add_argument('--save_dir', type=str, default='mot_results/Re-ID_results')
    parser.add_argument('--add_map', type=bool, default=True)
    parser.add_argument('--decay', action='store_true')
    args = parser.parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.data_path, verbose=True)
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    thresh = dataset_cfg.ASSOCIATION_THRESHOLD
    DATA_DIR = dataset_cfg.DATA_DIR
    type_name = dataset_cfg.ASSOCIATION_METHOD
    decay_factor= dataset_cfg.DECAY_FACTOR
    with open(args.result_path,'rb') as f:
        original_data = json.load(f)
    match_info_dir=Path(os.path.join(args.save_dir,'matching_info',args.version))
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
    refined_result=refine_association(result,nusc,thresh,type_name,similarity_matrices,id_lists,s_t_lists,DATA_DIR,map_similarity_matrices,decay_factor=decay_factor)
    data['results']=refined_result
    Path(os.path.join(args.save_dir,'motion_refine_with_map',args.version)).mkdir(parents=True, exist_ok=True) 
    with open(os.path.join(args.save_dir,'motion_refine_with_map',args.version,type_name+str(thresh)+'_tracking_result.json'), "w") as f:
        json.dump(data, f)