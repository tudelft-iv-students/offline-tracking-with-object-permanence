import copy
import pickle
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from numpy import linalg as LA
from collections import defaultdict
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, Union, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from data_extraction.prediction import PredictHelper_occ
from nuscenes.map_expansion.map_api import NuScenesMap
from train_eval.utils import create_logger
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from itertools import compress
import copy
from data_extraction.bbox import *
from data_extraction.data_extraction import *
import torch.utils.data as torch_data
import numpy.ma as ma

os.environ["MKL_NUM_THREADS"] = "18"
os.environ["NUMEXPR_NUM_THREADS"] = "18"
os.environ["OMP_NUM_THREADS"] = "18"


def add_lane_ctrs(encodings,mask):
    lane_ctrs=(ma.masked_array(encodings[:,:,:2],mask=mask[:,:,:2])).mean(axis=1).data
    lane_ctrs[(~(((1-mask).astype(bool))[:,:,:2].any(1)))]=np.inf
    return lane_ctrs

def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw

def global_to_local(origin: Tuple, global_pose: Tuple) -> Tuple:
    """
    Converts pose in global co-ordinates to local co-ordinates.
    :param origin: (x, y, yaw) of origin in global co-ordinates
    :param global_pose: (x, y, yaw) in global co-ordinates
    :return local_pose: (x, y, yaw) in local co-ordinates
    """
    # Unpack
    global_x, global_y, global_yaw = global_pose
    origin_x, origin_y, origin_yaw = origin

    # Translate
    local_x = global_x - origin_x
    local_y = global_y - origin_y

    # Rotate
    global_yaw = correct_yaw(global_yaw)
    theta = np.arctan2(-np.sin(global_yaw-origin_yaw), np.cos(global_yaw-origin_yaw))

    r = np.asarray([[np.cos(np.pi/2 - origin_yaw), np.sin(np.pi/2 - origin_yaw)],
                    [-np.sin(np.pi/2 - origin_yaw), np.cos(np.pi/2 - origin_yaw)]])
    local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

    local_pose = (local_x, local_y, theta)

    return local_pose

def list_to_tensor(feat_list: List[np.ndarray], max_num: int, max_len: int,
                    feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
    forming mini-batches

    :param feat_list: List of sequential features
    :param max_num: Maximum number of sequences in List
    :param max_len: Maximum length of each sequence
    :param feat_size: Feature dimension
    :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
        2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
    """
    feat_array = np.zeros((max_num, max_len, feat_size))
    
    mask_array = np.ones((max_num, max_len, feat_size))
    for n, feats in enumerate(feat_list):
        feat_array[n, :len(feats), :] = feats
        mask_array[n, :len(feats), :] = 0

    return feat_array, mask_array

class NuScenesDataset_MATCH_EXT(torch_data.Dataset):
    def __init__(self, scene_idx, dataset_cfg, class_name,  helper, mode,root_path=None, logger=None):
        # if root_path is None:
        #     raise NotImplementedError()
        super().__init__()
        if dataset_cfg.get('VERSION', None) is None:
            try:
                dataset_cfg.VERSION = dataset_cfg.version
            except:
                raise Exception('Specify version!')
        self.save_path = (Path(__file__).resolve().parent ).resolve() / 'extracted_mot_data' / 'final_version_nms'/ dataset_cfg.VERSION / class_name
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.info_path = (Path(__file__).resolve().parent ).resolve() / 'extracted_mot_data'/'final_version_nms'/ dataset_cfg.VERSION
        self.dataset_cfg=dataset_cfg
        self.version=dataset_cfg.VERSION
        self.logger = logger
        self.infos = []
        split_name = 'test'
        self.class_name = class_name
        self.helper=helper
        self.mode = mode

        self.include_nuscenes_data(split_name,scene_idx)
        
        map_locs = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.maps = {i: NuScenesMap(map_name=i, dataroot=self.helper.data.dataroot) for i in map_locs}
        self.polyline_length=20
        self.polyline_resolution=1
        self.map_extent=[-100, 100, -100, 150]
        self.no_velo=True
        
    def include_nuscenes_data(self, split_name,scene_idx):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.MOTION_INFO_PATH[split_name]:
            info_path = self.info_path / info_path
            if not info_path.exists():
                raise NotImplementedError()
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos[scene_idx]['scene_tracks'])
                self.scene_token=infos[scene_idx]['scene_token']

        self.logger.info('Total tracks in scene %d : %d' % (scene_idx,len(nuscenes_infos)))
        if self.mode != 'compute_stats':
            stats=self.load_data()
            scene_info=self.load_data(load_scene_info=True)
            self.mask=scene_info['mask']
            if self.mask.any(-1).sum() != 0 and self.mask is not None:
                self.max_length=stats['max_length']
                self.max_nodes=self.max_num_nodes=stats['num_lane_nodes']
                self.max_nbr_nodes=stats['num_nbrs']
                self.right_time=scene_info['right_time']
                self.look_up_list=scene_info['look_up_list']
                self.tracker_ids=scene_info['tracker_ids']
                self.skip = False
                self.class_tracks=scene_info['class_tracks']
                self.mask_row_indices=scene_info['mask_row_indices']
                # if 'decay_matrix' in scene_info.keys():
                #     self.decay_matrix=scene_info['decay_matrix']
            else:
                self.skip = True
                self.right_time=None
            if self.mode == 'add_time_diff' and not self.skip:
                left_times,right_times,num_obs,look_up_lists,tracker_ids,class_tracks=self.get_times(nuscenes_infos)
                self.diff_matrix=self.get_diff_mat(torch.Tensor(right_times[self.class_name]),torch.Tensor(left_times[self.class_name]))
                # scene_info['decay_matrix']=self.decay_matrix
                self.save_data(self.diff_matrix,save_time_diff=True)
        else:
            left_times,right_times,num_obs,look_up_lists,tracker_ids,class_tracks=self.get_times(nuscenes_infos)
            self.left_time=left_times[self.class_name]
            self.right_time=right_times[self.class_name]
            num_obs=num_obs[self.class_name]
            self.look_up_list=look_up_lists[self.class_name]
            self.tracker_ids=tracker_ids[self.class_name]
            
            frame_num_filter=torch.Tensor(num_obs)>1
            # frame_num_mask=(frame_num_filter.unsqueeze(0)*frame_num_filter.unsqueeze(-1))
            
            frame_num_mask=frame_num_filter.unsqueeze(-1).repeat(1,len(self.left_time))
            self.mask=self.get_similarity_mask(torch.Tensor(self.left_time),torch.Tensor(self.right_time))
            self.mask*=frame_num_mask
            self.mask_row_indices=list(compress(list(range(len(self.left_time))),list(self.mask.any(-1))))
            self.class_tracks=class_tracks
            # self.decay_matrix=self.get_time_decay_matrix(torch.Tensor(self.left_time),torch.Tensor(self.right_time))
            self.diff_matrix=self.get_diff_mat(torch.Tensor(self.right_time),torch.Tensor(self.left_time))
            scene_info={
                'right_time':self.right_time,
                'look_up_list':self.look_up_list,
                'tracker_ids':self.tracker_ids,
                'mask':self.mask,
                'class_tracks':self.class_tracks,
                'mask_row_indices':self.mask_row_indices,
                # 'decay_matrix':self.decay_matrix
            }
            if self.mask.any(-1).sum() != 0 and self.mask is not None:
                self.save_data(self.diff_matrix,save_time_diff=True)
            self.save_data(scene_info,save_scene_info=True)
        self.length=self.mask.any(-1).sum()
        self.map_radius_buffer=10
        if self.right_time is None:
            self.max_query_num = 0
        elif len(self.right_time) == 0:
            self.max_query_num = 0
        else:
            self.max_query_num = max(self.mask.sum(dim=1))
        self.logger.info('Total match queries in scene %d : %d' % (scene_idx,self.length))
    
            
    def get_diff_mat(self,times1,times2):
        times1=times1.unsqueeze(-1)
        times2=times2.unsqueeze(0)
        return times1-times2

    def get_similarity_mask(self,left_times,right_times,buffer_time=1.2):
        right_left_diffs=self.get_diff_mat(right_times,left_times)
        mask1=(right_left_diffs<-buffer_time)
        return mask1
    def get_time_decay_matrix(self,left_times,right_times,buffer_time=1.5,decay_factor=0.95,decay_thresh=3.0):
        right_left_diffs=self.get_diff_mat(right_times,left_times)
        mask1=(right_left_diffs>=-buffer_time)
        decay_matrix=decay_factor**(-decay_thresh-right_left_diffs)
        decay_matrix[mask1]=1
        return decay_matrix
    
    def get_times(self,scene_tracks):
        left_time_lists=defaultdict(list)
        right_time_lists=defaultdict(list)
        num_obs=defaultdict(list)
        look_up_lists=defaultdict(list)
        tracker_ids=defaultdict(list)
        class_tracks=[]
        for idx,track in enumerate(scene_tracks):
            name=track['instance_name'][0]
            if name ==self.class_name:
                class_tracks.append(track)
            left_time_lists[name].append(track['left_time'])
            right_time_lists[name].append(track['right_time'])
            num_obs[name].append(track['valid_frame_count'])
            look_up_lists[name].append(idx)
            tracker_ids[name].append(track['track_id'])
        return left_time_lists,right_time_lists,num_obs,look_up_lists,tracker_ids,class_tracks

    def get_target_tracks_input(self,target_info,visualize=False):
        last_obs=target_info['frame_info_list'][-1]['annotation_info']
        last_time=target_info['frame_info_list'][-1]['sample_info']['timestamp']
        # assert(init_time==last_time)
        location=last_obs['location'][0,:-1]
        if len(target_info['frame_info_list'])!=1:
            second_last_obs=target_info['frame_info_list'][-2]['annotation_info']
            second_last_time=target_info['frame_info_list'][-2]['sample_info']['timestamp']
            second_last_location=second_last_obs['location'][0,:-1]
            obs_yaw=correct_yaw(last_obs['rotation'][0])
            sample_token= last_obs=target_info['frame_info_list'][-1]['sample_info']['sample_token']
            disp_vec=location-second_last_location
            est_yaw=np.arctan2(disp_vec[1],disp_vec[0])
            est_yaw=np.array([correct_yaw(est_yaw)])
            single_obs=False
        else:
            single_obs=True
            obs_yaw=correct_yaw(last_obs['rotation'][0])
            sample_token= last_obs=target_info['frame_info_list'][-1]['sample_info']['sample_token']
        if not single_obs:
            if np.abs(np.abs(est_yaw-obs_yaw)-np.pi) <(np.pi/6):
                if est_yaw-obs_yaw > 0:
                    obs_yaw+=np.pi
                else:
                    obs_yaw-=np.pi
            if np.abs(est_yaw-obs_yaw) <(np.pi/4):
                yaw=(obs_yaw+est_yaw)/2
            else:
                if LA.norm(disp_vec)>0.75 and (last_time-second_last_time)<1.5:
                    yaw=est_yaw
                else:
                    yaw=obs_yaw
        else:
            yaw=obs_yaw
        origin=np.concatenate((location,yaw))
        
        hist_rec=np.empty([0,8])
        if visualize:
            glb_poses=np.empty([0,3])
        for idx,frame in enumerate(target_info['frame_info_list']):
            obs=frame['annotation_info']
            xy=obs['location'][0,:-1]
            t=frame['sample_info']['timestamp']-last_time
            global_yaw=obs['rotation'][0,0]
            if visualize:
                glb_poses=np.concatenate((glb_poses,np.array([[xy[0], xy[1], global_yaw]])),axis=0)
            local_pose = global_to_local(origin, (xy[0], xy[1], global_yaw))
            local_yaw = local_pose[-1]
            if idx==0:
                # if self.no_velo:
                #     local_velo=(0,0)
                # else:
                    # global_velo=obs['velocity'][0,:2]
                    # local_velo=global_to_local((0,0,origin[-1]), (global_velo[0], global_velo[1], 0))
                global_velo=obs['velocity'][0,:2]
                local_velo=global_to_local((0,0,origin[-1]), (global_velo[0], global_velo[1], 0))
                
            else:
                local_velo=np.asarray([local_pose[0]-prev_pose[0],local_pose[1]-prev_pose[1]])/(t-prev_t)
            hist_rec=np.concatenate((hist_rec,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),t,
                                                                            local_velo[0],local_velo[1]))])),0)
            prev_pose=local_pose
            prev_t=t
        if self.no_velo and not single_obs:
            hist_rec[0,-2:]=hist_rec[1,-2:]
        max_length=len(hist_rec)
        # assert(max_length>1)
        radius= np.max(LA.norm(hist_rec[:,:2],axis=1))
        return hist_rec,origin,radius,max_length,sample_token
    
    def get_future_tracks_input(self,origin,track_infos,radius,max_length,init_time):
        agent_list=[]
        for id,track_info in enumerate(track_infos):
            track_rec=np.empty([0,6])
            for idx,frame in enumerate(track_info['frame_info_list']):
                obs=frame['annotation_info']
                xy=obs['location'][0,:-1]
                t=frame['sample_info']['absolute_time_in_scene']-init_time
                global_yaw=obs['rotation'][0,0]
                local_pose = global_to_local(origin, (xy[0], xy[1], global_yaw))
                local_yaw = local_pose[-1]
                if idx==0:
                    global_velo=obs['velocity'][0,:2]
                    local_velo=np.asarray(global_to_local((0,0,origin[-1]), (global_velo[0], global_velo[1], 0)))[:2]
                # else:
                    # local_velo=np.asarray([local_pose[0]-prev_pose[0],local_pose[1]-prev_pose[1]])/(t-prev_t)
                track_rec=np.concatenate((track_rec,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),t))])),0)
                # prev_pose=local_pose
                # prev_t=t
            # assert(len(track_rec)>1)
            max_length=max(len(track_rec),max_length)
            radius=max(np.max(LA.norm(track_rec[:,:2],axis=1)),radius)
            if len(track_rec)>1:
                track_rec=np.flip(track_rec,axis=0)
                future_velo=(track_rec[:-1,:2]-track_rec[1:,:2])/(((track_rec[:-1,-1]-track_rec[1:,-1])).reshape(-1,1))
                future_velo=np.concatenate((future_velo,local_velo[np.newaxis,:]),axis=0)
            else:
                future_velo=local_velo[np.newaxis,:]
            track_rec=np.concatenate((track_rec,future_velo),axis=-1)
            agent_list.append(track_rec)
        return agent_list,radius,max_length

    def __len__(self):
        # if self._merge_all_iters_to_one_epoch:
        #     return len(self.infos) * self.total_epochs

        return self.length

    def __getitem__(self, index):
        # if self._merge_all_iters_to_one_epoch:
        #     index = index % len(self.infos)
        if self.mode=='debug_radius':
            row_ind=self.mask_row_indices[index]
            track_bool=self.mask[row_ind]
            target_info=copy.deepcopy(self.class_tracks[row_ind])
            init_time=self.right_time[row_ind]
            # match_indices=list(compress(np.arange(len(track_bool)),track_bool))
            candidate_tracks=list(compress(copy.deepcopy(self.class_tracks),track_bool))
            target_motion,origin,radius,max_length,sample_token = self.get_target_tracks_input(target_info,visualize=True)
            candidate_motion,radius,max_length  = self.get_future_tracks_input(origin,candidate_tracks,radius,max_length,init_time)

            return radius+self.map_radius_buffer
        
        elif self.mode=='compute_stats':
            row_ind=self.mask_row_indices[index]
            track_bool=self.mask[row_ind]
            target_info=copy.deepcopy(self.class_tracks[row_ind])
            init_time=self.right_time[row_ind]
            # match_indices=list(compress(np.arange(len(track_bool)),track_bool))
            candidate_tracks=list(compress(copy.deepcopy(self.class_tracks),track_bool))
            target_motion,origin,radius,max_length,sample_token = self.get_target_tracks_input(target_info,visualize=True)
            candidate_motion,radius,max_length  = self.get_future_tracks_input(origin,candidate_tracks,radius,max_length,init_time)
            num_nodes, num_nbrs,lane_node_feats,e_succ, e_prox = self.get_map_representation(sample_token,radius+self.map_radius_buffer, origin)
            data_dict = {'row_ind':row_ind,'target_motion':target_motion,'candidate_motion':candidate_motion,
                        'lane_node_feats': lane_node_feats,'origin':origin,'sample_token':sample_token,
                        'e_succ':e_succ, 'e_prox':e_prox}
            self.save_data(data_dict, row_ind=row_ind)
            
            stats={
                'num_lane_nodes':num_nodes,
                'num_nbrs':num_nbrs,
                'max_length':max_length
                
            }
            return stats
        
        elif self.mode=='extract_data':
            if self.skip:
                return 0
            row_ind=self.mask_row_indices[index]
            track_bool=self.mask[row_ind]
            matrix_row_inds=(torch.arange(len(track_bool))[track_bool]).unsqueeze(-1)
            data_dict=self.load_data(row_ind)
            if 'lane_ctrs' in data_dict.keys():
                return 0
            lane_feature = data_dict['lane_node_feats']
            target_motion=data_dict['target_motion']
            candidate_motion=data_dict['candidate_motion']
            target_motion,target_motion_mask=list_to_tensor([target_motion],1,self.max_length,8)
            candidate_motion,candidate_motion_mask=list_to_tensor(candidate_motion,self.max_query_num,self.max_length,8)
            lane_node_feats, lane_node_masks = list_to_tensor(lane_feature, self.max_num_nodes, self.polyline_length, 8)
            lane_ctrs = add_lane_ctrs(lane_node_feats, lane_node_masks)
            # e_succ=data_dict.pop('e_succ')
            # e_prox=data_dict.pop('e_prox')
            # e_succ=data_dict['e_succ']
            # e_prox=data_dict['e_prox']
            # s_next, edge_type = self.get_edge_lookup(e_succ, e_prox)
            data_dict['lane_node_feats'] = torch.from_numpy(lane_node_feats)
            data_dict['lane_node_masks'] = torch.from_numpy(lane_node_masks)
            data_dict['history'] = torch.from_numpy(target_motion).squeeze(0)
            data_dict['history_mask'] = torch.from_numpy(target_motion_mask).squeeze(0)
            data_dict['future'] = torch.from_numpy(candidate_motion)
            data_dict['future_mask'] = torch.from_numpy(candidate_motion_mask)
            data_dict['matrix_indices'] = torch.cat((torch.ones_like(matrix_row_inds)*row_ind,matrix_row_inds),axis=-1)
            data_dict['lane_ctrs'] = torch.from_numpy(lane_ctrs)
            # data_dict['s_next']=torch.from_numpy(s_next)
            # data_dict['edge_type']=torch.from_numpy(edge_type)
            self.save_data(data_dict, row_ind=row_ind)

            return 0
        
        elif self.mode=='load_data':
            row_ind=self.mask_row_indices[index]
            data_dict=self.load_data(row_ind)
            return data_dict
        
    def save_data(self,data_dict, row_ind= None, save_scene_info=False,save_time_diff=False):
        database_save_path = self.save_path / self.scene_token
        # db_info_save_path = save_path / f'nuscenes_occ_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        # all_db_infos = {}
        if row_ind is not None:
            filepath=database_save_path / (str(row_ind)+'.pickle')
        elif save_scene_info:
            filepath=database_save_path / 'scene_info.pickle'
        elif save_time_diff:
            filepath=database_save_path / 'time_diff.pickle'
        else:
            filepath=database_save_path / 'stats.pickle'

        with open(filepath, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return 
    
    def load_data(self, row_ind= None, load_scene_info=False,save_time_diff=False):
        database_save_path = self.save_path / self.scene_token
        if row_ind is not None:
            filepath=database_save_path / (str(row_ind)+'.pickle')
        elif load_scene_info:
            filepath=database_save_path / 'scene_info.pickle'
        elif save_time_diff:
            filepath=database_save_path / 'time_diff.pickle'
        else:
            filepath=database_save_path / 'stats.pickle'
        
        if not os.path.isfile(filepath):
            raise Exception('Could not find data. Please run the dataset in extract_data mode')

        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)
            
        return data
    
    def discard_poses_outside_extent( self,pose_set: List[np.ndarray],last_times:List =None,
                                    ids: List[str] = None, radius = None) -> Union[List[np.ndarray],
                                                                    Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []
        updated_last_times=[]

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if self.map_extent[2] <= pose[1]:
                    if LA.norm(pose[:2],ord=2)<radius:
                        flag = True
                        break

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])
                if last_times is not None:
                    updated_last_times.append(last_times[m])

        if ids is not None:
            return updated_pose_set, updated_ids
        elif last_times is not None:
            return updated_pose_set, updated_last_times
        else:
            return updated_pose_set
    @staticmethod
    def get_successor_edges(lane_ids: List[str], map_api: NuScenesMap) -> List[List[int]]:
        """
        Returns successor edge list for each node
        """
        e_succ = []
        for node_id, lane_id in enumerate(lane_ids):
            e_succ_node = []

            if node_id + 1 < len(lane_ids) and lane_id == lane_ids[node_id + 1]:
                e_succ_node.append(node_id + 1)
            else:
                outgoing_lane_ids = map_api.get_outgoing_lane_ids(lane_id)
                for outgoing_id in outgoing_lane_ids:
                    if outgoing_id in lane_ids:
                        e_succ_node.append(lane_ids.index(outgoing_id))
            for i in range(2,4):
                if node_id + i < len(lane_ids) and lane_id == lane_ids[node_id + i]:
                    e_succ_node.append(node_id + i)

            
            e_succ.append(e_succ_node)

        return e_succ

    @staticmethod
    def get_proximal_edges(lane_node_feats: List[np.ndarray], e_succ: List[List[int]],
                           dist_thresh=8, yaw_thresh=np.pi/4) -> List[List[int]]:
        """
        Returns proximal edge list for each node
        """
        e_prox = [[] for _ in lane_node_feats]
        for src_node_id, src_node_feats in enumerate(lane_node_feats):
            for dest_node_id in range(src_node_id + 1, len(lane_node_feats)):
                if dest_node_id not in e_succ[src_node_id] and src_node_id not in e_succ[dest_node_id]:
                    dest_node_feats = lane_node_feats[dest_node_id]
                    pairwise_dist = cdist(src_node_feats[:, :2], dest_node_feats[:, :2])
                    min_dist = np.min(pairwise_dist)
                    if min_dist <= dist_thresh:
                        yaw_src = np.arctan2(np.mean(np.sin(src_node_feats[:, 2])),
                                             np.mean(np.cos(src_node_feats[:, 2])))
                        yaw_dest = np.arctan2(np.mean(np.sin(dest_node_feats[:, 2])),
                                              np.mean(np.cos(dest_node_feats[:, 2])))
                        yaw_diff = np.arctan2(np.sin(yaw_src-yaw_dest), np.cos(yaw_src-yaw_dest))
                        if np.absolute(yaw_diff) <= yaw_thresh:
                            e_prox[src_node_id].append(dest_node_id)
                            e_prox[dest_node_id].append(src_node_id)

        return e_prox

    
    def add_boundary_flag(self,e_succ: List[List[int]], lane_node_feats: np.ndarray):
        """
        Adds a binary flag to lane node features indicating whether the lane node has any successors.
        Serves as an indicator for boundary nodes.
        """
        for n, lane_node_feat_array in enumerate(lane_node_feats):
            flag = 1 if len(e_succ[n]) == 0 else 0
            lane_node_feats[n] = np.concatenate((lane_node_feat_array, flag * np.ones((len(lane_node_feat_array), 1))),
                                                axis=1)
        
        return lane_node_feats

    def split_lanes(self, lanes: List[np.ndarray], max_len: int, lane_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses: (n+1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    def get_lane_flags(self,lanes: List[List[Tuple]], polygons: Dict[str, List[Polygon]]) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """

        lane_flags = [np.zeros((len(lane), len(polygons.keys()))) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                for n, k in enumerate(polygons.keys()):
                    polygon_list = polygons[k]
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][n] = 1
                            break

        return lane_flags
    def get_lanes_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap, radius=None) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return lanes: Dictionary of lane polylines
        """
        x, y, _ = global_pose
        if radius is None:
            radius = max(self.map_extent)
        lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']
        lanes = map_api.discretize_lanes(lanes, self.polyline_resolution)

        return lanes

    def get_lane_node_feats( self,origin: Tuple, lanes: Dict[str, List[Tuple]],
                                polygons: Dict[str, List[Polygon]]) -> Tuple[List[np.ndarray], List[str]]:
            """
            Generates vector HD map representation in the agent centric frame of reference
            :param origin: (x, y, yaw) of target agent in global co-ordinates
            :param lanes: lane centerline poses in global co-ordinates
            :param polygons: stop-line and cross-walk polygons in global co-ordinates
            :return:
            """

            # Convert lanes to list
            lane_ids = [k for k, v in lanes.items()]
            lanes = [v for k, v in lanes.items()]

            # Get flags indicating whether a lane lies on stop lines or crosswalks
            lane_flags = self.get_lane_flags(lanes, polygons)

            # Convert lane polylines to local coordinates:
            lanes = [np.asarray([global_to_local(origin, pose) for pose in lane]) for lane in lanes]

            # Concatenate lane poses and lane flags
            lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]

            # Split lane centerlines into smaller segments:
            lane_node_feats, lane_node_ids = self.split_lanes(lane_node_feats, self.polyline_length, lane_ids)

            return lane_node_feats, lane_node_ids

    def get_polygons_around_agent( self, global_pose: Tuple[float, float, float], map_api: NuScenesMap, radius=None) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
        """
        x, y, _ = global_pose
        if radius is None:
            radius = max(self.map_extent)
        record_tokens = map_api.get_records_in_radius(x, y, radius, ['stop_line', 'ped_crossing'])
        polygons = {k: [] for k in record_tokens.keys()}
        for k, v in record_tokens.items():
            for record_token in v:
                polygon_token = map_api.get(k, record_token)['polygon_token']
                polygons[k].append(map_api.extract_polygon(polygon_token))

        return polygons
    def get_map_representation(self, s_t, radius: float, global_pose: Tuple) -> Union[Tuple[int, int], Dict]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """

        map_name = self.helper.get_map_name_from_sample_token(s_t)
        map_api = self.maps[map_name]
            

        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(global_pose, map_api,radius=radius)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(global_pose, map_api,radius=radius)

        # Get vectorized representation of lanes
        lane_node_feats, lane_ids = self.get_lane_node_feats(global_pose, lanes, polygons)

        # Discard lanes outside map extent
        lane_node_feats, lane_ids = self.discard_poses_outside_extent(lane_node_feats, lane_ids,radius=radius)

        # Get edges: mapping index to successor or proximal lanes 
        e_succ = self.get_successor_edges(lane_ids, map_api)
        e_prox = self.get_proximal_edges(lane_node_feats, e_succ)

        # Concatentate flag indicating whether a node hassss successors to lane node feats
        lane_node_feats = self.add_boundary_flag(e_succ, lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, 6))]
            e_succ = [[]]
            e_prox = [[]]

        # # While running the dataset class in 'compute_stats' mode:
        # if mode == 'compute_stats':

        num_nbrs = [len(e_succ[i]) + len(e_prox[i]) for i in range(len(e_succ))]
        max_nbrs = max(num_nbrs) if len(num_nbrs) > 0 else 0
        num_nodes = len(lane_node_feats)
        
        for idx,lane in enumerate(lane_node_feats):
            yaws=lane[:,2].reshape(-1,1)
            cos=np.cos(yaws)
            sin=np.sin(yaws)
            lane_node_feats[idx]=np.concatenate((lane,cos,sin),axis=-1)


        return num_nodes, max_nbrs,lane_node_feats,e_succ, e_prox
    
    def get_edge_lookup(self, e_succ: List[List[int]], e_prox: List[List[int]]):
        """
        Returns edge look up tables
        :param e_succ: Lists of successor edges for each node
        :param e_prox: Lists of proximal edges for each node
        :return:

        s_next: Look-up table mapping source node to destination node for each edge. Each row corresponds to
        a source node, with entries corresponding to destination nodes. Last entry is always a terminal edge to a goal
        state at that node. shape: [max_nodes, max_nbr_nodes + 1]. Last

        edge_type: Look-up table of the same shape as s_next containing integer values for edge types.
        {0: No edge exists, 1: successor edge, 2: proximal edge, 3: terminal edge}
        """

        s_next = np.zeros((self.max_nodes, self.max_nbr_nodes + 1))
        edge_type = np.zeros((self.max_nodes, self.max_nbr_nodes + 1), dtype=int)

        for src_node in range(len(e_succ)):
            nbr_idx = 0
            successors = e_succ[src_node]
            prox_nodes = e_prox[src_node]

            # Populate successor edges
            for successor in successors:
                s_next[src_node, nbr_idx] = successor
                edge_type[src_node, nbr_idx] = 1
                nbr_idx += 1

            # Populate proximal edges
            for prox_node in prox_nodes:
                s_next[src_node, nbr_idx] = prox_node
                edge_type[src_node, nbr_idx] = 2
                nbr_idx += 1

            # Populate terminal edge
            s_next[src_node, -1] = src_node + self.max_nodes
            edge_type[src_node, -1] = 3

        return s_next, edge_type

def dummy_collate(batch_list):
    return batch_list



if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--data_root', type=str, required=True, help='nuscenes dataroot')
    parser.add_argument('--skip_compute_stats',  action='store_true')
    parser.add_argument('--type', default=None)
    # parser.add_argument('--result_path', type=str, default='../mot_results/tracking_result_cp_mini.json', help='')
    args = parser.parse_args()
    

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    
    version = dataset_cfg.VERSION
    if version == 'v1.0-trainval':
        val_scenes = splits.val
    elif version == 'v1.0-test':
        val_scenes = splits.val
    elif version == 'v1.0-mini':
        val_scenes = splits.mini_val
    
    nusc = NuScenes(version=version, dataroot=args.data_root, verbose=True)
    helper =PredictHelper_occ(nusc)
    # available_scenes = nuscenes_utils.get_available_scenes(nusc)
    # available_scene_names = [s['name'] for s in available_scenes]
    # val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    # for class_name in class_names:
    #     datasets=[]
    #     for scene_idx in range(len(val_scenes)):
    #         datasets.append(NuScenesDataset_MATCH_EXT(scene_idx,
    #             dataset_cfg=dataset_cfg, class_name=class_name,
    #             root_path=ROOT_DIR / 'data' / 'nuscenes',
    #             logger=create_logger(),helper=helper,mode='add_time_diff'
    #         ))
    # raise NotImplementedError()
    if args.type is None:
        class_names=[
                    'car',
                    'bus',
                    'truck',
                    'trailer'
                    ]
    else:
        class_names=[args.type]
    for class_name in class_names:
        if not args.skip_compute_stats:
            datasets=[]

            for scene_idx in range(len(val_scenes)):
                datasets.append(NuScenesDataset_MATCH_EXT(scene_idx,
                    dataset_cfg=dataset_cfg, class_name=class_name,
                    root_path=ROOT_DIR / 'data' / 'nuscenes',
                    logger=create_logger(),helper=helper,mode='compute_stats'
                ))
                # nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
            val_dls=[torch_data.DataLoader(dataset, dataset_cfg.BATCH_SIZE, shuffle=False,
                                num_workers=dataset_cfg.num_workers, pin_memory=True) for dataset in datasets]
            for scene_idx in tqdm(range(len(val_scenes))):
                scene_stats={}
                for index,mini_batch_stats in enumerate(val_dls[scene_idx]):
                    print('Computing scene',class_name,scene_idx)
                    print()
                    for k, v in mini_batch_stats.items():
                            # if k =='map_radius':
                            #     split_name=data_loader.dataset.name
                            #     stats[split_name+"_"+k]+=list(v)
                        if k in scene_stats.keys():
                            scene_stats[k] = max(scene_stats[k], torch.max(v).item())
                        else:
                            scene_stats[k] = torch.max(v).item()
                val_dls[scene_idx].dataset.save_data(scene_stats)
            del val_dls,datasets
    if args.type is None:
        class_names=[
                    'car',
                    'bus',
                    'truck',
                    'trailer'
                    ]
    else:
        class_names=[args.type]
    for class_name in class_names:
        datasets=[]
        for scene_idx in range(len(val_scenes)):
            datasets.append(NuScenesDataset_MATCH_EXT(scene_idx,
                dataset_cfg=dataset_cfg, class_name=class_name,
                root_path=ROOT_DIR / 'data' / 'nuscenes',
                logger=create_logger(),helper=helper,mode='extract_data'
            ))
            # nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
        val_dls=[torch_data.DataLoader(dataset, dataset_cfg.BATCH_SIZE, shuffle=False,
                            num_workers=dataset_cfg.num_workers, pin_memory=True) for dataset in datasets]
        for scene_idx in tqdm(range(len(val_scenes))):
            scene_stats={}
            for index,mini_batch_stats in enumerate(val_dls[scene_idx]):
                print('Extracting scene',class_name,scene_idx)
                # print(index)

    # for scene_idx in range(len(val_scenes)):
    #     datasets.append(NuScenesDataset_MATCH_EXT(scene_idx,
    #         dataset_cfg=dataset_cfg, class_name='car',
    #         root_path=ROOT_DIR / 'data' / 'nuscenes',
    #         logger=create_logger(),helper=helper,mode='debug_radius'
    #     ))
    # val_dls=[torch_data.DataLoader(dataset, dataset_cfg.BATCH_SIZE, shuffle=False,
    #                     num_workers=dataset_cfg.num_workers, pin_memory=True,collate_fn=dummy_collate) for dataset in datasets]
    # radius_list=[]
    # for scene_idx in tqdm(range(len(val_scenes))):
    #     scene_radius=[]
    #     for index,mini_batch_stats in enumerate(val_dls[scene_idx]):
    #         print(mini_batch_stats)
    #         scene_radius+=mini_batch_stats
    #     radius_list+=scene_radius
    # if not os.path.exists("extracted_mot_data/match/"):
    #     os.makedirs("extracted_mot_data/match/")

    # # Storing the lists in separate files
    # with open("extracted_mot_data/match/radius.txt", "w") as f:
    #     for item in radius_list:
    #         f.write(f"{item}\n")
        