import matplotlib.pyplot as plt
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from nuscenes.prediction.input_representation.static_layers import color_by_yaw
from nuscenes.prediction.input_representation.static_layers import correct_yaw
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from datasets.nuScenes.prediction import PredictHelper_occ
import numpy as np
from typing import Dict, Tuple, Union, List
from scipy.spatial.distance import cdist
from numpy import random
from numpy import linalg as LA
import numpy.ma as ma
import os,json,copy
from collections import defaultdict
import torch

def match_collate(batch_list):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    # batch_size = len(batch_list)
    ret = {}
    max_node_nums=max(data_dict['node_nums'])
    map_representation = {}

    for key, val in data_dict.items():
        if key in ['history', 'history_mask','future', 'future_mask']:
            ret[key] = torch.stack(val,dim=0)
            
        elif key in ['lane_node_feats','lane_node_masks','s_next','edge_type','lane_ctrs']:
            value=torch.stack(val,dim=0)
            map_representation[key]=value[:,:max_node_nums]
        # elif key =='map_representation':
        #     map_rep_dict = defaultdict(list)
        #     for cur_map in val:
        #         for k, v in cur_map.items():
        #             map_rep_dict[k].append(v)
        #     map_representation = {}
        #     for k,v in val.items():
        #         if k in ['lane_node_feats','lane_node_masks','s_next','edge_type','lane_ctrs']:
        #             value=torch.stack(v,dim=0)
        #             map_representation[k]=value[:,:max_node_nums]
        #     ret['map_representation']=map_representation
        else:
            ret[key]=val
    ret['map_representation']=map_representation
            
    return ret

class NuScenesGraphs_MATCH(NuScenesVector):
    """
    NuScenes dataset class for single agent prediction, using the graph representation from PGP for maps and agents
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper_occ ):
        """
        Initialize predict helper, agent and scene representations
        :param mode: Mode of operation of dataset, one of {'compute_stats', 'extract_data', 'load_data'}
        :param data_dir: Directory to store extracted pre-processed data
        :param helper: NuScenes PredictHelper
        :param args: Dataset arguments
        """
        super().__init__(mode, data_dir, args, helper)
        self.name=args['split_name']
        self.traversal_horizon = args['traversal_horizon']
        self.augment=args['augment']
        self.match = args['match']
        self.agent_radius_buffer=10
        self.random_rots=args['random_rots']
        if self.mode == "compute_stats" and self.random_rots:
            self.rot_rads=list((random.random(len(self.token_list))-0.5)*np.pi/2)
        # if self.mode == 'compute_stats':
        #     self.origin_list=[]
        #     self.radius_list=[]
        if self.mode != 'compute_stats':
            stats = self.load_stats()
            # print(len(self.origin_list))
            
        
        
        
        # if self.augment:
        # self.use_mid_origin=args['use_mid_origin']
        # self.augment_prob=args['augment_prob']
        # self.mask_prob=args['mask_prob']
        # self.augment_min=args['augment_min']
        # self.augment_count=0
        # self.dep_thresh=args['dep_thresh']
        # self.data_root = args['data_root']
        # self.deprecate_percent=args['dep_percent']
        # self.traj_mask_prob=args['traj_mask_prob']
        # self.adjust_yaw=args['adjust_yaw']
        # if self.random_rots:
        #     self.rdm_rot_prob=args['rdm_rot_prob']

        # Load dataset stats (max nodes, max agents etc.)
        if self.mode == 'extract_data':
            # stats = self.load_stats()
            # if self.augment:
            #     self.time_lengths = stats[self.name+"_times"]
            #     if self.name+"_token_list" in stats:
            #         self.token_list=stats[self.name+"_token_list"]
            if self.random_rots:
                self.rot_rads=stats['random_rotations']
            self.max_nbr_nodes = stats['max_nbr_nodes']
            self.num_agents=stats['num_agents']
            self.radius_list=stats[self.name+"_"+"map_radius"]


        
    def __getitem__(self, idx: int) -> Union[Dict, int]:
        """
        Get data point, based on mode of operation of dataset.
        :param idx: data index
        """
        if self.mode == 'compute_stats':
            return self.compute_stats(idx)
        elif self.mode == 'extract_data':
            self.extract_data(idx)
            return 0
        else:
            sample=self.load_data(idx)['inputs']
            
            sample_dict={
                'instance_token':sample['instance_token'],
                'sample_token':sample['sample_token'],
                'origin':sample['origin'],
                'node_seq_gt':sample['node_seq_gt']
                
            }
            
            for k,v in sample['map_representation'].items():
                if k != 'node_nums':
                    v=torch.Tensor(v)
                sample_dict[k]=v
            
            history_mask_time = -(random.random()*self.t_h+0.1)
            target_history_mask=sample['target_agent_representation']['history']['traj'][:,-1]>history_mask_time
            target_history=sample['target_agent_representation']['history']['traj'][target_history_mask]
            target_history_velo=sample['target_agent_representation']['history']['velo'][target_history_mask]
            target_history=np.concatenate((target_history,target_history_velo),axis=-1)
            target_history=np.concatenate((target_history,np.zeros([int(self.t_h*2)+1-target_history.shape[0],target_history.shape[1]])),axis=0)
            target_history_mask=np.concatenate((np.zeros([target_history_mask.sum()]),np.ones([int(self.t_h*2)+1-target_history_mask.sum()])),axis=0)
            target_history_mask=np.repeat(target_history_mask[:,np.newaxis],target_history.shape[1],axis=1)
            
            sample_dict['history']=torch.Tensor(target_history)
            sample_dict['history_mask']=torch.Tensor(target_history_mask)
            
            last_time=sample['target_agent_representation']['last_time']-0.1
            mask_time=random.random()*(last_time-1.5)+1.5
            target_future_mask=sample['target_agent_representation']['future']['traj'][:,-1]>mask_time
            future_velocity=sample['target_agent_representation']['future']['velo'][target_future_mask]
            target_future=np.concatenate((sample['target_agent_representation']['future']['traj'][target_future_mask],future_velocity),axis=-1)[-(int(self.t_h*2)+1):]
            
            last_times=np.array(sample['surrounding_agent_representation']['last_times'])-0.1
            mask_times=random.random(last_times.shape)*(last_times-1.5)+1.0
            # residual=last_times-mask_times
            # res_mask=residual>2.5
            # mask_times[res_mask]+=residual[res_mask]*random.random(residual[res_mask].shape)
            vehicles=sample['surrounding_agent_representation']['vehicles']
            gt=np.concatenate((target_future,np.zeros([vehicles.shape[1]-target_future.shape[0],target_future.shape[-1]])))[np.newaxis,:,:]
            gt_mask=(gt==0)
            mask_times=np.concatenate((mask_times,np.ones(len(vehicles)-len(mask_times)))).reshape(-1,1)
            agent_masks=(vehicles[:,:,-1]<mask_times)
            vehicles=np.concatenate((vehicles,sample['surrounding_agent_representation']['velocity']),axis=-1)
            vehicles[agent_masks]=0
            masks=np.repeat(agent_masks[:,:,np.newaxis],vehicles.shape[-1],axis=-1)

            masks=np.concatenate((gt_mask,masks),axis=0)
            all_vehicles=np.concatenate((gt,vehicles),axis=0)
            
            sample_dict['future']=torch.Tensor(all_vehicles)
            sample_dict['future_mask']=torch.Tensor(masks)
            
            return sample_dict

    def compute_stats(self, idx: int) -> Dict[str, int]:
        """
        Function to compute statistics for a given data point
        """     

        global_pose,target_disp = self.get_target_agent_representation(idx)
        # print("Target finished!")
        num_agents,map_radius = self.get_surrounding_agent_representation(idx, origin=global_pose,radius=target_disp)
        # print("Agents finished!")
        num_lane_nodes, max_nbr_nodes = self.get_map_representation(idx,map_radius,global_pose)
        # print("Map finished!")
        stats = {
            'num_lane_nodes': num_lane_nodes,
            'max_nbr_nodes': max_nbr_nodes,
            'num_agents': num_agents,
            'map_radius':map_radius
        }

        return stats

    def extract_data(self, idx: int):
        """
        Function to extract data. Bulk of the dataset functionality will be implemented by this method.
        :param idx: data index
        """
        inputs = self.get_inputs(idx)
        # ground_truth = self.get_ground_truth(idx)
        # node_seq_gt, evf_gt = self.get_visited_edges(idx, inputs['map_representation'])
        # init_node = self.get_initial_node(inputs['map_representation'])

        # ground_truth['evf_gt'] = evf_gt
        # inputs['init_node'] = init_node
        # inputs['node_seq_gt'] = node_seq_gt  # For pretraining with ground truth node sequence
        data = {'inputs': inputs}
        self.save_data(idx, data)

    def get_inputs(self, idx: int) -> Dict:
        i_t, s_t = self.token_list[idx].split("_")
        
        radius=self.radius_list[idx].item()
        target_agent_representation,origin,fut_xy = self.get_target_agent_representation(idx)
        surrounding_agent_representation = self.get_surrounding_agent_representation(idx,radius=radius,origin=origin)
        map_representation = self.get_map_representation(idx,radius=radius,origin=origin)
        node_seq_gt, evf_gt = self.get_visited_edges(idx, map_representation,fut_xy)
        map_representation = self.add_lane_ctrs(map_representation)
        inputs = {'instance_token': i_t,
                  'sample_token': s_t,
                  'map_representation': map_representation,
                  'surrounding_agent_representation': surrounding_agent_representation,
                  'target_agent_representation': target_agent_representation,
                  'node_seq_gt': node_seq_gt,
                  'evf_gt': evf_gt,
                  'origin': np.asarray(origin)}
        # a_n_masks_agnt = self.get_agent_node_masks(inputs['map_representation'], inputs['surrounding_agent_representation'])
        # a_n_masks_trgt = self.get_target_node_masks(inputs['map_representation'], inputs['target_agent_representation'])
        # inputs['agent_node_masks'] = {'agent':a_n_masks_agnt}
        # inputs['agent_node_masks'] = {'agent':a_n_masks_agnt}
        return inputs
    def add_lane_ctrs(self,map_representation):
        encodings=map_representation['lane_node_feats']
        mask=map_representation['lane_node_masks']
        lane_ctrs=(ma.masked_array(encodings[:,:,:2],mask=mask[:,:,:2])).mean(axis=1).data
        lane_ctrs[(~(((1-mask).astype(np.bool))[:,:,:2].any(1)))]=np.inf
        map_representation['lane_ctrs']=lane_ctrs
        return map_representation
    


    def get_target_agent_representation(self, idx: int, origin=None) :
        """
        Extracts target agent representation
        :param idx: data index
        :return hist: track history for target agent, shape: [t_h * 2, 5]
        """
        i_t, s_t = self.token_list[idx].split("_")
        global_pose = self.get_target_agent_global_pose(idx)    
        sample_annotation = self.helper.get_sample_annotation(i_t, s_t)
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        # Get future information
        coords_fut,global_yaw_fut,time_fut = self.helper.get_future_for_agent(i_t, s_t, seconds=self.t_h+self.t_f, in_agent_frame=False,add_yaw_and_time=True)
        future_rec=np.empty([0,6])
        # concat_future=np.empty([0,5])
        count=0
        for xy,r,t in zip(coords_fut,global_yaw_fut,time_fut):
            # if count==0:
            #     origin_fut=(xy[0], xy[1], correct_yaw(quaternion_yaw(Quaternion(r))))
            #     if self.use_mid_origin:
            #         origin=(np.asarray(global_pose)+np.asarray(origin_fut))/2
            #         if self.adjust_yaw:
            #             if np.abs(global_pose[-1]-origin_fut[-1])>np.pi:
            #                 adjusted_fut_yaw=origin_fut[-1]-np.pi*((origin_fut[-1]-global_pose[-1])//np.pi)
            #                 origin[-1]=global_pose[-1]+adjusted_fut_yaw
            #         if self.random_rots:
            #             if random.random()<self.rdm_rot_prob:
            #                 rot_rad=(random.random()-0.5)*np.pi
            #                 origin[-1]+=rot_rad
                            
            #         origin=tuple(origin)   
            #         origin_fut=origin
            if count == 0:
                origin = np.asarray(global_pose)
                if self.random_rots:
                    rot_rad=self.rot_rads[idx]
                    origin[-1]+=rot_rad
                origin=tuple(origin)
            global_yaw=quaternion_yaw(Quaternion(r))
            # print(origin)
            local_pose = self.global_to_local(origin, (xy[0], xy[1], global_yaw))
            local_yaw = local_pose[-1]
            future_rec=np.concatenate((future_rec,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),t))])),0)
            # concat_future=np.concatenate((concat_future,np.asarray([local_pose.__add__((t,1))])),0)
            count+=1
            last_t=t
            
        future=future_rec[::-1]
        if self.mode == "compute_stats":
            target_disp = LA.norm(future_rec[-1,:2],ord=2)
            return origin,max(target_disp,20)

        # x, y co-ordinates in agent's frame of reference
        coords,global_yaw,time_past = self.helper.get_past_for_agent(i_t, s_t, seconds=self.t_h, in_agent_frame=False,add_yaw_and_time=True)
        global_pose=global_pose[:-1].__add__((yaw,))
        local_pose=self.global_to_local(origin, global_pose)
        local_yaw=local_pose[-1]
        past_hist=np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),0))])
        # concat_hist=np.asarray([self.global_to_local(origin, global_pose).__add__((0,0))])
        count=0
        for xy,r,t in zip(coords,global_yaw,time_past):
            
            # if random.random()<self.traj_mask_prob and self.mode=='extract_data':
            #     continue
            glb_yaw=quaternion_yaw(Quaternion(r))
            local_pose = self.global_to_local(origin, (xy[0], xy[1], glb_yaw))
            local_yaw=local_pose[-1]
            past_hist=np.concatenate((past_hist,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),-t))])),0)
            # concat_hist=np.concatenate((concat_hist,np.asarray([local_pose.__add__((-t,0))])),0)
            count+=1
        hist=past_hist[::-1]
        past_velo=(hist[1:,:2]-hist[:-1,:2])/((hist[1:,-1]-hist[:-1,-1]).reshape(-1,1))
        past_velo=np.concatenate((np.zeros([1,2]),past_velo),axis=0)
        future_velo=(future[:-1,:2]-future[1:,:2])/(((future[:-1,-1]-future[1:,-1])).reshape(-1,1))
        future_velo=np.concatenate((future_velo,np.zeros([1,2])),axis=0)

        # gt_traj=np.flip(gt_coords,0)
        history={'traj':hist,'velo':past_velo}
        future={'traj':future,'velo':future_velo}
        target_representation={'history':history,'future':future,'last_time':last_t}
        return target_representation,origin,future_rec[:,:2]

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
    
    def get_polygons_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap, radius=None) -> Dict:
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
    
    def get_map_representation(self, idx: int, radius: float, origin: Tuple=None) -> Union[Tuple[int, int], Dict]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_name = self.helper.get_map_name_from_sample_token(s_t)
        map_api = self.maps[map_name]

        # Get agent representation in global co-ordinates
        if origin is None:
            global_pose = self.get_target_agent_global_pose(idx)
        else: 
            global_pose = origin
            

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

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == 'compute_stats':

            num_nbrs = [len(e_succ[i]) + len(e_prox[i]) for i in range(len(e_succ))]
            max_nbrs = max(num_nbrs) if len(num_nbrs) > 0 else 0
            num_nodes = len(lane_node_feats)

            return num_nodes, max_nbrs

        # Get edge lookup tables
        s_next, edge_type = self.get_edge_lookup(e_succ, e_prox)
        
        node_nums = len(lane_node_feats)

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 6)
        # lane_node_feats ~ [B,N,L,C] B: batch size; N: max_number of lanes (nodes) in the scene; L: max_length of a lane; 
        # C: channel number for a lane waypoint, which contains x,y location, theta angle, bools inidacting whether the point
        # locates on 'stop_line', 'ped_crossing' polygons and whether it has successor. 
        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
            's_next': s_next,
            'edge_type': edge_type,
            'node_nums':node_nums
        }

        return map_representation
    
    def discard_poses_outside_extent(self, pose_set: List[np.ndarray],last_times:List =None,
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
    
    def calculate_largest_radius(self, pose_set: List[np.ndarray], radius = None) -> Union[List[np.ndarray],
                                                                    Tuple[List[np.ndarray], List[str]]]:
        """
        Calculate largest radiius to include all agents
        """

        for m, poses in enumerate(pose_set):
            for n, pose in enumerate(poses):
                radius=max(LA.norm(pose[:2],ord=2),radius)
        return radius

    def filter_short_trajs(self,vehicles,last_times):
        from itertools import compress
        long_trajs=np.array(last_times)>1.5
        filtered_vehicles=list(compress(vehicles, long_trajs))
        last_times=list(compress(last_times, long_trajs))
        return filtered_vehicles,last_times
    
    def get_future_velos(self,vehicles):
        velos=[]
        for track in vehicles:
            velo=(track[:-1,:2]-track[1:,:2])/(((track[:-1,-1]-track[1:,-1])).reshape(-1,1))
            velo=np.concatenate((velo,np.zeros([1,2])),axis=0)
            velos.append(velo)
        return velos
        
    def get_surrounding_agent_representation(self, idx: int, radius=None, origin=None) -> \
            Union[Tuple[int, int], Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :return: ndarrays with surrounding pedestrian and vehicle track histories and masks for non-existent agents
        """

        # Get vehicles and pedestrian histories for current sample
        
        vehicles,last_times = self.get_agents_of_type(idx, 'vehicle', origin)
        # pedestrians = self.get_agents_of_type(idx, 'human', time, origin)
        # Discard poses outside map extent
        if self.mode == "compute_stats":
            vehicles,last_times = self.discard_poses_outside_extent(vehicles,last_times,radius=radius+self.agent_radius_buffer)
        else:
            vehicles,last_times = self.discard_poses_outside_extent(vehicles,last_times,radius=radius)
        # pedestrians = self.discard_poses_outside_extent(pedestrians,radius=radius)
        vehicles,last_times = self.filter_short_trajs(vehicles,last_times)

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == 'compute_stats':
            largest_radius=self.calculate_largest_radius(vehicles,radius)
            return len(vehicles), largest_radius
        velos=self.get_future_velos(vehicles)
        # Convert to fixed size arrays for batching
        if self.use_home:
            vehicles = self.list_to_tensor(vehicles, self.max_vehicles, int(self.t_h * 2 + 1), 5,True)
            pedestrians = self.list_to_tensor(pedestrians, self.max_pedestrians, int(self.t_h * 2 + 1), 5,True)

            surrounding_agent_representation = {
                'vehicles': vehicles,
                'pedestrians': pedestrians
            }
        else:
            
            vehicles, vehicle_masks = self.list_to_tensor(vehicles, self.num_agents, int((self.t_f+self.t_h) * 2 + 1), 6,False)
            velos,_ = self.list_to_tensor(velos, self.num_agents, int((self.t_f+self.t_h) * 2 + 1), 2,False)
            # pedestrians, pedestrian_masks = self.list_to_tensor(pedestrians, self.max_pedestrians, int((self.t_f) * 2 + 1), 4,False)

            surrounding_agent_representation = {
                'vehicles': vehicles,
                'vehicle_masks': vehicle_masks,
                'last_times':last_times,
                'velocity':velos
                # 'pedestrians': pedestrians,
                # 'pedestrian_masks': pedestrian_masks
            }
        if self.use_raster:
            i_t, s_t = self.token_list[idx].split("_")
            img = self.agent_rasterizer.make_representation(i_t, s_t)
            img = np.moveaxis(img, -1, 0)
            img = img.astype(float) / 255
            surrounding_agent_representation['image']=img

        return surrounding_agent_representation
    def get_agents_of_type(self, idx: int, agent_type: str, origin=None) -> List[np.ndarray]:
        """
        Returns surrounding agents of a particular class for a given sample
        :param idx: data index
        :param agent_type: 'human' or 'vehicle'
        :return: list of ndarrays of agent track histories.
        """
        i_t, s_t = self.token_list[idx].split("_")

        # Get agent representation in global co-ordinates
        if origin is None:
            origin = self.get_target_agent_global_pose(idx)

        # Load all agents for sample
        agent_details = self.helper.get_future_for_sample(s_t, seconds=self.t_h+self.t_f, in_agent_frame=False, just_xy=False)
        agent_hist = self.helper.get_future_for_sample(s_t, seconds=self.t_h+self.t_f, in_agent_frame=False, just_xy=True,add_yaw_and_time=True)

        # Add present time to agent histories
        present_time = self.helper.get_annotations_for_sample(s_t)
        for annotation in present_time:
            ann_i_t = annotation['instance_token']
            if ann_i_t in agent_hist.keys():
                present_pose = np.asarray(annotation['translation'][0:2]).reshape(1, 2)
                present_timestamp = np.zeros(1)
                present_yaw = np.asarray([annotation['rotation']])
                if agent_hist[ann_i_t]['coords'].any():
                    agent_hist[ann_i_t]['coords'] = np.concatenate((present_pose, agent_hist[ann_i_t]['coords'])) ## Order: Now -> Future
                    agent_hist[ann_i_t]['time_step'] = np.concatenate((present_timestamp, agent_hist[ann_i_t]['time']))
                    agent_hist[ann_i_t]['rotation'] = np.concatenate((present_yaw, agent_hist[ann_i_t]['global_yaw']))
                else:
                    agent_hist[ann_i_t]['coords'] = present_pose
                    agent_hist[ann_i_t]['time_step'] = present_timestamp
                    agent_hist[ann_i_t]['rotation'] = present_yaw

        # Filter for agent type
        agent_list = []
        agent_i_ts = []
        for k, v in agent_details.items():
            if v and agent_type in v[0]['category_name'] and v[0]['instance_token'] != i_t:
                agent_list.append(agent_hist[k])
                agent_i_ts.append(v[0]['instance_token'])

        # Convert to target agent's frame of reference
        for agent in agent_list:
            agent['xy_coords']=[]
            agent['yaw']=[]
            agent['cos(yaw)']=[]
            agent['sin(yaw)']=[]
            agent['time']=[]
            for n, pose in enumerate(agent['coords']):
                # if random.random()<self.traj_mask_prob and n>=1 and self.mode=='extract_data':
                #     continue
                local_pose = self.global_to_local(origin, (pose[0], pose[1], quaternion_yaw(Quaternion(agent['rotation'][n]))))
                agent['xy_coords'].append([local_pose[0], local_pose[1]])
                agent['yaw'].append(local_pose[2])
                agent['cos(yaw)'].append(np.cos(local_pose[2]))
                agent['sin(yaw)'].append(np.sin(local_pose[2]))
                agent['time'].append(agent['time_step'][n])
                

        # Flip history to have most recent time stamp last and extract past motion states
        last_times=[]
        for n, agent in enumerate(agent_list):
            xy = np.flip(agent['xy_coords'], axis=0)
            r = np.flip(agent['yaw'], axis=0)
            cos = np.flip(agent['cos(yaw)'], axis=0)
            sin = np.flip(agent['sin(yaw)'], axis=0)
            t = np.flip(agent['time'], axis=0)
            last_times.append(t[0])
            agent_list[n] = np.concatenate((xy, r.reshape(xy.shape[0],1),cos.reshape(xy.shape[0],1),sin.reshape(xy.shape[0],1),t.reshape(xy.shape[0],1)), axis=-1)

        return agent_list,last_times

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

            e_succ.append(e_succ_node)

        return e_succ

    @staticmethod
    def get_proximal_edges(lane_node_feats: List[np.ndarray], e_succ: List[List[int]],
                           dist_thresh=4, yaw_thresh=np.pi/4) -> List[List[int]]:
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

    @staticmethod
    def add_boundary_flag(e_succ: List[List[int]], lane_node_feats: np.ndarray):
        """
        Adds a binary flag to lane node features indicating whether the lane node has any successors.
        Serves as an indicator for boundary nodes.
        """
        for n, lane_node_feat_array in enumerate(lane_node_feats):
            flag = 1 if len(e_succ[n]) == 0 else 0
            lane_node_feats[n] = np.concatenate((lane_node_feat_array, flag * np.ones((len(lane_node_feat_array), 1))),
                                                axis=1)

        return lane_node_feats

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

    def get_initial_node(self, lane_graph: Dict) -> np.ndarray:
        """
        Returns initial node probabilities for initializing the graph traversal policy
        :param lane_graph: lane graph dictionary with lane node features and edge look-up tables
        """

        # Unpack lane node poses
        node_feats = lane_graph['lane_node_feats']
        node_feat_lens = np.sum(1 - lane_graph['lane_node_masks'][:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[:int(node_feat_lens[i]), :3])

        assigned_nodes = self.assign_pose_to_node(node_poses, np.asarray([0, 0, 0]), dist_thresh=3,
                                                  yaw_thresh=np.pi / 4, return_multiple=True)

        init_node = np.zeros(self.max_nodes)
        init_node[assigned_nodes] = 1/len(assigned_nodes)
        return init_node

    def get_visited_edges(self, idx: int, lane_graph: Dict, fut_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns nodes and edges of the lane graph visited by the actual target vehicle in the future. This serves as
        ground truth for training the graph traversal policy pi_route.

        :param idx: dataset index
        :param lane_graph: lane graph dictionary with lane node features and edge look-up tables
        :return: node_seq: Sequence of visited node ids.
                 evf: Look-up table of visited edges.
        """

        # Unpack lane graph dictionary
        node_feats = lane_graph['lane_node_feats']
        s_next = lane_graph['s_next']
        edge_type = lane_graph['edge_type']

        node_feat_lens = np.sum(1 - lane_graph['lane_node_masks'][:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[:int(node_feat_lens[i]), :3])

        # Initialize outputs
        current_step = 0
        node_seq = np.zeros(self.traversal_horizon)
        evf = np.zeros_like(s_next)

        # Get future trajectory:
        # i_t, s_t = self.token_list[idx].split("_")
        # fut_xy = self.helper.get_future_for_agent(i_t, s_t, self.t_f+self.t_h, True)
        fut_interpolated = np.zeros((fut_xy.shape[0] * 10 + 1, 2))
        param_query = np.linspace(0, fut_xy.shape[0], fut_xy.shape[0] * 10 + 1)
        param_given = np.linspace(0, fut_xy.shape[0], fut_xy.shape[0] + 1)
        val_given_x = np.concatenate(([0], fut_xy[:, 0]))
        val_given_y = np.concatenate(([0], fut_xy[:, 1]))
        fut_interpolated[:, 0] = np.interp(param_query, param_given, val_given_x)
        fut_interpolated[:, 1] = np.interp(param_query, param_given, val_given_y)
        fut_xy = fut_interpolated

        # Compute yaw values for future:
        fut_yaw = np.zeros(len(fut_xy))
        for n in range(1, len(fut_yaw)):
            fut_yaw[n] = -np.arctan2(fut_xy[n, 0] - fut_xy[n-1, 0], fut_xy[n, 1] - fut_xy[n-1, 1])

        # Loop over future trajectory poses
        query_pose = np.asarray([fut_xy[0, 0], fut_xy[0, 1], fut_yaw[0]])
        current_node = self.assign_pose_to_node(node_poses, query_pose)
        node_seq[current_step] = current_node
        for n in range(1, len(fut_xy)):
            query_pose = np.asarray([fut_xy[n, 0], fut_xy[n, 1], fut_yaw[n]])
            dist_from_current_node = np.min(np.linalg.norm(node_poses[current_node][:, :2] - query_pose[:2], axis=1))

            # If pose has deviated sufficiently from current node and is within area of interest, assign to a new node
            padding = self.polyline_length * self.polyline_resolution / 2
            if self.map_extent[0] - padding <= query_pose[0] <= self.map_extent[1] + padding and \
                    self.map_extent[2] - padding <= query_pose[1] <= self.map_extent[3] + padding:

                if dist_from_current_node >= 1.5:
                    assigned_node = self.assign_pose_to_node(node_poses, query_pose)

                    # Assign new node to node sequence and edge to visited edges
                    if assigned_node != current_node:

                        if assigned_node in s_next[current_node]:
                            nbr_idx = np.where(s_next[current_node] == assigned_node)[0]
                            nbr_valid = np.where(edge_type[current_node] > 0)[0]
                            nbr_idx = np.intersect1d(nbr_idx, nbr_valid)

                            if edge_type[current_node, nbr_idx] > 0:
                                evf[current_node, nbr_idx] = 1

                        current_node = assigned_node
                        if current_step < self.traversal_horizon-1:
                            current_step += 1
                            node_seq[current_step] = current_node

            else:
                break

        # Assign goal node and edge
        goal_node = current_node + self.max_nodes
        node_seq[current_step + 1:] = goal_node
        evf[current_node, -1] = 1

        return node_seq, evf

    @staticmethod
    def assign_pose_to_node(node_poses, query_pose, dist_thresh=5, yaw_thresh=np.pi/3, return_multiple=False):
        """
        Assigns a given agent pose to a lane node. Takes into account distance from the lane centerline as well as
        direction of motion.
        """
        dist_vals = []
        yaw_diffs = []

        for i in range(len(node_poses)):
            distances = np.linalg.norm(node_poses[i][:, :2] - query_pose[:2], axis=1)
            dist_vals.append(np.min(distances))
            idx = np.argmin(distances)
            yaw_lane = node_poses[i][idx, 2]
            yaw_query = query_pose[2]
            yaw_diffs.append(np.arctan2(np.sin(yaw_lane - yaw_query), np.cos(yaw_lane - yaw_query)))

        idcs_yaw = np.where(np.absolute(np.asarray(yaw_diffs)) <= yaw_thresh)[0]
        idcs_dist = np.where(np.asarray(dist_vals) <= dist_thresh)[0]
        idcs = np.intersect1d(idcs_dist, idcs_yaw)

        if len(idcs) > 0:
            if return_multiple:
                return idcs
            assigned_node_id = idcs[int(np.argmin(np.asarray(dist_vals)[idcs]))]
        else:
            assigned_node_id = np.argmin(np.asarray(dist_vals))
            if return_multiple:
                assigned_node_id = np.asarray([assigned_node_id])

        return assigned_node_id

    @staticmethod
    def get_agent_node_masks(hd_map: Dict, agents: Dict, dist_thresh=20, full_connect=True) -> Dict:
        """
        Returns key/val masks for agent-node attention layers. All agents except those within a distance threshold of
        the lane node are masked. The idea is to incorporate local agent context at each lane node.
        """

        lane_node_feats = hd_map['lane_node_feats']
        lane_node_masks = hd_map['lane_node_masks']
        vehicle_feats = agents['vehicles']
        vehicle_masks = agents['vehicle_masks']


        vehicle_node_masks = np.ones((len(lane_node_feats), len(vehicle_feats)))
        if full_connect:
            valid_vehicle=(vehicle_masks[:,:,0].sum(-1))==vehicle_feats.shape[1]
            vehicle_node_masks=np.repeat(valid_vehicle.reshape(1,-1),len(lane_node_feats),axis=0).astype(np.float32)
        else:
            for i, node_feat in enumerate(lane_node_feats):
                if (lane_node_masks[i] == 0).any():
                    node_pose_idcs = np.where(lane_node_masks[i][:, 0] == 0)[0]
                    node_locs = np.expand_dims((node_feat[node_pose_idcs, :2]),axis=0)

                    for j, vehicle_feat in enumerate(vehicle_feats):
                        if (vehicle_masks[j] == 0).any():
                            vehicle_pose_idcs = np.where(vehicle_masks[j][:, 0] == 0)[0]
                            vehicle_locs = np.expand_dims(vehicle_feat[vehicle_pose_idcs, :2],axis=1)
                            dist = np.min(np.linalg.norm(node_locs - vehicle_locs, axis=-1))
                            if dist <= dist_thresh:
                                vehicle_node_masks[i, j] = 0

                    for j, ped_feat in enumerate(ped_feats):
                        if (ped_masks[j] == 0).any():
                            ped_pose_idcs = np.where(ped_masks[j][:, 0] == 0)[0]
                            ped_locs = np.expand_dims(ped_feat[ped_pose_idcs, :2],axis=1)
                            dist = np.min(np.linalg.norm(node_locs - ped_locs, axis=-1))
                            if dist <= dist_thresh:
                                ped_node_masks[i, j] = 0

        agent_node_masks = {'vehicles': vehicle_node_masks, 'pedestrians': ped_node_masks}
        return agent_node_masks
    

    def get_target_node_masks(self,hd_map: Dict, target: Dict, dist_thresh=30) -> Dict:
        """
        Returns key/val masks for agent-node attention layers. All agents except those within a distance threshold of
        the lane node are masked. The idea is to incorporate local agent context at each lane node.
        """

        lane_node_feats = hd_map['lane_node_feats']
        lane_node_masks = hd_map['lane_node_masks']
        past_feats = target['history']['traj']
        past_masks = target['history']['mask']
        fut_feats = target['future']['traj']
        fut_masks = target['future']['mask']

        target_masks = np.ones((2, len(lane_node_feats)))
        past_flag=False
        future_flag=False
        for i, node_feat in enumerate(lane_node_feats):
            if (lane_node_masks[i] == 0).any():
                node_pose_idcs = np.where(lane_node_masks[i][:, 0] == 0)[0]
                node_locs = node_feat[node_pose_idcs, :2]


                if (past_masks == 0).any():
                    past_loc = past_feats[-1, :2]
                    dist = np.min(np.linalg.norm(node_locs - past_loc, axis=1))
                    if dist <= dist_thresh:
                        target_masks[0, i] = 0
                        past_flag=True


                if (fut_masks == 0).any():
                    fut_pose_idc = np.where(fut_masks[:, 0] == 0)[0][-1]
                    fut_loc = fut_feats[fut_pose_idc, :2]
                    dist = np.min(np.linalg.norm(node_locs - fut_loc, axis=1))
                    if dist <= dist_thresh:
                        target_masks[1, i] = 0
                        future_flag=True
                        
        if not past_flag or not future_flag:
            node_poses = []
            node_feat_lens = np.sum(1 - lane_node_masks[:, :, 0], axis=1)
            fut_pose_idc = np.where(fut_masks[:, 0] == 0)[0][-1]
            for i, node_feat in enumerate(lane_node_feats):
                if node_feat_lens[i] != 0:
                    node_poses.append(node_feat[:int(node_feat_lens[i]), :3])
            past_node_id=self.assign_pose_to_node(node_poses, past_feats[-1, :3])
            future_node_id=self.assign_pose_to_node(node_poses, fut_feats[fut_pose_idc, :3])
            target_masks[1, future_node_id] = 0
            target_masks[0, past_node_id] = 0

        return target_masks


