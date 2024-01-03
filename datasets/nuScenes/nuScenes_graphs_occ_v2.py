import matplotlib.pyplot as plt
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.static_layers import correct_yaw
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from datasets.nuScenes.prediction import PredictHelper_occ
import numpy as np
from typing import Dict, Tuple, Union, List
from scipy.spatial.distance import cdist
from nuscenes.utils.splits import create_splits_scenes
from nuscenes import NuScenes
from numpy import linalg as LA
import numpy.ma as ma
import os,copy,json
import random
import pickle

def get_categ(my_string):
    class0=my_string.split(".")[0]
    if class0=="vehicle" and (my_string.split(".")[1]=="bicycle" or my_string.split(".")[1]=="motorcycle"):
        class0="bicycle"
    return class0

def get_tr_scenes(split:str, nusc: NuScenes):
    if split not in {'mini_train', 'train'}:
        raise ValueError("split must be one of (mini_train,  train)")
    
    if split == 'train_val':
        split_name = 'train'
    else:
        split_name = split

    scenes = create_splits_scenes()
    scenes_for_split = scenes[split_name]
    
    if split == 'train':
        scenes_for_split = scenes_for_split[200:]
    scene_tokens=[]
    for scene in nusc.scene:
        if scene['name'] in scenes_for_split:
            scene_tokens.append(scene['token'])
    return scene_tokens

class NuScenesGraphs_OCC(NuScenesVector):
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
        self.map_radius_buffer=80
        self.match=False
        self.add_static_training_samples = args['add_static_training_samples']
        self.split=args['split']
        self.target = args['target']
        self.use_mid_origin=True
        self.adjust_yaw=True
        self.augment=False
        self.random_rots=False
        self.load_tokens()


        # Load dataset stats (max nodes, max agents etc.)
        if self.mode != 'compute_stats':
            stats = self.load_stats()
            self.max_nbr_nodes = stats['max_nbr_nodes']
            self.max_nodes = stats['num_lane_nodes']


    def load_tokens(self):
        file_path = os.path.join('vis_data',self.target+'.json')
        with open(file_path, 'r') as json_file:
            self.data_list=json.load(json_file)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Union[Dict, int]:
        """
        Get data point, based on mode of operation of dataset.
        :param idx: data index
        """
        token_dict=self.data_list[idx]
        if self.mode == 'compute_stats':
            return self.compute_stats(token_dict)
        elif self.mode == 'extract_data':
            self.extract_data(token_dict)
            return 0
        else:
            ret_dict=copy.deepcopy(self.load_data(token_dict))
            # ret_dict['inputs'].pop('surrounding_agent_representation')
            inputs=ret_dict['inputs']
            hist_length=inputs['target_agent_representation']['history'].shape[0]
            future_length=inputs['target_agent_representation']['future'].shape[0]

            history_frames=min(hist_length,5)
            future_frames=min(future_length,5)
            history=inputs['target_agent_representation']['history'][-history_frames:]
            future=inputs['target_agent_representation']['future'][-future_frames:]
            concat_motion=inputs['target_agent_representation']['concat_motion'][-future_length-history_frames:hist_length+future_frames]
            if future_length==future_frames:
                refine_input=inputs['target_agent_representation']['refine_input'][hist_length-history_frames:]
            else:
                refine_input=inputs['target_agent_representation']['refine_input'][hist_length-history_frames:-(future_length-future_frames)]
            hist, hist_masks = self.list_to_tensor([history], 1, int(self.t_h * 2 + 1), 6,False)
            future, future_masks = self.list_to_tensor([future], 1, int(self.t_h * 2 + 1), 6,False)
            concat_motion, concat_masks = self.list_to_tensor([concat_motion], 1, int(self.t_h * 2 + 1)*2, 7,False)
            concat_refine_input, concat_refine_mask = self.list_to_tensor([refine_input], 1, 1+int((self.t_f-self.t_h) * 2 + 1)+int(self.t_h * 2 + 1)*2, 6,False)
            
            reversed_query = inputs['target_agent_representation']['time_query']['query'].copy()
            temp_mask = (1-inputs['target_agent_representation']['time_query']['mask'][:,0]).astype(np.bool)
            reversed_query[temp_mask]=reversed_query[temp_mask][::-1]
            inputs['target_agent_representation']['time_query']['query']=reversed_query
            
            history={'traj':np.squeeze(hist,0),'mask':np.squeeze(hist_masks,0)}
            future={'traj':np.squeeze(future,0),'mask':np.squeeze(future_masks,0)}
            concat={'traj':np.squeeze(concat_motion,0),'mask':np.squeeze(concat_masks,0)}
            refine_input={'traj':np.squeeze(concat_refine_input,0),'mask':np.squeeze(concat_refine_mask,0)}
            target_agent_representation={'history':history,'future':future,'concat_motion':concat,'time_query':inputs['target_agent_representation']['time_query'],'refine_input':refine_input}
            ret_dict['inputs']['target_agent_representation']=target_agent_representation
            return ret_dict
    
    
    def compute_stats(self, token_dict: dict) -> Dict[str, int]:
        """
        Function to compute statistics for a given data point
        """     

        map_radius,origin = self.get_target_agent_representation(token_dict)
        # num_vehicles, num_pedestrians = self.get_surrounding_agent_representation(idx, self.time_lengths[idx],radius=map_radius+self.map_radius_buffer,origin=origin)
        num_lane_nodes, max_nbr_nodes = self.get_map_representation(token_dict,map_radius+self.map_radius_buffer,origin)

        stats = {
            'num_lane_nodes': num_lane_nodes,
            'max_nbr_nodes': max_nbr_nodes
        }

        return stats
    
    def save_data(self, token_dict: Dict, data: Dict):
        """
        Saves extracted pre-processed data
        :param idx: data index
        :param data: pre-processed data
        """
        token=token_dict['start']['ins_token']+"_"+token_dict['start']['sample_token']
        filename = os.path.join(self.data_dir, token + '.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, token_dict: Dict) -> Dict:
        """
        Function to load extracted data.
        :param idx: data index
        :return data: Dictionary with batched tensors
        """
        token=token_dict['start']['ins_token']+"_"+token_dict['start']['sample_token']
        filename = os.path.join(self.data_dir, token + '.pickle')

        if not os.path.isfile(filename):
            raise Exception('Could not find data. Please run the dataset in extract_data mode')

        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def extract_data(self, token_dict: dict):
        """
        Function to extract data. Bulk of the dataset functionality will be implemented by this method.
        :param idx: data index
        """
        inputs,ground_truth = self.get_inputs(token_dict)
        data = {'inputs': inputs, 'ground_truth': ground_truth}
        self.save_data(token_dict, data)

    def get_inputs(self, token_dict: dict) -> Dict:
        i_t=token_dict['start']['ins_token']
        s_t=token_dict['start']['sample_token']
        # time_length=len(token_dict['missing_frames'])
        target_agent_representation,ground_truth,map_radius,origin = self.get_target_agent_representation(token_dict)
        # surrounding_agent_representation = self.get_surrounding_agent_representation(idx,self.time_lengths[idx],radius=map_radius+self.map_radius_buffer,origin=origin)
        map_representation = self.get_map_representation(token_dict,radius=map_radius+self.map_radius_buffer,origin=origin)
        map_representation = self.add_lane_ctrs(map_representation)
        inputs = {'instance_token': i_t,
                  'sample_token': s_t,
                  'map_representation': map_representation,
                #   'surrounding_agent_representation': surrounding_agent_representation,
                  'target_agent_representation': target_agent_representation,
                  'origin': np.asarray(origin)}
        return inputs,ground_truth

    def add_lane_ctrs(self,map_representation):
        encodings=map_representation['lane_node_feats']
        mask=map_representation['lane_node_masks']
        lane_ctrs=(ma.masked_array(encodings[:,:,:2],mask=mask[:,:,:2])).mean(axis=1).data
        lane_ctrs[(~(((1-mask).astype(np.bool))[:,:,:2].any(1)))]=np.inf
        map_representation['lane_ctrs']=lane_ctrs
        return map_representation
    
        
    def get_target_agent_global_pose(self, ann:dict) -> Tuple[float, float, float]:
        """
        Returns global pose of target agent
        :param idx: data index
        :return global_pose: (x, y, yaw) or target agent in global co-ordinates
        """

        sample_annotation = ann
        x, y = sample_annotation['translation'][:2]
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        yaw = correct_yaw(yaw)
        global_pose = (x, y, yaw)

        return global_pose
    
    
    def get_target_agent_representation(self, token_dict: dict) :
        """
        Extracts target agent representation
        :param idx: data index
        :return hist: track history for target agent, shape: [t_h * 2, 5]
        """
        start=token_dict['start']
        end=token_dict['end']
        global_pose = start["ann_token"]
        missing_length=len(token_dict["missing_frames"])
        sample_annotation = self.helper.data.get('sample_annotation',start["ann_token"])
        global_pose=self.get_target_agent_global_pose(sample_annotation)
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        # Get future information
        i_t, s_t=end["ins_token"],end["sample_token"]
        start_sample = self.helper.data.get('sample',start["sample_token"])
        end_sample = self.helper.data.get('sample',s_t)
        time_offset=(end_sample['timestamp'] - start_sample['timestamp'])/1e6

        coords_fut,global_yaw_fut,time_fut = self.helper.get_future_for_agent(start["ins_token"],start["sample_token"], seconds=time_offset+self.t_h, in_agent_frame=False,add_yaw_and_time=True)
        future_rec=np.empty([0,6])
        concat_future=np.empty([0,7])
        count=0
        endpoint_time=time_offset
        if self.target == 'no_point':
            if not (len(coords_fut)>missing_length):
                # print("time offset", time_offset)
                # print("len(coords_fut): ",len(coords_fut))
                # print("missing_length: ",missing_length)
                raise Exception("future data is short")
            for xy,r,t in zip(coords_fut[missing_length:],global_yaw_fut[missing_length:],time_fut[missing_length:]):
                if count==0:
                    origin_fut=(xy[0], xy[1], correct_yaw(quaternion_yaw(Quaternion(r))))
                    if self.use_mid_origin:
                        origin=(np.asarray(global_pose)+np.asarray(origin_fut))/2
                        if self.adjust_yaw:
                            if np.abs(global_pose[-1]-origin_fut[-1])>np.pi:
                                adjusted_fut_yaw=origin_fut[-1]-np.pi*((origin_fut[-1]-global_pose[-1])//np.pi)
                                origin[-1]=(global_pose[-1]+adjusted_fut_yaw)/2
                                
                        origin=tuple(origin)   
                        origin_fut=origin
                local_pose = self.global_to_local(origin, (xy[0], xy[1], quaternion_yaw(Quaternion(r))))
                local_yaw = local_pose[-1]
                future_rec=np.concatenate((future_rec,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),t))])),0)
                concat_future=np.concatenate((concat_future,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),t,1))])),0)
                count+=1
        elif self.target == 'no_ann':
            for xy,r,t in zip(coords_fut,global_yaw_fut,time_fut):
                if count==0:
                    origin_fut=(xy[0], xy[1], correct_yaw(quaternion_yaw(Quaternion(r))))
                    if self.use_mid_origin:
                        origin=(np.asarray(global_pose)+np.asarray(origin_fut))/2
                        if self.adjust_yaw:
                            if np.abs(global_pose[-1]-origin_fut[-1])>np.pi:
                                adjusted_fut_yaw=origin_fut[-1]-np.pi*((origin_fut[-1]-global_pose[-1])//np.pi)
                                origin[-1]=(global_pose[-1]+adjusted_fut_yaw)/2
                                
                        origin=tuple(origin)   
                        origin_fut=origin
                local_pose = self.global_to_local(origin, (xy[0], xy[1], quaternion_yaw(Quaternion(r))))
                local_yaw = local_pose[-1]
                future_rec=np.concatenate((future_rec,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),t))])),0)
                concat_future=np.concatenate((concat_future,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),t,1))])),0)
                count+=1
            
        future=future_rec[::-1]
        try:
            map_radius = LA.norm(future_rec[-1,:2],ord=2)
        except:
            print(missing_length)
            print(time_fut)
            raise Exception("Target",self.target)
        if self.mode == "compute_stats":
            return max(map_radius,25),origin

        # x, y co-ordinates in agent's frame of reference
        try:
            i_t, s_t=start["ins_token"],start["sample_token"]
            coords,global_yaw,time_past = self.helper.get_past_for_agent(i_t, s_t, seconds=self.t_h, in_agent_frame=False,add_yaw_and_time=True)
            global_pose=global_pose[:-1].__add__((yaw,))
            local_pose=self.global_to_local(origin, global_pose)
            local_yaw=local_pose[-1]
            past_hist=np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),0))])
            concat_hist=np.asarray([self.global_to_local(origin, global_pose).__add__((np.cos(local_yaw),np.sin(local_yaw),0,0))])
            count=0
            for xy,r,t in zip(coords,global_yaw,time_past):
                glb_yaw=quaternion_yaw(Quaternion(r))
                local_pose = self.global_to_local(origin, (xy[0], xy[1], glb_yaw))
                local_yaw=local_pose[-1]
                past_hist=np.concatenate((past_hist,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),-t))])),0)
                concat_hist=np.concatenate((concat_hist,np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),-t,0))])),0)
                count+=1
        except:
            global_pose=global_pose[:-1].__add__((yaw,))
            local_pose=self.global_to_local(origin, global_pose)
            local_yaw=local_pose[-1]
            past_hist=np.asarray([local_pose.__add__((np.cos(local_yaw),np.sin(local_yaw),0))])
            concat_hist=np.asarray([self.global_to_local(origin, global_pose).__add__((np.cos(local_yaw),np.sin(local_yaw),0,0))])
        hist=past_hist[::-1]
        
        concat_motion=np.concatenate((concat_hist[::-1],concat_future),axis=0)        
        endpts_query=np.concatenate((np.zeros([1,2]),np.array([[endpoint_time,1.0]])),0)    
         
        time_query=np.empty([0,2])
        gt_poses=np.empty([0,3])
        if self.target == 'no_point':
            i_t, s_t=start["ins_token"],start["sample_token"]
            
            coords_fut,global_yaw_fut,time_fut = self.helper.get_future_for_agent(i_t, s_t, seconds=time_offset, in_agent_frame=False,add_yaw_and_time=True)

            for xy,r,t in zip(coords_fut[:missing_length],global_yaw_fut[:missing_length],time_fut[:missing_length]):
                local_pose = np.asarray([self.global_to_local(origin, (xy[0], xy[1], quaternion_yaw(Quaternion(r))))])
                gt_poses= np.concatenate((gt_poses,local_pose),0)
                time_query=np.concatenate((np.array([[t,t/endpoint_time]]),time_query),0)
            dummy_vals=np.ones([len(time_query),6])*np.inf
            # print(missing_length)
            # print(len(time_query))
            # print(time_fut.shape)
            dummy_vals[:,-1]=time_fut[:missing_length]
        elif self.target == 'no_ann':
            missing_frames=token_dict["missing_frames"]
            assert(len(missing_frames)==missing_length)
            for s_t in missing_frames:
                frame=self.helper.data.get('sample',s_t)
                t=(frame['timestamp'] - start_sample['timestamp'])/1e6
                time_query=np.concatenate((np.array([[t,t/endpoint_time]]),time_query),0)
            assert(len(time_query)==missing_length)
            dummy_vals=np.ones([len(time_query),6])*np.inf
            # print(missing_length)
            # print(len(time_query))
            # print(time_fut.shape)
            dummy_vals[:,-1]=time_query[:,0]

        concat_refine_input=np.concatenate((past_hist[::-1],dummy_vals,future_rec),axis=0)

        time_query, time_query_masks = self.list_to_tensor([time_query], 1, int((self.t_f) * 2 + 1), 2,False)
        query={'query':np.squeeze(time_query,0),'mask':np.squeeze(time_query_masks,0),'endpoints':endpts_query}
        gt_poses, gt_poses_masks = self.list_to_tensor([gt_poses], 1, int((self.t_f) * 2 + 1), 3,False)
        # concat_refine_input, concat_refine_mask = self.list_to_tensor(concat_refine_input, 1, 
        #                                                             int((self.t_f) * 2 + 1)+int(self.t_h * 2 + 1)*2, 6,False)
        # gt_rev, _ = self.list_to_tensor([gt_rev], 1, int((self.t_f-2) * 2 + 1), 3,False)
        ground_truth={'traj':np.squeeze(gt_poses,0),'mask':np.squeeze(gt_poses_masks,0)}
        target_representation={'history':hist,'future':future,'concat_motion':concat_motion,'time_query':query,'refine_input':concat_refine_input}
        return target_representation,ground_truth,max(map_radius,25),origin

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
    
    def get_map_representation(self, token_dict: dict, radius: float, origin: Tuple=None) -> Union[Tuple[int, int], Dict]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        start=token_dict["start"]
        i_t, s_t = start['ins_token'],start["sample_token"]
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
        # s_next, edge_type = self.get_edge_lookup(e_succ, e_prox)
        
        for idx,lane in enumerate(lane_node_feats):
            yaws=lane[:,2].reshape(-1,1)
            cos=np.cos(yaws)
            sin=np.sin(yaws)
            lane_node_feats[idx]=np.concatenate((lane,cos,sin),axis=-1)

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 8)
        # lane_node_feats ~ [B,N,L,C] B: batch size; N: max_number of lanes (nodes) in the scene; L: max_length of a lane; 
        # C: channel number for a lane waypoint, which contains x,y location, theta angle, bools inidacting whether the point
        # locates on 'stop_line', 'ped_crossing' polygons and whether it has successor. 
        
        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
            # 's_next': s_next,
            # 'edge_type': edge_type
        }

        return map_representation
    
    def discard_poses_outside_extent(self, pose_set: List[np.ndarray],
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

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if LA.norm(pose[:2],ord=2)<radius:
                    flag = True
                    break

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set
    
    def get_surrounding_agent_representation(self, idx: int, time: float, radius=None, origin=None) -> \
            Union[Tuple[int, int], Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :return: ndarrays with surrounding pedestrian and vehicle track histories and masks for non-existent agents
        """

        # Get vehicles and pedestrian histories for current sample
        
        vehicles = self.get_agents_of_type(idx, 'vehicle', time, origin)
        pedestrians = self.get_agents_of_type(idx, 'human', time, origin)

        # Discard poses outside map extent
        vehicles = self.discard_poses_outside_extent(vehicles,radius=radius)
        pedestrians = self.discard_poses_outside_extent(pedestrians,radius=radius)

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == 'compute_stats':
            return len(vehicles), len(pedestrians)

        # Convert to fixed size arrays for batching
        if self.use_home:
            vehicles = self.list_to_tensor(vehicles, self.max_vehicles, int(self.t_h * 2 + 1), 5,True)
            pedestrians = self.list_to_tensor(pedestrians, self.max_pedestrians, int(self.t_h * 2 + 1), 5,True)

            surrounding_agent_representation = {
                'vehicles': vehicles,
                'pedestrians': pedestrians
            }
        else:
            if self.random_tf:
                vehicles, vehicle_masks = self.list_to_tensor(vehicles, self.max_vehicles, int((self.t_f-2) * 2 + 1), 4,False)
                pedestrians, pedestrian_masks = self.list_to_tensor(pedestrians, self.max_pedestrians, int((self.t_f-2) * 2 + 1), 4,False)
            else:
                vehicles, vehicle_masks = self.list_to_tensor(vehicles, self.max_vehicles, int((self.t_f) * 2 + 1), 4,False)
                pedestrians, pedestrian_masks = self.list_to_tensor(pedestrians, self.max_pedestrians, int((self.t_f) * 2 + 1), 4,False)

            surrounding_agent_representation = {
                'vehicles': vehicles,
                'vehicle_masks': vehicle_masks,
                'pedestrians': pedestrians,
                'pedestrian_masks': pedestrian_masks
            }
        if self.use_raster:
            i_t, s_t = self.token_list[idx].split("_")
            img = self.agent_rasterizer.make_representation(i_t, s_t)
            img = np.moveaxis(img, -1, 0)
            img = img.astype(float) / 255
            surrounding_agent_representation['image']=img

        return surrounding_agent_representation
    def get_agents_of_type(self, idx: int, agent_type: str, time: float, origin=None) -> List[np.ndarray]:
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
        agent_details = self.helper.get_future_for_sample(s_t, seconds=time, in_agent_frame=False, just_xy=False)
        agent_hist = self.helper.get_future_for_sample(s_t, seconds=time, in_agent_frame=False, just_xy=True,add_yaw_and_time=True)

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
            agent['time']=[]
            for n, pose in enumerate(agent['coords']):
                if random.random()<self.traj_mask_prob and n>=1 and self.mode=='extract_data':
                    continue
                local_pose = self.global_to_local(origin, (pose[0], pose[1], quaternion_yaw(Quaternion(agent['rotation'][n]))))
                agent['xy_coords'].append([local_pose[0], local_pose[1]])
                agent['yaw'].append(local_pose[2])
                agent['time'].append(agent['time_step'][n])
                

        # Flip history to have most recent time stamp last and extract past motion states
        for n, agent in enumerate(agent_list):
            xy = np.flip(agent['xy_coords'], axis=0)
            r = np.flip(agent['yaw'], axis=0)
            t = np.flip(agent['time'], axis=0)
            agent_list[n] = np.concatenate((xy, r.reshape(xy.shape[0],1),t.reshape(xy.shape[0],1)), axis=-1)

        return agent_list

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

    def get_visited_edges(self, idx: int, lane_graph: Dict) -> Tuple[np.ndarray, np.ndarray]:
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
        i_t, s_t = self.token_list[idx].split("_")
        fut_xy = self.helper.get_future_for_agent(i_t, s_t, 6, True)
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
        ped_feats = agents['pedestrians']
        ped_masks = agents['pedestrian_masks']

        vehicle_node_masks = np.ones((len(lane_node_feats), len(vehicle_feats)))
        ped_node_masks = np.ones((len(lane_node_feats), len(ped_feats)))
        if full_connect:
            valid_vehicle=(vehicle_masks[:,:,0].sum(-1))==vehicle_feats.shape[1]
            valid_peds=(ped_masks[:,:,0].sum(-1))==ped_feats.shape[1]
            vehicle_node_masks=np.repeat(valid_vehicle.reshape(1,-1),len(lane_node_feats),axis=0).astype(np.float32)
            ped_node_masks = np.repeat(valid_peds.reshape(1,-1),len(lane_node_feats),axis=0).astype(np.float32)
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


