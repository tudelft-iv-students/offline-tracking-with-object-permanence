
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
import copy
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
# from pcdet.datasets.nuscenes import nuscenes_utils 
from itertools import compress
import numpy.ma as ma
from data_extraction.data_extraction import *
import torch.utils.data as torch_data
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.prediction.input_representation.static_layers import correct_yaw
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.loaders import interpolate_tracks,interpolate_tracking_boxes
class Track_completion_EXT(torch_data.Dataset):
    def __init__(self, refined_result_path:str, nusc:NuScenes,dataset_cfg, data_dir,mode='test', output_dir=None):
        super().__init__()
        self.refined_result_path=refined_result_path
        if dataset_cfg.get('VERSION', None) is None:
            try:
                dataset_cfg.VERSION = dataset_cfg.version
            except:
                raise Exception('Specify version!')
        if not os.path.isdir(Path(data_dir) / dataset_cfg.VERSION):
            Path(Path(data_dir) / dataset_cfg.VERSION).mkdir(parents=True, exist_ok=True)
        if output_dir is not None:
            self.save_path = Path(output_dir) / dataset_cfg.VERSION
            self.read_path = Path(data_dir) / dataset_cfg.VERSION
        else:
            self.read_path = self.save_path = Path(data_dir) / dataset_cfg.VERSION
        if not os.path.isdir(self.save_path):
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(self.read_path):
            Path(self.read_path).mkdir(parents=True, exist_ok=True)
        self.cfg_=config_factory('tracking_nips_2019')
        self.dataset_cfg=dataset_cfg
        self.version=dataset_cfg.VERSION
        self.nusc=nusc
        self.helper=PredictHelper_occ(nusc)
        self.mode = mode
        self.motion_time=self.t_h=dataset_cfg.t_h
        self.interpolate_dist_thresh=dataset_cfg.interpolate_dist_thresh
        self.interpolate_time_thresh=dataset_cfg.interpolate_time_thresh
        self.eval_split='val'
        self.token_id_list_file = os.path.join(self.read_path,'token_id_lists.txt')
 
        
        
        map_locs = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.maps = {i: NuScenesMap(map_name=i, dataroot=self.helper.data.dataroot) for i in map_locs}
        self.polyline_length=20
        self.polyline_resolution=1
        self.map_extent=[-100, 100, -100, 150]
        if self.mode in ['extract_data','compute_stats']:
            self.load_tracking_result(verbose=True)
            self.tracks=self.create_tracks()
            self.scene_tokens=list(self.tracks.keys())
            if self.mode == 'compute_stats':
                path = os.path.join( self.token_id_list_file) 
                if os.path.exists(path):
                    os.remove(path)
            else:
                stats=self.load_stats()
                self.max_nodes=stats['num_lane_nodes']
                self.max_missing_horizon=stats['max_missing_horizon']
        else:
            stats=self.load_stats()
            self.max_missing_horizon=stats['max_missing_horizon']
            with open(self.token_id_list_file, "r") as f:
                self.token_list= [str(line.strip()) for line in f]
            

    
    def __getitem__(self, idx: int) -> Union[Dict, int]:
        if self.mode in ['extract_data','compute_stats']:
            scene_token=self.scene_tokens[idx]
            return self.extract_data(copy.deepcopy(self.tracks[scene_token]),scene_token)
        elif self.mode == 'load_data':
            data=self.load_data(idx)
            if not ('endpoints' in data['target_agent_representation']['time_query'].keys()):
                future=copy.deepcopy(data['target_agent_representation']['future'])
                future_mask=~(future['mask'][:,0]).astype(bool)
                endpoint_time=future['traj'][:,-1][future_mask][-1]
                endpoint_query = np.concatenate((np.zeros([1,2]),np.array([[endpoint_time,1.0]])),0)   
                data['target_agent_representation']['time_query']['endpoints']=endpoint_query
            return data
        
    def __len__(self):
        """
        Size of dataset
        """
        if self.mode in ['extract_data','compute_stats']:
            return len(self.scene_tokens)
        else:
            return len(self.token_list)
        
        
    def load_stats(self) -> Dict[str, int]:
        """
        Function to load dataset statistics like max surrounding agents, max nodes, max edges etc.
        """
        filename = os.path.join(self.read_path, 'stats.pickle')
        if not os.path.isfile(filename):
            print('Tried to find ',filename)
            raise Exception('Could not find dataset statistics. Please run the dataset in compute_stats mode')

        with open(filename, 'rb') as handle:
            stats = pickle.load(handle)

        return stats
        
        
        
    def load_tracking_result(self,verbose):
        pred_boxes, _ = load_prediction(self.refined_result_path, self.cfg_.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
        pred_boxes = add_center_dist(self.nusc, pred_boxes)
        self.pred_boxes = filter_eval_boxes(self.nusc, pred_boxes, self.cfg_.class_range, verbose=verbose)
        return 
    def get_agents_of_type(self,track_dicts, left_time, right_time, origin,radius):
        agent_tracks=[]
        for track_id,track in track_dicts.items():
            single_track=[]
            for time,box in track.items():
                if left_time <= time <= right_time:
                    global_pose=box.translation[:-1].__add__((correct_yaw(quaternion_yaw(Quaternion(box.rotation))),))
                    local_pose=self.global_to_local(origin,global_pose)
                    local_time=(time-left_time)/1e6
                    step_info=local_pose.__add__((local_time,))
                    single_track.append(step_info)
            min_dist=np.min(LA.norm(np.array(single_track)[:,:2],axis=-1))
            if min_dist < radius:
                agent_tracks.append(np.array(single_track[::-1]))
        return agent_tracks
    @staticmethod     
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
        # global_yaw = correct_yaw(global_yaw)
        theta = np.arctan2(-np.sin(global_yaw-origin_yaw), np.cos(global_yaw-origin_yaw))

        r = np.asarray([[np.cos(np.pi/2 - origin_yaw), np.sin(np.pi/2 - origin_yaw)],
                        [-np.sin(np.pi/2 - origin_yaw), np.cos(np.pi/2 - origin_yaw)]])
        local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

        local_pose = (local_x, local_y, theta)

        return local_pose
    def get_target_motion(self,target_track: List, start_idx,track_timestamps,direction,origin,left_idx=None):
        assert(direction in ["left","right"])
        if direction == "left":
            left_acc_time=0
            idx=0
            left_motion_hist=[]
            left_time_hist=[]
            left_idx=start_idx
            
            while left_acc_time < self.motion_time:
                if left_idx-idx<0:
                    break
                if type(target_track[left_idx-idx]) is int:
                    idx+=1
                    continue
                time_diff=(track_timestamps[left_idx-idx]-track_timestamps[left_idx])/1e6
                left_time_hist.append(time_diff)
                global_pose=target_track[left_idx-idx].translation[:-1].__add__((correct_yaw(quaternion_yaw(Quaternion(target_track[left_idx-idx].rotation))),))
                left_motion_hist.append(self.global_to_local(origin,global_pose))
                idx+=1
                left_acc_time+=abs(time_diff)
            trans_hist=np.array(left_motion_hist[::-1])
            cosine_hist=np.cos(trans_hist[:,2]).reshape(-1,1)
            sine_hist=np.sin(trans_hist[:,2]).reshape(-1,1)
            time_hist=np.array([left_time_hist[::-1]])
        elif direction == "right":
            right_acc_time=0
            idx=0
            right_motion_hist=[]
            right_time_hist=[]
            right_idx=start_idx
            while right_acc_time < self.motion_time:
                if right_idx+idx>len(target_track)-1:
                    break
                time_diff=(track_timestamps[right_idx+idx]-track_timestamps[left_idx])/1e6
                right_time_hist.append(time_diff)
                global_pose=target_track[right_idx+idx].translation[:-1].__add__((correct_yaw(quaternion_yaw(Quaternion(target_track[right_idx+idx].rotation))),))
                right_motion_hist.append(self.global_to_local(origin,global_pose))
                right_acc_time+=abs((track_timestamps[right_idx+idx]-track_timestamps[right_idx])/1e6)
                idx+=1
            trans_hist=np.array(right_motion_hist[::-1])
            cosine_hist=np.cos(trans_hist[:,2]).reshape(-1,1)
            sine_hist=np.sin(trans_hist[:,2]).reshape(-1,1)
            time_hist=np.array([right_time_hist[::-1]])
        # if len(trans_hist) <=1:
        #     print("Warning: tracklet length smaller than 2!!!!")
        #     print(np.asarray(track_timestamps)/1e6)
        return np.concatenate((trans_hist,cosine_hist,sine_hist,time_hist.T),-1)
    
    def interpolate_tracks(self,tracks_by_timestamp: DefaultDict[int, List[TrackingBox]]) -> DefaultDict[int, List[TrackingBox]]:
        """
        Interpolate the tracks to fill in holes, especially since GT boxes with 0 lidar points are removed.
        This interpolation does not take into account visibility. It interpolates despite occlusion.
        :param tracks_by_timestamp: The tracks.
        :return: The interpolated tracks.
        """
        # Group tracks by id.

        tracks_by_timestamp = {k: v for k, v in sorted(tracks_by_timestamp.items())}
        tracks_by_id = defaultdict(list)
        track_timestamps_by_id = defaultdict(list)
        # vehicle_dict=defaultdict(dict)
        # pedestrain_dict=defaultdict(dict)
        # i=0
        for timestamp, tracking_boxes in tracks_by_timestamp.items():
            for tracking_box in tracking_boxes:
                tracks_by_id[tracking_box.tracking_id].append(tracking_box)
                track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)

                # if tracking_box.tracking_name in ["pedestrian"]:
                #     pedestrain_dict[tracking_box.tracking_id][timestamp]=tracking_box
                # else:
                #     vehicle_dict[tracking_box.tracking_id][timestamp]=tracking_box
            # i+=1

        # Interpolate missing timestamps for each track.
        timestamps = tracks_by_timestamp.keys()
        # print(len(timestamps))
        interpolate_count = 0
        for timestamp in timestamps:
            for tracking_id, track in tracks_by_id.items():
                if track_timestamps_by_id[tracking_id][0] <= timestamp <= track_timestamps_by_id[tracking_id][-1] and \
                        timestamp not in track_timestamps_by_id[tracking_id]:

                    # Find the closest boxes before and after this timestamp.
                    right_ind = bisect(track_timestamps_by_id[tracking_id], timestamp)
                    left_ind = right_ind - 1
                    right_timestamp = track_timestamps_by_id[tracking_id][right_ind]
                    left_timestamp = track_timestamps_by_id[tracking_id][left_ind]
                    right_tracking_box = tracks_by_id[tracking_id][right_ind]
                    left_tracking_box = tracks_by_id[tracking_id][left_ind]
                    right_ratio = float(right_timestamp - timestamp) / (right_timestamp - left_timestamp)
                    gap_dist=LA.norm(np.array(right_tracking_box.translation[:-1])-np.array(left_tracking_box.translation[:-1]),ord=2)
                    # tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                    # Interpolate.
                    if gap_dist<self.interpolate_dist_thresh or ((right_timestamp - left_timestamp)/1e6) < self.interpolate_time_thresh:
                        tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                        interpolate_count += 1
                        tracks_by_timestamp[timestamp].append(tracking_box)
        print("First interpolate_count: ",  interpolate_count)
        return tracks_by_timestamp
    
    
    def extract_data(self,tracks_by_timestamp: DefaultDict[int, List[TrackingBox]],scene_token:str):
        timestamps = tracks_by_timestamp.keys()
        if self.mode == 'compute_stats':
            num_lane_nodes=0
            max_missing_horizon=0
        tracks_by_timestamp = {k: v for k, v in sorted(tracks_by_timestamp.items())}
        tracks_by_id = defaultdict(list)
        track_timestamps_by_id = defaultdict(list)
        for timestamp, tracking_boxes in tracks_by_timestamp.items():
            for tracking_box in tracking_boxes:
                tracks_by_id[tracking_box.tracking_id].append(tracking_box)
                track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)

        interpolate_count=0
        for stamp_idx,timestamp in enumerate(timestamps):
            # print(stamp_idx,"/",len(timestamps))
            for tracking_id, track in tracks_by_id.items():
                if track_timestamps_by_id[tracking_id][0] <= timestamp <= track_timestamps_by_id[tracking_id][-1] and \
                        timestamp not in track_timestamps_by_id[tracking_id]:
                    interpolate_count+=1
                    right_ind = bisect(track_timestamps_by_id[tracking_id], timestamp)
                    left_ind = right_ind - 1
                    # if type(tracks_by_id[tracking_id][right_ind])==int or type(tracks_by_id[tracking_id][left_ind])==int:
                    #     raise Exception('Insertion error!!')


                    right_timestamp = track_timestamps_by_id[tracking_id][right_ind]
                    left_timestamp = track_timestamps_by_id[tracking_id][left_ind]
                    # right_tracking_box = tracks_by_id[tracking_id][right_ind]
                    # left_tracking_box = tracks_by_id[tracking_id][left_ind]
                    
                    right_tracking_box = tracks_by_id[tracking_id][right_ind]
                    left_tracking_box = tracks_by_id[tracking_id][left_ind]
                    box_size=(np.array(left_tracking_box.size)+np.array(right_tracking_box.size))/2
                    total_time=(right_timestamp-left_timestamp)/1e6
                    missing_timestamps=collect_times(timestamps,track_timestamps_by_id[tracking_id][left_ind:right_ind+1])
                    times=(np.array(missing_timestamps)-left_timestamp)/1e6
                    time_query=np.concatenate((times.reshape(-1,1),times.reshape(-1,1)/total_time),axis=1)
                    # right_ratio = float(right_timestamp - timestamp) / (right_timestamp - left_timestamp)
                    gap_dist=LA.norm(np.array(right_tracking_box.translation[:-1])-np.array(left_tracking_box.translation[:-1]),ord=2)
                    if gap_dist<self.interpolate_dist_thresh or ((right_timestamp - left_timestamp)/1e6) < self.interpolate_time_thresh:
                        print(tracking_id, timestamp)
                        raise Exception("Previous interpoltion goes wrong!!!")
                    # tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                    # Interpolate.
                    

                    left_yaw=correct_yaw(quaternion_yaw(Quaternion(tracks_by_id[tracking_id][left_ind].rotation)))
                    right_yaw=correct_yaw(quaternion_yaw(Quaternion(tracks_by_id[tracking_id][right_ind].rotation)))
                    disp_vec=np.asarray(right_tracking_box.translation[:-1])-np.asarray(left_tracking_box.translation[:-1])
                    est_yaw=np.arctan2(disp_vec[1],disp_vec[0])
                    est_yaw=correct_yaw(est_yaw)

                    left_pose=tracks_by_id[tracking_id][left_ind].translation[:-1].__add__((left_yaw,))
                    right_pose=tracks_by_id[tracking_id][right_ind].translation[:-1].__add__((right_yaw,))
                    origin=(np.asarray(left_pose)+np.asarray(right_pose))/2

                    origin[-1]=est_yaw
                    origin=tuple(origin)
                    left_motion=self.get_target_motion(tracks_by_id[tracking_id], left_ind, track_timestamps_by_id[tracking_id],"left",origin)
                    right_motion=self.get_target_motion(tracks_by_id[tracking_id], right_ind, track_timestamps_by_id[tracking_id],"right",origin,left_ind)
                    bounds=[track_timestamps_by_id[tracking_id][left_ind-len(left_motion)+1],
                            track_timestamps_by_id[tracking_id][right_ind+len(right_motion)-1]]
                    all_timestamps=[stamp for stamp in list(timestamps) if (stamp>=bounds[0] and stamp<=bounds[1])]
                    radius=max(np.max(LA.norm(left_motion[:,:2],axis=-1)),np.max(LA.norm(right_motion[:,:2],axis=-1)))+15.0
                    concat_left_motion=np.concatenate((left_motion,np.zeros([len(left_motion),1])),axis=1)
                    concat_right_motion=np.concatenate((right_motion,np.ones([len(right_motion),1])),axis=1)
                    concat_motion=np.concatenate((concat_left_motion,concat_right_motion[::-1]),axis=0)
                    dummy_vals=np.ones([len(time_query),6])*np.inf
                    dummy_vals[:,-1]=times
                    concat_refine_input=np.concatenate((left_motion,dummy_vals,right_motion[::-1]),axis=0)
                    # surrounding_vehicles=self.get_agents_of_type(vehicle_dict,left_timestamp,right_timestamp,origin,radius)
                    # # surrounding_bikes=get_agents_of_type(bicycle_dict,left_timestamp,right_timestamp,origin)
                    # surrounding_pedestrain=self.get_agents_of_type(pedestrain_dict,left_timestamp,right_timestamp,origin,radius)
                    num_nodes, max_nbrs,lane_node_feats=self.get_map_representation(radius,origin,scene_token)
                    
                    track_timestamps_by_id[tracking_id]+=missing_timestamps
                    track_timestamps_by_id[tracking_id].sort()
                    tracks_by_id[tracking_id]=tracks_by_id[tracking_id][:left_ind+1]+list(range(len(missing_timestamps)))+tracks_by_id[tracking_id][right_ind:]
                    token_id=scene_token+"_"+tracking_id+"_"+str(left_timestamp)+"_"+str(right_timestamp)
                    if self.mode == 'compute_stats':
                        num_lane_nodes=max(num_lane_nodes,num_nodes)
                        max_missing_horizon=max(max_missing_horizon,len(time_query))
                        with open(self.token_id_list_file, 'a') as handle:
                            handle.write(f"{token_id}\n")
                            
                    elif self.mode == 'extract_data':
                        ret_dict={}
                        hist, hist_masks = self.list_to_tensor([left_motion], 1, int(self.t_h * 2 + 1), 6)
                        future, future_masks = self.list_to_tensor([right_motion], 1, int(self.t_h * 2 + 1), 6)
                        concat_motion, concat_masks = self.list_to_tensor([concat_motion], 1, int(self.t_h * 2 + 1)*2, 7)
                        concat_refine_input, concat_refine_mask = self.list_to_tensor([concat_refine_input], 1, self.max_missing_horizon+int(self.t_h * 2 + 1)*2, 6)
                        history={'traj':np.squeeze(hist,0),'mask':np.squeeze(hist_masks,0)}
                        future={'traj':np.squeeze(future,0),'mask':np.squeeze(future_masks,0)}
                        concat={'traj':np.squeeze(concat_motion,0),'mask':np.squeeze(concat_masks,0)}
                        refine_input={'traj':np.squeeze(concat_refine_input,0),'mask':np.squeeze(concat_refine_mask,0)}
                        time_query, time_query_masks = self.list_to_tensor([time_query], 1, self.max_missing_horizon, 2)
                        endpoint_query=np.concatenate((np.zeros([1,2]),np.array([[total_time,1.0]])),0)  
                        query={'query':np.squeeze(time_query,0),'mask':np.squeeze(time_query_masks,0),'endpoints':endpoint_query}
                        target_agent_representation={'history':history,'future':future,'concat_motion':concat,'time_query':query,'refine_input':refine_input}
                        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 8)
                        map_representation = {'lane_node_feats': lane_node_feats,'lane_node_masks': lane_node_masks}
                        map_representation = self.add_lane_ctrs(map_representation)
                        ret_dict['target_agent_representation']=target_agent_representation
                        ret_dict['map_representation']=map_representation
                        ret_dict['missing_timestamps']=missing_timestamps
                        ret_dict['tracking_id']=tracking_id
                        ret_dict['origin']= np.asarray(origin)
                        ret_dict['scene_token']= scene_token
                        ret_dict['size']=box_size
                        ret_dict['all_timestamps']=all_timestamps
                        # ret_dict['translation']=np.asarray(tracks_by_id[tracking_id][left_ind].translation)
                        # ret_dict['rotation']=np.asarray(tracks_by_id[tracking_id][left_ind].translation)
                        self.save_data(token_id,ret_dict)
        print("Long gap to interpolate: ", interpolate_count)
        if self.mode == 'compute_stats':
            stats={
                'num_lane_nodes': num_lane_nodes,
                'max_missing_horizon': max_missing_horizon
            }
            return stats
        elif self.mode == 'extract_data':
            return 0
        
    def add_lane_ctrs(self,map_representation):
        encodings=map_representation['lane_node_feats']
        mask=map_representation['lane_node_masks']
        lane_ctrs=(ma.masked_array(encodings[:,:,:2],mask=mask[:,:,:2])).mean(axis=1).data
        lane_ctrs[(~(((1-mask).astype(np.bool))[:,:,:2].any(1)))]=np.inf
        map_representation['lane_ctrs']=lane_ctrs
        return map_representation
    
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
    
    def add_lane_ctrs(self,map_representation):
        encodings=map_representation['lane_node_feats']
        mask=map_representation['lane_node_masks']
        lane_ctrs=(ma.masked_array(encodings[:,:,:2],mask=mask[:,:,:2])).mean(axis=1).data
        lane_ctrs[(~(((1-mask).astype(np.bool))[:,:,:2].any(1)))]=np.inf
        map_representation['lane_ctrs']=lane_ctrs
        return map_representation
    
    def save_data(self, tokens: str, data: Dict):
        """
        Saves extracted pre-processed data
        :param idx: data index
        :param data: pre-processed data
        """
        dir_name = os.path.join(self.save_path,'preprocess_data')
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name,exist_ok=True)
        
        filename= os.path.join(dir_name, tokens + '.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, idx: int) -> Dict:
        """
        Function to load extracted data.
        :param idx: data index
        :return data: Dictionary with batched tensors
        """
        filename = os.path.join(self.read_path,'preprocess_data', self.token_list[idx] + '.pickle')

        if not os.path.isfile(filename):
            raise Exception('Could not find data. Please run the dataset in extract_data mode')

        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        return data
    
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

    def get_map_representation(self, radius: float, global_pose: Tuple, s_t:str) -> Union[Tuple[int, int], Dict]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """

        map_name = self.get_map_name_from_scene_token(s_t)
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


        return num_nodes, max_nbrs,lane_node_feats
    
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
        lanes = [np.asarray([self.global_to_local(origin, pose) for pose in lane]) for lane in lanes]

        # Concatenate lane poses and lane flags
        lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]

        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(lane_node_feats, self.polyline_length, lane_ids)

        return lane_node_feats, lane_node_ids
    
    def get_successor_edges(self,lane_ids: List[str], map_api: NuScenesMap) -> List[List[int]]:
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
            # for i in range(2,6):
            #     if node_id + i < len(lane_ids) and lane_id == lane_ids[node_id + i]:
            #         e_succ_node.append(node_id + i)

            
            e_succ.append(e_succ_node)

        return e_succ
    
    def get_map_name_from_scene_token(self, scene_token: str) -> str:

        scene = self.helper.data.get('scene', scene_token)
        log = self.helper.data.get('log', scene['log_token'])
        return log['location']


    def get_proximal_edges(self,lane_node_feats: List[np.ndarray], e_succ: List[List[int]],
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
    
    def create_tracks(self) -> Dict[str, Dict[int, List[TrackingBox]]]:
        """
        Returns all tracks for all scenes. Samples within a track are sorted in chronological order.
        This can be applied either to GT or predictions.
        :param all_boxes: Holds all GT or predicted boxes.
        :param nusc: The NuScenes instance to load the sample information from.
        :param eval_split: The evaluation split for which we create tracks.
        :param gt: Whether we are creating tracks for GT or predictions
        :return: The tracks.
        """
        # Only keep samples from this split.
        splits = create_splits_scenes()
        scene_tokens = set()
        for sample_token in self.pred_boxes.sample_tokens:
            scene_token = self.nusc.get('sample', sample_token)['scene_token']
            scene = self.nusc.get('scene', scene_token)
            if scene['name'] in splits[self.eval_split]:
                scene_tokens.add(scene_token)

        # Tracks are stored as dict {scene_token: {timestamp: List[TrackingBox]}}.
        tracks = defaultdict(lambda: defaultdict(list))

        # Init all scenes and timestamps to guarantee completeness.
        for scene_token in scene_tokens:
            # Init all timestamps in this scene.
            scene = self.nusc.get('scene', scene_token)
            cur_sample_token = scene['first_sample_token']
            while True:
                # Initialize array for current timestamp.
                cur_sample = self.nusc.get('sample', cur_sample_token)
                tracks[scene_token][cur_sample['timestamp']] = []

                # Abort after the last sample.
                if cur_sample_token == scene['last_sample_token']:
                    break

                # Move to next sample.
                cur_sample_token = cur_sample['next']

        # Group annotations wrt scene and timestamp.
        for sample_token in self.pred_boxes.sample_tokens:##For prediction, fill the empty tracks by adding tracking results to the corresponding time frame
            sample_record = self.nusc.get('sample', sample_token)
            scene_token = sample_record['scene_token']
            tracks[scene_token][sample_record['timestamp']] = self.pred_boxes.boxes[sample_token]

        # Replace box scores with track score (average box score). This only affects the compute_thresholds method and
        # should be done before interpolation to avoid diluting the original scores with interpolated boxes.
        car_tracks=defaultdict(lambda: defaultdict(list))
        for scene_id, scene_tracks in tracks.items():
            # For each track_id, collect the scores.
            track_id_scores = defaultdict(list)
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    track_id_scores[box.tracking_id].append(box.tracking_score)

            # Compute average scores for each track.
            track_id_avg_scores = {}
            for tracking_id, scores in track_id_scores.items():
                track_id_avg_scores[tracking_id] = np.mean(scores)

            # Apply average score to each box.
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    box.tracking_score = track_id_avg_scores[box.tracking_id]
            
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    if box.tracking_name == 'car':
                        car_tracks[scene_id][timestamp].append(box)

        # Interpolate GT and predicted tracks.
        for scene_token in car_tracks.keys():
            car_tracks[scene_token] = self.interpolate_tracks(car_tracks[scene_token])
            car_tracks[scene_token] = defaultdict(list, sorted(car_tracks[scene_token].items(), key=lambda kv: kv[0]))
            # if self.mode=='debug':
            #     return tracks[scene_token],tracks_by_ids[scene_token],track_timestamps_by_id
            # self.extract_data(copy.deepcopy(car_tracks[scene_token] ),scene_token)

        return car_tracks
    
    @staticmethod
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
    
def collect_times(timestamps,bounds):
    selected_timestamps=[stamp for stamp in list(timestamps) if (stamp>bounds[0] and stamp<bounds[1])]
    return selected_timestamps



if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict
    

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="track_completion_model/track_completion.yaml", help='specify the config of dataset')
    parser.add_argument('--result_path', type=str, help='tracking result json file')
    parser.add_argument('--data_root', type=str, required=True, help='nuscenes dataroot')
    parser.add_argument('--data_dir', type=str, default= 'extracted_track_completion_data',help='output dir')
    parser.add_argument('--tracker_name', type=str, default= 'CenterPoint',help='tracker name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--verbose', type=bool, default= True)
    parser.add_argument('--output_dir', type=str, default= None)
    parser.add_argument('--version', type=str, default= None)
    parser.add_argument('--skip_compute_stats', action= 'store_false')
    args = parser.parse_args()
    args.data_dir=os.path.join(args.data_dir,args.tracker_name)


    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    if args.version is not None:
        dataset_cfg.VERSION=args.version
    
    version = dataset_cfg.VERSION
    if version == 'v1.0-trainval':
        val_scenes = splits.val
    elif version == 'v1.0-test':
        val_scenes = splits.val
    elif version == 'v1.0-mini':
        val_scenes = splits.mini_val
    
    nusc = NuScenes(version=version, dataroot=args.data_root, verbose=True)
    if args.skip_compute_stats:
        print("Computing Stats ...")
        compute_dataset = Track_completion_EXT(args.result_path,nusc=nusc,dataset_cfg=dataset_cfg,data_dir=args.data_dir,mode='compute_stats',output_dir=args.output_dir )
        compute_dl=torch_data.DataLoader(compute_dataset, args.batch_size, shuffle=False,
                                        num_workers=dataset_cfg.num_workers, pin_memory=True)
        stats={}
        mini_batch_count=0
        num_mini_batches = len(compute_dl)
        print("Computing data stats...")
        for index,mini_batch_stats in enumerate(compute_dl):
            for k, v in mini_batch_stats.items():

                if k in stats.keys():
                    stats[k] = max(stats[k], torch.max(v).item())
                else:
                    stats[k] = torch.max(v).item()
            if args.verbose:
                print("mini batch " + str(mini_batch_count + 1) + '/' + str(num_mini_batches))
                mini_batch_count += 1
                
        filename = os.path.join(compute_dataset.save_path, 'stats.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del compute_dataset,compute_dl
    
    extract_dataset = Track_completion_EXT(args.result_path,nusc=nusc,dataset_cfg=dataset_cfg,data_dir=args.data_dir,mode='extract_data',output_dir=args.output_dir )
    extract_dl=torch_data.DataLoader(extract_dataset, args.batch_size, shuffle=False,
                                    num_workers=dataset_cfg.num_workers, pin_memory=True)
    # For printing progress
    print("Extracting pre-processed data...")
    mini_batch_count=0
    # Loop over splits and mini-batches
    for i, _ in enumerate(extract_dl):

        # Show progress
        if args.verbose:
            print("mini batch " + str(mini_batch_count + 1) + '/' + str(num_mini_batches))
            mini_batch_count += 1
