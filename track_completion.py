
from tqdm import tqdm
import torch,json
import torch.nn as nn
from numpy import linalg as LA
from collections import defaultdict
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, Union, List

import copy
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
# from pcdet.datasets.nuscenes import nuscenes_utils 
import numpy.ma as ma
from data_extraction.data_extraction import *
import torch.utils.data as torch_data
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.prediction.input_representation.static_layers import correct_yaw
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.loaders import interpolate_tracking_boxes
from track_completion_ext import *
from models.model import PredictionModel
from models.encoders.track_completion_encoder import Encoder_occ
from models.aggregators.attention_occ import Attention_occ
from models.decoders.mlp_occ import MLP_occ
import train_eval.utils as u
from torch.utils.data._utils.collate import default_collate
import numpy.linalg as LA

def build_and_load_model(ckpt_path,model_cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()
    print("Loading checkpoint from " + ckpt_path + " ...", end=" ")
    model = PredictionModel(
        Encoder_occ(model_cfg['encoder_args']),
        Attention_occ(model_cfg['aggregator_args']),
        MLP_occ(model_cfg['decoder_args'])
    ).float().to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path,map_location='cuda:0')
    else:
        checkpoint = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Done')
    return model

def collate_func(batch_list):
    missing_timestamps_list=[sample.pop('missing_timestamps') for sample in batch_list]
    all_timestamps_list=[sample.pop('all_timestamps') for sample in batch_list]
    box_size_list=[sample.pop('size') for sample in batch_list]
    batch=default_collate(batch_list)
    batch['missing_timestamps']=missing_timestamps_list
    batch['all_timestamps']=all_timestamps_list
    batch['box_sizes']=box_size_list
    return batch

def clamp_yaw(yaw):
    if yaw<-np.pi:
        clamped_yaw=2*np.pi+yaw
    elif yaw>np.pi:
        clamped_yaw=yaw-2*np.pi
    else:
        clamped_yaw=yaw
    return clamped_yaw

def correct_yaw_inverse(yaw):
    if type(yaw) is float:
        if yaw <= 0:
            yaw = -np.pi - yaw
        else:
            yaw = np.pi - yaw
        return yaw
    else:
        pos_rows=yaw>0
        neg_rows=yaw<0
        yaw[pos_rows]=np.pi-yaw[pos_rows]
        yaw[neg_rows]=-np.pi-yaw[neg_rows]
        return yaw

def get_global_rotation(origin,local_yaw):
    global_flipped_yaw =  float(origin[-1] -local_yaw)
    global_yaw=correct_yaw_inverse(clamp_yaw(global_flipped_yaw))
    return tuple(Quaternion._from_axis_angle(axis=np.array([0,0,1]),angle=global_yaw))

def local_to_global(origin: Tuple, local_pose: Tuple) -> Tuple:
    """
    Converts pose in global co-ordinates to local co-ordinates.
    :param origin: (x, y, yaw) of origin in global co-ordinates
    :param global_pose: (x, y, yaw) in global co-ordinates
    :return local_pose: (x, y, yaw) in local co-ordinates
    """
    # Unpack
    local_x, local_y = local_pose
    origin_x, origin_y, origin_yaw = origin

    # Rotate
    # global_yaw = correct_yaw(global_yaw)

    r = LA.inv(np.asarray([[np.cos(np.pi/2 - origin_yaw), np.sin(np.pi/2 - origin_yaw)],
                    [-np.sin(np.pi/2 - origin_yaw), np.cos(np.pi/2 - origin_yaw)]]))
    local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())
    
    # Translate
    global_x = local_x + origin_x
    global_y = local_y + origin_y

    global_pose = (global_x, global_y)

    return global_pose
class Track_filler():
    def __init__(self, refined_result_path:str, nusc,filling_info,dataset_cfg, data_dir, output_dir=None):

        self.refined_result_path=refined_result_path

        if output_dir is not None:
            self.save_path = Path(output_dir)
        else:
            self.save_path = Path(data_dir) / dataset_cfg.VERSION

        self.cfg_=config_factory('tracking_nips_2019')
        self.dataset_cfg=dataset_cfg
        self.version=dataset_cfg.VERSION
        self.filling_info = filling_info
        self.eval_split='val'
        self.nusc=nusc

        self.load_tracking_result(verbose=True)
        
    

   
    def load_tracking_result(self,verbose):
        pred_boxes, _ = load_prediction(self.refined_result_path, self.cfg_.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
        pred_boxes = add_center_dist(self.nusc, pred_boxes)
        self.pred_boxes = filter_eval_boxes(self.nusc, pred_boxes, self.cfg_.class_range, verbose=verbose)
        return 

    
    
    def interpolate_tracks(self,tracks_by_timestamp: DefaultDict[int, List[TrackingBox]],scene_token,scene_filling_info=None) -> DefaultDict[int, List[TrackingBox]]:
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

        for timestamp, tracking_boxes in tracks_by_timestamp.items():
            for tracking_box in tracking_boxes:
                tracks_by_id[tracking_box.tracking_id].append(tracking_box)
                track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)


        # Interpolate missing timestamps for each track.
        timestamps = tracks_by_timestamp.keys()

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
                    tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                    tracking_box.rotation=tuple(tracking_box.rotation)
                    if scene_filling_info is not None:
                        if tracking_id in scene_filling_info.keys():
                            if str(timestamp) in scene_filling_info[tracking_id].keys() or timestamp in scene_filling_info[tracking_id].keys():
                                assert(gap_dist>=self.dataset_cfg.interpolate_dist_thresh and ((right_timestamp - left_timestamp)/1e6) >= self.dataset_cfg.interpolate_time_thresh)
                                try: 
                                    frame_info=scene_filling_info[tracking_id][timestamp]
                                except:
                                    frame_info=scene_filling_info[tracking_id][str(timestamp)]
                                coord=frame_info['coord']
                                rotation=frame_info['rotation']
                                try:
                                    tracking_box.translation=coord.__add__((tracking_box.translation[-1],))
                                except:
                                    tracking_box.translation=coord.append(tracking_box.translation[-1])
                                tracking_box.rotation=rotation
                                interpolate_count += 1
                        tracks_by_timestamp[timestamp].append(tracking_box)
        print('Scene: ', scene_token)
        print('     Number of long interpolations in scene: ', interpolate_count)
        return tracks_by_timestamp
    

    
    def save_data(self, data: Dict):
        """
        Saves extracted pre-processed data
        :param idx: data index
        :param data: pre-processed data
        """
        
        filename= os.path.join(self.save_path, 'track_pred.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle,protocol=pickle.HIGHEST_PROTOCOL)



    
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

        for scene_id, scene_tracks in tracks.items():
            # For each track_id, collect the scores.
            track_id_scores = defaultdict(list)
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    track_id_scores[box.tracking_id].append(box.tracking_score)

            # # Compute average scores for each track.
            # track_id_avg_scores = {}
            # for tracking_id, scores in track_id_scores.items():
            #     track_id_avg_scores[tracking_id] = np.mean(scores)

            # # Apply average score to each box.
            # for timestamp, boxes in scene_tracks.items():
            #     for box in boxes:
            #         box.tracking_score = track_id_avg_scores[box.tracking_id]


        # Interpolate GT and predicted tracks.
        for scene_token in tracks.keys():
            if scene_token in self.filling_info.keys():
                scene_filling_info = self.filling_info[scene_token]
                tracks[scene_token] = self.interpolate_tracks(tracks[scene_token],scene_token,scene_filling_info)
            else:
                tracks[scene_token] = self.interpolate_tracks(tracks[scene_token],scene_token)
            tracks[scene_token] = defaultdict(list, sorted(tracks[scene_token].items(), key=lambda kv: kv[0]))

        # self.save_data(tracks)
        return tracks
def map_timestamps_to_sample(tracks,nusc):
    results={}
    for scene_token,timestamp_boxes in tracks.items():
        for timestamp,boxes in timestamp_boxes.items():
            assert(len(boxes)>0)
            sample_token=boxes[0].sample_token
            serialized_boxes=[box.serialize() for box in boxes]
            results[sample_token]=serialized_boxes
            
    scene_tokens=tracks.keys()
    for scene_token in scene_tokens:
        # Init all timestamps in this scene.
        scene = nusc.get('scene', scene_token)
        cur_sample_token = scene['first_sample_token']
        while True:
            # Initialize array for current timestamp.
            cur_sample = nusc.get('sample', cur_sample_token)
            assert(cur_sample_token in results.keys())

            # Abort after the last sample.
            if cur_sample_token == scene['last_sample_token']:
                break

            # Move to next sample.
            cur_sample_token = cur_sample['next']
    return results

if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict
    

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="track_completion_model/track_completion.yaml", help='specify the config of dataset')
    parser.add_argument('--result_path', type=str, help='tracking result json file')
    parser.add_argument('--data_dir', type=str, default= 'extrated_track_completion_data',help='data dir')
    parser.add_argument('--batch_size', type=int, default= 32)
    parser.add_argument('--verbose', type=bool, default= True)
    parser.add_argument('--output_dir', type=str, default= 'mot_results/track_completion_results')
    parser.add_argument('--ckpt_path', type=str, help='Trained model ckpt', required=True)

    # parser.add_argument('--skip_compute_stats', action= 'store_false')
    args = parser.parse_args()


    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    data_root = "/home/stanliu/data/mnt/nuScenes/"
    nusc = NuScenes(version=dataset_cfg.VERSION, dataroot=data_root, verbose=True)
    # if Path(os.path.join(args.data_dir,dataset_cfg.VERSION, 'filling_info.json')).exists():
    #     print("Found filling info, use the existing one!")
    #     with open(os.path.join(args.data_dir,dataset_cfg.VERSION, 'filling_info.json'), 'r') as handle:
    #         filled_dict=json.load(handle)
        
    # else:
        
    #     print("Did not find filling info, creating one!")
    model = build_and_load_model(args.ckpt_path,dataset_cfg.model_cfg)
    model.eval()
    test_dataset = Track_completion_EXT(args.result_path,nusc=nusc,dataset_cfg=dataset_cfg,data_dir=args.data_dir,mode='load_data',output_dir=args.output_dir )
    test_dl=torch_data.DataLoader(test_dataset, args.batch_size, shuffle=False,collate_fn=collate_func,
                                    num_workers=dataset_cfg.num_workers, pin_memory=True)

    # For printing progress
    print("Interpolating the fragmented track...")
    mini_batch_count = 0
    # Loop over splits and mini-batches
    filled_dict=defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
    with torch.no_grad():
        for i, data_batch in enumerate(test_dl):
            data = u.send_to_device(u.convert_double_to_float(data_batch))
            predictions = model(data_batch)
            # Show progress
            yaw=predictions['refined_yaw'].clone()
            traj=predictions['refined_traj'].clone().squeeze(1)
            mask=predictions['mask'].clone().squeeze(1)

            
            for idx,traj_submask in enumerate(mask):
                traj_submask=~traj_submask.bool()
                filled_traj=traj[idx][traj_submask].cpu().numpy()
                filled_yaw=yaw[idx][traj_submask].cpu().numpy()
                # motion=data['target_agent_representation']['concat_motion']
                # motion_mask=~motion['mask'][idx,:,0].bool()
                # motion_traj=motion['traj'][idx,:,:2][motion_mask]
                # motion_yaw=motion['traj'][idx,:,2][motion_mask]
                origin=tuple(data['origin'][idx].cpu().numpy())
                missing_timestamps=data['missing_timestamps'][idx]
                tracking_id=data['tracking_id'][idx]
                scene_token=data['scene_token'][idx]
                
                for idx,timestamp in enumerate(missing_timestamps):
                    global_coord=local_to_global(origin, tuple(filled_traj[idx]))
                    global_rotation=get_global_rotation(origin,tuple(filled_yaw[idx]))
                    box_info={'coord':global_coord,'rotation':global_rotation}
                    filled_dict[scene_token][tracking_id][timestamp]=box_info
            if args.verbose:
                print("mini batch " + str(mini_batch_count + 1) + '/' + str(len(test_dl)))
                mini_batch_count += 1
        # file_name=os.path.join(args.data_dir,dataset_cfg.VERSION, 'filling_info.json')
        # # print(filled_dict)
        # with open(file_name, 'w') as handle:
        #     json.dump(dict(filled_dict), handle)
    
    track_filler=Track_filler(args.result_path, nusc,filled_dict,dataset_cfg, args.data_dir, args.output_dir)
    tracks=track_filler.create_tracks()
    with open(args.result_path, 'r') as handle:
        ReID_data=json.load(handle)
    results=map_timestamps_to_sample(tracks,nusc)
    ReID_data['results']=results
    with open(os.path.join(args.output_dir,dataset_cfg.VERSION,'final_tracking_result.json'), "w") as f:
        json.dump(ReID_data, f)