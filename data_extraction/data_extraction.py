import tqdm,pickle,copy
from functools import reduce
from pathlib import Path
from nuscenes.utils import splits
import numpy as np
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.data_classes import EvalBoxes
from bisect import bisect
from typing import List, Dict, DefaultDict,Tuple,Union
from collections import defaultdict
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
import tqdm,os
from tqdm import trange
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import sys, os
sys.path.append(os.path.dirname(__file__))
from bbox import *
import torch

target_types={
    'car',
    'bus',
    'truck',
    'trailer'
    
}

def get_target_tracks(tracks_by_timestamp: DefaultDict[int, List[TrackingBox]]) -> DefaultDict[int, List[TrackingBox]]:
    """
    Interpolate the tracks to fill in holes, especially since GT boxes with 0 lidar points are removed.
    This interpolation does not take into account visibility. It interpolates despite occlusion.
    :param tracks_by_timestamp: The tracks.
    :return: The interpolated tracks.
    """
    # Group tracks by id.
    timestamps = tracks_by_timestamp.keys()
    tracks_by_timestamp = {k: v for k, v in sorted(tracks_by_timestamp.items())}
    tracks_by_id = defaultdict(list)
    track_timestamps_by_id = defaultdict(list)
    target_dict=defaultdict(dict)
    bicycle_dict=defaultdict(dict)
    pedestrain_dict=defaultdict(dict)
    # i=0
    for timestamp, tracking_boxes in tracks_by_timestamp.items():
        for tracking_box in tracking_boxes:
            tracks_by_id[tracking_box.tracking_id].append(tracking_box)
            track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)
            if tracking_box.tracking_name in target_types:
                target_dict[tracking_box.tracking_id][timestamp]=tracking_box
            elif tracking_box.tracking_name in ["bicycle","motorcycle"]:
                bicycle_dict[tracking_box.tracking_id][timestamp]=tracking_box
            elif tracking_box.tracking_name in ["pedestrian"]:
                pedestrain_dict[tracking_box.tracking_id][timestamp]=tracking_box

    return target_dict

def create_tracks_infos(all_boxes: EvalBoxes, nusc: NuScenes, eval_split: str, gt: bool, data_path: str ='/home/stanliu/data/mnt/nuScenes/nuscenes',trk_Score_thresh=0.2) \
        -> Dict[str, Dict[int, List[TrackingBox]]]:
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
    for sample_token in all_boxes.sample_tokens:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene = nusc.get('scene', scene_token)
        if scene['name'] in splits[eval_split]:
            scene_tokens.add(scene_token)

    # Tracks are stored as dict {scene_token: {timestamp: List[TrackingBox]}}.
    tracks = defaultdict(lambda: defaultdict(list))
    vec_tracks_scene_ids = defaultdict(lambda: defaultdict(list))
    # Init all scenes and timestamps to guarantee completeness.
    for scene_token in scene_tokens:
        # Init all timestamps in this scene.
        scene = nusc.get('scene', scene_token)
        cur_sample_token = scene['first_sample_token']
        while True:
            # Initialize array for current timestamp.
            cur_sample = nusc.get('sample', cur_sample_token)
            tracks[scene_token][cur_sample['timestamp']] = []

            # Abort after the last sample.
            if cur_sample_token == scene['last_sample_token']:
                break

            # Move to next sample.
            cur_sample_token = cur_sample['next']

    # Group annotations wrt scene and timestamp.
    for sample_token in all_boxes.sample_tokens:##For prediction, fill the empty tracks by adding tracking results to the corresponding time frame
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        tracks[scene_token][sample_record['timestamp']] = all_boxes.boxes[sample_token]

    # Replace box scores with track score (average box score). This only affects the compute_thresholds method and
    # should be done before interpolation to avoid diluting the original scores with interpolated boxes.
    if not gt:
        track_id_max_score_frame = defaultdict(lambda: defaultdict(int))
        track_id_avg_scores = defaultdict(lambda: defaultdict(float))
        for scene_id, scene_tracks in tracks.items():
            # For each track_id, collect the scores.
            track_id_scores = defaultdict(list)
            scene_tracks = defaultdict(list, sorted(scene_tracks.items(), key=lambda kv: kv[0]))
            
            
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    track_id_scores[box.tracking_id].append(box.tracking_score)

            # Compute average scores for each track.
            
            
            for tracking_id, scores in track_id_scores.items():
                track_id_avg_scores[scene_id][tracking_id] = np.mean(scores)
                track_id_max_score_frame[scene_id][tracking_id]=np.argmax(scores)


            # Apply average score to each box.
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    box.tracking_score = track_id_avg_scores[scene_id][box.tracking_id]
    test_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(tracks), desc='create_info', dynamic_ncols=True)
    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.
    for scene_token in tracks.keys():
        progress_bar.update()
        scene_tracks= get_target_tracks(tracks[scene_token])
        scene_info={'scene_token':scene_token,'scene_tracks':[]}
        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        init_time=nusc.get('sample', first_sample_token)['timestamp']

        for track_id in scene_tracks.keys():
            # print(track_id_avg_scores[scene_token][tracking_id] < trk_Score_thresh and len(scene_tracks[track_id])<2)
            if track_id_avg_scores[scene_token][track_id] < trk_Score_thresh and len(scene_tracks[track_id])<2:
                continue
            scene_tracks[track_id] = defaultdict(list, sorted(scene_tracks[track_id].items(),key= lambda kv: kv[0]))
            track_info={'track_id':track_id,'frame_info_list':[],'valid_frame_count':len(scene_tracks[track_id]),
                        'max_score_frame_idx':track_id_max_score_frame[scene_token][tracking_id],
                        'score':track_id_avg_scores[scene_token][tracking_id]}
            for idx,(timestep,det) in enumerate(scene_tracks[track_id].items()):
                sample_token=det.sample_token
                sample=nusc.get('sample', sample_token)
                ref_sd_token = sample['data'][ref_chan]
                ref_sd_rec = nusc.get('sample_data', ref_sd_token)
                ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
                ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
                curr_time = 1e-6 * (timestep-init_time)
                if idx == 0:
                    instance_name=det.tracking_name
                    track_info['instance_name']=np.array([instance_name])
                    initial_time = curr_time
                    track_info['left_time']=initial_time
                ref_lidar_path= nusc.get_sample_data_path(ref_sd_token)
                # Homogeneous transform from reference frame to ego car frame
                car_from_ref = transform_matrix(
                    ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=False
                )

                # Homogeneous transformation matrix from ego car to global frame
                global_from_car = transform_matrix(
                    ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=False,
                )
                
                # Homogeneous transformation matrix from sensor reference to global frame
                global_from_ref = reduce(np.dot, [global_from_car, car_from_ref])
                
                # Homogeneous transformation matrix from global frame to local frame
                local_from_global = transform_matrix(
                    det.translation, Quaternion(det.rotation), inverse=True,
                )
                
                # Homogeneous transformation matrix from global to _current_ ego car frame
                # Information for each sample
                sample_info = {
                    'sample_index':idx,
                    'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                    # 'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
                    # 'cam_intrinsic': ref_cam_intrinsic,
                    'sample_token': sample_token,
                    'global_from_ref': global_from_ref,
                    'local_from_global': local_from_global,
                    'timestamp': curr_time-initial_time,
                    'absolute_time_in_scene':curr_time
                }
                locs = np.asarray(det.translation).reshape(-1, 3)
                dims = np.array(det.size).reshape(-1, 3)[:, [1, 0, 2]]
                velocity = np.array(det.velocity).reshape(-1, 2)
                rots = np.array(quaternion_yaw(Quaternion(det.rotation))).reshape(-1, 1)
                boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1) #In global frame
                vis_box = BBox.array2bbox(np.asarray(det.translation.__add__((quaternion_yaw(Quaternion(det.rotation)),)).__add__((det.size[1],det.size[0],det.size[2]))))
                ann_info={'location':locs,
                        'dimension':dims,
                        'velocity':velocity,
                        'rotation':rots,
                        'box':boxes,
                        'vis_box':vis_box}
            
                frame_info={'annotation_info':ann_info,
                            'sample_info':sample_info}
                track_info['right_time']=curr_time
                track_info['frame_info_list'].append(frame_info)
                    
            scene_info['scene_tracks'].append(track_info)
        print('Number of tracks in scene: ',len(scene_info['scene_tracks']))
        test_nusc_infos.append(scene_info)
        vec_tracks_scene_ids[scene_token] = scene_tracks


    return test_nusc_infos

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes

def extract_data( version, data_path, save_path, result_path ,cfg,max_sweeps=15,verbose=False):
    
    save_path = save_path / version

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        eval_split = "val"
        val_scenes = splits.val
    elif version == 'v1.0-test':
        eval_split = "test"
        val_scenes = splits.test
    elif version == 'v1.0-mini':
        eval_split = "mini_val"
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError
    pred_boxes, _ = load_prediction(result_path, cfg.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    pred_boxes = add_center_dist(nusc, pred_boxes)
    # print(len(pred_boxes))
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])
    # print(save_path)
    print('%s:  val scene(%d)' % (version, len(val_scenes)))
    
    if not ((save_path / f'nuscenes_infos_occ_{max_sweeps}sweeps_test_temp.pkl').exists()):
        
        test_nusc_infos = create_tracks_infos(
            pred_boxes, nusc, eval_split, gt=False)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path / f'nuscenes_infos_occ_{max_sweeps}sweeps_test_temp.pkl', 'wb') as f:
            pickle.dump(test_nusc_infos, f)
            f.close()
    else:
        with open(save_path / f'nuscenes_infos_occ_{max_sweeps}sweeps_test_temp.pkl', 'rb') as f:
            test_nusc_infos = pickle.load(f)
            f.close()
    raise NotImplementedError()        
    test_nusc_infos = extract_pcl(save_path,test_nusc_infos,max_sweeps)

    with open(save_path / f'nuscenes_infos_occ_{max_sweeps}sweeps_test.pkl', 'wb') as f:
        pickle.dump(test_nusc_infos, f)
        f.close()

def toOpen3d(points):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:,0:3])
    return cloud

def get_lidar_all_sweeps(infos,segment_ground=False):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    from pcdet.utils import common_utils
    def remove_ego_points(points, center_radius=2.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]
    def segment_ground_points(pc_lidar,num_pts):
        _, inliers =pc_lidar.segment_plane(distance_threshold=0.08, ransac_n=10, num_iterations=8000)
        no_ground_mask=np.ones(num_pts).astype(bool)
        no_ground_mask[inliers]=False
        ground_mask=~no_ground_mask
        # pc_lidar_ground=pc_lidar.select_by_index(inliers, invert=False)
        # pc_lidar_noground=pc_lidar.select_by_index(inliers, invert=True)
        return no_ground_mask,ground_mask
    max_sweeps=15
    # if infos['gap_index'] is not None and infos['gap_index']>0:
    #     sep_index=[infos['gap_index'],infos['gap_index']+1]  
    #     sep_indices=[infos['frame_info_list'][idx]['sample_info']['timestamp'] for idx in sep_index]
    # else:
    agg_points=np.empty([5,0])
    for idx,frame_info in enumerate(infos['frame_info_list']):
        ann_info=frame_info['annotation_info']
        sample_info=frame_info['sample_info']
        lidar_path = Path('/home/stanliu/data/mnt/nuScenes/')/ 'nuscenes'/ sample_info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points).T
        ## Transorm to global frame
        global_from_ref = sample_info['global_from_ref']
        num_points = points_sweep.shape[1]
        points_sweep[:3, :] = global_from_ref.dot(np.vstack((points_sweep[:3, :], 
                                                                np.ones(num_points))))[:3, :]
        all_points=points_sweep.T
        if segment_ground:
            cloud= toOpen3d(all_points[:,:3])
            not_ground_mask,_=segment_ground_points(cloud,num_points)
        ## Retrieve points inside bbox
        box=ann_info['box'].copy()
        box[0,3:6]=box[0,3:6]*1.1
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(all_points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(box[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
        valid_mask=(box_idxs_of_pts==0)
        if segment_ground:
            valid_mask=valid_mask*not_ground_mask
        target_points=all_points[valid_mask].T
        ## Transform to local frame
        local_from_global = sample_info['local_from_global']
        target_points[:3, :]=local_from_global.dot(np.vstack((target_points[:3, :], 
                                                    np.ones(target_points.shape[1]))))[:3, :]
        target_points=np.vstack((target_points, 
                                np.ones(target_points.shape[1])*sample_info['timestamp']))
        agg_points=np.concatenate((agg_points,target_points),axis=1)
        if idx==0:
            gt_box=ann_info['box'].copy()
        else:
            gt_box+=ann_info['box']
    gt_box[:,:3]=0
    gt_box[:,6:]=0
    gt_box/=len(infos['frame_info_list'])

    return agg_points.T,gt_box
            
def extract_pcl(save_path,infos,max_sweeps, used_classes=None):

    database_save_path = save_path / f'nuscenes_occ_database_{max_sweeps}sweeps_withvelo'
    # db_info_save_path = save_path / f'nuscenes_occ_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

    database_save_path.mkdir(parents=True, exist_ok=True)
    # all_db_infos = {}

    for scene_idx in trange(len(infos)):
        scene_info=infos[scene_idx]
        scene_save_path=database_save_path / scene_info['scene_token']
        if not os.path.exists(scene_save_path):
            os.makedirs(scene_save_path)
        for idx,track_info in enumerate(scene_info['scene_tracks']):
            info = copy.deepcopy(track_info)
            agg_points,gt_box = get_lidar_all_sweeps(info)
            track_info['gt_box']=gt_box
            filename = '%s.bin' % (info['track_id'])
            filepath = scene_save_path / filename
            agg_points=agg_points.astype(np.float32)
            with open(filepath, 'w') as f:
                agg_points.tofile(f)

    return infos