import copy
import pickle
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import torch

from data_extraction.data_extraction import extract_data
from nuscenes.eval.common.config import config_factory

os.environ["MKL_NUM_THREADS"] = "18"
os.environ["NUMEXPR_NUM_THREADS"] = "18"
os.environ["OMP_NUM_THREADS"] = "18"

# class NuScenesDataset_OCC_EXT(DatasetTemplate):
#     def __init__(self, dataset_cfg, class_names, scene_idx, training=False, root_path=None, logger=None):
#         # if root_path is None:
#         #     raise NotImplementedError()
#         self.save_path = (Path(__file__).resolve().parent ).resolve() / 'extracted_mot_data' / dataset_cfg.VERSION
#         root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
#         self.version=dataset_cfg.VERSION
#         super().__init__(
#             dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
#         )
#         self.ref_lidar_path=dataset_cfg.REF_PCL_PATH
#         self.infos = []

#         self.include_nuscenes_data(self.mode,scene_idx)
        
#     def include_nuscenes_data(self, mode,scene_idx):
#         self.logger.info('Loading NuScenes dataset')
#         nuscenes_infos = []

#         for info_path in self.dataset_cfg.INFO_PATH[mode]:
#             info_path = self.save_path / info_path
#             if not info_path.exists():
#                 continue
#             with open(info_path, 'rb') as f:
#                 infos = pickle.load(f)
#                 nuscenes_infos.extend(infos[scene_idx]['scene_tracks'])
#                 self.scene_token=infos[scene_idx]['scene_token']

#         self.infos.extend(nuscenes_infos)
#         self.logger.info('Total tracks in scene %d : %d' % (scene_idx,len(self.infos)))

#     def get_sweep(self, sweep_info):
#         def remove_ego_points(points, center_radius=1.0):
#             mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
#             return points[mask]

#         lidar_path = Path('/home/stanliu/data/mnt/nuScenes/')/ 'nuscenes' / sweep_info['lidar_path']
#         points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
#         points_sweep = remove_ego_points(points_sweep).T
#         if sweep_info['transform_matrix'] is not None:
#             num_points = points_sweep.shape[1]
#             points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
#                 np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

#         cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
#         return points_sweep.T, cur_times.T


#     def __len__(self):
#         # if self._merge_all_iters_to_one_epoch:
#         #     return len(self.infos) * self.total_epochs

#         return len(self.infos)

#     def __getitem__(self, index):
#         # if self._merge_all_iters_to_one_epoch:
#         #     index = index % len(self.infos)

#         info = copy.deepcopy(self.infos[index])
#         i_t=info['track_id']
#         points = self.load_pcl(i_t)

#         input_dict = {
#             'points': points,
#             'left_time':info['left_time'],
#             'right_time':info['right_time'],
#             'track_id': i_t,
#             'frame_num':info['valid_frame_count'],
#             'box':info['gt_box'],
#             'track_names':info['instance_name']
#         }


#         data_dict = self.prepare_data(data_dict=input_dict)

#         if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
#             gt_boxes = data_dict['gt_boxes']
#             gt_boxes[np.isnan(gt_boxes)] = 0
#             data_dict['gt_boxes'] = gt_boxes

#         if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
#             data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

#         return data_dict
    
#     def load_pcl(self,i_t):
#         filename = os.path.join(self.save_path, self.ref_lidar_path, self.scene_token,i_t + '.bin')

#         if not os.path.isfile(filename):
#             raise Exception('Could not find data. Please run the dataset in extract_data mode')

#         with open(filename, 'rb') as handle:
#             # data = pickle.load(handle)
#             data = np.fromfile(handle, dtype=np.float32).reshape([-1, 5])
#         return data
    
#     def prepare_data(self, data_dict):
#         """
#         Args:
#             data_dict:
#                 points: optional, (N, 3 + C_in)
#                 gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
#                 gt_names: optional, (N), string
#                 ...

#         Returns:
#             data_dict:
#                 frame_id: string
#                 points: (N, 3 + C_in)
#                 gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
#                 gt_names: optional, (N), string
#                 use_lead_xyz: bool
#                 voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
#                 voxel_coords: optional (num_voxels, 3)
#                 voxel_num_points: optional (num_voxels)
#                 ...
#         """
#         if self.training:
#             assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
#             gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            
#             data_dict = self.data_augmentor.forward(
#                 data_dict={
#                     **data_dict,
#                     'gt_boxes_mask': gt_boxes_mask
#                 }
#             )
#             # data_dict={
#             #         **data_dict,
#             #         'gt_boxes_mask': gt_boxes_mask
#             #     }
#         if data_dict.get('sep_indices', None) is not None:
#             future_inds=data_dict['points'][:,-1]>=data_dict['sep_indices'][1]
#             future_points=(data_dict['points'][future_inds]).copy()
#             past_inds=data_dict['points'][:,-1]<=data_dict['sep_indices'][0]
#             past_points=(data_dict['points'][past_inds]).copy()
#             if self.training:
#                 noise_translate_std=[0.1,0.1,0.1]
#                 rot_range=[-0.25,0.25]
#                 future_points=translate_points(future_points,noise_translate_std)
#                 future_points=rotate_points(future_points,rot_range)
#                 past_points=translate_points(past_points,noise_translate_std)
#                 past_points=rotate_points(past_points,rot_range)
#             data_dict['future_points']=future_points
#             data_dict['past_points']=past_points

#         if data_dict.get('gt_boxes', None) is not None:
#             selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
#             data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
#             data_dict['gt_names'] = data_dict['gt_names'][selected]
#             gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
#             gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
#             data_dict['gt_boxes'] = gt_boxes

#             if data_dict.get('gt_boxes2d', None) is not None:
#                 data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

#         if data_dict.get('points', None) is not None:
#             data_dict = self.point_feature_encoder.forward(data_dict)

#         data_dict = self.data_processor.forward(
#             data_dict=data_dict
#         )

#         if self.training and len(data_dict['gt_boxes']) == 0:
#             new_index = np.random.randint(self.__len__())
#             return self.__getitem__(new_index)

#         data_dict.pop('gt_names', None)

#         return data_dict
        
#     def evaluation(self, det_annos, class_names, **kwargs):
#         import json
#         from nuscenes.nuscenes import NuScenes
#         from pcdet.datasets.nuscenes import nuscenes_utils
#         nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(Path('/home/stanliu/data/mnt/nuScenes/nuscenes')), verbose=True)
#         nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
#         nusc_annos['meta'] = {
#             'use_camera': False,
#             'use_lidar': True,
#             'use_radar': False,
#             'use_map': False,
#             'use_external': False,
#         }

#         output_path = Path(kwargs['output_path'])
#         output_path.mkdir(exist_ok=True, parents=True)
#         res_path = str(output_path / 'results_nusc.json')
#         with open(res_path, 'w') as f:
#             json.dump(nusc_annos, f)

#         self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

#         if self.dataset_cfg.VERSION == 'v1.0-test':
#             return 'No ground-truth annotations for evaluation', {}

#         from nuscenes.eval.detection.config import config_factory
#         from nuscenes.eval.detection.evaluate import NuScenesEval

#         eval_set_map = {
#             'v1.0-mini': 'mini_val',
#             'v1.0-trainval': 'val',
#             'v1.0-test': 'test'
#         }
#         try:
#             eval_version = 'detection_cvpr_2019'
#             eval_config = config_factory(eval_version)
#         except:
#             eval_version = 'cvpr_2019'
#             eval_config = config_factory(eval_version)

#         nusc_eval = NuScenesEval(
#             nusc,
#             config=eval_config,
#             result_path=res_path,
#             eval_set=eval_set_map[self.dataset_cfg.VERSION],
#             output_dir=str(output_path),
#             verbose=True,
#         )
#         metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

#         with open(output_path / 'metrics_summary.json', 'r') as f:
#             metrics = json.load(f)

#         result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
#         return result_str, result_dict

 
# def translate_points(points,noise_translate_std):
#     assert len(noise_translate_std) == 3
#     noise_translate = np.array([
#         np.random.normal(0, noise_translate_std[0], 1),
#         np.random.normal(0, noise_translate_std[1], 1),
#         np.random.normal(0, noise_translate_std[2], 1),
#     ], dtype=np.float32).T

#     points[:, :3] += noise_translate
#     return points
# def rotate_points(points,rot_range):
#     noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
#     points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
#     return points



# def toOpen3d(points):
#     cloud = o3d.geometry.PointCloud()
#     cloud.points = o3d.utility.Vector3dVector(points[:,0:3])
#     return cloud

# def segment_ground_points(pc_lidar,num_pts):
#     _, inliers =pc_lidar.segment_plane(distance_threshold=0.08, ransac_n=10, num_iterations=8000)
#     no_ground_mask=np.ones(num_pts).astype(bool)
#     no_ground_mask[inliers]=False
#     ground_mask=~no_ground_mask
#     # pc_lidar_ground=pc_lidar.select_by_index(inliers, invert=False)
#     # pc_lidar_noground=pc_lidar.select_by_index(inliers, invert=True)
#     return no_ground_mask,ground_mask

if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='extract_data', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--result_path', type=str, default='mot_results/v1.0-test/tracking_result.json', help='')
    args = parser.parse_args()

    if args.func == 'extract_data':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
        dataset_cfg.VERSION = args.version
        cfg_ = config_factory('tracking_nips_2019')
        cfg_.class_names=['car',
                        'bus',
                        'truck',
                        'trailer']
        
        extract_data(
            version=args.version,
            # data_path=ROOT_DIR / 'data' / 'nuscenes',
            data_path='/home/stanliu/data/mnt/nuScenes/nuscenes',
            save_path=(Path(__file__).resolve().parent ).resolve() / 'extracted_mot_data'/'final_version_nms',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            result_path=args.result_path,
            cfg=cfg_
        )
        # Use the infos extracted before
        # nuscenes_dataset = NuScenesDataset_OCC_EXT(
        #     dataset_cfg=dataset_cfg, class_names=None,
        #     root_path=ROOT_DIR / 'data' / 'nuscenes',
        #     # root_path=Path('/home/stanliu/data/mnt/nuScenes/'),
        #     logger=common_utils.create_logger(), training=False
        # )
        # nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
