import torch.optim
from typing import Dict, Union
import torch
import numpy as np
import os
from datasets.nuScenes.prediction import PredictHelper_occ
from nuscenes.prediction.input_representation.static_layers import *
from nuscenes.prediction.input_representation.combinators import Rasterizer
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
import logging
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as im


def return_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

device = return_device()

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


def send_to_device(data: Union[Dict, torch.Tensor]):
    """
    Utility function to send nested dictionary with Tensors to GPU
    """
    if type(data) is torch.Tensor:
        return data.to(device).clone().detach()
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v)
        return data
    else:
        return data


def convert2tensors(data):
    """
    Converts data (dictionary of nd arrays etc.) to tensor with batch_size 1
    """
    if type(data) is np.ndarray:
        return torch.as_tensor(data).unsqueeze(0)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert2tensors(v)
        return data
    else:
        return data
    
@torch.no_grad()
def local_pose_to_image(local_poses,pose_mask,resolution,img_size,arror_length=None):
    '''local_poses: [T,4] 
    mask: [T]
    '''
    if arror_length is None:
        arror_length=8
    y_m=np.asarray(local_poses[:,1][pose_mask].cpu())
    x_m=np.asarray(local_poses[:,0][pose_mask].cpu())
    img_origin=np.round(np.asarray(img_size)/2).astype(np.int)
    x=img_origin[1]+x_m*resolution
    y=img_origin[0]-y_m*resolution
    yaw=np.asarray((local_poses[:,2][pose_mask]).cpu())
    dy=-np.sin(yaw+np.pi/2)*arror_length
    dx=np.cos(yaw+np.pi/2)*arror_length
    return x,y,dx,dy
    
@torch.no_grad()
def visualize(inputs: Dict,ground_truth: Dict,predictions: Dict,helper: PredictHelper_occ, save_folder, file_name, mode='raw'):
    upper_limit=10
    batch_size=len(predictions['traj'])
    layer_names = ['drivable_area', 'ped_crossing']
    maps= load_all_maps(helper)
    colors = [(255, 255, 255), (119, 136, 153)]
    for sample_id in range(batch_size):
        if sample_id>upper_limit:
            return
        # try:
        instance_token=inputs['instance_token'][sample_id]
        sample_token=inputs['sample_token'][sample_id]
        future=inputs['target_agent_representation']['future']['traj'][sample_id]
        mask_fut=inputs['target_agent_representation']['future']['mask'][sample_id]
        hist=inputs['target_agent_representation']['history']['traj'][sample_id]
        nearest_idx=np.where(mask_fut[:, 0].cpu() == 0)[0][-1]
        prediction_horizon=future[nearest_idx,-1]
        sample_annotation = helper.get_sample_annotation(instance_token, sample_token)
        map_name = helper.get_map_name_from_sample_token(sample_token)
        x, y = sample_annotation['translation'][:2]
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        yaw_corrected = correct_yaw(yaw)
        global_pose=(x,y,yaw_corrected)
        if 'origin' in inputs:
            origin=tuple([inputs['origin'][sample_id,0].item(),inputs['origin'][sample_id,1].item(),inputs['origin'][sample_id,2].item()])
        else:
            coords_fut,global_yaw_fut,time_fut = helper.get_future_for_agent(instance_token, sample_token, seconds=2+prediction_horizon, in_agent_frame=False,add_yaw_and_time=True)

            sep_idx= np.searchsorted(time_fut, (prediction_horizon-0.001).item())
            origin_fut=coords_fut[sep_idx][0],coords_fut[sep_idx][1],correct_yaw(quaternion_yaw(Quaternion(global_yaw_fut[sep_idx])))
            origin=tuple((np.asarray(global_pose)+np.asarray(origin_fut))/2)
        dist=LA.norm(future[0,:2].cpu(),ord=2)
        image_side_length = 2 * max(25,dist+10)
        image_side_length_pixels = 400
        resolution=image_side_length_pixels/image_side_length
        patchbox = get_patchbox(origin[0], origin[1], image_side_length)

        angle_in_degrees = angle_of_rotation(origin[2]) * 180 / np.pi

        canvas_size = (image_side_length_pixels, image_side_length_pixels)
        masks = maps[map_name].get_map_mask(patchbox, angle_in_degrees, layer_names, canvas_size=canvas_size)
        
        images = []
        for mask, color in zip(masks, colors):
            images.append(change_color_of_binary_mask(np.repeat(mask[::-1, :, np.newaxis], 3, 2), color))
        if mode=='refine':
            traj = predictions['refined_traj'][sample_id].squeeze(0)
            yaw = predictions['refined_yaw'][sample_id]
        elif mode=='raw':
            traj = predictions['traj'][sample_id].squeeze(0)
            yaw = predictions['yaw'][sample_id]
        lanes=inputs['map_representation']['lane_node_feats'][sample_id].flatten(0,1).clone()
        lanes_mask=inputs['map_representation']['lane_node_feats'][sample_id].flatten(0,1)[:,0].bool()
        pred = torch.cat((traj,yaw),-1)
        pose_pred_mask=~(predictions['mask'][sample_id]).bool()
        gt = ground_truth['traj'][sample_id]
        image = Rasterizer().combine(images)
        pose_future_mask=~inputs['target_agent_representation']['future']['mask'][sample_id][:,0].bool()
        pose_hist_mask=~inputs['target_agent_representation']['history']['mask'][sample_id][:,0].bool()
        xs, ys, dxs, dys=local_pose_to_image(future,pose_future_mask,resolution,canvas_size)
        xsh, ysh, dxsh, dysh=local_pose_to_image(hist,pose_hist_mask,resolution,canvas_size)
        xsp, ysp, dxsp, dysp=local_pose_to_image(pred,pose_pred_mask,resolution,canvas_size,5)
        xsg, ysg, dxsg, dysg=local_pose_to_image(gt,pose_pred_mask,resolution,canvas_size,5)
        xsl, ysl, dxsl, dysl=local_pose_to_image(lanes,lanes_mask,resolution,canvas_size,2.5)
        # plt.imshow(image)
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        
        for x, y, dx, dy in zip(xs, ys, dxs, dys):
            ax.arrow(x, y, dx, dy, width=0.8, color=(1,0,0,1))
        for x, y, dx, dy in zip(xsh, ysh, dxsh, dysh):
            ax.arrow(x, y, dx, dy, width=0.8, color=(0,1,0,1))
        for x, y, dx, dy in zip(xsp, ysp, dxsp, dysp):
            ax.arrow(x, y, dx, dy, width=1.0, color=(0.0,0,1,1))
        for x, y, dx, dy in zip(xsg, ysg, dxsg, dysg):
            ax.arrow(x, y, dx, dy, width=1.0, color=(1,0,1,0.3))
        for x, y, dx, dy in zip(xsl, ysl, dxsl, dysl):
            ax.arrow(x, y, dx, dy, width=0.5, color=(1,0.5,0,0.3))
        ax.imshow(image)
        ax.grid(False)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        im.imsave(os.path.join(save_folder,file_name+'_sample'+str(sample_id)+'_'+mode), image_from_plot)
        # except:
        #     continue
    return

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger