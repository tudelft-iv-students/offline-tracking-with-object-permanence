import sys
import train_eval.utils as u
import math 
from train_eval.trainer import Trainer
import yaml
import matplotlib.pyplot as plt
# from train_eval.trainer import Trainer
import torch
import os
import pickle
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
occ_times=[]
scores=[]
map_scores=[]
gt_occ_times=[]
gt_scores=[]
gt_map_scores=[]
with open("configs/match_train.yml", 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)
# from torch.utils.tensorboard import SummaryWriter
# import os
# writer = SummaryWriter(log_dir=os.path.join('/home/stanliu/code/pgp/PGP/output/test_ram', 'tensorboard_logs'))
# trainer = Trainer(cfg, "/home/stanliu/data/mnt/nuScenes/", "/home/stanliu/code/pgp/PGP/preprocess_home","output/test_home_original_9/checkpoints/7.tar")
# visualizer = Visualizer(cfg, "/home/stanliu/data/mnt/nuScenes/", "/home/stanliu/code/pgp/PGP/preprocess_home","output/test_home_original_7/checkpoints/7.tar")
# trainer = Trainer(cfg, "/home/stanliu/data/mnt/nuScenes/", "preprocess_graph_match_mini_test",'output/test_match/checkpoints/16.tar')
trainer = Trainer(cfg, "/home/stanliu/data/mnt/nuScenes/", "preprocess_graph_match_add_lane_info",'output/test_match/augmentation_w1.0/checkpoints/54.tar')
with torch.no_grad():
    for i,data in enumerate(trainer.val_dl):
        # torch.cuda.empty_cache()
        # Load data
        sys.stdout.write('processing %d, %d/%d\r' % (i, i+1, len(trainer.val_dl)))
        sys.stdout.flush()

        data = u.send_to_device(u.convert_double_to_float(data))
        data_test=data['inputs']
        gt_test=data['ground_truth']
        
        predictions=trainer.model(data_test)
        times=data_test['future'][:,1:,:,-3]
        gt_times=data_test['future'][:,0,:,-3]
        mask=data_test['future_mask'][:,1:,:,-3].bool()
        gt_mask=data_test['future_mask'][:,0,:,-3].bool()
        times[mask]=math.inf
        gt_times[gt_mask]=math.inf
        occ_time,inds=torch.min(times,dim=-1)
        occ_time=occ_time[~torch.isinf(occ_time)]
        gt_time,_=torch.min(gt_times,dim=-1)
        score=predictions['scores'][:,1:][(~(predictions['masks'][:,1:,:,0]).bool()).any(dim=-1)].flatten()
        map_score=predictions['map_scores'][:,1:][(~(predictions['masks'][:,1:,:,0]).bool()).any(dim=-1)].flatten()
        gt_score=predictions['scores'][:,0].flatten()
        gt_map_score=predictions['map_scores'][:,0].flatten()
        occ_times+=list(occ_time.cpu().numpy())
        gt_occ_times+=list(gt_time.cpu().numpy())
        scores+=list(score.cpu().numpy())
        map_scores+=list(map_score.cpu().numpy())
        gt_scores+=list(gt_score.cpu().numpy())
        gt_map_scores+=list(gt_map_score.cpu().numpy())

data={"occ_times":occ_times,"gt_occ_times":gt_occ_times,"scores":scores,"gt_scores":gt_scores,'map_scores':map_scores,"gt_map_scores":gt_map_scores}

filename = os.path.join("temp" + '.pickle')
with open(filename, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


