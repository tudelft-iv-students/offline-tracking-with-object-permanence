# Code adapted from PGP https://github.com/nachiket92/PGP

import torch.utils.data as torch_data
from typing import Dict
from train_eval.initialization import initialize_prediction_model, initialize_metric,\
    initialize_dataset, get_specific_args
import torch
import os
import train_eval.utils as u
import numpy as np
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.data_classes import Prediction
import json
from datasets.nuScenes.prediction import PredictHelper_occ
from datasets.nuScenes.nuScenes_graphs_match import match_collate
from models.library.baselines import *

# Initialize device:
device = u.return_device()



class Match_Evaluator:
    """
    Class for evaluating trained models
    """
    def __init__(self, cfg1: Dict, data_root: str, data_dir: str, checkpoint_path1: str):
        """
        Initialize evaluator object
        :param cfg1: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        """

        # Initialize dataset
        ds_type = cfg1['dataset'] + '_' + cfg1['agent_setting'] + '_' + cfg1['input_representation']
        spec_args = get_specific_args(cfg1['dataset'], data_root, cfg1['version'] if 'version' in cfg1.keys() else None)
        test_set = initialize_dataset(ds_type, ['load_data', data_dir, cfg1['test_set_args']] + spec_args)
        self.teacher_force=False
        # Initialize dataloader
        if "match" in cfg1:
            if cfg1['match'] == True:
                self.dl = torch_data.DataLoader(test_set, cfg1['batch_size'], shuffle=True, num_workers=cfg1['num_workers'],collate_fn=match_collate)
                self.add_baseline=True
        else:
            self.dl = torch_data.DataLoader(test_set, cfg1['batch_size'], shuffle=True, num_workers=cfg1['num_workers'])
            self.add_baseline=False

        # Initialize model
        self.model1 = initialize_prediction_model(cfg1['encoder_type'], cfg1['aggregator_type'], cfg1['decoder_type'],
                                                 cfg1['encoder_args'], cfg1['aggregator_args'], cfg1['decoder_args'])
#         print('before model sent to device=',get_freer_gpu())
        self.model1 = self.model1.float().to(device)
        self.model1.eval()
#         print('after model sent to device=',get_freer_gpu())

        # Load checkpoint
        checkpoint1 = torch.load(checkpoint_path1,map_location='cuda:0')
        self.model1.load_state_dict(checkpoint1['model_state_dict'], strict=False)
        
        cfg2=cfg1['cfg2']
        self.model2 = initialize_prediction_model(cfg2['encoder_type'], cfg2['aggregator_type'], cfg2['decoder_type'],
                                                 cfg2['encoder_args'], cfg2['aggregator_args'], cfg2['decoder_args'])
#         print('before model sent to device=',get_freer_gpu())
        self.model2 = self.model2.float().to(device)
        self.model2.eval()
#         print('after model sent to device=',get_freer_gpu())
        checkpoint_path2=cfg2['checkpoint_path2']
        # Load checkpoint
        checkpoint2 = torch.load(checkpoint_path2,map_location='cuda:0')
        self.model2.load_state_dict(checkpoint2['model_state_dict'], strict=False)

        # Initialize metrics
        self.metrics = [initialize_metric(cfg1['val_metrics'][i], cfg1['val_metric_args'][i])
                        for i in range(len(cfg1['val_metrics']))]

        self.add_img=False
        self.add_img_period=len(self.dl)//8
        self.visualize_start_epoch=cfg1['visualize_start_epoch']
        if self.add_img:
            self.vis_helper=PredictHelper_occ(self.dl.dataset.helper.data)

    def evaluate(self, output_dir: str):
        """
        Main function to evaluate trained model
        :param output_dir: Output directory to store results
        """

        # Initialize aggregate metrics
        agg_metrics = self.initialize_aggregate_metrics()

        with torch.no_grad():
            for i, data in enumerate(self.dl):

                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))
                if self.teacher_force:
                    data['inputs']['gt_traj']=data['ground_truth']['traj']
                else:
                    data['inputs']['gt_traj']= None
                # Forward pass
                predictions = self.model1(data['inputs'])
                predictions2 = self.model2(data['inputs'])
                predictions['scores']=predictions2['scores']
                if self.add_baseline:
                    baseline_score = CVM_pred(data['inputs'])
                    predictions['baseline'] = baseline_score
                if self.add_img and (i+1) % self.add_img_period==0:
                    file_name='Batch'+str(i)
                    u.visualize(data['inputs'],data['ground_truth'],predictions,self.vis_helper, os.path.join(output_dir,'imgs'), file_name, 'raw')
                    u.visualize(data['inputs'],data['ground_truth'],predictions,self.vis_helper, os.path.join(output_dir,'imgs'), file_name, 'refine')
                # Aggregate metrics
                agg_metrics = self.aggregate_metrics(agg_metrics, predictions, data['ground_truth'])

                self.print_progress(i)

        # compute and print average metrics
        self.print_progress(len(self.dl))
        with open(os.path.join(output_dir, 'results', "results.txt"), "w") as out_file:
            for metric in self.metrics:
                avg_metric = agg_metrics[metric.name]/agg_metrics['sample_count']
                output = metric.name + ': ' + format(avg_metric, '0.4f')
                print(output)
                out_file.write(output + '\n')

    def initialize_aggregate_metrics(self):
        """
        Initialize aggregate metrics for test set.
        """
        agg_metrics = {'sample_count': 0}
        for metric in self.metrics:
            agg_metrics[metric.name] = 0

        return agg_metrics

    def aggregate_metrics(self, agg_metrics: Dict, model_outputs: Dict, ground_truth: Dict):
        """
        Aggregates metrics for evaluation
        """
        minibatch_metrics = {}
        for metric in self.metrics:
            minibatch_metrics[metric.name] = metric.compute(model_outputs, ground_truth).item()
        try:
            batch_size = ground_truth['traj'].shape[0]
        except:
            batch_size = ground_truth.shape[0]
        agg_metrics['sample_count'] += batch_size

        for metric in self.metrics:
            agg_metrics[metric.name] += minibatch_metrics[metric.name] * batch_size

        return agg_metrics

    def print_progress(self, minibatch_count: int):
        """
        Prints progress bar
        """
        epoch_progress = minibatch_count / len(self.dl) * 100
        print('\rEvaluating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar, format(epoch_progress, '0.2f'), '%', end="\n" if epoch_progress == 100 else " ")

    def generate_nuscenes_benchmark_submission(self, output_dir: str):
        """
        Sets up list of Prediction objects for the nuScenes benchmark.
        """

        # NuScenes prediction helper
        helper = self.dl.dataset.helper

        # List of predictions
        preds = []

        with torch.no_grad():
            for i, data in enumerate(self.dl):

                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))

                # Forward pass
                predictions = self.model(data['inputs'])
                traj = predictions['traj']
                probs = predictions['probs']

                # Load instance and sample tokens for batch
                instance_tokens = data['inputs']['instance_token']
                sample_tokens = data['inputs']['sample_token']

                # Create prediction object and add to list of predictions
                for n in range(traj.shape[0]):

                    traj_local = traj[n].detach().cpu().numpy()
                    probs_n = probs[n].detach().cpu().numpy()
                    starting_annotation = helper.get_sample_annotation(instance_tokens[n], sample_tokens[n])
                    traj_global = np.zeros_like(traj_local)
                    for m in range(traj_local.shape[0]):
                        traj_global[m] = convert_local_coords_to_global(traj_local[m],
                                                                        starting_annotation['translation'],
                                                                        starting_annotation['rotation'])

                    preds.append(Prediction(instance=instance_tokens[n], sample=sample_tokens[n],
                                            prediction=traj_global, probabilities=probs_n).serialize())

                # Print progress bar
                self.print_progress(i)

            # Save predictions to json file
            json.dump(preds, open(os.path.join(output_dir, 'results', "evalai_submission.json"), "w"))
            self.print_progress(len(self.dl))
