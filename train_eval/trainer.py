import torch.optim
import torch.utils.data as torch_data
from typing import Dict
from train_eval.initialization import initialize_prediction_model, initialize_metric,\
    initialize_dataset, get_specific_args
import torch
import time
import math
import os
import train_eval.utils as u
from torchvision.utils import make_grid
from models.decoders.ram_decoder import get_index,get_dense
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ["MKL_NUM_THREADS"] = "6"
# os.environ["NUMEXPR_NUM_THREADS"] = "6"
# os.environ["OMP_NUM_THREADS"] = "6"

from return_device import return_device
device = return_device()


class Trainer:
    """
    Trainer class for running train-val loops
    """
    def __init__(self, cfg: Dict, data_root: str, data_dir: str, checkpoint_path=None, just_weights=False, writer=None):
        """
        Initialize trainer object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        :param just_weights: Load just weights from checkpoint
        :param writer: Tensorboard summary writer
        """

        # Initialize datasets:
        ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
        spec_args = get_specific_args(cfg['dataset'], data_root, cfg['version'] if 'version' in cfg.keys() else None)
        train_set = initialize_dataset(ds_type, ['load_data', data_dir, cfg['train_set_args']] + spec_args)
        val_set = initialize_dataset(ds_type, ['load_data', data_dir, cfg['val_set_args']] + spec_args)
        datasets = {'train': train_set, 'val': val_set}
        self.pretrain_mlp = cfg['decoder_args']['pretrain_mlp']
        # Initialize dataloaders
        self.tr_dl = torch_data.DataLoader(datasets['train'], cfg['batch_size'], shuffle=True,
                                           num_workers=cfg['num_workers'], pin_memory=True)
        self.val_dl = torch_data.DataLoader(datasets['val'], cfg['batch_size'], shuffle=False,
                                            num_workers=cfg['num_workers'], pin_memory=True)

        # Initialize model
        self.model = initialize_prediction_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                                 cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float().to(device)
        self.model.aggregator.pretrain_mlp = self.pretrain_mlp 
        self.model.decoder.pretrain_mlp = self.pretrain_mlp 

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['optim_args']['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg['optim_args']['scheduler_step'],
                                                         gamma=cfg['optim_args']['scheduler_gamma'])

        # Initialize epochs
        self.current_epoch = 0

        # Initialize losses
        self.losses = [initialize_metric(cfg['losses'][i], cfg['loss_args'][i]) for i in range(len(cfg['losses']))]
        self.loss_weights = cfg['loss_weights']
        self.teacher_force =True

        # Initialize metrics
        self.train_metrics = [initialize_metric(cfg['tr_metrics'][i], cfg['tr_metric_args'][i])
                              for i in range(len(cfg['tr_metrics']))]
        self.val_metrics = [initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
                            for i in range(len(cfg['val_metrics']))]
        self.val_metric = math.inf
        self.min_val_metric = math.inf

        # Print metrics after these many minibatches to keep track of training
        self.add_img=True
        self.log_period = len(self.tr_dl)//cfg['log_freq']
        self.add_img_freq=200
        if self.add_img:
            self.nrow=cfg['nrow']

        # Initialize tensorboard writer
        self.writer = writer
        self.tb_iters = 0

        # Load checkpoint if checkpoint path is provided
        if checkpoint_path is not None:
            print()
            print("Loading checkpoint from " + checkpoint_path + " ...", end=" ")
            self.load_checkpoint(checkpoint_path, just_weights=just_weights)
            print("Done")

        # Generate anchors if using an anchor based trajectory decoder
        if hasattr(self.model.decoder, 'anchors') and torch.as_tensor(self.model.decoder.anchors == 0).all():
            print()
            print("Extracting anchors for decoder ...", end=" ")
            self.model.decoder.generate_anchors(self.tr_dl.dataset)
            print("Done")

    def train(self, num_epochs: int, output_dir: str):
        """
        Main function to train model
        :param num_epochs: Number of epochs to run training for
        :param output_dir: Output directory to store tensorboard logs and checkpoints
        :return:
        """

        # Run training, validation for given number of epochs
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + num_epochs):

            # Set current epoch
            self.current_epoch = epoch
            print()
            print('Epoch (' + str(self.current_epoch + 1) + '/' + str(start_epoch + num_epochs) + ')')
            self.model.aggregator.teacher_force = True
            self.model.decoder.teacher_force = True
            self.teacher_force = True
            # # Train
            train_epoch_metrics = self.run_epoch('train', self.tr_dl,output_dir)
            self.print_metrics(train_epoch_metrics, self.tr_dl, mode='train')
            self.model.aggregator.teacher_force = False
            self.model.decoder.teacher_force = False
            self.teacher_force = False
            # with torch.no_grad():
            #     train_epoch_metrics = self.run_epoch('selection', self.tr_dl)
            # self.print_metrics(train_epoch_metrics, self.tr_dl, mode='train')
            # Validate
            try:
                with torch.no_grad():
                    val_epoch_metrics = self.run_epoch('val', self.val_dl)
                self.print_metrics(val_epoch_metrics, self.val_dl, mode='val')

                # Scheduler step
                self.scheduler.step()

                # Update validation metric
                self.val_metric = val_epoch_metrics[self.val_metrics[0].name] / val_epoch_metrics['minibatch_count']

                # save best checkpoint when applicable
                if self.val_metric < self.min_val_metric:
                    self.min_val_metric = self.val_metric
                    self.save_checkpoint(os.path.join(output_dir, 'checkpoints', 'best.tar'))

                # Save checkpoint
                self.save_checkpoint(os.path.join(output_dir, 'checkpoints', str(self.current_epoch) + '.tar'))
            except:
                self.save_checkpoint(os.path.join(output_dir, 'checkpoints', str(self.current_epoch) + '.tar'))
    
    
    def grad_norm(self):
        total_norm=0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm


    def run_epoch(self, mode: str, dl: torch_data.DataLoader,output_dir=None):
        """
        Runs an epoch for a given dataloader
        :param mode: 'train' or 'val'
        :param dl: dataloader object
        """
        if mode == 'val':
            self.model.eval()
        else:
            self.model.train()

        # Initialize epoch metrics
        epoch_metrics = self.initialize_metrics_for_epoch(mode)
        # if mode=='selection':
        #     file_object  = open("selection_idx.txt", "w+")
        # focal = self.losses[0]
        # ade_loss = self.losses[1]
        # Main loop
        st_time = time.time()
        for i, data in enumerate(dl):
            # time0=time.time()
            torch.cuda.empty_cache()
            # Load data
            data = u.send_to_device(u.convert_double_to_float(data))
            if self.pretrain_mlp or self.teacher_force:
                data['inputs']['gt_traj']=data['ground_truth']['traj']
            else: 
                data['inputs']['gt_traj']= None
            # Forward pass
            # time1=time.time()
            predictions = self.model(data['inputs'])
            # if i%6==0:
            #     print('     Time for forward pass =',time.time()-time1)
            def visualize(mode):
                if mode == 'endpoints':
                    shape=[predictions['mask'].shape[0],1,self.losses[0].H,self.losses[0].W]
                    endpoint_map=predictions['mask'].clone().view(shape)
                    endpoints=predictions['endpoints']
                    for batch in range(shape[0]):
                        for point in endpoints[batch]:
                            x,y=point
                            endpoint_map[batch,0,max(0,x-1):min(x+2,shape[-2]),max(0,y-1):min(shape[-1],y+2)]=0
                    self.writer.add_image(
                        "endpoints", make_grid(endpoint_map.float(), nrow=self.nrow,padding=0 ,normalize=True),self.tb_iters
                    )
                if mode == 'last_step_heatmap':
                    if self.losses[0].reduce_map:
                        nodes_2D=get_index(predictions['pred'],predictions['mask'],self.losses[0].H,self.losses[0].W)
                        sparse_rep=(predictions['pred'][:,-1]).clone().detach()
                        # normalize_factor,_=torch.max(sparse_rep,dim=-1,keepdim=True)
                        # sparse_rep/=normalize_factor
                        dense_pred=get_dense(sparse_rep.unsqueeze(1),nodes_2D,self.model.decoder.H,self.model.decoder.W).cpu()
                        self.writer.add_image(
                            "last_step_heatmap", make_grid(dense_pred, nrow=self.nrow,padding=0 ,normalize=True, scale_each=True),self.tb_iters
                        )
                    else:
                        mask_map=predictions['mask'].unsqueeze(1).clone().detach().float()
                        prob=predictions['pred'][:,-1].clone().detach().unsqueeze(1)*255
                        heatmap=torch.cat((prob,torch.zeros_like(prob),mask_map*10),dim=1)
                        origin=torch.round(self.losses[0].compensation).int()
                        heatmap[:,:,origin[0]-2:origin[0]+2,origin[1]-2:origin[1]+2]=20
                        self.writer.add_image(
                            "last_step_heatmap", make_grid(heatmap.cpu(), nrow=self.nrow,padding=0 ,normalize=True, scale_each=True),self.tb_iters
                        )
                if mode == 'traj':
                    gt_map=self.losses[0].generate_gtmap(predictions['traj'].view(predictions['traj'].shape[0],-1,predictions['traj'].shape[-1]).clone().detach(),predictions['mask'],visualize=True)
                    gt_map=torch.clamp(torch.sum(gt_map,dim=1,keepdim=True),0.0,1.0)
                    mask_map=gt_map.clone()
                    mask_map[predictions['mask'].view(gt_map.shape)]=0.5
                    gt_map+=mask_map
                    gt_map=gt_map.repeat(1,3,1,1)
                    gt_map*=127
                    self.writer.add_image(
                        "trajectory_heatmap", make_grid(gt_map.cpu(), nrow=self.nrow, normalize=True,scale_each=True),self.tb_iters
                    )
                if mode == 'all':
                    traj_idx=0
                    gt_map=self.losses[0].generate_gtmap(predictions['traj'][:,traj_idx].view(predictions['traj'].shape[0],-1,predictions['traj'].shape[-1]).clone().detach(),predictions['mask'],visualize=True)
                    gt_map=torch.clamp(torch.sum(gt_map,dim=1,keepdim=True),0.0,1.0)
                    mask_map=gt_map.clone()
                    mask_map[predictions['mask'].view(gt_map.shape)]=0.5
                    gt_map+=mask_map
                    gt_map=gt_map.repeat(1,3,1,1)
                    gt_map*=127
                    endpoints=predictions['endpoints']
                    for batch in range(gt_map.shape[0]):
                        for i,point in enumerate(endpoints[batch]):
                            if i== traj_idx:
                                x,y=point
                                gt_map[batch,:-1,max(0,x-1):min(x+2,self.losses[0].H),max(0,y-1):min(self.losses[0].W,y+2)]=0
                                gt_map[batch,-1,max(0,x-1):min(x+2,self.losses[0].H),max(0,y-1):min(self.losses[0].W,y+2)]=255
                                
                            else:
                                x,y=point
                                gt_map[batch,1:,max(0,x-1):min(x+2,self.losses[0].H),max(0,y-1):min(self.losses[0].W,y+2)]=0
                                gt_map[batch,0,max(0,x-1):min(x+2,self.losses[0].H),max(0,y-1):min(self.losses[0].W,y+2)]=255
                    self.writer.add_image(
                        "all", make_grid(gt_map.cpu(), nrow=self.nrow, normalize=True),self.tb_iters
                    )
            if self.add_img:
                with torch.no_grad():
                    if i%self.add_img_freq==0:                                            
                        if ((not self.pretrain_mlp) and (not self.teacher_force)):
                            visualize('all')
                            visualize('last_step_heatmap')
                            visualize('traj')
                            if self.losses[0].reduce_map:
                                nodes_2D=get_index(predictions['pred'],predictions['mask'],self.losses[0].H,self.losses[0].W)
                                dense_pred=get_dense(predictions['pred'].clone().detach(),nodes_2D,self.model.decoder.H,self.model.decoder.W)
                                pred=torch.sum(dense_pred,dim=1,keepdim=True)
                                self.writer.add_image(
                                    "pred", make_grid(pred.cpu(), nrow=3,padding=0 ,normalize=True),self.tb_iters
                                )
                            else:
                                pred=torch.sum(predictions['pred'],dim=1,keepdim=True)
                                self.writer.add_image(
                                    "pred", make_grid(pred.cpu(), nrow=self.nrow,padding=0 ,normalize=True,scale_each=True),self.tb_iters
                                )
                            gt_map=self.losses[0].generate_gtmap(data['ground_truth']['traj'],predictions['mask'],visualize=True)
                            gt_map=torch.sum(gt_map,dim=1,keepdim=True)
                            self.writer.add_image(
                                "gt_heatmap", make_grid(gt_map.cpu(), nrow=self.nrow, normalize=True,scale_each=True),self.tb_iters
                            )
                        else:
                            visualize('last_step_heatmap')
                            shape=[predictions['mask'].shape[0],1,self.model.decoder.H,self.model.decoder.W]
                            swapped=torch.zeros_like(data['ground_truth']['traj'])
                            swapped[:,:,0],swapped[:,:,1]=-data['ground_truth']['traj'][:,:,1],data['ground_truth']['traj'][:,:,0]
                            coord=torch.round(swapped/self.model.decoder.resolution+self.model.decoder.compensation.to(swapped.device)).int()
                            coord=torch.clamp(coord,0,shape[-1])
                            gt_map=torch.zeros(shape)
                            mask_map=gt_map.clone()
                            mask_map[predictions['mask'].view(gt_map.shape)]=0.5
                            gt_map+=mask_map
                            gt_map=gt_map.repeat(1,3,1,1)
                            gt_map*=127
                            for batch in range(shape[0]):
                                for t in range(coord.shape[1]):
                                    x,y=coord[batch,t]
                                    gt_map[batch,0,x-2:x+3,y-2:y+3]=0##Only one ground truth in each heatmap layer
                            # gt_map=torch.sum(gt_map,dim=1,keepdim=True)
                            self.writer.add_image(
                                "gt_traj_heatmap", make_grid(gt_map.cpu(), nrow=self.nrow, normalize=True, scale_each=True),self.tb_iters
                            )
                            traj=predictions['traj'].squeeze(1).clone().detach()
                            shape=[predictions['mask'].shape[0],1,self.model.decoder.H,self.model.decoder.W]
                            swapped=torch.zeros_like(data['ground_truth']['traj'])
                            swapped[:,:,0],swapped[:,:,1]=-traj[:,:,1],traj[:,:,0]
                            coord=torch.round(swapped/self.model.decoder.resolution+self.model.decoder.compensation.to(swapped.device)).int()
                            coord=torch.clamp(coord,0,shape[-1])
                            gt_map=torch.zeros(shape)
                            mask_map=gt_map.clone()
                            mask_map[predictions['mask'].view(gt_map.shape)]=0.5
                            gt_map+=mask_map
                            gt_map=gt_map.repeat(1,3,1,1)
                            gt_map*=127
                            for batch in range(shape[0]):
                                for t in range(coord.shape[1]):
                                    x,y=coord[batch,t]
                                    gt_map[batch,1,x-2:x+3,y-2:y+3]=0##Only one ground truth in each heatmap layer
                            # gt_map=torch.sum(gt_map,dim=1,keepdim=True)
                            self.writer.add_image(
                                "traj_heatmap", make_grid(gt_map.cpu(), nrow=self.nrow, normalize=True, scale_each=True),self.tb_iters
                            )
                            
                            # endpoint_map=torch.zeros(shape)
                            # endpoints=predictions['endpoints']
                            # for batch in range(shape[0]):
                            #     for point in endpoints[batch]:
                            #         x,y=point
                            #         endpoint_map[batch,0,max(0,x-2):min(x+3,shape[-2]),max(0,y-2):min(shape[-1],y+3)]=1
                            # self.writer.add_image(
                            #     "endpoints", make_grid(endpoint_map, nrow=4,padding=0 ,normalize=True),self.tb_iters
                            # )
                        torch.cuda.empty_cache()
            # if mode== 'selection':
            #     loss0=focal.compute(predictions,data['ground_truth']) 
            #     loss1=ade_loss.compute(predictions,data['ground_truth']) 
            #     if loss0>65.0 or loss1>2.0:
            #         visualize('all')
            #         visualize('last_step_heatmap')
            #         visualize('traj')
            #         gt_map=self.losses[0].generate_gtmap(data['ground_truth']['traj'],predictions['mask'],visualize=True)
            #         gt_map=torch.sum(gt_map,dim=1,keepdim=True)
            #         self.writer.add_image(
            #             "gt_heatmap", make_grid(gt_map.cpu(), nrow=self.nrow, normalize=True,scale_each=True),self.tb_iters
            #         )
            #         minibatch_time = time.time() - st_time
            #         st_time = time.time()

            #         # Aggregate metrics
            #         minibatch_metrics, epoch_metrics = self.aggregate_metrics(epoch_metrics, minibatch_time,
            #                                                                 predictions, data['ground_truth'], mode)


                # if loss0>65.0:
                #     file_object.write("focal %d\r\n" % (i))
                #     print('focal',loss0.item())
                # if loss1>2.0:
                #     file_object.write("ade %d\r\n" % (i))
                #     print('ade',loss1.item())
                # self.log_tensorboard_train(minibatch_metrics)
                # continue

                


            # Compute loss and backprop if training
            if mode == 'train':
                loss = self.compute_loss(predictions, data['ground_truth'])
                # print('Iteration',i,' loss: ',loss.item())
                # print( '    Weight norm of conv1d',torch.norm(self.model.encoder.target_agent_encoder.conv1d.conv1.weight).item())
                if torch.isnan(loss):
                    with torch.no_grad():
                        # visualize('all')
                        visualize('last_step_heatmap')
                        # visualize('traj')
                        # nodes_2D=get_index(predictions['pred'],predictions['mask'],self.losses[0].H,self.losses[0].W)
                        # dense_pred=get_dense(predictions['pred'].clone().detach(),nodes_2D,self.model.decoder.H,self.model.decoder.W)
                        pred=torch.sum(predictions['pred'],dim=1,keepdim=True)
                        self.writer.add_image(
                            "pred", make_grid(pred.cpu(), nrow=2,padding=0 ,normalize=True),self.tb_iters
                        )
                        # gt_map=self.losses[0].generate_gtmap(data['ground_truth']['traj'],predictions['mask'],visualize=True)
                        # gt_map=torch.sum(gt_map,dim=1,keepdim=True)
                        # self.writer.add_image(
                        #     "gt_heatmap", make_grid(gt_map.cpu(), nrow=2, normalize=True),self.tb_iters
                        # )
                        sum=predictions['pred'].flatten(1).sum(dim=-1)
                        self.save_checkpoint(os.path.join(output_dir, 'checkpoints', str(self.current_epoch) + '.tar'))
                        # corrupt_idx=torch.argmin(sum).item()
                        # corrupt_map=predictions['mask'][corrupt_idx].view(self.model.decoder.H,self.model.decoder.W)
                        print(sum)
                        # print(corrupt_map[self.model.decoder.compensation[0]-1:self.model.decoder.compensation[0]+2,self.model.decoder.compensation[1]-1:self.model.decoder.compensation[1]+2])
                        for loss in self.losses:
                            print(loss.name,':')
                            print(loss.compute(predictions, data['ground_truth']).item())
                    raise NotImplementedError()
                # print('################ Memory usage after compute loss ####################')
                # print(torch.cuda.memory_summary(device=device, abbreviated=False))
                # time2=time.time()
                self.back_prop(loss)
                # print("     Grad norm:",self.grad_norm(),'  \n')
                # if i%6==0:
                #     print('     Time for backward pass =',time.time()-time2)
            
            

            # Keep time
            minibatch_time = time.time() - st_time
            st_time = time.time()

            # Aggregate metrics
            minibatch_metrics, epoch_metrics = self.aggregate_metrics(epoch_metrics, minibatch_time,
                                                                      predictions, data['ground_truth'], mode)

            # Log minibatch metrics to tensorboard during training
            if mode == 'train':
                self.log_tensorboard_train(minibatch_metrics)

            # Display metrics at a predefined frequency
            if i % self.log_period == self.log_period - 1:
                self.print_metrics(epoch_metrics, dl, mode)
            # if i%6==0:
            #     print('Time for one cycle =',time.time()-time0)

        # Log val metrics for the complete epoch to tensorboard
        if mode == 'val':
            self.log_tensorboard_val(epoch_metrics)

        return epoch_metrics

    def compute_loss(self, model_outputs: Dict, ground_truth: Dict) -> torch.Tensor:
        """
        Computes loss given model outputs and ground truth labels
        """
        loss_vals = [loss.compute(model_outputs, ground_truth) for loss in self.losses]
        total_loss = torch.as_tensor(0, device=device).float()
        for n in range(len(loss_vals)):
            total_loss += self.loss_weights[n] * loss_vals[n]

        return total_loss

    def back_prop(self, loss: torch.Tensor, grad_clip_thresh=10):
        """
        Backpropagate loss
        """
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        # print('################ Memory usage before back propagation ####################')
        # print(torch.cuda.memory_summary(device=device, abbreviated=False))
        try:
            loss.backward()
            # print('################ Memory usage after back propagation ####################')
            # print(torch.cuda.memory_summary(device=device, abbreviated=False))
            torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
            self.optimizer.step()
        except: 
            print('Error happens in back propagation! Probably memory not enough!')

    def initialize_metrics_for_epoch(self, mode: str):
        """
        Initialize metrics for epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        epoch_metrics = {'minibatch_count': 0, 'time_elapsed': 0}
        for metric in metrics:
            epoch_metrics[metric.name] = 0

        return epoch_metrics

    def aggregate_metrics(self, epoch_metrics: Dict, minibatch_time: float, model_outputs: Dict, ground_truth: Dict,
                          mode: str):
        """
        Aggregates metrics by minibatch for the entire epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics

        minibatch_metrics = {}
        for metric in metrics:
            minibatch_metrics[metric.name] = metric.compute(model_outputs, ground_truth).item()

        epoch_metrics['minibatch_count'] += 1
        epoch_metrics['time_elapsed'] += minibatch_time
        for metric in metrics:
            epoch_metrics[metric.name] += minibatch_metrics[metric.name]

        return minibatch_metrics, epoch_metrics

    def print_metrics(self, epoch_metrics: Dict, dl: torch_data.DataLoader, mode: str):
        """
        Prints aggregated metrics
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        minibatches_left = len(dl) - epoch_metrics['minibatch_count']
        eta = (epoch_metrics['time_elapsed']/epoch_metrics['minibatch_count']) * minibatches_left
        epoch_progress = int(epoch_metrics['minibatch_count']/len(dl) * 100)
        print('\rTraining:' if mode == 'train' else '\rValidating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar, str(epoch_progress), '%', end=", ")
        print('ETA:', int(eta), end="s, ")
        print('Metrics', end=": { ")
        for metric in metrics:
            metric_val = epoch_metrics[metric.name]/epoch_metrics['minibatch_count']
            print(metric.name + ':', format(metric_val, '0.2f'), end=", ")
        print('\b\b }', end="\n" if eta == 0 else "")

    def load_checkpoint(self, checkpoint_path, just_weights=False):
        """
        Loads checkpoint from given path
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path,map_location='cuda:0')
        else:
            checkpoint = torch.load(checkpoint_path,map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not just_weights:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.val_metric = checkpoint['val_metric']
            self.min_val_metric = checkpoint['min_val_metric']

    def save_checkpoint(self, checkpoint_path):
        """
        Saves checkpoint to given path
        """
        torch.save({
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metric': self.val_metric,
            'min_val_metric': self.min_val_metric
        }, checkpoint_path)

    def log_tensorboard_train(self, minibatch_metrics: Dict):
        """
        Logs minibatch metrics during training
        """
        for metric_name, metric_val in minibatch_metrics.items():
            self.writer.add_scalar('train/' + metric_name, metric_val, self.tb_iters)
        self.tb_iters += 1

    def log_tensorboard_val(self, epoch_metrics):
        """
        Logs epoch metrics for validation set
        """
        for metric_name, metric_val in epoch_metrics.items():
            if metric_name != 'minibatch_count' and metric_name != 'time_elapsed':
                metric_val /= epoch_metrics['minibatch_count']
                self.writer.add_scalar('val/' + metric_name, metric_val, self.tb_iters)

        self.tb_iters += 1
