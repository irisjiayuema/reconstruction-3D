import os
import wandb
import numpy as np
import torch
import mmengine
from mmengine.optim import build_optim_wrapper
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.distributed as dist
from mmengine.dist import get_dist_info, collect_results_cpu, collect_results_gpu
from mmengine import print_log
import torch.nn.functional as F
from tqdm import tqdm
from estimator.utils import colorize

class Trainer:
    def __init__(self, config, train_sampler, train_dataloader, val_dataloader, model):
        self.config = config
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model

        # Build optimizer and scheduler
        self.optimizer_wrapper = build_optim_wrapper(self.model, config.optim_wrapper)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer_wrapper.optimizer, 
            [l['lr'] for l in self.optimizer_wrapper.optimizer.param_groups], 
            epochs=config.train_cfg.max_epochs, 
            steps_per_epoch=len(train_dataloader),
            cycle_momentum=config.param_scheduler.cycle_momentum, 
            base_momentum=config.param_scheduler.get('base_momentum', 0.85), 
            max_momentum=config.param_scheduler.get('max_momentum', 0.95),
            div_factor=config.param_scheduler.div_factor, 
            final_div_factor=config.param_scheduler.final_div_factor, 
            pct_start=config.param_scheduler.pct_start, 
            three_phase=config.param_scheduler.three_phase
        )

        self.train_step = 0
        self.val_step = 0
        self.iters_per_train_epoch = len(train_dataloader)
        self.iters_per_val_epoch = len(val_dataloader)
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def log_images(self, log_dict, prefix="", scalar_cmap="turbo_r", min_depth=1e-3, max_depth=80, step=0):
        # Custom log images. Please add more items to the log dict returned from the model
        ...

    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor) and k in self.config.collect_input_args:
                collect_batch_data[k] = v.cuda()
        return collect_batch_data

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        dataset = self.val_dataloader.dataset

        for idx, batch_data in enumerate(self.val_dataloader):
            self.val_step += 1

            batch_data_collect = self.collect_input(batch_data)
            result, log_dict = self.model(mode='infer', **batch_data_collect)
            
            # Process result and metrics
            ...

            # Log images
            if idx % self.config.train_cfg.val_log_img_interval == 0:
                self.log_images(log_dict=log_dict, prefix="Val", step=self.val_step)

        self.model.train()

    def train_epoch(self, epoch_idx):
        self.model.train()

        for idx, batch_data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            self.train_step += 1
            batch_data_collect = self.collect_input(batch_data)
            loss_dict, log_dict = self.model(mode='train', **batch_data_collect)

            total_loss = loss_dict['total_loss']
            self.optimizer_wrapper.update_params(total_loss)
            self.scheduler.step()

            # Logging and image visualization
            ...

            # Validation check
            if self.train_step % self.config.train_cfg.val_interval == 0:
                self.val_epoch()

    def save_checkpoint(self, epoch_idx):
        model_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        checkpoint_dict = {
            'epoch': epoch_idx,
            'model_state_dict': model_dict,
            'optim_state_dict': self.optimizer_wrapper.state_dict(),
            'schedule_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint_dict, os.path.join(self.config.runner_info.work_dir, f'checkpoint_{epoch_idx:02d}.pth'))

    def run(self):
        for epoch_idx in range(self.config.train_cfg.max_epochs):
            self.train_epoch(epoch_idx)
            if epoch_idx % self.config.train_cfg.val_interval == 0:
                self.val_epoch()
            if epoch_idx % self.config.train_cfg.save_checkpoint_interval == 0:
                self.save_checkpoint(epoch_idx)
