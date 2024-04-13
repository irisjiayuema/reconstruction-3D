import os
import argparse
from unittest import runner
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger

from estimator.utils import RunnerInfo, log_env, fix_random_seed
from estimator.models.builder import build_model
from estimator.datasets.builder import build_dataset
from estimator.trainer import Trainer  # Assuming this is the single-GPU Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--log_name', type=str, default='', help='log_name for wandb')
    parser.add_argument('--tags', type=str, default='', help='tags for wandb')
    parser.add_argument('--seed', type=int, default=621, help='for debug')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir or cfg.work_dir
    mkdir_or_exist(cfg.work_dir)
    cfg.debug = args.debug

    # fix seed
    fix_random_seed(args.seed)

    logger_name = args.log_name if args.log_name else 'default_logger'

    log_file = os.path.join(cfg.work_dir, 'train.log')
    logger = MMLogger.get_instance(log_level='INFO', log_file=log_file, name=logger_name)

    runner_info = RunnerInfo()
    runner_info.config = cfg
    runner_info.logger = logger
    runner_info.work_dir = cfg.work_dir
    runner_info.seed = args.seed
    runner_info.launcher = 'torch'
    runner_info.distributed = False

    if not cfg.debug:
        wandb.init(
            project=cfg.project,
            name=args.log_name,
            tags=args.tags.split(',') if args.tags else [],
            dir=cfg.work_dir,
            config=cfg,
            settings=wandb.Settings(start_method="spawn"))

    env_cfg = cfg.get('env_cfg')
    if env_cfg is None:
        env_cfg = {}

    log_env(cfg, env_cfg, runner_info, logger)

    model = build_model(cfg.model).cuda()

    # build dataloaders
    train_dataset = build_dataset(cfg.train_dataloader.dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.train_dataloader.num_workers,
        pin_memory=True)

    val_dataset = build_dataset(cfg.val_dataloader.dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=True)

    # save config
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))

    # initialize trainer and start training
    trainer = Trainer(
        config=cfg,
        train_sampler=None,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model)

    trainer.run()
    wandb.finish()

if __name__ == '__main__':
    main()
