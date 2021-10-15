import argparse
import os

import torch

from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader

import models
import wandb
from dataset import myDataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch
def train(CFG):
    seed_torch(42)
    train_dataset = myDataset(**CFG["data_set"], mode="training")
    train_loader = DataLoader(
        train_dataset, CFG.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    test_dataset = myDataset(**CFG["data_set"], mode="test")
    test_loader = DataLoader(test_dataset, CFG.batch_size, shuffle=False,  num_workers=16, pin_memory=True)

    logger.info('The patch number of train is %d' % len(train_dataset))

    model = get_instance(models, 'model', CFG)

    logger.info(f'\n{model}\n')
    # LOSS
    loss = get_instance(losses, 'loss', CFG)
    # TRAINING
    trainer = Trainer(
        model=model,
        mode="train",
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        test_loader=test_loader
    )

    return trainer.train()


def test(save_path, checkpoint_name, CFG, show):
    checkpoint = torch.load(os.path.join(save_path, checkpoint_name))
    CFG_ck = checkpoint['config']
    # DATA LOADERS
    test_dataset = myDataset(**CFG["data_set"], mode="test")
    test_loader = DataLoader(test_dataset, CFG.batch_size,
                             shuffle=False,  num_workers=16, pin_memory=True)
    # MODEL
    model = get_instance(models, 'model', CFG_ck)

    # LOSS
    loss = get_instance(losses, 'loss', CFG_ck)
    
    # TEST
    tester = Trainer(model=model, mode="test", loss=loss, CFG=CFG, checkpoint=checkpoint,
                     test_loader=test_loader, save_path=save_path,show=show)
    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.yaml', type=str,
                        help='Path to the config file (default: config.yaml)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--name', default=None, type=str,
                        help='the wandb name of run')
    args = parser.parse_args()

    # yaml = YAML(typ='safe')
    with open('config.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))  # 为列表类型
    os.environ["CUDA_VISIBLE_DEVICES"] = CFG["CUDA_VISIBLE_DEVICES"]
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
  
   
    wandb.init(project="MAA_Net_vessel-segmentation", config=CFG,
               sync_tensorboard=True, name=f"{CFG['data_set']['name']} {CFG['model']['type']} {CFG['loss']['type']} {args.name}")
    path = train(CFG)
    checkpoint_name = "best_model.pth"
    test(path, checkpoint_name, CFG,show = True)
