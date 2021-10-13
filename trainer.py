import json
import math
import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
import wandb
from utils.helpers import dir_exists, get_instance, seedgrow,remove_files,read_pickle,recompone_overlap
from utils.metrics import AverageMeter, get_metrics, get_metrics_seed, count_connect_component


class Trainer:
    def __init__(self, mode, model, resume=None, CFG=None, loss=None,
                 train_loader=None,
                 val_loader=None,
                 checkpoint=None,
                 test_loader=None,
                 save_path=None,show = False):
        self.CFG = CFG
        self.show = show
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss = loss
        self.model = nn.DataParallel(model.cuda())

        self.stride = self.CFG["data_process"]["test"]["stride"]
        self.patch_size = self.CFG["data_process"]["test"]["patch_size"]
        test_data_path = os.path.join(self.CFG["data_set"]["path"],self.CFG["data_set"]["name"], "test_pro")
        self.full_imgs = read_pickle(test_data_path,"full_img")
        self.full_gts = read_pickle(test_data_path,"full_gt")
        wandb.watch(self.model)
        cudnn.benchmark = True
        # train and val
        if mode == "train":
            # OPTIMIZER
            self.optimizer = get_instance(
                torch.optim, "optimizer", CFG, self.model.parameters())
            self.lr_scheduler = get_instance(
                torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)

            
            # MONITORING
            self.improved = True
            self.not_improved_count = 0
            self.mnt_best = -math.inf if self.CFG.mnt_mode == 'max' else math.inf

            # CHECKPOINTS & TENSOBOARD
            start_time = datetime.now().strftime('%y%m%d%H%M%S')
            self.checkpoint_dir = os.path.join(
                CFG.save_dir, self.CFG['model']['type'], start_time)
            self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
            dir_exists(self.checkpoint_dir)
           
            # config_save_path = os.path.join(self.checkpoint_dir, 'config.yaml')
            self.train_logger_save_path = os.path.join(
                self.checkpoint_dir, 'runtime.log')
            logger.add(self.train_logger_save_path)
            logger.info(self.checkpoint_dir)
            # with open(config_save_path, 'w') as handle:
            #     json.dump(self.config, handle, indent=4, sort_keys=True)
            if resume:
                self._resume_checkpoint(resume)

        # test
        if mode == "test":
            self.model.load_state_dict(checkpoint['state_dict'])
            self.checkpoint_dir = save_path
          

    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            # RUN TRAIN (AND VAL)
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:
                results = self._valid_epoch(epoch)
                # LOGGING INFO
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
                    # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
                if self.mnt_mode != 'off' and epoch >= 10:
                    try:
                        if self.mnt_mode == 'min':
                            self.improved = (
                                results[self.CFG.mnt_metric] <= self.mnt_best )
                        else:
                            self.improved = (
                                results[self.CFG.mnt_metric] >= self.mnt_best )
                    except KeyError:
                        logger.warning(
                            f'The metrics being tracked ({self.CFG.mnt_metric}) has not been calculated. Training stops.')
                        break

                    if self.improved:
                        self.mnt_best = results[self.CFG.mnt_metric]
                        self.not_improved_count = 0

                    else:
                        self.not_improved_count += 1

                    if self.not_improved_count >= self.CFG.early_stopping:
                        logger.info(
                            f'\nPerformance didn\'t improve for {self.CFG.early_stop} epochs')
                        logger.warning('Training Stoped')
                        break
            if self.test_loader is not None and epoch % self.CFG.test_per_epochs == 0:
                self.test(log="tensorboard", epoch=epoch, metrics_type="FULL")
            # SAVE CHECKPOINT
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

        return self.checkpoint_dir

    def _train_epoch(self, epoch):

        self.model.train()
        wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            

            # LOSS & OPTIMIZE

            self.optimizer.zero_grad()
            if self.CFG.amp is True:
                with torch.cuda.amp.autocast(enabled=True):
                    pre = self.model(img)
                    loss = self.loss(pre, gt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre = self.model(img)
                loss = self.loss(pre, gt)
                loss.backward()
                self.optimizer.step()
            self.total_loss.update(loss.item())
            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()
            # FOR EVAL and INFO
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *
                    self._metrics_ave().values(), self.batch_time.average,
                    self.data_time.average))
        # METRICS TO TENSORBOARD
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)
        self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for img, gt, mask in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                # LOSS
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        predict = self.model(img)
                        loss = self.loss(predict, gt)
                else:
                    predict = self.model(img)
                    loss = self.loss(predict, gt)
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)

        # LOGGING & TENSORBOARD
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        # self.writer.add_image(
        #     f'{wrt_mode}/inputs_targets_predictions', val_img, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def test(self, epoch=1, log="wandb", metrics_type="FULL"):

        logger.info(f"###### TEST {metrics_type} EVALUATION ######")
        wrt_mode = 'test'
        self.model.eval()
        tbar = tqdm(self.test_loader, ncols=50)
        pres = []
  
       
        tic1 = time.time()
        total_loss = AverageMeter()
        self._reset_metrics()
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        pre = self.model(img)
                        loss = self.loss(pre, gt)
                      
                else:
                    pre = self.model(img)
                    loss = self.loss(pre, gt)
                pres.extend(pre)
             
                total_loss.update(loss.item())
        tic2 = time.time()
        test_time = tic2 - tic1
        logger.info(f'test time:  {test_time}')

        pres = torch.stack(pres, 0).cpu()

        H,W = self.full_imgs[0][0].shape
        pad_h = self.stride - (H - self.patch_size) % self.stride
        pad_w = self.stride - (W - self.patch_size) % self.stride
        new_h = H + pad_h
        new_w = W + pad_w

        pres = recompone_overlap(pres.numpy(), new_h, new_w, self.stride, self.stride)  # predictions
        pres = TF.crop(torch.from_numpy(pres), 0, 0, H, W)

        if self.show == True:
            dir_exists("save_picture")
            remove_files("save_picture")
            n,_,_,_= self.full_imgs.shape
            for i in range(n):
                predict = torch.sigmoid(pres[i]).detach().numpy()
                predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                cv2.imwrite(f"save_picture/img{i}.png", np.uint8(self.full_imgs[i][0]*255))
                cv2.imwrite(f"save_picture/gt{i}.png", np.uint8(self.full_gts[i][0]*255))
                cv2.imwrite(f"save_picture/pre{i}.png", np.uint8(predict[0]*255))
                cv2.imwrite(f"save_picture/pre_b{i}.png", np.uint8(predict_b[0]*255))

        if self.CFG.DTI is True:
            pre_seed = seedgrow(pres, self.CFG.threshold, self.CFG.threshold_low, True)
            metrics = get_metrics_seed(pres, pre_seed, self.full_gts)
            pre_num, gt_num = count_connect_component(pre_seed, self.full_gts)
        else:
            metrics = get_metrics(pres, self.full_gts, threshold=self.CFG.threshold)
            pre_num, gt_num = count_connect_component(pres, self.full_gts, threshold=self.CFG.threshold)

        # LOGGING & TENSORBOARD
        tic3 = time.time()
        metrics_time = tic3 - tic1
        logger.info(f'metrics time:  {metrics_time}')
        if log == "wandb":
            wandb.log({f'{wrt_mode}_{metrics_type}/loss': total_loss.average})
            for k, v in list(metrics.items())[:-1]:
                wandb.log({f'{wrt_mode}_{metrics_type}/{k}': v})
        else:
            self.writer.add_scalar(
                f'{wrt_mode}_{metrics_type}/loss',  total_loss.average, epoch)
            for k, v in list(metrics.items())[:-1]:
                self.writer.add_scalar(
                    f'{wrt_mode}_{metrics_type}/{k}', v, epoch)
        logger.info(f'         loss: {total_loss.average}')
        for k, v in metrics.items():
            logger.info(f'         {str(k):15s}: {v}')
        logger.info(f'         pre_num: {pre_num}')
        logger.info(f'         gt_num: {gt_num}')

    def _save_checkpoint(self, epoch, save_best=True):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            logger.info("Saving current best: best_model.pth")
        return filename

    def _resume_checkpoint(self, resume_path):
        logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['mode']['type'] != self.config['mode']['type']:
            logger.warning(
                {'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            logger.warning(
                {'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # if self.lr_scheduler:
        #     self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.train_logger = checkpoint['logger']
        logger.info(
            f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }
