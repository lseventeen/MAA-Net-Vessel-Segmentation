import numpy as np
import torch
from sklearn.metrics import (f1_score, jaccard_score, precision_recall_curve,
                             roc_auc_score)

import cv2
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def get_metrics(predict, target,threshold=0.5):
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    predict_b = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else: 
        target = target.flatten()

    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    auc = roc_auc_score(target, predict)
    # precision, recall, thresholds = precision_recall_curve(target, output)
    # precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    # recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    # pra = np.trapz(precision, recall)
    # jc = jaccard_score(target, output_b)

    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
        # "PRA": np.round(pra, 4),
        # "jc ": np.round(jc, 4),
    }


def get_metrics_seed(predict, predict_b, target):
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    predict_b = predict_b.flatten()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else: 
        target = target.flatten()

    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    auc = roc_auc_score(target, predict)
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
        # "PRA": np.round(pra, 4),
        # "jc ": np.round(jc, 4),
    }

def count_connect_component(predict, target,threshold=None,connectivity=8):
    gt_num,pre_num = 0,0
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)

    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    for i in range(len(predict)):
        pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(predict[i,0,:,:], dtype=np.uint8)*255, connectivity=connectivity)
        gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(target[i,0,:,:], dtype=np.uint8)*255, connectivity=connectivity)
        pre_num += pre_n
        gt_num += gt_n
    return pre_num,gt_num