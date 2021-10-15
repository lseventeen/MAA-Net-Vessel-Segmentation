import os
import pickle
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
    
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_pickle(path, type):
    with open(file=path + f"/{type}.pkl", mode='rb') as file:
        img = pickle.load(file)
    return img


def save_pickle(path, type, img_list):
    with open(file=path + f"/{type}.pkl", mode='wb') as file:
        pickle.dump(img_list, file)





def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))




def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape) == 4)  # 4D arrays
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)  # check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    # print("N_patches_h: " + str(N_patches_h))
    # print("N_patches_w: " + str(N_patches_w))
    # print("N_patches_img: " + str(N_patches_img))
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    # print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(
    #     img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros(
        (N_full_imgs, preds.shape[1], img_h, img_w))  # itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[
                    k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    # print(final_avg.shape)
    # assert (np.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
    # assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
    return final_avg 