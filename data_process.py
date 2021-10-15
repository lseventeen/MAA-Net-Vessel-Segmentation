import argparse
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import safe_load
from scipy import ndimage, stats
from skimage import color, measure
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists,remove_files,removeConnectedComponents

def data_process(path, name, patch_size, stride, mode):
    data_path = os.path.join(path, name)
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)

    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")

        file_list = list(sorted(os.listdir(img_path)))
    
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
  
    img_list = []
    gt_list = []

    for file in file_list:
        if name == "DRIVE":

            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

        elif name == "DCA1":
            if len(file) <= 7:

                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))

                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))

       



    mean, std = getMeanStd(img_list)
    img_list = normalization(img_list, mean, std)
    

    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)



    save_patch(img_patch, save_path, "img_patch", name)
    save_patch(gt_patch, save_path, "gt_patch", name)
    if mode == "test":
        save_image(img_list, save_path, "full_img", name)
        save_image(gt_list, save_path, "full_gt", name)


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list


def save_patch(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')

def save_image(imgs_list, path, type, name):
    imgs = torch.stack(imgs_list, 0).numpy()
    with open(file=os.path.join(path, f'{type}.pkl'), mode='wb') as file:
        pickle.dump(imgs, file)
        print(f'save {name} {type} : {type}.pkl')



def getMeanStd(imgs_list):

    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    return mean, std


def normalization(imgs_list, mean, std):
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()


    with open('config.yaml', encoding='utf-8') as file:
        config = safe_load(file)  # 为列表类型

    data_process(config["data_set"]["path"], name="DRIVE",
                 mode="training", **config["data_process"]["training"])
    data_process(config["data_set"]["path"], name="DRIVE",
                 mode="test", **config["data_process"]["test"])

    data_process(config["data_set"]["path"], name="DCA1",
                 mode="training", **config["data_process"]["training"])

    data_process(config["data_set"]["path"], name="DCA1",
                 mode="test", **config["data_process"]["test"])
  