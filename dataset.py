import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.transforms import (Compose, RandomHorizontalFlip,
                                    RandomVerticalFlip)

from utils.helpers import Fix_RandomRotation
class myDataset(Dataset):
    def __init__(self, path, name, mode):
        self.mode = mode
        self.data_path = os.path.join(path, name, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_file)
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        # gt = self.gt_file[idx]
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()
        
        if self.mode != "test":
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            
            torch.manual_seed(seed)
            gt = self.transforms(gt)


        return img, gt 
    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)
            
        return img_list
    def __len__(self):
        return len(self.img_file)


