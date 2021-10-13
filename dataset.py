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
        
        # lgt_file = "lgt" + img_file[3:]
        # with open(file=os.path.join(self.data_path, lgt_file), mode='rb') as file:
        #     lgt = torch.from_numpy(pickle.load(file)).float()
        
        # sgt_file = "sgt" + img_file[3:]
        # with open(file=os.path.join(self.data_path, lgt_file), mode='rb') as file:
        #     sgt = torch.from_numpy(pickle.load(file)).float()
        

        
        if self.mode != "test":
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)

  
            torch.manual_seed(seed)
            gt = self.transforms(gt)

            # torch.manual_seed(seed)
            # lgt = self.transforms(lgt)

            # torch.manual_seed(seed)
            # sgt = self.transforms(sgt)

    
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 3, 1)
            # ax1.imshow(torch.squeeze(img), cmap="gray")
            # ax2 = fig.add_subplot(1, 3, 2)
            # ax2.imshow(torch.squeeze(mask), cmap="gray")
            # ax3 = fig.add_subplot(1, 3, 3)
            # ax3.imshow(torch.squeeze(gt), cmap="gray")
            # plt.show()

        return img, gt # , lgt, sgt
    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)
            
        return img_list
    def __len__(self):
        return len(self.img_file)


