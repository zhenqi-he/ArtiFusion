
import os
import torch
from typing import Optional, Any
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class ImgSet(ImageFolder):
    def __init__(
        self, 
        root, 
        transform=None, 
        transform_list=None,
    ):

        super(ImgSet, self).__init__(root=root, transform=transform)
        self.transform_list = transform_list


    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(img.copy()))
            img = torch.stack(img_transformed)      
        else:
            img = self.transform(img)
        return img, target


class NameImgSet(ImgSet):
    def __init__(
        self, 
        root, 
        transform=None, 
        transform_list=None,
    ):

        super(NameImgSet, self).__init__(root=root, transform=transform)
        self.transform_list = transform_list


    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(img.copy()))
            img = torch.stack(img_transformed)      
        else:
            img = self.transform(img)
        return img, target, path

class NameImgPredSet(Dataset):
    def __init__(
        self, 
        root, 
        transform=None, 
    ):

        super(NameImgPredSet, self).__init__()
        
        self.transform = transform
        self.loader = default_loader
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root,img) for img in imgs]
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        path = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        return img, path