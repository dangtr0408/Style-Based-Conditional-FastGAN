import os
import cv2
import numpy as np
from PIL import ImageOps

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import itertools

import torchvision.transforms as transforms
from torchsampler import ImbalancedDatasetSampler

from catalyst.data.sampler import DistributedSamplerWrapper

import config

#Dataloader

def cycle(iterable):
    for item in itertools.cycle(iterable):
        yield item

def Multi_DataLoader(input_dataset, rank, world_size, batch_size=32, shuffle=False, pin_memory=False, num_workers=0):
    sampler = ImbalancedDatasetSampler(input_dataset)
    distributed_sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank, shuffle=shuffle)
    dataloader = DataLoader(input_dataset, batch_size=batch_size, pin_memory=pin_memory, 
                            num_workers=num_workers, drop_last=True, shuffle=False, sampler=distributed_sampler)
    if config.INF_SAMPLER:
        return cycle(dataloader)
    return dataloader

class PadToSize:
    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w < self.size or h < self.size:
            padding = (
                (self.size - w) // 2 if w < self.size else 0,
                (self.size - h) // 2 if h < self.size else 0,
                (self.size - w + 1) // 2 if w < self.size else 0,
                (self.size - h + 1) // 2 if h < self.size else 0
            )
            return ImageOps.expand(img, padding, fill=self.fill)
        return img

class EnsureRGB:
    def __call__(self, img):
        # Convert image to RGB if it has an alpha channel or is grayscale
        if img.mode in ('RGBA', 'LA', 'L'):
            img = img.convert('RGB')
        return img

class DATASET(Dataset):
    def __init__(self, directory, augment=False):
        self.directory = directory
        self.class_names = os.listdir(directory)
        self.n_classes = np.arange(len(self.class_names))
        self.img_path = [os.listdir(os.path.join(directory,f"{self.class_names[i]}")) for i in range(len(self.class_names))]
        self.len_classes = [len(self.img_path[i]) for i in range(len(self.img_path))]
        self.group = np.insert(np.cumsum(np.array(self.len_classes)),0,0)-1
        self.labels = [[i]*self.len_classes[i] for i in range(len(self.len_classes))]
        self.labels_flatten = np.array([item for sublist in self.labels for item in sublist])
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                EnsureRGB(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((int(config.IMG_SIZE),int(config.IMG_SIZE))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.ToTensor()
    def __len__(self):
        return sum(self.len_classes)
    def get_class(self, idx):
        if idx==0: return 0, 0
        for i in range(len(self.group)):
            if idx <= self.group[i]:
                #(class, class_idx)
                return i-1, idx-(self.group[i-1]+1)
        return "Errors occurred in get_class()!"
    def cv2_imread(self, path):
        """imread that excepts utf-8 path"""
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    def bgr2rgb(self, image):
        return image[[2,1,0],:,:]
    def __getitem__(self, idx):
        class_num, class_idx = self.get_class(idx)
        class_path = os.path.join(self.directory, self.class_names[class_num])
        img_path = os.path.join(class_path, self.img_path[class_num][class_idx])
        img = self.cv2_imread(img_path)
        img = self.transform(img)
        img = self.bgr2rgb(img)
        return img, self.labels[class_num][class_idx]
    def get_labels(self):
        return self.labels_flatten
    def get_num_classes(self):
        return len(self.n_classes)