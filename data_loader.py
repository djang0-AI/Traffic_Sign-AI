import os
import random

import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transform

from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from fastprogress.fastprogress import master_bar

from utils import get_mirrors, get_symmetric


def split_data(root_dir, n_folds = 4):
    trainids = {}
    valids = {}
    mb = master_bar(range(len(os.listdir(root_dir))))
    for subfolder in os.listdir(root_dir):
        subfolderpath = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolderpath):
            unique_schilder = {int(f.split("_")[1]) for f in os.listdir(subfolderpath) if f.endswith(".png")}
            unique_schilder = np.array([f for f in unique_schilder])
            np.random.seed(42)
            np.random.shuffle(unique_schilder)
            folds = np.array_split(unique_schilder, n_folds)
            fold = folds.pop(0)
            valids[subfolder] = [os.path.join(subfolderpath, f) for f in os.listdir(subfolderpath) if  f.endswith(".png") and int(f.split("_")[1]) in fold]
            trainids[subfolder]= [os.path.join(subfolderpath, f) for f in os.listdir(subfolderpath) if  f.endswith(".png") and int(f.split("_")[1]) in np.concatenate(folds)]

    return trainids, valids


def get_full_traindata(root_dir):
    trainids = {}
    mb = master_bar(range(len(os.listdir(root_dir))))
    for subfolder in os.listdir(root_dir):
        subfolderpath = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolderpath):
            trainids[subfolder]= [os.path.join(subfolderpath, f) for f in os.listdir(subfolderpath) if  f.endswith(".png")]
    return trainids


def get_full_testdata(root_dir):
    testids = {}
    gt = pd.read_csv(os.path.join("./input/traffic-sign/Test.csv"))
    for index, row in gt.iterrows():
        path = os.path.join("./input/traffic-sign/", row["Path"])
        if os.path.isfile(path):
            klasse = row["ClassId"]
            try:
                testids[klasse] += [path]
            except:
                testids[klasse] = [path]
        else:
            assert False
                
    return testids


class GTSRB_Dataset(Dataset):
    def __init__(self, ids, use_mirrored, rebalance, mirrored_classes, symmetric, augmentation, augmentation_size , size = [64, 64]):
        self.data = {key : [] for key in ids}
        augmented= {key : [] for key in ids}
        hflipper =  transform.RandomHorizontalFlip(p=0.5)
        for key,paths in ids.items():
            for path in paths:
                image = torch.tensor(cv2.imread(path),dtype = torch.float)/255
                image = image.transpose(0,2)
                image = image.transpose(1,2)
                image = image.flip(0)
                image = resize(image, size)
                self.data[key].append(image.unsqueeze(0))
                
        print("Done")
        if augmentation_size > 0 :
            print("--> Continuing with Augmentation")
            for key, images in self.data.items():
                missing = len(images)*augmentation_size
                augmentation_pool = images.copy()
                if mirrored_classes[str(key).zfill(2)]:
                    augmentation_pool += [image.flip(2) for image in self.data[mirrored_classes[str(key).zfill(2)]]]
                used_for_augmentation = []
                while missing > len(augmentation_pool):
                    used_for_augmentation += augmentation_pool
                    missing -= len(augmentation_pool)
                random.seed(42)
                used_for_augmentation += random.sample(augmentation_pool, missing)
                for image in used_for_augmentation:
                    if symmetric[str(key).zfill(2)]:
                        image = hflipper(image)
                    width = image.shape[2]
                    height = image.shape[3]
                    image = resize(image,(width*2, height*2))
                    image = augmentation(image)
                    image = transform.RandomCrop((int(width*1.9), int(height*1.9)))(image)
                    image  = resize(image, (width,height))
                    augmented[key].append(image)
                        
            for key,images in augmented.items():
                self.data[key] += images
        self.classes = torch.tensor([int(key) for key,images in self.data.items() for i in images], dtype = torch.uint8)
        self.data = torch.cat([torch.cat(images) for key, images in self.data.items()])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] , self.classes[idx]


def get_transforms():
    augmentation = transform.Compose([
    transform.ColorJitter(0.5,0.1,0.2,0),
    transform.RandomRotation(degrees=(-10, 10)),
    transform.RandomPerspective(distortion_scale=.1, p=0.8),
    transform.RandomAdjustSharpness(0.8, 0.8)]
    )
    
    return augmentation


def get_data():

    mirrors = get_mirrors()
    symmetric = get_symmetric()
    augmentation = get_transforms()

    trainfolder = "./input/traffic-sign/Train/"
    trainids, valids = split_data(trainfolder,n_folds = 4)
    testfolder ="./input/traffic-sign/Test/"
    testids = get_full_testdata(testfolder)

    traindata = GTSRB_Dataset(trainids,  True, False , mirrors, symmetric, augmentation, augmentation_size = 1, size = [32,32])
    valdata = GTSRB_Dataset(valids,  False, False , mirrors, symmetric, augmentation, augmentation_size = 0, size = [32,32])
    testdata = GTSRB_Dataset(testids,  False, False , mirrors, symmetric, augmentation, augmentation_size = 0, size = [32,32])

    return traindata, valdata, testdata