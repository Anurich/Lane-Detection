import torch
import torch.nn.functional as F
from contructdata import constructdataset
import torch.utils.data as data
import pandas as pd
from PIL import Image
import os
import numpy as np
import cv2

class dataset(data.Dataset):
    def __init__(self, train=True, transform = None):
        super().__init__()
        self.image_folder = "dataset/"
        self.transforms = transform
        self.train = train
        if train:
            # read the dataset
            self.df = pd.read_csv("train.csv",names=["imagename","lane1","lane2"]).iloc[1:]
        else:
            self.df = pd.read_csv("test.csv",names=["imagename","lane1","lane2"]).iloc[1:]

    def _preprocesslane(self, lane):
        lane  = lane.replace("[","").replace("]","").replace(",","")
        lane  = list(map(float,lane.split()))
        return lane

    def read_data_from_csv(self, data_frame, idx):
        row = data_frame.iloc[idx]
        imagename, lane1, lane2 = row
        if not self.train:
            orgimg  = cv2.imread("dataset/"+imagename)
            orgimg  = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img  =  Image.open("dataset/"+imagename).convert("RGB")
        if self.transforms != None:
            img  = self.transforms(img)
        lane1  = torch.Tensor(self._preprocesslane(lane1))[:60]
        lane2  = torch.Tensor(self._preprocesslane(lane2))[:60]
        #print(np.array(lane1).reshape(-1,2).shape, np.array(lane2).reshape(-1,2).shape)
        if not self.train:
            return img, lane1, lane2,orgimg
        else:
            lane1 = (lane1 - lane1.min())/(lane1.max() - lane1.min())
            lane2 = (lane2 - lane2.min())/(lane2.max() - lane2.min())
            return img, lane1, lane2

    def __getitem__(self, idx):
        if self.train:
            # get the data from csv file
            return self.read_data_from_csv(self.df, idx)
        else:
            return self.read_data_from_csv(self.df, idx)
    def __len__(self):
        if self.train:
            return self.df.shape[0]
        else:
            return self.df.shape[0]

def data_loader(transform, train, batch_size):
    train_files = ["dataset/label_data_0313.json", "dataset/label_data_0531.json"]
    test_files  = ["dataset/label_data_0601.json"]
    # this function will only run once if file exists it will not run
    constructdataset(train_files, train=True)
    constructdataset(test_files, train=False)
    data_ = dataset(train, transform)
    index = np.arange(data_.df.shape[0])
    sampler = data.RandomSampler(index)
    batch_sampler =  data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    loader  = data.DataLoader(data_, batch_sampler= batch_sampler)
    return loader



