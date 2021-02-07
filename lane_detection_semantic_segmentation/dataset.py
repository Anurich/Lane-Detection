import torch
import torch.nn as nn
import pickle
import numpy as np
class dataset(torch.utils.data.Dataset):
    def __init__(self, train,transform=None):
        super().__init__()
        # read the train and test from pkl file
        self.train =  train
        if self.train:
            self.train = pickle.load(open("full_CNN_train.p","rb"))
            self.train_label  = pickle.load(open("full_CNN_labels.p","rb"))
            self.transform = transform
        else:
            pass
    def __getitem__(self, idx):
        if self.train:
            img_x, img_y = self.train[idx], self.train_label[idx]
            if self.transform:
                img_x = self.transform(img_x)
                img_y = self.transform(img_y)
            else:
                img_x = torch.Tensor(img_x).permute(2,0,1)
                img_y = torch.Tensor(img_y).permute(2,0,1)/255.
            return img_x, img_y

        else:
            pass
    def sample(self, batch_size):
        range_dataset =  np.arange(len(self.train))
        return np.random.choice(range_dataset, batch_size)

    def __len__(self):
        if self.train:
            return len(self.train_label)
        else:
            pass

def data_loader(batch_size, transform=None, train=True):
    data = dataset(train, transform)
    # so we sample the data from the
    if train:
        index = data.sample(batch_size)
        #index = np.arange(len(data.train))
        random_sampler = torch.utils.data.SubsetRandomSampler(index)
        batchSampler   = torch.utils.data.BatchSampler(random_sampler,batch_size,drop_last=False)
        train_data_loader=  torch.utils.data.DataLoader(data,batch_sampler=batchSampler)
        return train_data_loader
    else:
        pass
