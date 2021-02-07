import torch
from model import *
from dataset import *
import numpy as np
import sys

def main():
    batch_size = 128
    train_data = data_loader(batch_size)
    # step to run
    step = np.math.ceil(int(train_data.dataset.__len__()/batch_size))
    network = model()
    #define the loss and optimizer
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    #train loop
    epoch = 10
    for i in range(epoch):
        for stp in range(step):
            index = train_data.dataset.sample(batch_size)
            sampler = torch.utils.data.SubsetRandomSampler(index)
            train_data.batch_sampler.sampler = sampler
            imgX, imgY = next(iter(train_data))
            optimizer.zero_grad()
            # now we pass the img through model
            predictedY  = network(imgX)
            # calculate the loss
            loss = criteria(predictedY, imgY)
            loss.backward()
            optimizer.step()
            stats = "[%d/%d/%d] Loss: %.4f "%(stp, i, epoch, loss.item())
            print("\r "+stats, end="  ")
            sys.stdout.flush()
            if stp%50==0 and stp != 0:
                torch.save({
                    "model_state":network.state_dict()
                },"weights/model.pth")
                print("\r"+stats)

main()
