import torch
import torch.nn as nn
import torchvision.models as models

class model(nn.Module):
    def __init__(self):
        super().__init__()
        # load the pretrained model

        self.alex = models.alexnet(pretrained=True)
        self.alex = nn.Sequential(*list(self.alex.children()))[:-1]
        '''
        for param in self.alex.parameters():
            param.requires_grad = False
        '''
        # next we define two dense layer
        #self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu    = nn.ReLU()
        self.dense1 = nn.Linear(9216, 120)
        self.lane1  = nn.Linear(120, 60)

        self.dense2 = nn.Linear(9216, 120)
        self.lane2  = nn.Linear(120, 60)


    def forward(self, x):
        dense = self.alex(x)
        dense = self.flatten(dense)
        lane_first     = self.dense1(dense)
        lane_first     = self.relu(lane_first)
        lane_first_output= self.lane1(lane_first)
        # dense layer
        lane_second    = self.dense2(dense)
        lane_second    = self.relu(lane_second)
        lane_second_output = self.lane2(lane_second)

        return lane_first_output, lane_second_output
