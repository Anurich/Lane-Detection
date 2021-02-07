import torch
import torch.nn as nn


class convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding = 1)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.relu      = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class deconvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        #self.batchNorm = nn.BatchNorm2d(out_channels)
        self.relu   = nn.ReLU()
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.deconv(x)
        #x = self.batchNorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class model(nn.Module):
    # The architecture idea of this model is taken from segnet
    # it is not the actual segnet architecture
    def __init__(self):
        super().__init__()
        self.batchNorm = nn.BatchNorm2d(3)
        #layer1
        self.conv1 = convolution(3, 8, 3, 1)
        self.conv2 = convolution(8, 16, 3, 1)
        #layer 2
        self.conv3 = convolution(16, 16, 3, 1)
        self.conv4 = convolution(16, 32, 3, 1)
        # layer3
        self.conv5  = convolution(32, 32, 3, 1)
        self.conv6  = convolution(32, 64, 3, 1)
        self.conv7  =  convolution(64, 64, 3, 1)
        self.maxpool = nn.MaxPool2d(2)
        #layer 4
        #self.conv8 = convolution(64, 128, 3, 1)
        #self.conv9 = convolution(128, 128, 3, 1)
        # deconvolution
        self.upsample = nn.Upsample(scale_factor=2)
        #self.convtranspose1  = deconvolution(128, 128, 3)
        #self.convtranspose2  = deconvolution(128, 64, 3)
        self.convtranspose3 = deconvolution(64, 64, 3)
        self.convtranspose4 = deconvolution(64, 32, 3)
        self.convtranspose5 = deconvolution(32, 32, 3)
        #decon layer 2
        self.convtranspose6 = deconvolution(32, 16, 3)
        self.convtranspose7 = deconvolution(16, 16, 3)
        self.convtranspose8 = deconvolution(16, 8, 3)
        #output
        self.output  = deconvolution(8, 1, 3)


    def forward(self, x):
        # layer 1
        x  = self.batchNorm(x)
        x  = self.conv1(x)
        x  = self.conv2(x)
        x  = self.maxpool(x)
        # layer 2
        x  = self.conv3(x)
        x  = self.conv4(x)
        x  = self.maxpool(x)
        #layer3
        x  = self.conv5(x)
        x  = self.conv6(x)
        x  = self.conv7(x)
        x  = self.maxpool(x)
        #layer4
        #x  = self.conv8(x)
        #x  = self.conv9(x)
        #x  = self.maxpool(x)

        # decon layer 1
        x = self.upsample(x)
        #x = self.convtranspose1(x)
        #x = self.convtranspose2(x)
        x = self.convtranspose3(x)
        # decon layer 2
        #x = self.upsample(x)
        x = self.convtranspose4(x)
        x = self.convtranspose5(x)
        # output layers
        x = self.upsample(x)
        x = self.convtranspose6(x)
        x = self.convtranspose7(x)
        x = self.upsample(x)
        x = self.convtranspose8(x)
        x = self.output(x)

        return x

