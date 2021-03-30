import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cupy as cp


class FDN(nn.Module):

    def __init__(self):
        super(FDN, self).__init__()

        # TO-DO: Encoder blocks

        # The depthwise separable layer #1
        self.dconv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=6, bias=False)
        self.pconv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)
        
        # The depthwise separable layer #2
        self.dconv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=16, bias=False)
        self.pconv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)
        
        # The depthwise separable layer #3
        self.dconv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=16, bias=False)
        self.pconv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)
        
        # The depthwise separable layer #4
        self.dconv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=16, bias=False)
        self.pconv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)
        
        # The depthwise separable layer #5
        self.dconv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=16, bias=False)
        self.pconv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)
        
        # The depthwise separable layer #6
        self.dconv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=16, bias=False)
        self.pconv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)
        
        # The depthwise separable layer #7
        self.dconv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=16, bias=False)
        self.pconv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)
        
        # The depthwise separable layer #8
        self.dconv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), \
                                stride=(1, 1), padding = 1, groups=16, bias=False)
        self.pconv8 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), \
                                stride=(1, 1), groups=1, bias=False)

    def forward(self, x):
        x = F.relu(self.pconv1(self.dconv1(x))) # depthwise separable 1
        x = F.relu(self.pconv2(self.dconv2(x))) # depthwise separable 2

        x = F.max_pool2d(x,(2, 2))              # Sampling -> shape/2
        x = F.relu(self.pconv3(self.dconv3(x))) # depthwise separable 3
        x = F.relu(self.pconv4(self.dconv4(x))) # depthwise separable 4

        x = F.max_pool2d(x,(2, 2))              # Sampling -> shape/4
        x = self.pconv5(self.dconv5(x))         # depthwise separable 5
        x = F.relu(F.instance_norm(x))
        
        x = F.interpolate(x,scale_factor=2.0,mode='nearest')   # Upsampling -> shape/2
        x = self.pconv6(self.dconv6(x))         # depthwise separable 6
        x = F.relu(F.instance_norm(x))          # Instance norm 1 + ReLU

        x = F.interpolate(x,scale_factor=2.0,mode='nearest')   # Upsampling -> shape
        x = self.pconv7(self.dconv7(x))         # depthwise separable 7
        x = F.relu(F.instance_norm(x))          # Instance norm 2 + ReLU

        x = F.hardsigmoid(self.pconv8(self.dconv8(x)))
        return x

    def foreground_loss(self, y_pred, y_true):
        criterion = nn.BCELoss()
        loss = criterion(y_pred, y_true)
        return loss

    def foreground_accuracy(self, y_pred, y_true):
        round_pred = torch.round(y_pred)
        return torch.mean(torch.eq(round_pred, y_true).float())



# ############################################
# # TO-DO: test cross entropy loss
# Batch = 1
# H = 3
# W = 5
# C = 1

# y_pred = torch.rand(size=[Batch, C, H, W]).float()
# y_true = torch.randint(0,2,size=[Batch, C, H, W]).float()

# criterion = nn.BCELoss(reduction='none')
# loss = torch.mean(criterion(y_pred, y_true))
# acc = torch.mean(torch.eq(torch.round(y_pred), y_true).float())

# print(y_pred)
# print(torch.round(y_pred))
# print(y_true)
# print(loss)
# print(acc)


############################################
# TO-DO: test model shape
# Batch = 1
# H = 720
# W = 480
# C = 6

# # input shape with dimension [batch, height, width, channel]
# x = torch.randint(0, 256, size = [Batch, C, H, W]).float()

# # forward to the fdn network
# model = FDN()
# out = model(x)

# # check input-output shape
# print(x.shape)
# print(out.shape)

