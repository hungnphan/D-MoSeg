import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import params

class CDN(nn.Module):

    def __init__(self, kmixture):
        super(CDN, self).__init__()

        self.K_MIXTURES = kmixture

        self.dconv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 7), \
                    stride=(1, 7), padding = (0, 3), groups=3, bias=False)
        self.pconv1 = nn.Conv2d(in_channels=3, out_channels=7, kernel_size=(1, 1))
        self.dconv2 = nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(1, 7), \
                    stride=(1, 7), padding = (0, 3), groups=7, bias=False)
        self.pconv2 = nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(1, 1))

        self.fc_pi = nn.Linear(35, self.K_MIXTURES)  # 6*6 from image dimension
        self.fc_sigma = nn.Linear(35, self.K_MIXTURES)
        self.fc_mu = nn.Linear(35, self.K_MIXTURES*3)


    def forward(self, x):
        """
            This method returns the CDNBackgroundModeling network architecture.
            Input of the network is a [BATCH, CHANNEL, 1, FPS] numpy array or TF tensor that is normalized into range [0, 1]
            Output of the network is a (N, 5*K_MIXTURES) including
                model_pi =      [N, K_MIXTURES]
                model_sigma =   [N, K_MIXTURES]
                model_mu =      [N, K_MIXTURES * 3]    
        """        
    
        x = F.relu(self.pconv1(self.dconv1(x))) # depthwise separable
        x = F.relu(self.pconv2(self.dconv2(x))) # depthwise separable
        
        x = x.view(-1, self.num_flat_features(x))   # flatten

        pi = F.softmax(self.fc_pi(x), dim=1)        # FC to pi      [Batch, KMIX]
        sigma = F.hardsigmoid(self.fc_sigma(x))     # FC to sigma^2 [Batch, KMIX]
        mu = F.hardsigmoid(self.fc_mu(x))           # FC to muy     [Batch, KMIX * 3]

        out = torch.cat([pi, sigma, mu], -1)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



# # Test network
# net = CDN(kmixture=4)
# print(net)

# HEIGHT = 320
# WIDTH = 240
# FPS = 240
# CHANNEL = 3

# # [BATCH, 1, FPS, CHANNELS]
# input_batch_img = np.random.randint(0, 256, size=[HEIGHT * WIDTH, 1, FPS, CHANNEL])

# # [BATCH, CHANNEL, 1, FPS]
# x = torch.from_numpy(input_batch_img).permute(0,3,1,2).type(torch.FloatTensor)

# print(x.shape)

# out = net(x)
# print(out.shape)
