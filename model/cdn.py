import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import params

class CDN(nn.Module):

    def __init__(self, kmixture, img_height, img_width, img_channel):
        super(CDN, self).__init__()

        self.K_MIXTURES = kmixture
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.IMG_CHANNEL = img_channel

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
            Input of the network is a [BATCH, CHANNEL, 1, FPS] numpy array or tensor that is normalized into range [0, 1]
            Output of the network is a (N, 5*K_MIXTURES) including
                model_pi =      [N, K_MIXTURES]
                model_sigma =   [N, K_MIXTURES]
                model_mu =      [N, K_MIXTURES * 3]    
        """        

        input_shape = x.shape

        x = F.relu(self.pconv1(self.dconv1(x))) # depthwise separable
        x = F.relu(self.pconv2(self.dconv2(x))) # depthwise separable
        
        x = x.view(-1, self.num_flat_features(x))   # flatten

        pi = F.softmax(self.fc_pi(x), dim=1)        # FC to pi      [Batch, KMIX]
        sigma = F.hardsigmoid(self.fc_sigma(x))     # FC to sigma^2 [Batch, KMIX]
        mu = F.hardsigmoid(self.fc_mu(x))           # FC to muy     [Batch, KMIX * 3]

        out = torch.cat([pi, sigma, mu], -1)

        # print("\tIn Model: input size", input_shape,
        #       "output size", out.size())

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def calculate_background(self, reshaped_frames_stack, batch_size):
        """
            This method calculate the most suitable current background of the video and
                saves the background into self.background_model attributes (numpy array)

            Input:
                reshaped_frames_stack: a normalized (H*W, FPS, C) cupy array containing
                    the values of the pixels of the frames used to calculate the background
                    model

            Ouput: calculated background as a cp.ndarray
        """

        out_mix_pi = torch.zeros(size=[self.IMG_HEIGHT*self.IMG_WIDTH, self.K_MIXTURES])     # [H*W, KMIX]
        out_mix_var = torch.zeros(size=[self.IMG_HEIGHT*self.IMG_WIDTH, self.K_MIXTURES])   # [H*W, KMIX]
        out_mix_mean = torch.zeros(size=[self.IMG_HEIGHT*self.IMG_WIDTH, self.IMG_CHANNEL, self.K_MIXTURES])    # [H*W, C*KMIX]

        # Predict the GMM distribution w.r.t input batch of frames
        net_output = self.forward(reshaped_frames_stack)

        # Split (N, 5*K_MIXTURES) -> (N, K_MIXTURES) + (N, K_MIXTURES) + (N, 3K_MIXTURES)
        out_mix_pi, out_mix_var, out_mix_mean = \
            torch.split(net_output, [self.K_MIXTURES, self.K_MIXTURES, 3*self.K_MIXTURES], dim=-1)

        # print(f"out_mix_pi.shape = {out_mix_pi.shape}")
        # print(f"out_mix_var.shape = {out_mix_var.shape}")
        # print(f"out_mix_mean.shape = {out_mix_mean.shape}")

        # Post-processing: reshape the output, denormalize the variance, compute mixture mask
        mix_pi = out_mix_pi.view(self.IMG_HEIGHT, self.IMG_WIDTH, self.K_MIXTURES, 1)
        mix_var = out_mix_var.view(self.IMG_HEIGHT, self.IMG_WIDTH, self.K_MIXTURES, 1)
        mix_mean = out_mix_mean.view(self.IMG_HEIGHT, self.IMG_WIDTH, self.K_MIXTURES, self.IMG_CHANNEL)

        # Extract background: at each pixel, we pickout the mixture components that
        # have the greatest value of (weight / variance)
        mix_var = 16.0 + 16.0 * mix_var
        div = mix_pi / mix_var                                          # [H, W, KMIX, 1]
        mixture_mask = torch.amax(div, dim=-2, keepdim=True) == div     # [H, W, KMIX, 1]
        background_img = torch.amax(mix_mean * mixture_mask, dim=-2)    # [H, W, 3] (keep_dim = False)

        # Reverse normalization: (0,1) -> (0,255) uint8
        background_img = (background_img * 255.0).type(torch.uint8).cpu().data.numpy()

        # cv.imwrite('background.png', background_img)
        return background_img







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
