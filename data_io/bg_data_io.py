import numpy as np
import cupy as cp
import cv2 as cv
import torch
import os

class BgDataLoader:
    def __init__(self, args, scenario_name, sequence_name):
        self.sequence_dir = args.sequence_dir    # path to CDnet/scenario/sequence
        self.scenario_name = scenario_name
        self.sequence_name = sequence_name


        self.FPS = args.FPS
        self.data_path = None
        
        self.data_frame = None

        self.img_heigh = None
        self.img_width = None
        self.img_channel = None

        self.data_len = -1
        self.current_frame_idx = -1

        # Get img info: height, width, channels
        if self.__init_img_info__():
            # Init data frame for training
            self.__init_data_frame__()
            print("Successfully init data loader ...")
        else:
            print("Fail to init data loader ...")

        return

    def __init_img_info__(self):
        self.data_path = os.path.join(self.sequence_dir, self.scenario_name, self.sequence_name, 'input', 'in%06d.jpg')
        # print(self.data_path)
        if os.path.exists(self.data_path % 1):
            sample_frame = cv.imread(self.data_path % 1, cv.IMREAD_COLOR)

            if sample_frame is not None:
                self.img_heigh, self.img_width, self.img_channel = sample_frame.shape
                self.data_len = len([file_name for file_name in os.listdir(os.path.join(self.sequence_dir, self.scenario_name, self.sequence_name, 'input')) \
                                        if os.path.isfile(os.path.join(self.sequence_dir, self.scenario_name, self.sequence_name, 'input', file_name))])
                return True
        return False

    def __init_data_frame__(self):

        self.data_frame = torch.cuda.FloatTensor(self.img_heigh*self.img_width, \
                                                 self.img_channel, 1, self.FPS).fill_(0)

        for frame_idx in range(self.FPS):
            # count the reading frame
            self.current_frame_idx = self.current_frame_idx + 1

            # [H, W, C]: read image from file
            img = cv.imread(self.data_path % (frame_idx+1), cv.IMREAD_COLOR)

            # [H, W, C] -> [H*W, C, 1]: reshape image in format of PyTorch
            img_reshape = img.reshape([self.img_heigh * self.img_width, self.img_channel, 1])

            # [H*W, C, 1, FPS]: append image to Torch tensor
            self.data_frame[..., self.current_frame_idx % self.FPS] = (torch.from_numpy(img_reshape).float() / 255.0).cuda()
       
        return

    def load_next_k_frame(self, k):
        for _ in range(k):
            self.load_next_frame()

    def load_next_frame(self):

        if os.path.exists(self.data_path % (self.current_frame_idx+1)):

            # count the reading frame
            self.current_frame_idx = self.current_frame_idx + 1

            # [H, W, C]: read image from file
            img = cv.imread(self.data_path % (self.current_frame_idx+1), cv.IMREAD_COLOR)

            # [H, W, C] -> [H*W, C, 1]: reshape image in format of PyTorch
            img_reshape = img.reshape([self.img_heigh * self.img_width, self.img_channel, 1])

            # [H*W, C, 1, FPS]: append image to Torch tensor
            self.data_frame[..., self.current_frame_idx % self.FPS] = (torch.from_numpy(img_reshape).float() / 255.0).cuda()

            return True 

        return False
