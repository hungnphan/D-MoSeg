# from torch.utils.data import Dataset
import torch
from torch.utils.data import IterableDataset
import numpy as np
import cv2 as cv
import torch
import os
import math


class BackgroundIterableDataset(IterableDataset):
    def __init__(self, args):
        super(BackgroundIterableDataset).__init__()        

        self.sequence_dir = args.sequence_dir     # path to CDnet/scenario/sequence
        self.scenario_name = args.scenario_name
        self.sequence_name = args.sequence_name

        self.FPS = args.FPS
        self.data_path = None
        
        self.data_frame = None

        self.img_heigh = None
        self.img_width = None
        self.img_channel = None

        self.stride = args.WINDOW_STRIDE
        self.data_len = -1
        self.current_frame_idx = -1

        # Get img info: height, width, channels
        if self.__init_img_info__():
            # Init data frame for training
            self.__init_data_frame__()

            # We calculate the number of pixel batch for training in whole dataset
            self.dataset_size = self.img_heigh*self.img_width * ((self.data_len - self.FPS ) // self.stride)

            print("Successfully init data loader ...")
        else:
            print("Fail to init data loader ...")

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

        self.data_frame = torch.FloatTensor(self.img_heigh*self.img_width, \
                                            self.img_channel, 1, self.FPS).fill_(0)

        for frame_idx in range(self.FPS):
            # count the reading frame
            self.current_frame_idx = self.current_frame_idx + 1

            # [H, W, C]: read image from file
            img = cv.imread(self.data_path % (frame_idx+1), cv.IMREAD_COLOR)

            # [H, W, C] -> [H*W, C, 1]: reshape image in format of PyTorch
            img_reshape = img.reshape([self.img_heigh * self.img_width, self.img_channel, 1])

            # [H*W, C, 1, FPS]: append image to Torch tensor
            self.data_frame[..., self.current_frame_idx % self.FPS] = torch.from_numpy(img_reshape).float() / 255.0
       
    def load_next_k_frame(self):
        for it in range(self.stride):
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

    def __iter__(self):
        # Get worker information
        worker_info = torch.utils.data.get_worker_info()

        # single-process data loading, return the full iterator
        if worker_info is None:  
            # We iterate all the pixel in the dataset
            for pixel_batch_idx in range(self.dataset_size):
                # Calculate the local index of pixel batch within a frame
                local_pixel_idx = pixel_batch_idx % (self.img_heigh * self.img_width)

                # When we iterate all a frame, then move the striding window
                if pixel_batch_idx != 0 and local_pixel_idx == 0:
                    self.load_next_k_frame()

                # Get the value of pixel batch
                yield self.data_frame[local_pixel_idx % (self.img_heigh * self.img_width), ...]

        # in a worker process
        else:  
            # Split workload among workers
            # per_worker = int(math.ceil(self.dataset_size / float(worker_info.num_workers)))

            # print(f"worker_info.id = {worker_info.id}")

            # We iterate all the pixel in the dataset
            current_step = 1
            for pixel_batch_idx in range(worker_info.id,
                                         self.dataset_size,
                                         worker_info.num_workers):
                # Calculate the local index of pixel batch within a frame
                local_pixel_idx = pixel_batch_idx % (self.img_heigh * self.img_width)

                # When we iterate all a frame, then move the striding window
                if pixel_batch_idx >= current_step*(self.img_heigh * self.img_width):
                    current_step = current_step + 1
                    self.load_next_k_frame()

                # Get the value of pixel batch
                yield self.data_frame[local_pixel_idx % (self.img_heigh * self.img_width), ...]

    def __len__(self):
        # count = self.img_heigh*self.img_width
        count = self.img_heigh*self.img_width * ((self.data_len - self.FPS ) // self.stride)
        return count 

