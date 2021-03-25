from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torch
import os


class BackgroundDataset(Dataset):
    def __init__(self, args):          
        # super(BackgroundDataset, self).__init__()

        # self.local_rank = local_rank

        self.sequence_dir = args.sequence_dir    # path to CDnet/scenario/sequence

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
            print("Successfully init data loader ...")
        else:
            print("Fail to init data loader ...")

    def __init_img_info__(self):
        self.data_path = os.path.join(self.sequence_dir, 'input', 'in%06d.jpg')
        if os.path.exists(self.data_path % 1):
            sample_frame = cv.imread(self.data_path % 1, cv.IMREAD_COLOR)

            if sample_frame is not None:
                self.img_heigh, self.img_width, self.img_channel = sample_frame.shape
                self.data_len = len([file_name for file_name in os.listdir(os.path.join(self.sequence_dir, 'input')) \
                                        if os.path.isfile(os.path.join(self.sequence_dir, 'input', file_name))])
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

    def __getitem__(self, index):
        value_at_index = self.data_frame[index % (self.img_heigh * self.img_width), ...]

        if index == (self.img_heigh * self.img_width -1):
            self.load_next_k_frame()

        return value_at_index

    def __len__(self):
        count = self.img_heigh*self.img_width
        # count = self.img_heigh*self.img_width * ((self.data_len - self.FPS ) // self.stride)
        return count 


# Testing
from arg_parser import parse_config_from_json
import torch.distributed as dist

def work_with_processes(rank, world_size):
    print(f"Setting up DataDistributedParallel on rank {rank}.")
    setup(rank, world_size)

    config = parse_config_from_json(config_file='config.json')
    train_dataset = BackgroundDataset(config)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=world_size,
    	rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.PIXEL_BATCH,
        shuffle=False,            
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )    

    total_step = len(train_loader)
    
    for sample in train_loader:
        print(type(sample), sample.shape)
        break

    cleanup()

import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6759'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Destructor for group of processes with multi-GPUs
def cleanup():
    dist.destroy_process_group()

def process(target_fnc, world_size):
    mp.spawn(target_fnc,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    print("starting .. ")
    process(target_fnc = work_with_processes, world_size=1)

