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

###############################
# Testing
from arg_parser import parse_config_from_json
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6759'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Destructor for group of processes with multi-GPUs
def cleanup():
    dist.destroy_process_group()

def work_with_processes(rank, world_size):
    print(f"Setting up DataDistributedParallel on rank {rank}.")
    setup(rank, world_size)

    # train_dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=False,
    #                             transform=torchvision.transforms.Compose([
    #                                 torchvision.transforms.ToTensor(),
    #                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #                             ]))

    config = parse_config_from_json(config_file='config.json')
    train_dataset = BackgroundIterableDataset(config)

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
        pin_memory=True
        # sampler=train_sampler
    )    

    # print(type(train_loader))

    # total_step = len(train_loader)
    # print(f"total_step = {total_step}")
    
    # generator = iter(train_loader)
    # for _ in iter(train_loader):
    #     try:
    #         # Samples the batch
    #         pixel_batch = next(generator)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         generator = iter(train_loader)
    #         pixel_batch = next(generator)


    # for i, sample in enumerate(train_loader):
        # print(i, type(sample))
        # break

    cleanup()

def process(target_fnc, world_size):
    mp.spawn(target_fnc,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    print("starting .. ")
    process(target_fnc = work_with_processes, world_size=2)


