import numpy as np
import cupy as cp
import cv2 as cv
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
from arg_parser import parse_config_from_json
from cdn import CDN
from likelihood_loss_function import TrainingCriterion
from data_io import DataLoader


# Init group of processes for multi-GPUs
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6789'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Destructor for group of processes with multi-GPUs
def cleanup():
    dist.destroy_process_group()

def train_with_multi_gpus(rank, world_size):
    print(f"Setting up DataDistributedParallel on rank {rank}.")
    setup(rank, world_size)

    # Specify device
    device = rank

    # Read method params from json config file
    config = parse_config_from_json(config_file='config.json')

    # Initialize data_loader
    data_loader = DataLoader(config)

    # Define Convolutional Density Network
    cdn_net = CDN(config.KMIXTURE, \
                  data_loader.img_heigh, data_loader.img_width, data_loader.img_channel).to(device)

    # construct DDP model
    ddp_cdn_net = DDP(cdn_net, device_ids=[rank])

    # Define loss function
    train_criterion = TrainingCriterion(config)

    # Define network optimizer
    optimizer = torch.optim.Adam(cdn_net.parameters(), lr=config.LEARNING_RATE)

    # Calculate the number of pixel in 2D image space
    N_PIXEL = data_loader.img_heigh * data_loader.img_width

    # input shape = [num_pixels, C, 1, FPS]
    num_pixels, C, FPS = config.PIXEL_BATCH, 3, config.FPS
    example_input = np.random.randint(0, 256,[num_pixels,C,1,FPS])
    example_input = (torch.from_numpy(example_input).float() / 255.0).to(device)
    print(f"example_input.shape = {example_input.shape}")

    example_output = cdn_net(example_input)
    print(f"example_output.shape = {example_output.shape}")

    cleanup()


def process(target_fnc, world_size):
    mp.spawn(target_fnc,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    process(target_fnc = train_with_multi_gpus, world_size=1)
