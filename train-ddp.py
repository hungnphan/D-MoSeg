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

    print("\nStart the model training ...")
    print(f"There are {round(N_PIXEL/config.PIXEL_BATCH)} batches for training\n")

    # Train on each of sliding window's move
    for sliding_step in range(config.SLIDING_STEP):
        print(f"---Training with the position #{sliding_step} of sliding windows---")
        train_data_frame = data_loader.data_frame

        # Train on a position of sliding window [config.EPOCHS] times
        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            step_loss = 0.0

            # Train on batches with a size of [config.PIXEL_BATCH] pixels
            for train_step in range(round(N_PIXEL/config.PIXEL_BATCH)):
                # zero the parameter gradients
                optimizer.zero_grad()

                # Split the training batch
                batch_start = train_step * config.PIXEL_BATCH                   
                batch_end = min(N_PIXEL, (train_step+1) * config.PIXEL_BATCH)                
                training_batch = data_loader.data_frame[batch_start:batch_end, ...]     # input shape = [num_pixels, C, 1, FPS]
               
                # Feed forward on CDN
                output = cdn_net(training_batch)    # output shape = [N, 5 * KMIX]

                # x.shape = [N, 1, FPS, 3]  y.shape = [N, K_MIXTURES*5]
                loss = train_criterion.likelihood_loss(training_batch.permute(0, 2, 3, 1), output).to(device)
                
                # Backward + optimize
                loss.backward()
                optimizer.step()

                # Accumulated loss values
                step_loss += loss.item()        # Accumulated loss for each train_step (train on batches)
                epoch_loss += loss.item()       # Accumulated loss for each
                
                # Report the average loss every 200 mini-batches
                if (train_step+1) % 200 == 0:
                    print('[epoch=%d, train_step=%5d]\tloss: %.3f' %
                        (epoch + 1, train_step + 1, step_loss / 200,))   
                    step_loss = 0.0         
            
            # Report the average loss at each position of sliding window
            print('---> Everage loss (at each position of sliding window) = %.5f\n' %(epoch_loss / round(N_PIXEL/config.PIXEL_BATCH)))

        data_loader.load_next_k_frame(2)

    cleanup()


def process(target_fnc, world_size):
    mp.spawn(target_fnc,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    process(target_fnc = train_with_multi_gpus, world_size=1)


