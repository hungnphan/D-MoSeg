import numpy as np
import cupy as cp
import cv2 as cv
import h5py
import glob
import os

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP

from arg_parser import parse_config_from_json
from cdn import CDN
from likelihood_loss_function import TrainingCriterion
from data_io import DataLoader
from bg_dataset_iterable import BackgroundIterableDataset

cdnet_data = {
    "badWeather"                : ["blizzard","skating","snowFall","wetSnow"],
    "baseline"                  : ["highway","office","pedestrians","PETS2006"],
    "cameraJitter"              : ["badminton","boulevard","sidewalk","traffic"],
    "dynamicBackground"         : ["boats","canoe","fall","fountain01","fountain02","overpass"],
    "intermittentObjectMotion"  : ["abandonedBox","parking","sofa","streetLight","tramstop","winterDriveway"],
    "lowFramerate"              : ["port_0_17fps","tramCrossroad_1fps","tunnelExit_0_35fps","turnpike_0_5fps"],
    "nightVideos"               : ["bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet"],
    "PTZ"                       : ["continuousPan","intermittentPan","twoPositionPTZCam","zoomInZoomOut"],
    "shadow"                    : ["backdoor","bungalows","busStation","copyMachine","cubicle","peopleInShade"],
    "thermal"                   : ["corridor","diningRoom","lakeSide","library","park"],
    "turbulence"                : ["turbulence0","turbulence1","turbulence2","turbulence3"]
}

def train_with_ddp(world_size=1, config=None, backend="gloo"):
    # mp.spawn(train,
    #         args=(world_size,),
    #         nprocs=world_size,
    #         join=True)
    
    processes = []
    for rank in range(world_size):
        p = Process(target=train, args=(rank, world_size, config, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def train(rank, world_size, config, backend):
    
    # Setup DDP worker/process group for training
    def setup(rank, world_size, backend):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '6760'

        # initialize the process group
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Destructor for group of processes with multi-GPUs
    def cleanup():
        dist.destroy_process_group()

    print(f"Setting up DataDistributedParallel on rank {rank}.")
        
    # Init process group in Torch
    setup(rank, world_size, backend)    

    # Specify device
    device = rank

    # Initialize BGS data_loader
    bgs_dataset = BackgroundIterableDataset(config)
    
    # Initialize data sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        bgs_dataset,
        num_replicas=world_size,
        rank=rank
    )

    # Initialize data loader API from Torch
    train_loader = torch.utils.data.DataLoader(
        dataset=bgs_dataset,
        batch_size=config.PIXEL_BATCH,
        shuffle=False,            
        num_workers=0,
        pin_memory=True
        # sampler=train_sampler
    )    

    # Define Convolutional Density Network
    cdn_net = CDN(config.KMIXTURE, \
                  bgs_dataset.img_heigh, bgs_dataset.img_width, bgs_dataset.img_channel)    #.to(rank)

    # Define loss function for background training
    train_criterion = TrainingCriterion(config)

    # Define network optimizer
    optimizer = torch.optim.Adam(cdn_net.parameters(), lr=config.LEARNING_RATE)

    # Restore model from last checkpoint
    ckpt_dir = os.path.join(config.CKPT_DIR, config.scenario_name, config.sequence_name)
    path_to_checkpoint = os.path.join(ckpt_dir,'cdn.pth')
    if os.path.exists(path_to_checkpoint):
        print(f"Loading checkpoint from {path_to_checkpoint} ..")

        map_location = None #{'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(path_to_checkpoint, map_location=map_location)

        cdn_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_sliding_step   = checkpoint['sliding_step']
        # self.last_epoch          = checkpoint['epoch']
    else:
        os.makedirs(ckpt_dir)
        print('No checkpoint is found ! Start to train from scratch.')
        last_sliding_step   = 0

    # construct DDP model
    cdn_net = DDP(cdn_net) # , device_ids=[rank]

    # # Set model mode to continue training
    cdn_net.train()
    
    print(f"\nStart the model training on rank {rank}...")

    data_iter = iter(train_loader)
    batch_idx = -1
    while True:
        try:
            # get the next training_batch
            # input shape = [num_pixels, C, 1, FPS]
            training_batch = next(data_iter)
            batch_idx = batch_idx + 1

            # Train on a position of sliding window [config.EPOCHS] times
            for epoch in range(config.EPOCHS):
                epoch_loss = 0.0
                step_loss = 0.0

                # zero the parameter gradients
                optimizer.zero_grad()

                # Feed forward on CDN
                output = cdn_net(training_batch)    # output shape = [N, 5 * KMIX]

                # x.shape = [N, 1, FPS, 3]  y.shape = [N, K_MIXTURES*5]
                loss = train_criterion.likelihood_loss(training_batch.permute(0, 2, 3, 1), output)
                
                # Backward + optimize
                loss.backward()
                optimizer.step()

                # Accumulated loss values
                step_loss += loss.item()        # Accumulated loss for each train_step (train on batches)
                epoch_loss += loss.item()       # Accumulated loss for each

                # Report the average loss at each position of sliding window
                print('---> Everage loss at batch {%02d} = %.5f\n' %(batch_idx, epoch_loss))

        except StopIteration:
            # if StopIteration is raised, break from loop
            break




    # Close the process group
    cleanup()


#############################################
# Read method params from json config file
config_file = 'config.json'
config = parse_config_from_json(config_file=config_file)  

# Set scenario + sequence name
scenario_name = config.scenario_name
sequence_name = config.sequence_name

train_with_ddp(world_size=2, config=config, backend="gloo")




