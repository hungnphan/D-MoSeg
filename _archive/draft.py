import os

from torchvision import datasets
import torch
import torchvision.transforms as transforms



# Testing
from arg_parser import parse_config_from_json
import torch.distributed as dist

def do_something(rank, world_size):
    print(f"Setting up DataDistributedParallel on rank {rank}.")
    setup(rank, world_size)

    config = parse_config_from_json(config_file='config.json')
    # train_dataset = BackgroundDataset(config)
    train_dataset = datasets.MNIST("mnist_data", train=True, transform=transforms.ToTensor(), download=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=world_size,
    	rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=False,            
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )    

    total_step = len(train_loader)
    print(f"total_step = {total_step}")

    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        print(type(images))
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
    process(target_fnc = do_something, world_size=1)

