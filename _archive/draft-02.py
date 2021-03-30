import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class RandomDataset(Dataset):
    def __init__(self, length, local_rank):
        self.len = length
        self.data = torch.stack([torch.ones(1), 
                                 torch.ones(1)*2,
                                 torch.ones(1)*3,
                                 torch.ones(1)*4,
                                 torch.ones(1)*5,
                                 torch.ones(1)*6,
                                 torch.ones(1)*7,
                                 torch.ones(1)*8]).to('cuda')
        self.local_rank = local_rank
        
        # [8, 1]
        # print(f"self.data.shape = {self.data.shape}")

    def __getitem__(self, index):

        # [1]
        # print(f"self.data[index].shape = {self.data[index].shape}")
        return self.data[index]

    def __len__(self):
        return self.len
 
torch.distributed.init_process_group(backend="nccl")
 
batch_size = 4
data_size = 8
 
local_rank = torch.distributed.get_rank()
print(local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


# dataset = RandomDataset(data_size, local_rank)

from bg_dataset import BackgroundDataset
from arg_parser import parse_config_from_json

config = parse_config_from_json(config_file='config.json')
dataset = BackgroundDataset(local_rank, config)

sampler = DistributedSampler(dataset,num_replicas=1,rank=local_rank)
 
#rand_loader =DataLoader(dataset=dataset,batch_size=batch_size,sampler=None,shuffle=True)
rand_loader = DataLoader(dataset=dataset,batch_size=batch_size,sampler=sampler)

epoch = 0
while epoch < 1:
    sampler.set_epoch(epoch)
    for data in rand_loader:
        # print(data)
        print(type(data))
    epoch+=1