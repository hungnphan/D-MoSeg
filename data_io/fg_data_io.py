import numpy as np
import cupy as cp
import cv2 as cv
import torch
import h5py
import glob
import os
import torch
from torch.utils.data import IterableDataset

class ForegroundIterableDataset(IterableDataset):
    def __init__(self, rank, world_size, config, n_replica):
        super(ForegroundIterableDataset).__init__()

        self.config = config
        self.shuffle_buffer_size = 10
        self.n_replica = n_replica

        # Get process-dist information
        self.proc_id = rank
        self.num_proc = world_size

        # Get scenario name + data_dir
        self.scenario_name = self.config.scenario_name
        self.fg_training_dir = self.config.FG_TRAINING_DATA

        # Scan data sequence for specific scenario
        self.train_data = self.__scan_training_data__()
        self.num_sequence = len(self.train_data)

    def __scan_training_data__(self):
        file_names = []
        for file_name in os.listdir(self.fg_training_dir):
            if file_name.startswith(self.scenario_name) and file_name.endswith('.hdf5'): 
                file_names.append(file_name)
        return file_names

    def __iter__(self):
        # Get worker information
        self.worker_info = torch.utils.data.get_worker_info()
        self.worker_id = 0
        self.num_workers = 1
        if self.worker_info is not None:
            self.worker_id = self.worker_info.id
            self.num_workers = self.worker_info.num_workers

        # Calculate the local rank of specific worker in process
        self.local_rank = self.num_workers*self.proc_id + self.worker_id
        self.global_size = self.num_workers*self.num_proc

        # Use all allocated data for each process
        for _ in range(self.n_replica):
            for data_name in self.train_data:
                # Init shuffle buffer
                shuffle_buffer = []
                it = self.__get_item_data__(data_name)
                for _ in range(self.shuffle_buffer_size):
                    try:
                        data_item = next(it)
                        shuffle_buffer.append(data_item)
                    except:
                        self.shuffle_buffer_size = len(shuffle_buffer)
                        break
                        
                try:
                    while True:
                        try:
                            # load data to refill the buffer
                            data_item = next(it)

                            # yield data
                            random_idx = np.random.randint(len(shuffle_buffer))
                            yield_value = shuffle_buffer[random_idx]
                            shuffle_buffer[random_idx] = data_item
                            yield yield_value
                            
                        except StopIteration:
                            break
                        
                    while len(shuffle_buffer) > 0:
                        random_idx = np.random.randint(len(shuffle_buffer))
                        yield shuffle_buffer.pop(random_idx)
                except GeneratorExit:
                    pass

                break

    def __get_item_data__(self, data_name):
        path_to_data = os.path.join(self.config.FG_TRAINING_DATA, data_name)
        with h5py.File(path_to_data, "r") as data_block:

            # Extract frame index in data_block
            data_keys = sorted(list(data_block.keys()))

            # Iterate over data_block
            for data_idx in range(len(data_keys)):
                
                # Distribute the data to each worker
                if data_idx%self.global_size == self.local_rank:
                    
                    # Extract data_key (frame_idx)
                    data_key = data_keys[data_idx]

                    # Get data from h5py using key 'data_idx'
                    # inp_img, bg_img, fg_img
                    data_tensor = data_block[data_key]

                    yield data_tensor[...,:6], data_tensor[...,6]
                    # yield data_key

    def __len__(self):
        return 200*len(self.train_data)




# #########################################
# from args.arg_parser import parse_config_from_json

# config = parse_config_from_json(config_file='config/config.json')
# fg_dataset = ForegroundIterableDataset(0,1,config)

# # Initialize data loader API from Torch
# batch_size = 5
# num_worker = 1
# train_loader = torch.utils.data.DataLoader(
#     dataset=fg_dataset,
#     batch_size=batch_size,
#     # shuffle=True,            
#     num_workers=num_worker,
#     # pin_memory=True
# )

# # for frame_idx in train_loader:
# #     print(frame_idx)


# for train_x, train_y in train_loader:

#     for item in range(batch_size):
#         cv.imshow('Input image',train_x[item,:, :,:3].cpu().detach().numpy())
#         cv.imshow('Background image',train_x[item,:, :,3:6].cpu().detach().numpy())
#         cv.imshow('Foreground image',train_y[item,:, :].cpu().detach().numpy())

#         # Toogle pause video process
#         key = cv.waitKey(50)
#         if key == ord('p'):
#             while(cv.waitKey(1) != ord('p')):
#                 continue

# cv.destroyAllWindows()





# #########################################
# from args.arg_parser import parse_config_from_json
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.multiprocessing import Process

# def train_with_ddp(world_size=1, backend="gloo"):    
#     processes = []
#     for rank in range(world_size):
#         p = Process(target=train, args=(rank, world_size, backend))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

# def train(rank, world_size, backend):
    
#     # Setup DDP worker/process group for training
#     def setup(rank, world_size, backend):
#         os.environ['MASTER_ADDR'] = '127.0.0.1'
#         os.environ['MASTER_PORT'] = '6760'

#         # initialize the process group
#         dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

#     # Destructor for group of processes with multi-GPUs
#     def cleanup():
#         dist.destroy_process_group()

#     print(f"Setting up DataDistributedParallel on rank {rank}.")
        
#     # Init process group in Torch
#     setup(rank, world_size, backend)    

#     # Config
#     config = parse_config_from_json(config_file='config/config.json')

#     # Initialize BGS data_loader
#     fg_dataset = ForegroundIterableDataset(rank,world_size,config)
    
#     # Initialize data loader API from Torch
#     train_loader = torch.utils.data.DataLoader(
#         dataset=fg_dataset,
#         batch_size=5,
#         shuffle=False,            
#         num_workers=2,
#         pin_memory=True
#     )    

#     # if rank == 0:
#     for frame_idx in train_loader:
#         print(rank, frame_idx)
#         # pass

#     # Close the process group
#     cleanup()


# train_with_ddp(world_size=15, backend="gloo")