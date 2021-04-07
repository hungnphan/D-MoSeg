# from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
import numpy as np
import cv2 as cv
import torch
import os
import math
import torch


class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # single-process data loading, return the full iterator
        if worker_info is None:  
            iter_start = self.start
            iter_end = self.end
        # in a worker process
        else:  
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        # print(type(iter(range(iter_start, iter_end))))
        # return iter(range(iter_start, iter_end))

        print("This method we use yield instead of return")
        for data_value in range(iter_start, iter_end):
            print(f"yield data {data_value}")
            yield data_value

# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
# [tensor([3]), tensor([4]), tensor([5]), tensor([6])]

# Mult-process loading with two worker processes
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
# [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

# With even more workers
print(list(torch.utils.data.DataLoader(ds, num_workers=5)))
# [tensor([3]), tensor([4]), tensor([5]), tensor([6])]

