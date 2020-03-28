import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torchvision import datasets, transforms
import os

rank_ = int(os.environ['RANK'])
world_size_ = int(os.environ['WORLD_SIZE'])

distributed.init_process_group(os.environ['BACKEND'], rank=rank_,
                               world_size=world_size_)

tensor = torch.zeros(1)

if rank_ == 0:
    tensor += 1
    for i in range(1,4):
        distributed.send(tensor=tensor, dst=i)
else:
    distributed.recv(tensor=tensor, src=0)
print('Rank {}: {}'.format(rank_, tensor))
