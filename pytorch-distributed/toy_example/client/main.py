import os
import time

import torch
from torch import distributed as dist

COORDINATION_SERVER = 0

def init_network():
    rank_ = int(os.environ['RANK'])
    world_size_ = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(os.environ['BACKEND'], rank=rank_,
                            world_size=world_size_)
    return rank_, world_size_


def train_model(model):
    x = torch.randn(1)
    return x


if __name__ == '__main__':
    rank, world_size = init_network()

    # initialize state
    model = torch.zeros(1)
    n_rounds = torch.zeros(1)

    # get number of rounds
    dist.broadcast(n_rounds, COORDINATION_SERVER)

    # simulate rounds of federated learning
    for _ in range(int(n_rounds)):
        dist.broadcast(model, COORDINATION_SERVER)
        update = train_model(model)
        dist.reduce(update, COORDINATION_SERVER)
        time.sleep(0.1)
