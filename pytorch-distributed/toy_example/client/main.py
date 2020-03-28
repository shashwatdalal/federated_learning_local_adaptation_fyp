import random
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
    model += random.random()
    return model


if __name__ == '__main__':
    rank, world_size = init_network()

    # simulate rounds of federated learning

    # initialize state
    model = torch.zeros(1)
    n_rounds = torch.zeros(1)

    dist.recv(tensor=n_rounds, src=COORDINATION_SERVER)

    for round in range(int(n_rounds)):
        dist.recv(tensor=model, src=COORDINATION_SERVER)
        model = train_model(model)
        dist.send(tensor=model, dst=COORDINATION_SERVER)
        time.sleep(0.1)
