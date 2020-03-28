import os

import torch
from torch import distributed as dist

COORDINATION_SERVER = 0

def init_network():
    rank_ = int(os.environ['RANK'])
    world_size_ = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(os.environ['BACKEND'], rank=rank_,
                            world_size=world_size_)
    return rank_, world_size_

def collect_and_update(model):
    update = torch.zeros(1)
    # reduce will take <update> value of this node
    dist.reduce(update, COORDINATION_SERVER, op=dist.ReduceOp.SUM)
    # average update
    model += (update / (world_size - 1))
    return model

if __name__ == '__main__':

    rank, world_size = init_network()

    # send number of rounds
    n_rounds = torch.zeros(1) + int(os.environ['N_ROUNDS'])
    dist.broadcast(n_rounds, COORDINATION_SERVER)

    model = torch.zeros(1)
    print("Round {} Model: {}".format(0, model))

    # simulate rounds of federated learning
    for round in range(int(n_rounds)):
        dist.broadcast(model, COORDINATION_SERVER)
        # collect updates
        model = collect_and_update(model)
        print("Round {} Model: {}".format(round+1, model))
