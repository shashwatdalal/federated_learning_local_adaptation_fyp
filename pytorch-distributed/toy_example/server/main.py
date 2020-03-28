import os

import torch
from torch import distributed as dist


def init_network():
    rank_ = int(os.environ['RANK'])
    world_size_ = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(os.environ['BACKEND'], rank=rank_,
                            world_size=world_size_)
    return rank_, world_size_

def broadcast(model, world_size_):
    for i in range(1, world_size_):
        try:
            dist.send(tensor=model, dst=i)
        except RuntimeError:
            print('Training Finished')
            exit(0)

def collect_and_update(model):
    updates = 0
    for i in range(1, world_size):
        update = torch.zeros(1)
        dist.recv(tensor=update, src=i)
        # print("Rank {} sent: {}".format(i, update))
        updates += update
    # average update
    model += (updates / (world_size - 1))
    return model


if __name__ == '__main__':

    rank, world_size = init_network()

    # exchange meta-data

    # send number of rounds
    N_ROUNDS = 100
    N_ROUNDS = torch.zeros(1) + N_ROUNDS
    broadcast(N_ROUNDS, world_size)

    # send initial model
    model = torch.zeros(1)
    broadcast(model, world_size)
    print("Round {} Model: {}".format(0, model))

    # simulate rounds of federated learning
    for round in range(int(N_ROUNDS)):
        # collect updates
        model = collect_and_update(model)
        broadcast(model, world_size)
        print("Round {} Model: {}".format(round+1, model))
