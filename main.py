import flwr as fl
from flwr.server.strategy import FedAvg
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from collections import OrderedDict
import argparse

import client as client
from model import SimCLR, SimCLRPredictor, NTXentLoss
import centralized_eval
import utils
from test import evaluate_gb_model 
import os 

DEVICE = utils.DEVICE


fl.common.logger.configure(identifier="debug", filename="./log_files/log.txt")

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default= 2,
    help="Number of CPUs to use during simulation",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default= 0.2,
    help="Number of GPUs to use during simulation",
)

parser.add_argument(
    "--num_clients",
    type=int,
    default=10,
    help="Number of clients",
)

parser.add_argument(
    "--use_resnet18",
    type=bool,
    default=False,
    help="Use Resnet18 over Resnet50",
)

parser.add_argument(
    "--num_rounds",
    type=int,
    default=1000,
    help="Number of FL training rounds",
)

parser.add_argument(
    "--num_anchors",
    type=int,
    default= 300,
    help="Number of Anchors to Select",
)

def fit_config_fn(server_round: int):
    fit_config = {}
    fit_config["current_round"] = server_round
    return fit_config


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            
            gb_simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(gb_simclr.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            gb_simclr.load_state_dict(state_dict, strict=True)
            
            args = parser.parse_args()
            
            path = f"./fl_checkpoints/num_clients_{args.num_clients}/"
            file = f"checkpoint_round_{server_round}.pth"
            
            if not os.path.exists(path):
                os.makedirs(path)
                
            
            if server_round % 5 == 0 or server_round == 1:
                centralized_eval.calculate_metrics(gb_simclr, server_round)
                

        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(on_fit_config_fn = fit_config_fn)

if __name__ == "__main__":

    print("Cuda?:", torch.cuda.is_available())
    # print("name:", torch.cuda.get_device_name(0))
    
    print(torch.__version__)
    print(torch.version.cuda)
    
    args = parser.parse_args()
    
    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    useResnet18 = args.use_resnet18
    NUM_ANCHORS = args.num_anchors
    
    #client sim code
    gpu_alloc = 0.2
    cpu_alloc = 2
    if NUM_CLIENTS < 5:
        gpu_alloc = float(1) / float(NUM_CLIENTS)
        cpu_alloc = int(12 / NUM_CLIENTS)
    
    print('GPU ALLOC:', gpu_alloc)
    print('CPU ALLOC:', cpu_alloc)
    
    #NO GPU
    # gpu_alloc = 0
    
    client_resources = {
        "num_cpus": cpu_alloc,
        "num_gpus": gpu_alloc
    }
    
    print("Num Clients: ", NUM_CLIENTS)
    print("Num Rounds: ", NUM_ROUNDS)
    print("Resnet18", useResnet18)
    
    print("Num Rounds: ", NUM_ANCHORS)
    
    
        
    fds = utils.get_anchored_fds(NUM_CLIENTS, NUM_ANCHORS)
    anchor_data, _ = utils.load_partition(fds, NUM_CLIENTS, split = 'train', apply_augment = False)
    test_data, _ = utils.load_partition(fds, 0, split = 'test', apply_augment = False)
    
    
    #get batch
    
    anchorloader = DataLoader(dataset=anchor_data, batch_size = NUM_ANCHORS)
    testloader = DataLoader(dataset=test_data, batch_size = 512)

    for batch in anchorloader:
        anchor_data = batch['img']

    
    centralized_eval.init(anchor_data, test_data)
    
    fl.simulation.start_simulation(
        client_fn=client.get_client_fn(fds, useResnet18, NUM_CLIENTS),
        num_clients= NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds= NUM_ROUNDS),
        client_resources=client_resources,
        strategy=strategy,
    )
    
    loss, accuracy = evaluate_gb_model(utils.useResnet18, NUM_CLIENTS)
    
    print("FINAL GLOBAL MODEL RESULTS:")
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    
    
    data = ['fine_tune', utils.fineTuneEncoder, useResnet18, NUM_CLIENTS, loss, accuracy]
    
    
    