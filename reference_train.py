import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimCLR, SimCLRPredictor, NTXentLoss
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torchvision.transforms as transforms
from eval_metric import EvalMetric
import os

import flwr as fl
import utils
import csv

simclr = None

DEVICE = utils.DEVICE

EPOCHS = 2000

finetune_fraction = 0.1

log_path = "./log.txt"




def main(useResnet18):
    global simclr
    print(DEVICE)
            
    
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    simclr_predictor = SimCLRPredictor(10, DEVICE, useResnet18=useResnet18, tune_encoder = False).to(DEVICE)

    epochLoad = load_model(simclr)
    epochLoad = 100

    simclr_optimizer = optim.SGD(simclr.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(simclr_optimizer, T_max = EPOCHS)


    
    ntxent = NTXentLoss(device=DEVICE).to(DEVICE)


    trainset, testset = utils.load_centralized_data()



    trainloader = DataLoader(trainset, batch_size = 512, shuffle = True, num_workers = utils.num_workers)

    
    for epoch in range(epochLoad + 1, EPOCHS):
        avgLoss = ssl_train(epoch, simclr, trainloader, simclr_optimizer, ntxent)
        if epoch > 100 and epoch % 5 == 0:
            save_model(epoch)

        utils.sim_log([epoch, avgLoss], './log_files/reference_train.csv')
        
        cosine_scheduler.step()
        
def load_model(simclr):
    
    list_of_files = [fname for fname in glob.glob("./reference_models/ssl_centralized_model_*.pth")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from:", latest_round_file)
    count = latest_round_file.split('_')[5].split('.')[0]
    
    state_dict = torch.load(latest_round_file)
    
    simclr.load_state_dict(state_dict, strict = False)
    
    return int(count)


def ssl_train(epoch, net, trainloader, optimizer, criterion):
    net.train()
    
    num_batches = len(trainloader)
    batch = 0
    
    avgLoss = 0
    
    for item in trainloader:
        _, x_i, x_j = item['img']
        
        optimizer.zero_grad()
        
        x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
        
        z_i, z_j = net(x_i), net(x_j)
    
        loss = criterion(z_i, z_j)
        
        loss.backward()
        optimizer.step()
        
        avgLoss += loss.item()
        
        print(f"SSL Train Batch Epoch{epoch}: {batch} / {num_batches}")
        batch += 1

        
    return avgLoss / num_batches  

count = 0
def save_model(epoch):
    global count, simclr

    
    if not os.path.isdir('./reference_models'):
        os.mkdir('./reference_models')

    torch.save(simclr.state_dict(), f"./reference_models/ssl_centralized_model_csa_{epoch}.pth")

if __name__ == "__main__":
    main(False)
