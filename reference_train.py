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

EPOCHS = 1000

finetune_fraction = 0.1

log_path = "./log.txt"

useLinearPred = False



def main(useResnet18):
    global simclr
    print(DEVICE)
    #Data: load augmented train data for SimCLR, test data, 
    #unsupervised SSL learning of simCLR
    #apply relative representations to guess accuracy
    #find actualy representation accuracy with linear predictors & MLP's on frozen encoder
    
        
    
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    simclr_predictor = SimCLRPredictor(10, DEVICE, useResnet18=useResnet18, tune_encoder = False, linear_predictor = useLinearPred).to(DEVICE)

    simclr_optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4)
    predictor_optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)

    
    ntxent = NTXentLoss(device=DEVICE).to(DEVICE)
    cross_entropy = nn.CrossEntropyLoss()


    trainset, testset = utils.load_augmented()


    
    print('yuh')
    trainloader = DataLoader(trainset, batch_size = 512, shuffle = True)
    testloader = DataLoader(testset, batch_size = 256, shuffle = True)
    print('cuh')
    
    for epoch in range(EPOCHS):
        ssl_train(epoch, simclr, trainloader, simclr_optimizer, ntxent)
        if epoch > 100 and epoch % 5 == 0:
            save_model(epoch)
        
def load_model(simclr):
    
    list_of_files = [fname for fname in glob.glob("./ssl_centralized/ssl_centralized_model_*.pth")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from:", latest_round_file)
    count = latest_round_file.split('_')[3].split('.')[0]
    
    state_dict = torch.load(latest_round_file)
    
    simclr.load_state_dict(state_dict)
    
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

    
    if not os.path.isdir('ssl_centralized'):
        os.mkdir('ssl_centralized')

    torch.save(simclr.state_dict(), f"./ssl_centralized/ssl_centralized_model_{epoch}.pth")

if __name__ == "__main__":
    main(False)

