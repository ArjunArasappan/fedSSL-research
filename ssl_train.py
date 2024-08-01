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
from eval import EvalMetric
import os

import flwr as fl
import utils
import csv

simclr = None

DEVICE = utils.DEVICE
EPOCHS = 1
count = 0

log_path = "./log.txt"



def ssl_train(useResnet18):
    #Data: load augmented train data for SimCLR, test data, 
    #unsupervised SSL learning of simCLR
    #apply relative representations to guess accuracy
    #find actualy representation accuracy with linear predictors & MLP's on frozen encoder
        
    
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    ntxent = NTXentLoss( device=DEVICE).to(DEVICE)
    
        
    trainset, testset = utils.load_augmented()
    
    eval = EvalMetric(trainset, testset)
    
    eval.load_reference()
    eval.evaluateReference()
    eval.selectAnchors()
    

    


def train(net, trainloader, optimizer, criterion, epochs):
    net.train()
    num_batches = len(trainloader)
    batch = 0
    total_loss = 0
    
    for epoch in range(epochs):

        for item in trainloader:
            x_i, x_j = item['img']
   
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            optimizer.zero_grad()
            
            z_i = net(x_i).to(DEVICE)
            z_j = net(x_j).to(DEVICE)
            
            loss = criterion(z_i, z_j)
            total_loss += loss

            loss.backward()
            optimizer.step()

            print("Client Train Batch:", batch, "/", num_batches)
            
            batch += 1
            
    return {'Loss' : float(total_loss / batch)}
                 

def test(net, testloader, criterion):
    
    if testloader == None:
        return -1, -1
    
    net.eval()
    loss_epoch = 0
    batch = 0
    num_batches = len(testloader)
    
    with torch.no_grad():
        for item in testloader:
            x_i, x_j = item['img']
            
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            
            z_i = net(x_i).to(DEVICE)
            z_j = net(x_j).to(DEVICE)
            loss = criterion(z_i, z_j)
            
            loss_epoch += loss.item()
            
            print("Client Train Batch:", batch, "/", num_batches)
            
            batch += 1
    return loss_epoch / (batch), -1


def save_model(acc):
    global count
    
    if not os.path.isdir('weights_nosched'):
        os.mkdir('weights_nosched')
        
    torch.save(simclr_predictor.state_dict(), f"./weights_nosched/centralized_model_{acc}.pth")
    count += 1

if __name__ == "__main__":
    ssl_train(False)

