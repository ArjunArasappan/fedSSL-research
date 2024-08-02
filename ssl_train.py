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

EPOCHS = 10
SEGMENTS = 10

count = 0

log_path = "./log.txt"

useLinearPred = False



def main(useResnet18):
    #Data: load augmented train data for SimCLR, test data, 
    #unsupervised SSL learning of simCLR
    #apply relative representations to guess accuracy
    #find actualy representation accuracy with linear predictors & MLP's on frozen encoder
    
        
    trainset, testset = utils.load_augmented()
    
    relative_eval = EvalMetric(trainset, testset)
    
    relative_eval.load_reference()
    # eval.evaluateReference()
    relative_eval.selectAnchors()
    relative_eval.calcReferenceLatents()
    
    ssl_simulation(trainset, testset, useResnet18, relative_eval)
    
def ssl_simulation(trainset, testset, useResnet18, relative_eval):
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    simclr_predictor = SimCLRPredictor(10, DEVICE, useResnet18=useResnet18, tune_encoder = False, linear_predictor = useLinearPred).to(DEVICE)

    simclr_optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4)
    predictor_optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)

    
    ntxent = NTXentLoss(device=DEVICE).to(DEVICE)
    cross_entropy = nn.CrossEntropyLoss()
    
    
    trainloader = DataLoader(trainset, batch_size = 256, shuffle = True, pin_memmory = True)

    
    for epoch in range(EPOCHS * SEGMENTS):
        train(simclr, trainloader, simclr_optimizer, ntxent)
        similarities, mean, mode = computeSimilarities(simclr, simclr_predictor, relative_eval)
        accuracy = supervised_train(simclr, simclr_predictor, trainloader, cross_entropy)
    

def train(net, trainloader, optimizer, criterion):
    net.train()
    
    num_batches = len(trainloader)
    batch = 0
    total_loss = 0
    
    for item in trainloader:
        _, x_i, x_j = item['img']
        
        optimizer.zero_grad()
    
        loss = criterion(z_i, z_j)
        total_loss += loss

        loss.backward()
        optimizer.step()

        print(f"Client Train Batch: {batch} / {num_batches}")
        batch += 1
        
        if batch >= num_batches / SEGMENTS:
            break
        
        
    return total_loss / batch

def computeSimilarities(simclr, data, relative_eval):
    
    loader = DataLoader(data, pin_memmory = True, batch_size = len(data))
    
    for idx, item in enumerate(loader):
        print(f"Relatuve Eval: {idx}")
        
        batch = item['img']
        relative_eval.calcModelLatents(simclr)
        similarities, mean, median = relative_eval.computeSimilarities(batch, model)
        print(f"Relative Eval DONE: {mean}, {median}")
        return similarities, mean, median
    

def save_model(acc):
    global count
    
    if not os.path.isdir('weights_nosched'):
        os.mkdir('weights_nosched')
        
    torch.save(simclr_predictor.state_dict(), f"./weights_nosched/centralized_model_{acc}.pth")
    count += 1

if __name__ == "__main__":
    main(False)

