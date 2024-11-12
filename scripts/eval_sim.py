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
SEGMENTS = 1

#epochs to train linear
fine_tune_epochs = 3


finetune_fraction = 0.3

log_path = "./log.txt"

useLinearPred = True



def main(useResnet18):
    #Data: load augmented train data for SimCLR, test data, 
    #unsupervised SSL learning of simCLR
    #apply relative representations to guess accuracy
    #find actualy representation accuracy with linear predictors & MLP's on frozen encoder
    
        
    trainset, testset = utils.load_augmented()
    
    #setup eval metric
    
    reference_path = './reference_models/ssl_centralized_model_csa_1225.pth'
    
    reference = SimCLR(DEVICE, useResnet18=False).to(DEVICE)
    state_dict = torch.load(reference_path,  map_location=torch.device('cpu'))
    reference.load_state_dict(state_dict, strict = True)
    
    reference.eval()
    reference.setInference(True)
        
    relative_eval = EvalMetric(reference)
    anchors = relative_eval.selectAnchors(trainset)
    
    relative_eval.setAnchors(anchors)


    relative_eval.calcReferenceAnchorLatents()
    
    ssl_simulation(trainset, testset, useResnet18, relative_eval)
    
def ssl_simulation(trainset, testset, useResnet18, relative_eval):
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    simclr_predictor = SimCLRPredictor(10, DEVICE, useResnet18=useResnet18, tune_encoder = False, linear_predictor = useLinearPred).to(DEVICE)

    simclr_optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4)
    predictor_optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)

    
    ntxent = NTXentLoss(device=DEVICE)
    cross_entropy = nn.CrossEntropyLoss()
    
    
    trainloader = DataLoader(trainset, batch_size = 512, shuffle = True)
    testloader = DataLoader(testset, batch_size = 1024, shuffle = True)\
    

    
    
    for epoch in range(EPOCHS * SEGMENTS):
        
        mean, median = computeSimilarities(testloader, simclr, relative_eval)

        supervised_train(simclr, simclr_predictor, trainloader, predictor_optimizer, cross_entropy)
        loss, accuracy = supervised_test(simclr_predictor, testloader, cross_entropy)
        
        train(simclr, trainloader, simclr_optimizer, ntxent)
        
        utils.sim_log([SEGMENTS, epoch, mean.item(), median.item(), loss, accuracy])
        
    

def train(net, trainloader, optimizer, criterion):
    net.train()
    
    num_batches = len(trainloader)
    batch = 0
    
    for item in trainloader:
        _, x_i, x_j = item['img']
        
        optimizer.zero_grad()
        
        x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
        
        z_i, z_j = net(x_i), net(x_j)
    
        loss = criterion(z_i, z_j)

        loss.backward()
        optimizer.step()

        print(f"Train Batch: {batch} / {num_batches}")
        batch += 1
        
        if batch >= num_batches / SEGMENTS:
            break
        
    
def computeSimilarities(testloader, simclr, relative_eval):
    
    
    similarity, similarity_normed = [], []
    batch = 0
    for item in testloader:
        
        x = item['img']
        
        x = x.to(DEVICE)
        
        relative_eval.calcModelLatents(simclr)
        
        sim, normed_sim = relative_eval.computeSimilarity(x, simclr)
        print(f"Computing sims {batch}/{len(testloader)}: {sim[0]}, {sim[1]}")
        similarity.append(sim)
        similarity_normed.append(normed_sim)
        batch += 1
        
    sim = torch.mean(torch.Tensor(similarity), dim = 1)
    normed_sim = torch.mean(torch.Tensor(similarity), dim = 1)
    print(f"Relative Eval DONE: {sim}, {normed_sim}")

    return sim, normed_sim

def supervised_train(simclr, simclr_predictor, trainloader, optimizer, criterion):
    state_dict = simclr.state_dict()
    weights = [v.cpu().numpy() for v in state_dict.values()]

    simclr_predictor.set_encoder_parameters(weights)
    
    simclr_predictor.train()
    
    num_batches = len(trainloader)

    for i in range(fine_tune_epochs):    
        for idx, item in enumerate(trainloader):
            (x, _, _), labels = item['img'], item['label']
            
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            z = simclr_predictor(x)
        
            loss = criterion(z, labels)

            loss.backward()
            optimizer.step()

            print(f"Client Train Batch: {idx} / {num_batches}")
        
def supervised_test(simclr_predictor, testloader, criterion):
    simclr_predictor.eval()
    
    total = 0
    correct = 0
    loss = 0

    batch = 0
    num_batches = len(testloader)

    with torch.no_grad():
        for item in testloader:
            x , labels = item['img'], item['label']
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            logits = simclr_predictor(x)
            values, predicted = torch.max(logits, 1)  
            
            total += labels.size(0)
            loss += criterion(logits, labels).item()
            
            correct += (predicted == labels).sum().item()
            if batch >= num_batches / 2:
                break
            print(f"Test Batch: {batch} / {num_batches}")
            batch += 1
  
    return loss / batch, correct / total
              

    

def save_model(acc):
    global count
    
    if not os.path.isdir('weights_nosched'):
        os.mkdir('weights_nosched')
        
    torch.save(simclr_predictor.state_dict(), f"./weights_nosched/centralized_model_{acc}.pth")
    count += 1

if __name__ == "__main__":
    main(False)

