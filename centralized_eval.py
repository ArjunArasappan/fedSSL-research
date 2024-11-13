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

EPOCHS = 100
SEGMENTS = 1

#epochs to train linear
fine_tune_epochs = 3


finetune_fraction = 0.3

log_path = "./log.txt"



reference_model = None
relative_eval_metric = None
useResnet18 = False
simclr_predictor = None

def init(anchors):
    global reference_model, relative_eval_metric, trainset, testset, simclr_predictor
    
    trainset, testset = utils.load_centralized_data()
        
    
    
    reference_path = './reference_models/ssl_centralized_model_csa_1225.pth'
    
    reference_model = SimCLR(DEVICE, useResnet18=False).to(DEVICE)
    
    reference_model.load_state_dict(torch.load(reference_path,  map_location=torch.device('cpu')), strict = True)
    # reference.load_state_dict(torch.load(reference_path), strict = True)
    
    reference_model.eval()
    reference_model.setInference(True)
        
    relative_eval_metric = EvalMetric(reference_model)
    anchors = relative_eval_metric.selectAnchors(trainset)
    
    relative_eval_metric.setAnchors(anchors)

    relative_eval_metric.calcReferenceAnchorLatents()
    
    simclr_predictor = SimCLRPredictor(10, DEVICE, useResnet18=useResnet18, tune_encoder = False).to(DEVICE)
    
    random_init = SimCLR(DEVICE, useResnet18=False).to(DEVICE)
    
    calculate_metrics(random_init, 0)
    


    
def calculate_metrics(simclr, round):
    global reference_model, relative_eval_metric, trainset, testset, useResnet18, simclr_predictor
    
    state_dict = simclr.state_dict()
    weights = [v.cpu().numpy() for v in state_dict.values()]

    simclr_predictor.set_encoder_parameters(weights)
    
    predictor_optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)
    
    ntxent = NTXentLoss(device=DEVICE)
    cross_entropy = nn.CrossEntropyLoss()
    
    trainloader = DataLoader(trainset, batch_size = 512, shuffle = True)
    testloader = DataLoader(testset, batch_size = 512, shuffle = True)\
    

    mean, median = computeSimilarities(testloader, simclr, relative_eval_metric)

    supervised_train(simclr_predictor, trainloader, predictor_optimizer, cross_entropy)
    loss, accuracy = supervised_test(simclr_predictor, testloader, cross_entropy)
    
    
    utils.sim_log([SEGMENTS, round, mean.item(), median.item(), loss, accuracy])
        
    
    
def computeSimilarities(testloader, simclr, relative_eval):
    
    
    similarity, similarity_normed = [], []
    batch = 0
    for item in testloader:
        
        x, _, _ = item['img']
        
        x = x.to(DEVICE)
        
        relative_eval.calcModelLatents(simclr)
        
        sim, normed_sim = relative_eval.computeSimilarity(x, simclr)
        print(f"Computing sims {batch}/{len(testloader)}: {sim}, {normed_sim}")
        similarity.append(sim)
        similarity_normed.append(normed_sim)
        batch += 1
        
    sim = torch.mean(torch.Tensor(similarity), dim = 1)
    normed_sim = torch.mean(torch.Tensor(similarity), dim = 1)
    print(f"Relative Eval DONE: {sim}, {normed_sim}")

    return sim, normed_sim

def supervised_train(simclr_predictor, trainloader, optimizer, criterion):
    
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
            x, _, _, labels = item['img'], item['label']
            
            
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
