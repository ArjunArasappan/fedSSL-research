import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimCLRPredictor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torchvision.transforms as transforms
import os

import flwr as fl
import utils
import csv

simclr_predictor = None

DEVICE = utils.DEVICE
EPOCHS = 1
count = 0

log_path = "./log.txt"


def load_model(useResnet18):
    global simclr_predictor
    
    list_of_files = [fname for fname in glob.glob("./centralized_weights/centralized_model_*.pth")]
    if list_of_files == []:
        return 0
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from:", latest_round_file)
    print(latest_round_file)
    accuracy = latest_round_file.split('/')[2].split('_')[2].split('.pth')[0]
    accuracy = float(accuracy)
    
    state_dict = torch.load(latest_round_file)
    
    simclr_predictor.load_state_dict(state_dict, strict = True)
    
    return accuracy

def centralized_train(useResnet18):
    global simclr_predictor
    simclr_predictor = SimCLRPredictor(utils.NUM_CLASSES, DEVICE, useResnet18=utils.useResnet18, tune_encoder=utils.fineTuneEncoder).to(DEVICE)
    
    # preacc = load_model(False)
    preacc = 0
    
    trainset, testset = load_data()
    
    trainloader = DataLoader(trainset, batch_size = 128, shuffle = True)
    testloader = DataLoader(testset, batch_size = 128)   

    optimizer = optim.SGD(simclr_predictor.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    criterion = nn.CrossEntropyLoss()
    
    train(trainloader, testloader, optimizer, criterion, scheduler, preacc)
    
    
def train(trainloader, testloader, optimizer, criterion, scheduler, preacc):

    maxAccuracy = preacc
    
    iters = 0
    
    while maxAccuracy < 0.94:
        trainAccuracy = train_predictor(trainloader, optimizer, criterion )

    
        print(f"Train Accuracy: {trainAccuracy}")
        
        data = [iters, "train accuracy", int(trainAccuracy * 10000) / 10000]

        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        
        
        
        testLoss, testAccuracy = evaluate(testloader, criterion)
        
        print(f"Test Accuracy: {testAccuracy}")
        
        data = [iters, "test accuracy", int(testAccuracy * 10000) / 10000]

        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            

        
        if maxAccuracy <= testAccuracy:
            acc = int(testAccuracy * 1000) / 1000
            save_model(acc)
            
        maxAccuracy = max(testAccuracy, maxAccuracy)
            
        # scheduler.step()
        iters += 1
        

    
    
    





def train_predictor(trainloader, optimizer, criterion):
    global simclr_predictor
    simclr_predictor.train()
    
    num_batches = len(trainloader)
    total = 0
    correct = 0
    
    for epoch in range(EPOCHS):
        batch = 0
        
        for item in trainloader:
            x, labels = item['img'], item['label']
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = simclr_predictor(x)
            
            values, predicted = torch.max(outputs, 1)  
            
            loss = criterion(outputs, labels)
        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch: {epoch} / {EPOCHS} Predictor Train Batch: {batch} / {num_batches}")
            batch += 1
            
    return correct / total
            

def evaluate(testloader, criterion):
    global simclr_predictor

    simclr_predictor.eval()
    
    total = 0
    correct = 0
    
    loss = 0
    batch = 0
    
    num_batches = len(testloader)

    with torch.no_grad():
        for item in testloader:
            x, labels = item['img'], item['label']
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            logits = simclr_predictor(x)
            values, predicted = torch.max(logits, 1)  
            
            total += labels.size(0)
            loss += criterion(logits, labels).item()
            correct += (predicted == labels).sum().item()

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
    centralized_train(False)

