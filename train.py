import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimCLR, SimCLRPredictor
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


def applyTrainTransform(batch):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        
    batch["img"] = [transform_train(img) for img in batch["img"]]
    return batch

def applyTestTransform(batch):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        
    batch["img"] = [transform_test(img) for img in batch["img"]]
    return batch

def load_data():
    fds = FederatedDataset(dataset="cifar10", partitioners = {'train' : IidPartitioner(1), 'test' : IidPartitioner(1)})
    
    train_data = fds.load_split("train")
    train_data = train_data.with_transform(applyTrainTransform)

    test_data = fds.load_split("test")
    test_data = test_data.with_transform(applyTestTransform)

    return train_data, test_data

def load_model(useResnet18):
    global simclr_predictor
    
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)

    list_of_files = [fname for fname in glob.glob("./centralized_weights/centralized_model_*.pth")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from:", latest_round_file)
    state_dict = torch.load(latest_round_file)
    
    simclr.load_state_dict(state_dict)
    
    weights = [v.cpu().numpy() for v in simclr.state_dict().values()]
    
    simclr_predictor.set_encoder_parameters(weights)

def centralized_train(useResnet18):
    global simclr_predictor
    simclr_predictor = SimCLRPredictor(utils.NUM_CLASSES, DEVICE, useResnet18=utils.useResnet18, tune_encoder=utils.fineTuneEncoder).to(DEVICE)
    
    trainset, testset = load_data()
    
    trainloader = DataLoader(trainset, batch_size = 128, shuffle = True)
    testloader = DataLoader(testset, batch_size = 128)   

    optimizer = optim.SGD(simclr_predictor.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    criterion = nn.CrossEntropyLoss()
    
    train(trainloader, testloader, optimizer, criterion, scheduler )
    
    
def train(trainloader, testloader, optimizer, criterion, scheduler ):

    maxAccuracy = 0
    
    iters = 0
    
    while maxAccuracy < 0.90:
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
            
        maxAccuracy = max(testAccuracy, maxAccuracy)
        
        if maxAccuracy == testAccuracy:
            acc = int(testAccuracy * 1000) / 1000
            save_model(acc)
            
        scheduler.step()
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
    
    if not os.path.isdir('centralized_weights'):
        os.mkdir('centralized_weights')
        
    torch.save(simclr_predictor.state_dict(), f"./centralized_weights/centralized_model_{acc}.pth")
    count += 1

if __name__ == "__main__":
    centralized_train(False)

