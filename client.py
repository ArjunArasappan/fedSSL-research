import flwr as fl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset
import csv

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from transform import SimCLRTransform
from model import SimCLR, NTXentLoss, SimCLRPredictor
import utils



DEVICE = utils.DEVICE

def train(net, trainloader, optimizer, criterion, epochs):
    net.train()
    num_batches = len(trainloader)
    batch = 0
    total_loss = 0
    
    for epoch in range(epochs):

        for item in trainloader:
            x, x_i, x_j = item['img']
   
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
            x, x_i, x_j = item['img']
            
            x, x_i, x_j = x.to(DEVICE), x_i.to(DEVICE), x_j.to(DEVICE)
            
            z_i = net(x_i).to(DEVICE)
            z_j = net(x_j).to(DEVICE)
            loss = criterion(z_i, z_j)
            
            loss_epoch += loss.item()
            
            print("Client Train Batch:", batch, "/", num_batches)
            
            batch += 1
    return loss_epoch / (batch), -1


round = 0
class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid, simclr, trainset, testset, useResnet18, num_clients, loss):
        self.cid = cid
        self.simclr = simclr
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=3e-4)
        self.loss = loss
        
        self.useResnet18 = useResnet18
        self.num_clients = num_clients
        

        self.trainloader = DataLoader(trainset, batch_size = utils.BATCH_SIZE)
        
        if testset == None:
            self.testloader = None
        else:
            self.testloader = DataLoader(testset, batch_size = utils.BATCH_SIZE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.simclr.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.simclr.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.simclr.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        global round
        
        self.set_parameters(parameters)
        self.simclr.setInference(False)
        results = train(self.simclr, self.trainloader, self.optimizer, self.loss, epochs=1)
        
        data = ['client train', config['current_round'], self.useResnet18, self.num_clients, self.cid, results['Loss']]
        utils.sim_log(data)
        
        return self.get_parameters(config={}), len(self.trainloader), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.simclr.setInference(False)
        
        loss, accuracy = test(self.simclr, self.testloader, self.loss)

        return float(loss), 1, {"accuracy": accuracy}

def get_client_fn(fds, useResnet18, num_clients):
    
    ntxent = NTXentLoss( device=DEVICE).to(DEVICE)
    
    

    def client_fn(cid):
        clientID = int(cid)
        
        train, test = utils.load_partition(fds, clientID, client_test_split = 0)
        simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
        
        return CifarClient(clientID, simclr, train, test, useResnet18, num_clients, ntxent).to_client()
    
    return client_fn


