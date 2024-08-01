import utils
from model import SimCLRPredictor
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
DEVICE = utils.DEVICE

class EvalMetric:
    def __init__(self, trainset, testset):
        self.reference_model = SimCLRPredictor(10, DEVICE, useResnet18=False).to(DEVICE)
        self.anchor_num = 100
        self.cross_entropy = nn.CrossEntropyLoss()
        self.trainset = trainset
        self.testset = testset 
        
        
    def load_reference(self):
        
        state_dict = torch.load("./weights/centralized_model_0.901.pth")
        self.reference_model.load_state_dict(state_dict, strict = True)
        
    def evaluateReference(self):
        train = DataLoader(self.trainset, batch_size = 512)
        test = DataLoader(self.testset, batch_size = 512)
        acc = []
        
        self.reference_model.eval()
        
        with torch.no_grad():
            correct = 0
            total = 0

            for item in train:
                (x, x_i, x_j), labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                logits = self.reference_model(x)
                values, predicted = torch.max(logits, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                
                print(f"Eval Reference: {batch} / {size}")
                
            acc.append(correct / total)
            
            correct = 0
            total = 0
            
            for item in test:
                x, labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                logits = self.reference_model(x)
                values, predicted = torch.max(logits, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                
                print(f"Eval Reference: {batch} / {size}")
                
            acc.append(correct / total)
            
        # print(f"Reference Train: loss - {acc[0][0]}")
        print(f"Reference Train: accuracy - {acc[0]}")
        print(f"Reference Test: accuracy - {acc[1]}")

        
        
        
        