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
        self.reference = SimCLRPredictor(10, DEVICE, useResnet18=False).to(DEVICE)
        self.anchor_num = 100
        self.cross_entropy = nn.CrossEntropyLoss()
        self.trainset, self.testset = trainset, testset
        
        
    def load_reference(self):
        
        state_dict = torch.load("./weights/centralized_model_0.901.pth")
        self.reference.load_state_dict(state_dict, strict = True)
        
    def evaluateReference(self):
        train = DataLoader(self.trainset, batch_size = len(trainset))
        test = DataLoader(self.testset, batch_size = len(trainset))
        acc = []
        
        reference.eval()
        with torch.no_grad():
            for item in [train, test]:
                x, labels = item[0]['img'], item[0]['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                logits = self.reference(x)
                values, predicted = torch.max(logits, 1)  
                
                total = labels.size(0)
                loss = self.cross_entropy(logits, labels).item()
                correct = (predicted == labels).sum().item()

                acc.append((loss, correct / total))
            
        # print(f"Reference Train: loss - {acc[0][0]}")
        print(f"Reference Train: accuracy - {acc[0][1]}")
        print(f"Reference Test: accuracy - {acc[1][1]}")

        
        
        
        