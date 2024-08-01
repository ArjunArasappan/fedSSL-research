import utils
from model import SimCLRPredictor
import glob
import os
import torch
from torch.utils.data import DataLoader
DEVICE = utils.DEVICE

class EvalMetric:
    def __init__(self, trainset, testset):
        self.reference = SimCLRPredictor(10, DEVICE, useResnet18=False).to(DEVICE)
        self.anchor_num = 100
        
        self.trainset, self.testset = trainset, testset
        
        
    def load_reference(self):
        
        state_dict = torch.load("./weights/centralized_model_0.901.pth")
        self.reference.load_state_dict(state_dict, strict = True)
        
    def evaluateReference(self):
        print(type(self.trainset))

        
        
        
        