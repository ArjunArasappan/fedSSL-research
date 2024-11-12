import flwr as fl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

import utils


class NTXentLoss(nn.Module):
    def __init__(self, device, temperature=0.5, ):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.batch_size = None
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        

    def forward(self, z_i, z_j):

        self.batch_size = z_i.size(0)
        
        z_i = F.normalize(z_i).to(self.device)
        z_j = F.normalize(z_j).to(self.device)
        
        z = torch.cat((z_i, z_j), dim=0).to(self.device)
        

        sim_matrix = torch.matmul(z, z.T).to(self.device) / self.temperature
        sim_matrix.fill_diagonal_(-float('inf'))
        

        labels = torch.arange(self.batch_size, 2 * self.batch_size, device=self.device)
        labels = torch.cat((labels, labels - self.batch_size))  
        
        loss = self.criterion(sim_matrix, labels)
        return loss


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.in_features = dim
        self.out_features = projection_size
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    def __init__(self, device, useResnet18, image_size=32, projection_size=2048, projection_hidden_size=4096, num_layer = 2) -> None:
        super(SimCLR, self).__init__()
        
        if useResnet18:
            self.encoder = resnet18(weights = None).to(device)
        else:
            self.encoder = resnet50(weights = None).to(device)


        self.encoded_size = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        self.projected_size = projection_size                
        self.proj_head = MLP(self.encoded_size, projection_size, projection_hidden_size).to(device)

        
        self.isInference = False
        
    def setInference(self, isInference):
        self.isInference = isInference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder(x)
        if self.isInference:
            return e1
        return self.proj_head(e1)

class SimCLRPredictor(nn.Module):
    def __init__(self, num_classes, device, useResnet18 = True, tune_encoder = False):
        super(SimCLRPredictor, self).__init__()
                
        self.simclr = SimCLR(device, useResnet18 = useResnet18).to(device)
        
        self.predictor = nn.Linear(self.simclr.encoded_size, num_classes)
        
        self.simclr.setInference(True)
        
        if tune_encoder:
            for param in self.simclr.parameters():
                param.requires_grad = True
        else:
            for param in self.simclr.parameters():
                param.requires_grad = False
                
    def set_encoder_parameters(self, weights):
        
        params_dict = zip(self.simclr.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.simclr.load_state_dict(state_dict, strict=True)
    
    def getLatent(self, x):
        self.simclr.setInference(True)
        return self.simclr(x)

    def forward(self, x):
        self.simclr.setInference(True)
        features = self.simclr(x)
        output = self.predictor(features)
        return output
    