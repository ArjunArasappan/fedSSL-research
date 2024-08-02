import utils
from model import SimCLRPredictor
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

DEVICE = utils.DEVICE

class EvalMetric:
    def __init__(self, trainset, testset, num_anchors = 300):
        self.reference_model = SimCLRPredictor(10, DEVICE, useResnet18=False).to(DEVICE)
        self.num_anchors = num_anchors
        self.cross_entropy = nn.CrossEntropyLoss()
        self.trainset = trainset
        self.testset = testset
         
        self.ref_anchor_latents = None
        self.model_anchor_latents = None
        
        
        self.eval_batches = 10
        
        
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
            batch = 0
            size = len(train)

            for item in train:
                (x, x_i, x_j), labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                logits = self.reference_model(x)
                values, predicted = torch.max(logits, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                batch += 1
                if batch == self.eval_batches:
                    break
                print(f"Eval Reference: {batch} / {self.eval_batches}")
                
            acc.append(correct / total)
            
            correct = 0
            total = 0
            batch = 0
            size = len(test)
            
            for item in test:
                x, labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                logits = self.reference_model(x)
                values, predicted = torch.max(logits, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if batch == self.eval_batches:
                    break
                batch += 1
                print(f"Eval Reference: {batch} / {self.eval_batches}")
                
            acc.append(correct / total)
            
        # print(f"Reference Train: loss - {acc[0][0]}")
        print(f"Reference Train: accuracy - {acc[0]}")
        print(f"Reference Test: accuracy - {acc[1]}")
    
    def selectAnchors(self):
        shuffled_train = self.trainset.shuffle(seed=42)

        anchors_list = shuffled_train.select(range(self.num_anchors))
        self.anchors = [anchor['img'][0] for anchor in anchors_list]
        
        self.anchors = torch.stack(self.anchors)
        self.anchors = self.anchors.to(DEVICE)
        #anchors have shape of (200, 3, 32, 32)
        
    def calcReferenceAnchorLatents(self):
        #caluclate latent representations of anchors for reference model and normalize latents
        self.reference_model.eval()
        
        
        with torch.no_grad():
            # num_anchors * latentsize
            self.ref_anchor_latents = self.reference_model.getLatent(self.anchors)


    
    def getAnchors(self):
        return self.anchors
    
    def calcModelLatents(self, model):
        model.eval()
        model.setInference(True)
        
        with torch.no_grad():
            #num_anchors * latentsize
            self.model_anchor_latents = model(self.anchors)
            
    
    
    def computeSimilarity(self, testbatch, model):
        testbatch = testbatch.to(DEVICE)
        
        model.setInference(True)
        
        #batchsize x latentsize
        abs_model_latent = model(testbatch)
        # norm_model_latent = abs_model_latent / torch.norm(abs_model_latent, p=2, dim=1, keepdim=True)

        abs_ref_latent = self.reference_model.getLatent(testbatch)
        # norm_ref_latent = abs_ref_latent / torch.norm(abs_ref_latent, p=2, dim=1, keepdim=True)

        
        #batchsize x num_anchors
        relative_model = abs_model_latent @ self.model_anchor_latents.T
        relative_ref = abs_ref_latent @ self.ref_anchor_latents.T

        rel_model_normed = relative_model / torch.norm(relative_model, p=2, dim=1, keepdim=True)
        rel_ref_normed = relative_ref / torch.norm(relative_ref, p=2, dim=1, keepdim=True)
    

        
        similarities = torch.sum(rel_model_normed * rel_ref_normed, dim=1)
        
        return list(similarities), torch.mean(similarities).item(), torch.median(similarities).item()
        
        
        
        
        
        