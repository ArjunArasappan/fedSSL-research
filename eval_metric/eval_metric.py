import utils
from model import SimCLR
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

# import detectors
# import timm

DEVICE = utils.DEVICE

class EvalMetric:
    def __init__(self, ref_model, num_anchors = 300):
        self.reference_model = ref_model
        self.num_anchors = num_anchors
         
        self.ref_anchor_latents = None
        self.model_anchor_latents = None
        
        self.eval_batches = 10
        
    
    def selectAnchors(self, trainset):
        shuffled_train = trainset.shuffle(seed=42)

        anchors_list = shuffled_train.select(range(self.num_anchors))
        self.anchors = [anchor['img'][0] for anchor in anchors_list]
        
        anchors = torch.stack(self.anchors)
        return anchors
    
    def setAnchors(self, anchors):
        self.anchors = anchors
        
    def calcReferenceAnchorLatents(self):
        
        self.reference_model.eval()
        
        with torch.no_grad():
            # num_anchors * latentsize
            self.ref_anchor_latents = self.reference_model(self.anchors).to(DEVICE)


    
    def getAnchors(self):
        return self.anchors
    
    def calcModelLatents(self, model):
        model.eval()
        model.setInference(True)
        
        with torch.no_grad():
            #num_anchors * latentsize
            self.model_anchor_latents = model(self.anchors).to(DEVICE)
            
    
    def computeSimilarity(self, testbatch, model):
        testbatch = testbatch.to(DEVICE)
        
        #batchsize x latentsize
        abs_model_latent = model(testbatch)
        # norm_model_latent = abs_model_latent / torch.norm(abs_model_latent, p=2, dim=1, keepdim=True)

        abs_ref_latent = self.reference_model(testbatch)
        # norm_ref_latent = abs_ref_latent / torch.norm(abs_ref_latent, p=2, dim=1, keepdim=True)

        
        #batchsize x num_anchors
        relative_model = abs_model_latent @ self.model_anchor_latents.T
        relative_ref = abs_ref_latent @ self.ref_anchor_latents.T

        #normalize relative representations

        rel_model_normed = relative_model / torch.norm(relative_model, p=2, dim=1, keepdim=True)
        rel_ref_normed = relative_ref / torch.norm(relative_ref, p=2, dim=1, keepdim=True)
        
        print(torch.norm(rel_model_normed, p=2, dim=1, keepdim=True))
        print(torch.norm(rel_ref_normed, p=2, dim=1, keepdim=True))
        
        similarities = torch.sum(relative_model * relative_ref, dim=1)
        print(similarities)

        similarities_normed = torch.sum(rel_model_normed * rel_ref_normed, dim=1)
        print(similarities_normed)

        
        sim_data = [torch.mean(similarities).item(), torch.median(similarities).item(), torch.var(similarities).item()]
        sim_norm_data = [torch.mean(similarities_normed).item(), torch.median(similarities_normed).item(), torch.var(similarities_normed).item()]

        
        
        return sim_data, sim_norm_data
        
        
        
        
        
        