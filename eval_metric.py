import utils
from model import SimCLR
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

import torch.nn.functional as F


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
        
    
    def setAnchors(self, anchors):
        print('setanchors')
        self.anchors = anchors.to(DEVICE)
        
    def calcReferenceAnchorLatents(self):
        
        self.reference_model.eval()
        
        with torch.no_grad():
            # num_anchors * latentsize
            self.ref_anchor_latents = self.reference_model(self.anchors).to(DEVICE)
            self.ref_anchor_normed = F.normalize(self.ref_anchor_latents, p=2, dim=1)
            
            latents = self.ref_anchor_latents.matmul(self.ref_anchor_latents.T)
            


    
    def getAnchors(self):
        return self.anchors
    
    def calcModelLatents(self, model):
        model.eval()
        model.setInference(True)
        
        with torch.no_grad():
            #num_anchors * latentsize
            self.model_anchor_latents = model(self.anchors).to(DEVICE)
            self.model_anchor_normed = F.normalize(self.model_anchor_latents, p=2, dim=1)
            
            latents = self.model_anchor_latents.matmul(self.model_anchor_latents.T)
                        
    # def cos_similarity()
    
    def computeSimilarity(self, testbatch, model):
        testbatch = testbatch.to(DEVICE)
        
        
        
        print('computing sims....')
        
        #batchsize x latentsize
        abs_model_latent = model(testbatch)
        abs_reference_latent = self.reference_model(testbatch)
        
        abs_model_normed = F.normalize(abs_model_latent, p=2, dim=1).to(DEVICE)
        abs_reference_normed = F.normalize(abs_reference_latent, p=2, dim=1).to(DEVICE)
        
        
        
        relative_model_representations = [0] * 4
        relative_reference_representations = [0] * 4
        
        # no normalization: batchsize x anchor size
        relative_model_representations[0] = abs_model_latent @ (self.model_anchor_latents.T)
        relative_reference_representations[0] = abs_reference_latent @ (self.ref_anchor_latents.T)
        
        # normalize anchor latents: batchsize x anchor size
        relative_model_representations[1] = abs_model_latent @ (self.model_anchor_normed.T)
        relative_reference_representations[1] = abs_reference_latent @ (self.ref_anchor_normed.T)
        
        # no normalization: batchsize x anchor size
        relative_model_representations[2] = abs_model_normed @ (self.model_anchor_latents.T)
        relative_reference_representations[2] = abs_reference_normed @ (self.ref_anchor_latents.T)
        
        
        # normalize anchor latents: batchsize x anchor size
        relative_model_representations[3] = abs_model_normed @ (self.model_anchor_normed.T)
        relative_reference_representations[3] = abs_reference_normed @ (self.ref_anchor_normed.T)
        
        
        
        #comparision schemes: dot product, cosine similarity
        
        data = []
        

        
        
        for i, (model, reference) in enumerate(zip(relative_model_representations, relative_reference_representations)):
            dot_sim = torch.sum(model * reference, dim=1)
            cos_sim = F.cosine_similarity(model, reference, dim = 1)
            dot_data = [torch.mean(dot_sim).item(), torch.median(dot_sim).item(), torch.var(dot_sim).item()]
            cos_data = [torch.mean(cos_sim).item(), torch.median(cos_sim).item(), torch.var(cos_sim).item()]
            
            tensorized_data = torch.tensor([dot_data, cos_data])
            print(tensorized_data.shape)
            
            
            print(i, dot_data[0])
            print(i, cos_data[0])
            
            data.append(tensorized_data)

        
        data = torch.stack(data)
        print(data.shape)
        
        #return 4x2x3 matrix
        return data
 
    
        
        
        


        