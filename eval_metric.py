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
        
    def batch_normalize(self, batch):
        #all features should be centered around 0
        
        means = torch.mean(batch, dim = 0)
        stdev = torch.std(batch, dim = 0)
        
        normalized_batch = (batch - means)
        # normalized_batch = normalized_batch / stdev
        # normalized_batch = torch.nan_to_num(normalized_batch, nan = 0)

        return normalized_batch
        
        
        
    def calcReferenceAnchorLatents(self):
        
        self.reference_model.eval()
        
        with torch.no_grad():
            # num_anchors * latentsize
            self.ref_anchor_latents = self.reference_model(self.anchors).to(DEVICE)
            self.ref_anchor_latents = self.batch_normalize(self.ref_anchor_latents)

            self.ref_anchor_latents = F.normalize(self.ref_anchor_latents, p = 2, dim = 1)

                        


    
    def getAnchors(self):
        return self.anchors
    
    def calcModelLatents(self, model):
        model.eval()
        model.setInference(True)
        
        with torch.no_grad():
            #num_anchors * latentsize
            self.model_anchor_latents = model(self.anchors).to(DEVICE)
            self.model_anchor_latents = self.batch_normalize(self.model_anchor_latents)
            self.model_anchor_latents = F.normalize(self.model_anchor_latents, p = 2, dim = 1)
                                    
    # def cos_similarity()
    
    def computeSimilarity(self, testbatch, model):
        
        
        
        
        
        testbatch = testbatch.to(DEVICE)
        
        # print(testbatch)
        
        model.eval()
        model.setInference(True)
        
        self.reference_model.eval()
        # self.reference_model.setInference(True)
        
        #batchsize x latentsize
        abs_model_latent = model(testbatch)
        abs_model_latent = self.batch_normalize(abs_model_latent)
        # norm_model_latent = abs_model_latent / torch.norm(abs_model_latent, p=2, dim=1, keepdim=True)

        abs_ref_latent = self.reference_model(testbatch)
        abs_ref_latent = self.batch_normalize(abs_ref_latent)

        # norm_ref_latent = abs_ref_latent / torch.norm(abs_ref_latent, p=2, dim=1, keepdim=True)
        
        # print(abs_model_latent[:10, :10])
        # print(abs_ref_latent[:10, :10])
        
        # print(abs_model_latent.mean(dim = 0))
        # print(abs_ref_latent.mean(dim = 0))


        
        #batchsize x num_anchors
        relative_model = abs_model_latent @ self.model_anchor_latents.T
        relative_ref = abs_ref_latent @ self.ref_anchor_latents.T

        rel_model_normed = relative_model / torch.norm(relative_model, p=2, dim=1, keepdim=True)
        rel_ref_normed = relative_ref / torch.norm(relative_ref, p=2, dim=1, keepdim=True)
        
        similarities = F.cosine_similarity(rel_model_normed, rel_ref_normed, dim=1)
        # print(similarities)
        return torch.tensor([torch.mean(similarities).item(), torch.median(similarities).item()])
        
        
        
        