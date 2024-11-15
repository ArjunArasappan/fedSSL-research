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
        stdev = torch.sqrt(torch.var(batch, dim = 0))
        
        normalized_batch = (batch - means)

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
        
        model.setInference(True)
        
        #batchsize x latentsize
        abs_model_latent = model(testbatch)
        abs_ref_latent = self.reference_model(testbatch)
        
        abs_model_latent = self.batch_normalize(abs_model_latent)
        abs_ref_latent = self.batch_normalize(abs_ref_latent)
        # norm_model_latent = abs_model_latent / torch.norm(abs_model_latent, p=2, dim=1, keepdim=True)
        
        # abs_model_latent = abs_model_latent / 


        
        # print(abs_ref_latent)
        # print("mean dim 0", torch.mean(abs_model_latent, dim = 0)[:20])
        # print("var dim 0", torch.var(abs_model_latent, dim = 0)[:20])
        # print("mean dim 1", torch.mean(abs_model_latent, dim = 1)[:20])
        # print("var dim 1", torch.var(abs_model_latent, dim = 1)[:20])
        
        # print("mean", torch.mean(abs_ref_latent).mean().item())
        # print("var", torch.var(abs_ref_latent).mean().item())

        
        # print(abs_model_latent[:10, :10])
        # print(abs_ref_latent[:10, :10])

        # norm_ref_latent = abs_ref_latent / torch.norm(abs_ref_latent, p=2, dim=1, keepdim=True)

        
        #batchsize x num_anchors
        relative_model = abs_model_latent @ self.model_anchor_latents.T
        relative_ref = abs_ref_latent @ self.ref_anchor_latents.T
        
        relative_model = relative_model / torch.norm(abs_model_latent, p=2, dim=1, keepdim=True)
        relative_ref = relative_ref / torch.norm(abs_ref_latent, p=2, dim=1, keepdim=True)
        
        # print(relative_model[:6, :6])
        # print(abs_model_latent[:6, :6])
        # print(self.model_anchor_normed.T[:6, :6])
        # print(torch.norm(abs_model_latent, p=2, dim=1, keepdim=True))
        
     

        
        # print(rel_model_normed[:6, :6])
        # print(rel_ref_normed[:6, :6])

        
        
        # similarities = torch.sum(relative_model * relative_ref, dim=1)
        similarities = F.cosine_similarity(relative_model, relative_ref, dim = 1)
        # print(similarities)
        return torch.tensor([torch.mean(similarities).item(), torch.var(similarities).item()])
        
        
        
        
        
        # print('computing sims....')
        
        # #batchsize x latentsize
        # abs_model_latent = model(testbatch)
        # abs_reference_latent = self.reference_model(testbatch)
        
        # abs_model_normed = F.normalize(abs_model_latent, p=2, dim=1).to(DEVICE)
        # abs_reference_normed = F.normalize(abs_reference_latent, p=2, dim=1).to(DEVICE)
        
        
        
        # relative_model_representations = [0] * 4
        # relative_reference_representations = [0] * 4
        
        # # no normalization: batchsize x anchor size
        # relative_model_representations[0] = abs_model_latent @ (self.model_anchor_latents.T)
        # relative_reference_representations[0] = abs_reference_latent @ (self.ref_anchor_latents.T)
        
        # # normalize anchor latents: batchsize x anchor size
        # relative_model_representations[1] = abs_model_latent @ (self.model_anchor_normed.T)
        # relative_reference_representations[1] = abs_reference_latent @ (self.ref_anchor_normed.T)
        
        # # no normalization: batchsize x anchor size
        # relative_model_representations[2] = abs_model_normed @ (self.model_anchor_latents.T)
        # relative_reference_representations[2] = abs_reference_normed @ (self.ref_anchor_latents.T)
        
        
        # # normalize anchor latents: batchsize x anchor size
        # relative_model_representations[3] = abs_model_normed @ (self.model_anchor_normed.T)
        # relative_reference_representations[3] = abs_reference_normed @ (self.ref_anchor_normed.T)
        
        
        
        # #comparision schemes: dot product, cosine similarity
        
        # data = []
        

        
        
        # for i, (model, reference) in enumerate(zip(relative_model_representations, relative_reference_representations)):
        #     dot_sim = torch.sum(model * reference, dim=1)
        #     cos_sim = F.cosine_similarity(model, reference, dim = 1)
        #     dot_data = [torch.mean(dot_sim).item(), torch.median(dot_sim).item(), torch.var(dot_sim).item()]
        #     cos_data = [torch.mean(cos_sim).item(), torch.median(cos_sim).item(), torch.var(cos_sim).item()]
            
        #     tensorized_data = torch.tensor([dot_data, cos_data])
        #     print(tensorized_data.shape)
            
            
        #     print(i, dot_data[0])
        #     print(i, cos_data[0])
            
        #     data.append(tensorized_data)

        
        # data = torch.stack(data)
        # print(data.shape)
        
        # #return 4x2x3 matrix
        # return data
 
    
        
        
        


        