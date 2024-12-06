import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimCLR, SimCLRPredictor, NTXentLoss
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torchvision.transforms as transforms
from eval_metric import EvalMetric
import os
import timm

import flwr as fl
import utils
import csv

simclr = None

DEVICE = utils.DEVICE

EPOCHS = 1000
SEGMENTS = 1

#epochs to train linear
fine_tune_epochs = 1


finetune_fraction = 0.3

log_path = "./log.txt"

useLinearPred = True



def main(useResnet18):
    #Data: load augmented train data for SimCLR, test data, 
    #unsupervised SSL learning of simCLR
    #apply relative representations to guess accuracy
    #find actualy representation accuracy with linear predictors & MLP's on frozen encoder
    
        

        
    fds = utils.get_anchored_fds(1, 300)
    anchor_data, _ = utils.load_partition(fds, 1, split = 'train', apply_augment = False)
    test_data, _ = utils.load_partition(fds, 0, split = 'test', apply_augment = False)
    train_data, _ = utils.load_partition(fds, 0, split = 'train', apply_augment = True)

    
    
    #get batch
    
    anchorloader = DataLoader(dataset=anchor_data, batch_size = 300)
    testloader = DataLoader(dataset=test_data, batch_size = 512)

    for batch in anchorloader:
        anchor_data = batch['img']
    

    
    #setup eval metric
    
    reference_path = '/home/harsh/arjun/fedSSL-research/log_files/ssl_centralized_model_csa_1225.pth'
    
    reference = SimCLR(DEVICE, useResnet18=False).to(DEVICE)
    state_dict = torch.load(reference_path)
    reference.load_state_dict(state_dict, strict = True)
    

    
        

    relative_eval_metric = EvalMetric(reference)
        
    relative_eval_metric.setAnchors(anchor_data)
    relative_eval_metric.calcReferenceAnchorLatents()
    
    ssl_simulation(train_data, test_data, useResnet18, relative_eval_metric)
    
def load_model():
    simclr = SimCLR(DEVICE, useResnet18=False).to(DEVICE)
    
    reference_path = '/home/harsh/arjun/fedSSL-research/reference_models/ssl_centralized_new_390.pth'
    
    reference = SimCLR(DEVICE, useResnet18=False).to(DEVICE)
    state_dict = torch.load(reference_path)
    reference.load_state_dict(state_dict, strict = True)
    
    return reference

    
    # Load the default ResNet50 model
    model = timm.create_model("resnet50", pretrained=False)

    # Replace the first convolutional layer with a 3x3 kernel
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,  # Adjust stride for smaller inputs (e.g., CIFAR-10)
        padding=1,
        bias=False
    )

    # Load the pre-trained weights non-strictly to account for the `conv1` change
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet50_simclr_cifar10/resolve/main/pytorch_model.bin"
    )
    model.load_state_dict(state_dict, strict=False)  # Ignore mismatch in layers like `conv1`


    model.fc = nn.Identity()
    model.eval()


    return model.to(DEVICE)
    
    
    
def ssl_simulation(trainset, testset, useResnet18, relative_eval):
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    
    
    
    simclr = load_model()
    
    
    simclr_predictor = SimCLRPredictor(10, DEVICE, useResnet18=useResnet18, tune_encoder = False).to(DEVICE)

    simclr_optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4)
    predictor_optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)

    
    ntxent = NTXentLoss(device=DEVICE)
    cross_entropy = nn.CrossEntropyLoss()
    
    
    trainloader = DataLoader(trainset, batch_size = 512, shuffle = True)
    testloader = DataLoader(testset, batch_size = 1024, shuffle = True)
    

    
    utils.sim_log(["SEGMENTS", "epoch", "mean", "median", "loss", "accuracy"], path = '/home/harsh/arjun/fedSSL-research/log_files/centralized_results.csv')


    for epoch in range(EPOCHS * SEGMENTS):
        
        # simclr.setInference(True)
        simclr.eval()
        
        mean, median = computeSimilarities(testloader, simclr, relative_eval)

        supervised_train(simclr, simclr_predictor, trainloader, predictor_optimizer, cross_entropy)
        loss, accuracy = supervised_test(simclr_predictor, testloader, cross_entropy)
        
        train(simclr, trainloader, simclr_optimizer, ntxent)
        
        utils.sim_log([SEGMENTS, epoch, mean.item(), median.item(), loss, accuracy], path = '/home/harsh/arjun/fedSSL-research/log_files/centralized_results.csv')
        
    

def train(net, trainloader, optimizer, criterion):
    net.train()
    
    num_batches = len(trainloader)
    batch = 0
    
    for item in trainloader:
        _, x_i, x_j = item['img']
        
        optimizer.zero_grad()
        
        x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
        
        z_i, z_j = net(x_i), net(x_j)
    
        loss = criterion(z_i, z_j)

        loss.backward()
        optimizer.step()

        print(f"Train Batch: {batch} / {num_batches}")
        batch += 1
        
        if batch >= num_batches / SEGMENTS:
            break
        
    
def computeSimilarities(testloader, simclr, relative_eval):
    
    
    means, medians = [], []
    batch = 0
    relative_eval.calcModelLatents(simclr)
    for item in testloader:
        
        x = item['img']
        
        x = x.to(DEVICE)
        

        mean, median = relative_eval.computeSimilarity(x, simclr)
        print(f"Computing sims {batch}/{len(testloader)}: {mean}, {median}")
        means.append(mean)
        medians.append(median)
        batch += 1
        
    mean, median = torch.mean(torch.Tensor(means)), torch.median(torch.Tensor(medians))
    print(f"Relative Eval DONE: {mean}, {median}")

    return mean, median

def supervised_train(simclr, simclr_predictor, trainloader, optimizer, criterion):
    state_dict = simclr.state_dict()
    weights = [v.cpu().numpy() for v in state_dict.values()]

    simclr_predictor.set_encoder_parameters(weights)
    
    simclr_predictor.train()
    
    num_batches = len(trainloader)

    for i in range(fine_tune_epochs):
        idx = 0    
        for item in trainloader:
            (x, _, _), labels = item['img'], item['label']
            
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            z = simclr_predictor(x)
        
            loss = criterion(z, labels)

            loss.backward()
            optimizer.step()

            print(f"Client Train Batch: {idx} / {num_batches}")
            idx += 1
        
def supervised_test(simclr_predictor, testloader, criterion):
    simclr_predictor.eval()
    
    total = 0
    correct = 0
    loss = 0

    batch = 0
    num_batches = len(testloader)

    with torch.no_grad():
        for item in testloader:
            x , labels = item['img'], item['label']
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            logits = simclr_predictor(x)
            values, predicted = torch.max(logits, 1)  
            
            total += labels.size(0)
            loss += criterion(logits, labels).item()
            
            correct += (predicted == labels).sum().item()
            # if batch >= num_batches / 2:
            #     break
            print(f"Test Batch: {batch} / {num_batches}")
            batch += 1
  
    return loss / batch, correct / total
              

    

def save_model(acc):
    global count
    
    if not os.path.isdir('weights_nosched'):
        os.mkdir('weights_nosched')
        
    torch.save(simclr_predictor.state_dict(), f"/home/harsh/arjun/fedSSL-research/log_files/centralized_model_{acc}.pth")
    count += 1

if __name__ == "__main__":
    main(False)

