import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
from model import SimCLR, SimCLRPredictor
import os, glob


DEVICE = utils.DEVICE



def evaluate_gb_model():
    simclr_predictor = SimCLRPredictor(10, DEVICE, useResnet18=False, tune_encoder=False).to(DEVICE)
    
    file = load_model(simclr_predictor)
    
    utils.ssl_log([file])
    
    train, test = utils.load_data()
    
    trainloader = DataLoader(train, batch_size = 512)
    testloader = DataLoader(test, batch_size = 512)   

    optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_epochs = 20
    
    fine_tune_predictor(simclr_predictor, trainloader, optimizer, criterion, train_epochs)
    
    loss, accuracy = evaluate(simclr_predictor, testloader, criterion)
    
    data = [file, loss, accuracy]
    
    utils.ssl_log(data)
    


    return loss, accuracy


def load_model(simclr_predictor):
    simclr = SimCLR(DEVICE, useResnet18=False).to(DEVICE)

    
    list_of_files = [fname for fname in glob.glob("./ssl_centralized/ssl_centralized_model_csa_*.pth")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    # latest_round_file = '/root/fedSSL-research/ssl_centralized/checkpoint_round_1000.pth'
    print("Loading pre-trained model from:", latest_round_file)
    state_dict = torch.load(latest_round_file)
    
    simclr.load_state_dict(state_dict)
    
    weights = [v.cpu().numpy() for v in simclr.state_dict().values()]
    
    simclr_predictor.set_encoder_parameters(weights)
    
    return latest_round_file


def fine_tune_predictor(simclr_predictor, trainloader, optimizer, criterion, train_epochs):
    simclr_predictor.train()

    for epoch in range(train_epochs):
        batch = 0
        num_batches = len(trainloader)

        for item in trainloader:
            x , labels = item['img'], item['label']
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = simclr_predictor(x)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch: {epoch} Predictor Train Batch: {batch} / {num_batches}")
            batch += 1

def evaluate(simclr_predictor, testloader, criterion):
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

            print(f"Test Batch: {batch} / {num_batches}")
            batch += 1
  
    return loss / batch, correct / total

if __name__ == "__main__":

    loss, accuracy = evaluate_gb_model()
    print(f"Loss: {loss}, Accuracy: {accuracy}")
