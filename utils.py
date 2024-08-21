import torch
from transform import SimCLRTransform
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torchvision.transforms as transforms

import csv

ssl_transform = SimCLRTransform()

NUM_CLASSES = 10
useResnet18 = False
fineTuneEncoder = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_path = './ssl_log.txt'

def get_mean_stddev(data):
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    
    return mean, std

def applyTrainTransform(batch):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        
    batch["img"] = [transform_train(img) for img in batch["img"]]
    return batch

def applyTestTransform(batch):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        
    batch["img"] = [transform_test(img) for img in batch["img"]]
    return batch

def applySSLAugment(batch):
    batch["img"] = [ssl_transform(img) for img in batch["img"]]
    return batch

# def load_central_test():
#     fds = FederatedDataset(dataset="cifar10")
    
#     train_data = fds.load_split("train")

#     train_data = train_data.with_transform(applyTrainTransform)

#     test_data = fds.load_split("test")
#     test_data = test_data.with_transform(applyTestTransform)
    
#     return train_data, test_data


def load_data():
    fds = FederatedDataset(dataset="cifar10", partitioners = {'train' : IidPartitioner(1), 'test' : IidPartitioner(1)})
    
    train_data = fds.load_split("train")
    train_data = train_data.with_transform(applyTrainTransform)

    test_data = fds.load_split("test")
    test_data = test_data.with_transform(applyTestTransform)


    return train_data, test_data

def load_augmented():
    fds = FederatedDataset(dataset="cifar10", partitioners = {'train' : IidPartitioner(1), 'test' : IidPartitioner(1)})
    
    train_data = fds.load_split("train")
    train_data = train_data.with_transform(applySSLAugment)

    test_data = fds.load_split("test")
    test_data = test_data.with_transform(applyTestTransform)
    

    return train_data, test_data

def ssl_log(data, path = log_path):
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        
def sim_log(data):
    with open('./sim_log.txt', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
