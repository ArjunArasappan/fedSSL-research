import torch

NUM_CLASSES = 10
useResnet18 = False
fineTuneEncoder = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
