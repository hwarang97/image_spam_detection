import torch
from data_loader import get_loaders
from data_loader import split_dataset
from model import CNNModel
from train import train_model
from test import test_model

# hyper-parameters
laerning_rate = 0.0001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_avalilable() else 'cpu')
save_interval = 1

# load data
train_loader = get_loaders('asdf')
valid_loader = get_loaders('asdf')