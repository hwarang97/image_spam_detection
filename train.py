import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNModel
from data_loader import get_loaders

# hyper-parameters
laerning_rate = 0.0001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_avalilable() else 'cpu')

# load data
train_loader = get_loaders('asdf')
valid_loader = get_loaders('asdf')

# model, loss func, optimizer
model = CNNModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=laerning_rate)