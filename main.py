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
batch_size = 32
save_interval = 1

def main():
    # load data
    train_data, val_data, test_data = split_dataset()
    train_loader, val_loader, test_loader = get_loaders(train_data, val_data, test_data, batch_size=batch_size)

    # init modle
    model = CNNModel().to(device)

    # train, val
    train_model(model, train_loader, val_loader, num_epochs, laerning_rate, device)

    # test
    test_model(model, test_loader, device)

if __name__ == '__main__':
    main()