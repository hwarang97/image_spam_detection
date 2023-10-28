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

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for images in train_loader:
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss += train_loss / len(train_loader.dataset)

# valdiataion
model.eval()
valid_loss = 0.0
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item() * images.size(0)

    valid_loss = valid_loss / len(valid_loader.dataset)

print(f"Epoch {epoch+1}/{num_epochs}, 
      Train Loss: {train_loss:.4f}, 
      Valid Loss: {valid_loss:.4f}")