import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import CNNModel
from data_loader import get_loaders
from sklearn.metrics import f1_score



def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_interval=1):
    
    # loss func, optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store value for plotting
    train_losses = []
    val_losses = []
    f1_scores = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device), labels.to(device)

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
        valid_preds, valid_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                valid_preds.extend(predicted.cpu().numpy())
                valid_targets.extend(labels.cpu().numpy())

            valid_loss = valid_loss / len(val_loader.dataset)
            f1 = f1_score(valid_targets, valid_preds, average='binary')

        print(f"Epoch {epoch+1}/{num_epochs}, 
            Train Loss: {train_loss:.4f}, 
            Valid Loss: {valid_loss:.4f}, 
            F1 Score: {f1:.4f}")
        
        if epoch % save_interval == 0:
            torch.save(model.stat_dict(), f'model_checkpoint_{epoch}.pth')

    # plotting the train and val loss
    plt.figure(figsize=(10,8))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train & Val Loss over Epochs')
    plt.legend()
    plt.show()

    # plotting the F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(f1_scores, label='F1 score (Val)')
    plt.xlabel('Epochs')
    plt.ylabel('F1 socre')
    plt.title('F1 score over epochs')
    plt.legend()
    plt.show()