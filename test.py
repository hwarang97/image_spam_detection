import torch
import torch.nn as nn
from model import CNNModel
from data_loader import get_loaders
from sklearn.metrics import f1_score

# hyper-parameters
device = torch.device('cuda' if torch.cuda.is_avalilable() else 'cpu')

# Load the model
model = CNNModel().to(device)
model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()

# load test dataset
test_loader = get_loaders('asdf')

# criterion
criterion = nn.BCELoss()

# eval on test set
test_loss = 0.0
test_preds, test_targets = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_loader.dataset)
f1 = f1_score(test_targets, test_preds, average='binary')

print(f"Test Loss: {test_loss:.4f}, F1 Score: {f1:.4f}")