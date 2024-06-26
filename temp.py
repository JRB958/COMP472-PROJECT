import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
import numpy as np

# Define your CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess your dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='path/to/your/data', transform=transform)
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize results array
results = {}

# Early stopping parameters
patience = 3

# K-fold Cross Validation model evaluation
for fold, (train_val_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Create the train+val and test sets
    train_val_subsampler = Subset(dataset, train_val_ids)
    test_subsampler = Subset(dataset, test_ids)  # This line ensures the test set changes

    # Further split the train_val_subsampler into training and validation sets
    train_size = int(0.85 * len(train_val_subsampler))
    val_size = len(train_val_subsampler) - train_size
    train_subsampler, val_subsampler = random_split(train_val_subsampler, [train_size, val_size])

    # Define data loaders for training, validation, and testing data in this fold
    train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subsampler, batch_size=32, shuffle=False)

    # Initialize model
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Train the model
    num_epochs = 50  # max number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}')

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

        # Check early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                break

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy for fold {fold}: {accuracy}%')
    results[fold] = accuracy

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum / len(results.items())} %')
