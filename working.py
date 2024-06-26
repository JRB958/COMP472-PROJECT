# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:42:57 2024

@author: g_alave
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
import numpy as np

data_path = 'C:\\Users\\g_alave\\Desktop\\COMP472-PROJECT\\yui'
best_model_path = 'best_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_layer = nn.Sequential(
            
            # First convolutional layer: input (3x224x224), output (32x224x224)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            # Second convolutional layer: input (32x224x224), output (32x224x224)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            # First subsampling (MaxPooling) layer: input (32x224x224), output (32x112x112)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional layer: input (32x112x112), output (64x112x112)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            # Second subsampling (MaxPooling) layer: input (64x112x112), output (64x56x56)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
      
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            
            # First fully connected layer: input (64*56*56), output (1000)
            nn.Linear(64*56*56, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            
            # Second fully connected layer: input (1000), output (512)
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            
            # Third fully connected layer: input (512), output (4) - assuming 4 classes for classification
            nn.Linear(512, 4)
        )
        
    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_layer(x)
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers
        x = self.fc_layer(x)
        return x


# Load and preprocess your dataset
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_path, transform=transform)
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
    model = CNN()
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
        print("EPOCH ",epoch)
        for i, data in enumerate(train_loader, 0):
            print("i ",i)
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
