import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torchvision.datasets as datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import metrics
import time
import re
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Hyperparameters and settings
batch_size = 32
num_epochs = 10
learning_rate = 0.001
patience = 5
random_seed = 42

# Global vars
data_path = './Dataset'
best_model_path = 'best_model.pth'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Create datasets for training and testing
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Function to load datasets
def custom_dataset_loader(batch_size):
    # Split dataset into training, validation, and testing
    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(random_seed))
    
    # Create data loaders for training and testing
    train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = td.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader

# Load datasets
train_loader, valid_loader, test_loader = custom_dataset_loader(batch_size)

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

# Training and validation function
def train_and_validate_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, best_model_path):
    best_val_loss = float('inf') # Tracks the best (lowest) validation loss observed so far. Initialized to infinity.
    patience_counter = 0 # Counts the number of consecutive epochs during which the validation loss has not improved.
    
    # main training loop
    for epoch in range(num_epochs):
        
        # model Training
        # **************
        
        model.train() # Sets the model to training mode.
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        
        # batch loop iterates through batches of training data
        for i, (images, labels) in enumerate(train_loader):        
            outputs = model(images) # forward pass, outputs are the predictions of the model
            loss = criterion(outputs, labels) # loss between the predictions and the actual labels
            loss_list.append(loss.item())
            
            # backpropagation 
            optimizer.zero_grad() # clears the gradients of all model parameters.
            loss.backward() # computes the gradients of the loss with respect to the model parameters.
            optimizer.step() # updates the model parameters using the computed gradients.
            
            # Calculate training accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total) # not used
            
            # Print training status every 5 steps
            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))
    
        # model Evaluation
        # ****************
        
        model.eval() # sets the model to evaluation mode. This is crucial as it turns off certain layers like 
                     # dropout and batch normalization, which behave differently during training and inference.
                     
        with torch.no_grad(): # disables gradient calculation, reducing memory consumption and speeding up computations.
            val_loss = 0
            correct = 0
            total = 0
            
            # batch loop iterates through batches of validation data
            for images, labels in valid_loader:
                outputs = model(images) # forward pass, outputs are the predictions of the model
                loss = criterion(outputs, labels) # # loss between the predictions and the actual labels
                val_loss += loss.item() # accumulates the total validation loss
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            val_loss /= len(valid_loader) # averages the loss over the entire validation set
            accuracy = (correct / total) * 100 # calculate validation accuracy
        
        if epoch > 3: 
            if val_loss < best_val_loss: # the validation loss is stored if it's the lowest as the best loss and the model is saved as best model
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else: # if it's not the best, the counter increases up to the patience parameter
                patience_counter += 1
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f} %, Best Validation Loss: {best_val_loss:.4f}')
        
        if patience_counter >= patience:
            print("Early stopping")
            break
        
        # add a wait period to allow the device to recover
        if (num_epochs > 1 ):
            time.sleep(300)


# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in data_loader:        
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return all_labels, all_preds

def clean_class_names(classes):
    # Use regular expression to remove numbers and brackets
    return [re.sub(r'\(\d+\)', '', class_name).strip() for class_name in classes]

def plot_confusion_matrix(labels, preds, classes):
    # Clean the class names
    classes = clean_class_names(classes)
    
    cm = metrics.confusion_matrix(labels, preds)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    # Plot the confusion matrix with additional customization
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display.plot(ax=ax, cmap="Blues")
    
    # Add labels and title
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    
    # Ensure labels are set correctly
    ax.set_xticklabels(classes, rotation=0, ha="right")
    ax.set_yticklabels(classes, rotation=0)
    
    plt.show()

def evaluate_bias(model, dataset, attribute_indices, attribute_names):
    model.eval()
    results = []
    
    for attribute_idx, attribute_name in zip(attribute_indices, attribute_names):
        subset_indices = [i for i, (_, attr) in enumerate(dataset.samples) if attr[attribute_idx] == attribute_name]
        subset = td.Subset(dataset, subset_indices)
        loader = td.DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        labels, preds = evaluate_model(model, loader)
        
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        
        results.append({
            'Attribute': attribute_name,
            'Macro P': precision,
            'Macro R': recall,
            'Macro F': f1,
            'Accuracy': accuracy
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(f'bias_analysis_{attribute_names[0]}.csv', index=False)


def k_fold_cross_validation(model, dataset, num_epochs, patience, best_model_path, k=10):
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{k}')
        
        train_subset = td.Subset(dataset, train_idx)
        val_subset = td.Subset(dataset, val_idx)
        
        train_loader = td.DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = td.DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, best_model_path)
        
        model.load_state_dict(torch.load(best_model_path))
        labels, preds = evaluate_model(model, val_loader)
        
        accuracy = accuracy_score(labels, preds)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
        
        results.append({
            'Fold': fold + 1,
            'Macro P': precision_macro,
            'Macro R': recall_macro,
            'Macro F': f1_macro,
            'Micro P': precision_micro,
            'Micro R': recall_micro,
            'Micro F': f1_micro,
            'Accuracy': accuracy
        })
    
    results_df = pd.DataFrame(results)
    results_df.loc['Average'] = results_df.mean()
    print(results_df)
    results_df.to_csv('k_fold_performance_metrics.csv', index=False)


# Main code execution
if __name__ == '__main__':
    model = CNN();

    k_fold_cross_validation(model, full_dataset, num_epochs, patience, best_model_path, k=10)
    evaluate_bias(model, full_dataset, [0], ['male', 'female'])

