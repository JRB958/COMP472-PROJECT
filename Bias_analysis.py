import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime

# Function to load datasets for age groups
def segment_dataset_by_age(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dictionary to hold datasets for each age group
    age_groups = {'young': [], 'mid': [], 'senior': []}
    
    for age_group in age_groups.keys():
        group_path = os.path.join(data_path, 'age', age_group)
        if os.path.exists(group_path):
            dataset = datasets.ImageFolder(root=group_path, transform=transform)
            age_groups[age_group] = dataset
    
    return age_groups

# Function to load datasets for gender groups
def segment_dataset_by_gender(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dictionary to hold datasets for each gender group
    gender_groups = {'male': [], 'female': []}
    
    for gender in gender_groups.keys():
        group_path = os.path.join(data_path, 'gender', gender)
        if os.path.exists(group_path):
            dataset = datasets.ImageFolder(root=group_path, transform=transform)
            gender_groups[gender] = dataset
    
    return gender_groups

# set path
data_path = 'C:\\Users\g_alave\Desktop\dataset\categories' 
age_groups = segment_dataset_by_age(data_path)
gender_groups = segment_dataset_by_gender(data_path)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64 * 56 * 56, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)  # assuming 4 classes for classification
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def evaluate_model_on_subset(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

# Load the best model
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_model_path = 'C:\\Users\g_alave\Desktop\\best_model\\best_model.pth'
model.load_state_dict(torch.load(best_model_path))

# Function to evaluate and collect metrics for a group
def collect_metrics_for_group(groups, attribute):
    results = []
    for group_key, dataset in groups.items():
        if dataset:
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
            accuracy, precision, recall, f1 = evaluate_model_on_subset(model, data_loader)
            results.append({
                'Attribute': attribute,
                'Group': group_key.capitalize(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                '#Images': len(dataset)
            })
    return results

# Collect metrics for age groups
age_results = collect_metrics_for_group(age_groups, 'Age')

# Collect metrics for gender groups
gender_results = collect_metrics_for_group(gender_groups, 'Gender')


# Convert results into a DataFrame
# Combine results
all_results = age_results + gender_results
bias_analysis_table = pd.DataFrame(all_results)

# Calculate totals for #Images and averages for Accuracy, Precision, Recall, F1-Score
group_totals = bias_analysis_table.groupby(['Attribute', 'Group']).agg({
    '#Images': 'sum',
    'Accuracy': lambda x: round(x.mean() * 100, 2),
    'Precision': lambda x: round(x.mean() * 100, 2),
    'Recall': lambda x: round(x.mean() * 100, 2),
    'F1-Score': lambda x: round(x.mean() * 100, 2)
}).reset_index()

# Append Total/Average row for each Attribute
totals = group_totals.groupby('Attribute').agg({
    '#Images': 'sum',
    'Accuracy': lambda x: round(x.mean(), 2),
    'Precision': lambda x: round(x.mean(), 2),
    'Recall': lambda x: round(x.mean(), 2),
    'F1-Score': lambda x: round(x.mean(), 2)
}).reset_index()
totals['Group'] = 'Total/Average'

bias_analysis_table = pd.concat([group_totals, totals], ignore_index=True)

# Function to display table using Matplotlib
def display_table(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=["#f2f2f2"]*len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"C:\\Users\g_alave\Desktop\\bias_analysis_table_{current_time}.png")
    plt.show()

# Display the table
display_table(bias_analysis_table)
