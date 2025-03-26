import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
data_dir = 'tugas/data_susu'  # dataset directory
batch_size = 32
num_epochs = 10
num_classes = 9
img_size = (224, 224)
validation_split = 0.2

# Data transforms (resizing and normalization, no augmentation)
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
dataset_size = len(dataset)
val_size = int(validation_split * dataset_size)
train_size = dataset_size - val_size

# Split dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load pretrained ResNet18 and modify final layer
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Prepare logging file
log_file = 'training_log.txt'
with open(log_file, 'w') as f:
    f.write("Training Log\n")

# Containers to hold training history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Variable to track the best validation accuracy
best_val_acc = 0.0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0
    
    # Training Phase
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total_train += labels.size(0)
    
    epoch_loss = running_loss / total_train
    epoch_acc = running_corrects / total_train
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data).item()
            total_val += labels.size(0)
    
    val_epoch_loss = val_running_loss / total_val
    val_epoch_acc = val_running_corrects / total_val
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)
    
    # Check if current model has the best validation accuracy so far
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.state_dict(), 'best_model.pth')
        best_model_message = " -- Best model saved"
    else:
        best_model_message = ""
    
    # Log epoch summary to txt file and print it
    log_message = (f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, "
                   f"Train Acc={epoch_acc:.4f}, Val Loss={val_epoch_loss:.4f}, "
                   f"Val Acc={val_epoch_acc:.4f}{best_model_message}\n")
    print(log_message)
    with open(log_file, 'a') as f:
        f.write(log_message)

# Print final summary
print("Training Completed. Summary:")
print("Train Losses:", train_losses)
print("Train Accuracies:", train_accuracies)
print("Validation Losses:", val_losses)
print("Validation Accuracies:", val_accuracies)

# Plotting training and validation curves
epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_summary.png')
plt.show()
