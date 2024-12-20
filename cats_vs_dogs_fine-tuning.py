# Import necessary libraries
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import drive
drive.mount('/content/drive')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize image to 224x224
    transforms.Lambda(lambda img: img.convert("RGB")), # convert all images to RGB
    transforms.ToTensor(), # image normalization to tensor values from 0 to 1
])

# Load and transform images from a folder 
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"): # extract all jpg files
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = transform(img) # image processing
            images.append(img)
    return torch.stack(images) # stack all images as a tensor for vectorization
    
# Load path to my data (folder is subdivided into two folders of train and tesr)
base_path = "/content/drive/My Drive/..." # put your file directory from google drive

# For train set (train folder is subdivided into two folders of cats and dogs)
train_cats_path = os.path.join(base_path, "train/cats")
train_dogs_path = os.path.join(base_path, "train/dogs")

# For test set (train folder is subdivided into two folders of cats and dogs)
test_cats_path = os.path.join(base_path, "test/cats")
test_dogs_path = os.path.join(base_path, "test/dogs")

# Load tensors for train and test sets
train_cats_tensors = load_images_from_folder(train_cats_path)
train_dogs_tensors = load_images_from_folder(train_dogs_path)
test_cats_tensors = load_images_from_folder(test_cats_path)
test_dogs_tensors = load_images_from_folder(test_dogs_path)

# Create labels (0 for cat and 1 for dog)
train_labels = torch.cat([torch.zeros(train_cats_tensors.size(0)), torch.ones(train_dogs_tensors.size(0))])
test_labels = torch.cat([torch.zeros(test_cats_tensors.size(0)), torch.ones(test_dogs_tensors.size(0))])

# Concatenate cats and dogs tensor for train and test sets as well as their labels
train_dataset = TensorDataset(torch.cat([train_cats_tensors, train_dogs_tensors]), train_labels)
test_dataset = TensorDataset(torch.cat([test_cats_tensors, test_dogs_tensors]), test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet-50 model
model = models.resnet50(pretrained=True)

# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Modify the fully connected layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),  # Change output to 1 feature for binary classification
    nn.Sigmoid()  # Apply sigmoid to get probabilities
)

# Move model to GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0005)

# Lists to store accuracy for each epoch
train_accuracies = []
test_accuracies = []

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward ppropagation
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward propagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy on the train data
        predicted = (outputs > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100*correct_train/total_train
    train_accuracies.append(train_accuracy)

    # Evaluate on test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    
# Plotting training and testing accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Save the model for website deployment
torch.save(model.state_dict(), '/content/drive/My Drive/cats and dogs/model.pth')