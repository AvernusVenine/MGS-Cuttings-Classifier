import tempfile
from pathlib import Path
import ImagePreprocessor, dataset
from torch.optim import SGD
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss, Linear
from torchvision.models import resnet50 # Change as needed for testing
import torchvision.transforms as T
import ray.tune as tune
from ray.tune.search.optuna import OptunaSearch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchNorm2d(x)
        x = self.relu(x)
        return x

class ClassificationCNN(nn.Module):
    def __init__(self, n_classes=2):
        super(ClassificationCNN, self).__init__()

        self.convBlock1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=0)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Inception Layer 1
        #self.

        self.dense = nn.LazyLinear(out_features=n_classes)

def train_loop(dataloader, model, loss_func, optimizer):
    model.train()

    for _, (X, y) in enumerate(dataloader):
        # Forward Propagation
        pred = model(X)
        loss = loss_func(pred, y)

        #Backward Propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_func):
    model.eval()

    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss = test_loss / len(dataloader)
    accuracy = accuracy / len(dataloader.dataset)

    print(f'Accuracy: {100 * accuracy} | Test Loss: {test_loss}')
    return test_loss, accuracy

def train_model(max_epochs : int = 10, batch_size : int = 64, lr : float = 1e-4, momentum : float = 0.9, l2 = 0):
    mean, std = (0.4435, 0.4484, 0.4092), (0.1288, 0.1286, 0.1236)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        A.ColorJitter(brightness=.1, contrast=.1),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5),
        ToTensorV2()
    ])

    data = ImageFolder(root=f'grain_data/', transform=lambda img: transform(image=np.array(img))["image"])

    train_data, test_data = random_split(data, [.8, .2])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = resnet50()

    n_features = model.fc.in_features
    model.fc = Linear(n_features, 2)
    model.load_state_dict(torch.load('trained_classification_model.pth'))

    loss_func = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2)

    best_loss = float('inf')
    patience = 10
    triggers = 0

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}')
        train_loop(train_loader, model, loss_func, optimizer)
        loss, accuracy = test_loop(test_loader, model, loss_func)

        print(f'Loss: {loss} | Accuracy: {accuracy}')

        if loss < best_loss:
            best_loss = loss
            triggers = 0

            torch.save(model.state_dict(), 'trained_model.pth')
        else:
            triggers += 1

            if triggers >= patience:
                print(f'No improvement for {patience} epochs.  Stopping early with {best_loss}')
                break