import os
from torch.optim import SGD
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn import CrossEntropyLoss, Linear
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
import torchvision.transforms as T
import ray.tune as tune
from ray.tune.search.optuna import OptunaSearch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn

class GrainDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []


    def __getitem__(self, idx):

        pass


class ClassificationModel(nn.Module):
    def __init__(self, in_features):
        super(ClassificationModel, self).__init__()

        model = resnet18()
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()

        self.age = nn.Linear(in_features, 4) # Precambrian, Paleozoic, Cretaceous, Other

        self.other_type = nn.Linear(in_features, 4) # Chert, Unknown, Secondary, Gypsum

        self.cretaceous_type = nn.Linear(in_features, 2) # Shale, Non-Shale
        self.shale_type = nn.Linear(in_features, 2) # Gray Shale, Speckled Shale
        self.cretaceous_other_type = nn.Linear(in_features, 7)

        self.paleozoic_type = nn.Linear(in_features, 2) # Carbonate, Sandstone Shale

        self.precambrian_type = nn.Linear(in_features, 2) # Crystalline, Other
        self.crystalline_type = nn.Linear(in_features, 3) # Light, Dark, Red
        self.light_type = nn.Linear(in_features, 3)
        self.dark_type = nn.Linear(in_features, 2)
        self.red_type = nn.Linear(in_features, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)

        return {
            'age' : self.age(x),
            'other_type' : self.other_type(x),
            'cretaceous_type' : self.cretaceous_type(x),
            'shale_type' : self.shale_type(x),
            'cretaceous_other_type' : self.cretaceous_other_type(x),
            'paleozoic_type' : self.paleozoic_type(x),
            'precambrian_type' : self.precambrian_type(x),
            'crystalline_type' : self.crystalline_type(x),
            'light_type' : self.light_type(x),
            'dark_type' : self.dark_type(x),
            'red_type' : self.red_type(x)
        }

def compute_loss(outputs, targets):
    loss = 0

    age = targets['age']

    age_loss = F.cross_entropy(outputs['age'], targets['age'])
    loss += age_loss

    # Other Grains Loss
    other_mask = (age == 0)
    if other_mask.any():
        loss += F.cross_entropy(outputs['other_type'][other_mask],
                                targets['other_type'][other_mask])

    # Cretaceous Grains Loss
    cretaceous_mask = (age == 1)
    if cretaceous_mask.any():
        cretaceous_type = targets['cretaceous_type'][cretaceous_mask]

        loss += F.cross_entropy(outputs['cretaceous_type'][cretaceous_mask],
                                targets['cretaceous_type'][cretaceous_mask])

        shale_mask = (cretaceous_type == 0)
        if shale_mask.any():
            loss += F.cross_entropy(outputs['shale_type'][shale_mask],
                                    targets['shale_type'][shale_mask])

        cretaceous_other_mask = (cretaceous_type == 1)
        if cretaceous_other_mask.any():
            loss += F.cross_entropy(outputs['cretaceous_other_type'][cretaceous_other_mask],
                                    targets['cretaceous_other_type'][cretaceous_other_mask])

    # Paleozoic Grains Loss
    paleozoic_mask = (age == 2)
    if paleozoic_mask.any():
        loss += F.cross_entropy(outputs['paleozoic_type'][paleozoic_mask],
                                targets['paleozoic_type'][paleozoic_mask])

    # Precambrian Grains Loss
    precambrian_mask = (age == 3)
    if precambrian_mask.any():
        precambrian_type = targets['precambrian_type'][precambrian_mask]

        loss += F.cross_entropy(outputs['precambrian_type'][precambrian_mask],
                                targets['precambrian_type'][precambrian_mask])

        crystalline_mask = (precambrian_type == 0)
        if crystalline_mask.any():
            crystalline_type = targets['crystalline_type'][crystalline_mask]

            loss += F.cross_entropy(outputs['crystalline_type'][crystalline_mask],
                                    targets['crystalline_type'][crystalline_mask])

            light_mask = (crystalline_type == 0)
            if light_mask.any():
                loss += F.cross_entropy(outputs['light_type'][light_mask],
                                        targets['light_type'][light_mask])

            dark_mask = (crystalline_type == 1)
            if dark_mask.any():
                loss += F.cross_entropy(outputs['dark_type'][dark_mask],
                                        targets['dark_type'][dark_mask])

            red_mask = (crystalline_type == 2)
            if red_mask.any():
                loss += F.cross_entropy(outputs['red_type'][red_mask],
                                        targets['red_type'][red_mask])

    return loss

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

def load_model(path : str = None):
    model = resnet18(pretrained=False)

    pass

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