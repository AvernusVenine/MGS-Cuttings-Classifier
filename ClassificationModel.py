import os
import cv2
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn import CrossEntropyLoss, Linear
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, resnet34
from torch.optim import Adam
import torchvision.transforms as T
import ray.tune as tune
from ray.tune.search.optuna import OptunaSearch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn

import dataset


class GrainDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

        self.img_files = [
            os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk(root_dir)
            for filename in filenames
            if filename.lower().endswith(('.jpg', '.bmp', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        basename = os.path.basename(os.path.dirname(img_path))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=img)
        img = transformed['image']

        target = dataset.GRAIN_MAP[basename].get_target()

        return img, target

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()

        model = resnet50()
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()

        self.age = nn.LazyLinear(4) # Precambrian, Paleozoic, Cretaceous, Other

        self.other_type = nn.LazyLinear(4) # Chert, Unknown, Secondary, Gypsum

        self.cretaceous_type = nn.LazyLinear(2) # Shale, Non-Shale
        self.shale_type = nn.LazyLinear(2) # Gray Shale, Speckled Shale
        self.cretaceous_other_type = nn.LazyLinear(6)

        self.paleozoic_type = nn.LazyLinear(2) # Carbonate, Sandstone Shale

        self.precambrian_type = nn.LazyLinear(2) # Crystalline, Other
        self.crystalline_type = nn.LazyLinear(3) # Light, Dark, Red
        self.light_type = nn.LazyLinear(3)
        self.dark_type = nn.LazyLinear(2)
        self.red_type = nn.LazyLinear(4)

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

def collate_fn(batch):
    return tuple(zip(*batch))

def compute_loss(outputs, targets, device):
    keys = targets[0].keys()
    batch_targets = {}

    for key in keys:
        batch_targets[key] = torch.stack([t[key] for t in targets]).to(device)

    targets = batch_targets

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
        loss += F.cross_entropy(outputs['cretaceous_type'][cretaceous_mask],
                                targets['cretaceous_type'][cretaceous_mask])

        cretaceous_type = targets['cretaceous_type']
        shale_mask = (cretaceous_type == 0) & cretaceous_type
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
        loss += F.cross_entropy(outputs['precambrian_type'][precambrian_mask],
                                targets['precambrian_type'][precambrian_mask])

        precambrian_type = targets['precambrian_type']
        crystalline_mask = (precambrian_type == 0) & precambrian_type
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

def train_loop(dataloader, model, optimizer, device):
    model.train()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward Propagation
        output = model(torch.stack(images, dim=0))
        loss = compute_loss(output, targets, device)

        #Backward Propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, device):
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(torch.stack(images, dim=0))
            test_loss += compute_loss(output, targets, device)

    test_loss = test_loss / len(dataloader)

    print(f'Loss: {test_loss}')
    return test_loss

def load_model(path : str = None):
    model = ClassificationModel()

    if path:
        model.load_state_dict(torch.load(path))

    return model

def train_model(max_epochs : int = 10, batch_size : int = 64, lr : float = 1e-4, l2 = 0):
    mean, std = 0.45, 0.12

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=mean, std=std),
        A.RandomBrightnessContrast(brightness_limit=(-.2, .2), contrast_limit=(-.2, .2), p=.75),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5),
        A.Affine(rotate=(-45, 45), shear=(-10, 10), p=.75),
        ToTensorV2()
    ])

    data = GrainDataset(root_dir=f'grain_data/', transform=transform)

    train_size = int(0.75 * len(data))
    train_data, test_data = random_split(data, [train_size, len(data) - train_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2)

    best_loss = float('inf')
    patience = 100
    triggers = 0

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}')
        train_loop(train_loader, model, optimizer, device)
        loss = test_loop(test_loader, model, device)

        if loss < best_loss:
            best_loss = loss
            triggers = 0

            torch.save(model.state_dict(), 'Classifier_50.pth')
        else:
            triggers += 1

            if triggers >= patience:
                print(f'No improvement for {patience} epochs.  Stopping early with {best_loss}')
                break
