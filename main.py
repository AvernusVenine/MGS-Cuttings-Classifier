import tempfile
from pathlib import Path

import FasterRCNNModel
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

from dataset import GrainLabel
import DetectionModel
import ClassificationModel

DATA_PATH = 'grain_data'

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

# Load data into two sets, test and train
def load_data():
    mean, std = (0.4435, 0.4484, 0.4092), (0.1288, 0.1286, 0.1236)

    transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=mean, std=std),
        T.ColorJitter(brightness=.1, contrast=.1),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5),
        ToTensorV2
    ])

    data = ImageFolder(root=f'{DATA_PATH}/', transform=transform)

    train_data, test_data = random_split(data, [.8, .2])

    return train_data, test_data

# Uses ray to hyperparameter train the model on a dictionary of configurations
def hyperparameter_train(config):
    max_epochs = 10
    train_data, _ = load_data()

    # Split train data into train/validation subsets
    split = int(len(train_data) * .85)
    train_subset, val_subset = random_split(
        train_data, [split, len(train_data) - split]
    )

    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=True)

    model = resnet50()

    n_features = model.fc.in_features
    model.fc = Linear(n_features, len(train_data.classes))

    loss_func = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['l2'])

    # Train/validate loop
    for epoch in range(max_epochs):
        train_loop(train_loader, model, loss_func, optimizer)
        loss, accuracy = test_loop(val_loader, model, loss_func)

        tune.report({'loss': loss, 'accuracy': accuracy})

# Used to train the model on a single set of parameters
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

    data = ImageFolder(root=f'{DATA_PATH}/', transform=lambda img: transform(image=np.array(img))["image"])

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

def trial_dir_creator(trial):
    return f'{trial.trainable_name}_{trial.trial_id}'

def tune_model(num_samples : int = 10):
    config = {
        'batch_size': tune.choice([8, 16, 32, 64]),
        'lr': tune.loguniform(1e-4, 1e-1),
        'momentum': tune.uniform(0, .9),
        'l2': tune.choice([0, 2, 4, 6, 8]),
    }

    tuner = tune.Tuner(
        hyperparameter_train,
        tune_config=tune.TuneConfig(
            mode='max',
            metric='accuracy',
            search_alg=OptunaSearch(),
            num_samples=num_samples,
            trial_dirname_creator=trial_dir_creator,
        ),
        param_space=config,
    )

    results = tuner.fit()

    print(f'Best config: {results.get_best_result().config}')

ClassificationModel.train_model(max_epochs=200, batch_size=128)
#FasterRCNNModel.train_model(max_epochs=200, batch_size=16)