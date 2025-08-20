import tempfile
from pathlib import Path
import image_preprocessor, dataset
from torch.optim import SGD
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss, Linear
from torchvision.models import resnet50 # Change as needed for testing
import torchvision.transforms as T
import ray.tune as tune
from ray.train import get_checkpoint, Checkpoint
import ray.cloudpickle as pickle
from ray import train

TRAIN_DATA_PATH = 'train_data'
TEST_DATA_PATH = 'test_data'

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
    mean, std = image_preprocessor.compute_mean_and_std(TRAIN_DATA_PATH, [])

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomApply([
            T.RandomPerspective(),
            T.RandomRotation(degrees=(0, 360)),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomAutocontrast(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ], p=0.5)
    ])

    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    train_data = ImageFolder(root=f'{TRAIN_DATA_PATH}/', transform=train_transform)
    test_data = ImageFolder(root=f'{TEST_DATA_PATH}/', transform=test_transform)

    return train_data, test_data

# Uses ray to hyperparameter train the model on a dictionary of configurations
def hyperparameter_train(config, max_epochs : int):
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

    # Load checkpoint if it exists
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            path = Path(checkpoint_dir) / "data.pkl"

            with open(path, 'rb') as fp:
                state = pickle.load(fp)

            start_epoch = state['epoch']
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
    else:
        start_epoch = 0

    # Test/validate loop
    for epoch in range(start_epoch, max_epochs):
        train_loop(train_loader, model, loss_func, optimizer)
        loss, accuracy = test_loop(val_loader, model, loss_func)

        checkpoint_data = {
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            path = Path(checkpoint_dir) / 'data.pkl'

            with open(path, 'wb') as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {'loss' : loss, 'accuracy' : accuracy},
                checkpoint=checkpoint
            )
# Used to train the model on a single set of parameters
def  train_model(max_epochs : int = 10, batch_size : int = 64, lr : float = 1e-2, momentum : float = 0, l2 = 0):
    mean, std = image_preprocessor.compute_mean_and_std(TRAIN_DATA_PATH, [])

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomApply([
            T.RandomPerspective(),
            T.RandomRotation(degrees=(0, 360)),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomAutocontrast(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ], p=0.5)
    ])

    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    train_data = ImageFolder(root=f'{TRAIN_DATA_PATH}/', transform=train_transform)
    test_data = ImageFolder(root=f'{TEST_DATA_PATH}/', transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = resnet50()

    n_features = model.fc.in_features
    model.fc = Linear(n_features, len(train_data.classes))

    loss_func = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2)

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}')
        train_loop(train_loader, model, loss_func, optimizer)
        loss, accuracy = test_loop(test_loader, model, loss_func)

        print(f'Loss: {loss} | Accuracy: {accuracy}')

param_config = {
    'batch_size' : tune.choice([8, 16, 32, 64]),
    'lr' : tune.loguniform(1e-4, 1e-1),
    'momentum' : tune.choice([0, .3, .6, .9]),
    'l2' : tune.choice([0, 2, 4, 6, 8]),
}

hyperparameter_train(param_config, 10)