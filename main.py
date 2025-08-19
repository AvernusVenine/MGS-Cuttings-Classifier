import image_preprocessor, dataset
from torch.optim import SGD
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Linear
from torchvision.models import resnet50 # Change as needed for testing
import torchvision.transforms as T

TRAIN_DATA_PATH = 'train_data'
TEST_DATA_PATH = 'test_data'

NUM_EPOCHS = 10
DEVICE = 'cpu'

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

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

#TODO: Possibly dont use pretrained?
model = resnet50()

n_features = model.fc.in_features
model.fc = Linear(n_features, len(train_data.classes))

#TODO: Change when dealing with a binary set vs a multiclass one
loss_func = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=1e-2, momentum=.9)


for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}')
    train_loop(train_loader, model, loss_func, optimizer)
    test_loop(test_loader, model, loss_func)



