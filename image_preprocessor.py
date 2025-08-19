import torchvision.transforms as T
from PIL import Image
import torch
from pathlib import Path

from dataset import CuttingLabel

# Finds the mean and std of the data set for normalization
def compute_mean_and_std(path : str, labels : list):
    channels = 3

    mean = torch.zeros(channels)
    std = torch.zeros(channels)

    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32)
    ])

    total_images = 0

    for label in labels:
        images = Path(f'{path}/{label}').glob('*.jpg')

        for image in images:
            image = Image.open(image)
            image = transform(image)

            mean += image.mean([1, 2])
            std += image.std([1, 2])

            total_images += 1

    return mean/total_images, std/total_images

# Divides an image into 6 equal parts (2 high, 3 long)
def divide_image(image : Image):
    w, h = image.size

    half = h//2
    third = w//3

    images = []

    images.append(image.crop((0, 0, third, half)))
    images.append(image.crop((0, half, third, h)))
    images.append(image.crop((third, 0, 2*third, half)))
    images.append(image.crop((third, half, 2 * third, h)))
    images.append(image.crop((2*third, 0, w, half)))
    images.append(image.crop((2 * third, half, w, h)))

    return images

# Divides a given dataset into its smaller images for preprocessing
def divide_dataset(data_path : str, save_path : str, labels : list):
    for label in labels:
        images = Path(f'{data_path}/{label}').glob('*.jpg')

        idx = 0
        for image in images:
            image = Image.open(image)
            result = divide_image(image)

            for r in result:
                r.save(f'{save_path}/{label}/{label}_{idx}.jpg')
                idx += 1

#divide_dataset('raw_images', 'train_data', [CuttingLabel.SHALE])