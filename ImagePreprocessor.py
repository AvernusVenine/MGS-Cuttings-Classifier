import torchvision.transforms as T
from PIL import Image
import torch
from pathlib import Path
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import GrainLabel

# Finds the mean and std of the data set for normalization
def compute_mean_and_std(path : str, labels = None):
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

# Divides an image into n x m equal parts, where n is the number of columns and m is rows
def divide_image(image : Image, n : int = 2, m: int = 3):
    w, h = image.size

    columns = h//n
    rows = w//m

    images = []

    for i in range(m):
        for j in range(n):
            images.append(image.crop((rows * i, columns * j, rows * (i+1), columns * (j+1))))

    return images

# Divides a given dataset into its smaller images for preprocessing
def preprocess_dataset(data_path : str, save_path : str, label : str, file_type : str):
    images = Path(f'{data_path}/{label}').glob(f'**/*{file_type}')

    idx = 0
    for image in images:
        image = Image.open(image)

        result = divide_image(image, m=5)

        for r in result:
            r.save(f'{save_path}/{label}/{label}_{idx}.jpg')
            idx += 1

def separate_grains(data_path : str, save_path : str, label : str, file_type : str):
    images = Path(f'{data_path}/{label}').glob(f'**/*{file_type}')

    model = fasterrcnn_resnet50_fpn_v2()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    model.load_state_dict(torch.load('trained_detection_model_temp.pth'))
    model.eval()

    threshold = .6
    mean, std = (0.4435, 0.4484, 0.4092), (0.1288, 0.1286, 0.1236)

    count = 0

    for image in images:
        print(f'{image} | {count}')

        image = Image.open(image)

        arr = np.array(image)

        transform = A.Compose([
            A.Resize(256, 512),
            A.ToGray(num_output_channels=3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        X = transform(image=arr)['image']

        model.eval()
        with torch.no_grad():
            prediction = model([X])[0]

        w, h = image.size

        resize_w, resize_h = 512, 256
        scale_x = w / resize_w
        scale_y = h / resize_h

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for idx, box in enumerate(boxes):

            if scores[idx] < threshold:
                continue

            x1, y1, x2, y2 = map(float, box)

            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y

            cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
            cropped.save(f'{save_path}/{label}/{label}_{count}.jpg')

            count += 1

#separate_grains('raw_images', 'grain_data', CuttingLabel.SHALE, '.jpg')