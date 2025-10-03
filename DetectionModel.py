import os
from pathlib import Path
from turtledemo.chaos import coosys

import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import ImagePreprocessor, dataset
import xml.etree.ElementTree as ET
from torch.optim import SGD, Adam
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import time

import cv2

from dataset import GrainLabel

DATA_PATH = 'annotated_data'

class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvTransBlock, self).__init__()

        self.conv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchNorm2d(x)
        x = self.relu(x)
        return x

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

    def visualize_kernels(self):
        weights = self.conv2d.weight.data.cpu().numpy()

        num_kernels = weights.shape[0]
        fig_cols = 8
        fig_rows = (num_kernels + fig_cols - 1) // fig_cols

        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 2, fig_rows * 2))
        axes = axes.flatten()

        for i in range(num_kernels):
            kernel = weights[i, 0, :, :]
            axes[i].imshow(kernel, cmap='seismic')
            axes[i].set_title(f'Kernel {i}')
            axes[i].axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

class DetectionCNN(nn.Module):
    def __init__(self):
        super(DetectionCNN, self).__init__()

        self.convBlock1 = ConvBlock(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4) # 128x128x1 -> 128x128x64
        self.convBlock2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2) # 128x128x64 -> 128x128x64
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 128x128x64 -> 64x64x64
        self.convBlock3 = ConvBlock(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0) # 64x64x64 -> 64x64x256

        #TODO: Might need to change this
        self.upsample = nn.Sequential(
            ConvTransBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Determines the chance of a pixel being a center
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

        # Determines the (x, y) offset of the top left
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

        # Determines the height and width of bounding box
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        batch, _, input_h, input_w = x.shape

        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.maxPool1(x)
        x = self.convBlock3(x)

        x = self.upsample(x)

        heatmap = torch.sigmoid(self.heatmap_head(x))
        offset = self.offset_head(x)
        size = torch.sigmoid(self.size_head(x))

        if self.training:
            return heatmap, offset, size
        else:
            return self.decode_predictions(heatmap, offset, size, input_h, input_w)

    @staticmethod
    def decode_predictions(heatmap, offset, size, input_h, input_w, threshold=0, kernel_size=3):
        batch, _, feat_h, feat_w = heatmap.shape
        detections = []

        for b in range(batch):

            hm_max = F.max_pool2d(
                heatmap[b:b+1],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            )

            # Only keep local maxima that are greater than some threshold
            local_max = (heatmap[b, 0] == hm_max) & (heatmap[b, 0] > threshold)

            if local_max.sum() == 0:
                continue

            peak = torch.where(local_max)
            peak_y, peak_x = peak[0], peak[1]
            scores = heatmap[b, 0][peak_y, peak_x]

            batch_detections = []

            for idx, (y, x, score) in enumerate(zip(peak_y, peak_x, scores)):
                offset_x = offset[b, 0, y, x]
                offset_y = offset[b, 1, y, x]

                x = (x.float() + offset_x)
                y = (y.float() + offset_y)

                ix = x.long().clamp(0, feat_w - 1)
                iy = y.long().clamp(0, feat_h - 1)

                w = size[b, 0, iy, ix] * input_w
                h = size[b, 1, iy, ix] * input_h

                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2

                x1 = torch.clamp(x1, 0, input_w-1)
                y1 = torch.clamp(y1, 0, input_h-1)
                x2 = torch.clamp(x2, 0, input_w-1)
                y2 = torch.clamp(y2, 0, input_h-1)

                if x2 > x1 and y2 > y1:
                    batch_detections.append([
                        score.item(), x1.item(), y1.item(), x2.item(), y2.item()
                    ])

            detections.append(batch_detections)

        return detections

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None and hasattr(m, 'final_layer'):
                    nn.init.constant_(m.bias, -2.19)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        prior_prob = 0.10  # Assume 10% of pixels are "positive" due to Gaussian spread
        bias_init = -np.log((1 - prior_prob) / prior_prob)

        # Initialize final heatmap layer
        nn.init.constant_(self.heatmap_head[-1].bias, bias_init)


class CenterNetLoss(nn.Module):
    def __init__(self, pos_weight=2, lambda_offset=1.0, lambda_size=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.lambda_offset = lambda_offset
        self.lambda_size = lambda_size

    def modified_focal_loss(self, pred, gt, alpha=2, beta=4):
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)  # for numerical stability
        gt = torch.clamp(gt, 0, 1)

        # Positive samples are where gt == 1 (exact centers have value 1.0)
        pos_inds = gt.eq(1).float()
        # Negative samples are where gt < 1
        neg_inds = gt.lt(1).float()

        # Reduce penalty for predictions close to ground truth Gaussian
        neg_weights = torch.pow(1 - gt, beta)

        # Positive loss: penalize false negatives
        pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
        # Negative loss: penalize false positives, weighted by distance from GT
        neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()

        # Normalize by number of objects, not pixels
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

    def masked_l1_loss(self, pred, gt, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask = mask.expand_as(pred).float()
        loss = F.l1_loss(pred * mask, gt * mask, reduction='sum')
        num_pos = mask.sum()
        return loss / num_pos.clamp(min=1)

    def forward(self, predictions, targets):
        pred_hm, pred_offset, pred_size = predictions

        target_hm = targets['heatmap']
        target_offset = targets['offset']
        target_size = targets['size']
        mask = targets['mask']

        heatmap_loss = self.modified_focal_loss(pred_hm, target_hm)
        offset_loss = self.masked_l1_loss(pred_offset, target_offset, mask)
        size_loss = self.masked_l1_loss(pred_size, target_size, mask)

        total_loss = heatmap_loss + self.lambda_offset * offset_loss + self.lambda_size * size_loss

        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'offset_loss': offset_loss,
            'size_loss': size_loss
        }

def generate_gaussian_heatmap(center_x, center_y, width, height, output_h, output_w, sigma_scale=0.3):
    # Adaptive sigma based on object size
    sigma_x = sigma_scale * width
    sigma_y = sigma_scale * height
    sigma = min(sigma_x, sigma_y)
    sigma = max(sigma, 1)  # Minimum sigma

    # Create coordinate grids
    x = torch.arange(0, output_w, dtype=torch.float32)
    y = torch.arange(0, output_h, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    # Calculate Gaussian
    gaussian = torch.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * sigma ** 2))

    return gaussian

def prepare_centers(bboxes, input_width, input_height, output_width, output_height):
    heatmap = torch.zeros(1, output_height, output_width)
    offset = torch.zeros(2, output_height, output_width)
    size = torch.zeros(2, output_height, output_width)
    mask = torch.zeros(output_height, output_width)

    scale_x = output_width / input_width
    scale_y = output_height / input_height

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # Convert to output coordinates
        x1_out = x1 *scale_x
        y1_out = y1 * scale_y
        x2_out = x2 * scale_x
        y2_out = y2 * scale_y

        # Calculate center and size
        center_x = (x1_out + x2_out) / 2
        center_y = (y1_out + y2_out) / 2
        width = x2_out - x1_out
        height = y2_out - y1_out

        # Get integer center coordinates
        center_x_int = int(center_x)
        center_y_int = int(center_y)

        if 0 <= center_x_int < output_width and 0 <= center_y_int < output_height:
            gaussian = generate_gaussian_heatmap(
                center_x, center_y, width, height,
                output_height, output_width
            )

            heatmap[0] = torch.maximum(heatmap[0], gaussian)

            offset[0, center_y_int, center_x_int] = center_x - center_x_int
            offset[1, center_y_int, center_x_int] = center_y - center_y_int

            size[0, center_y_int, center_x_int] = width / input_width
            size[1, center_y_int, center_x_int] = height / input_height

            mask[center_y_int, center_x_int] = 1

    return {
        'heatmap': heatmap,
        'offset': offset,
        'size': size,
        'mask': mask
    }


class GrainDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

        self.img_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.root_dir, img_filename)
        xml_path = img_path.replace('.jpg', '.xml')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_height, original_width = img.shape[:2]

        boxes = self._parse_xml(xml_path)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        transformed = self.transform(image=img, bboxes=boxes, labels=[1]*len(boxes))
        img = transformed['image']

        # For some reason Albumentations ToGray doesn't always force it to a single channel
        if img.shape[0] != 1:
            img = img[:1, :, :]

        boxes = transformed['bboxes']

        # Prepare CenterNet targets
        targets = prepare_centers(
            boxes,
            original_width, # Should be 1920
            original_height, # Should be 1080
            128,  # output_width
            128  # output_height
        )

        return img, targets

    @staticmethod
    def _parse_xml(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            ]
            boxes.append(bbox)

        return boxes

def collate_fn(batch):
    return tuple(zip(*batch))

def train_loop(dataloader, model, optimizer, device, loss_func):
    model.train()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        images = torch.stack(images).to(device)

        targets = {
            key: torch.stack([t[key] for t in targets]).to(device)
            for key in targets[0]
        }

        # Forward Propagation
        outputs = model(images)
        loss_dict = loss_func(outputs, targets)
        loss = sum(loss for loss in loss_dict.values())

        #Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, device, loss_func):
    model.train()

    total_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            images = torch.stack(images).to(device)

            targets = {
                key: torch.stack([t[key] for t in targets]).to(device)
                for key in targets[0]
            }

            outputs = model(images)
            loss_dict = loss_func(outputs, targets)
            loss = sum(loss for loss in loss_dict.values())

            total_loss += loss.item()

    test_loss = total_loss / len(dataloader)

    return test_loss


def train_model(max_epochs : int = 10, batch_size : int = 32, lr : float = 1e-5, momentum : float = 0.9, l2 = 0):
    mean, std = 0.45, 0.12

    transform = A.Compose([
        A.ToGray(num_output_channels=1),
        A.Resize(128, 128),
        A.ColorJitter(brightness=.1, contrast=.1),
        A.Affine(translate_percent={'x': (-.25, .25), 'y': (-.25, .25)}),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=5.0,
        min_visibility=.1,
        check_each_transform=True
    ))

    data = GrainDataset(root_dir=f'{DATA_PATH}/', transform=transform)

    train_size = int(0.8 * len(data))
    train_data, test_data = random_split(data, [train_size, len(data) - train_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DetectionCNN()
    model.load_state_dict(torch.load('trained_detection_model_custom.pth'))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_func = CenterNetLoss()

    best_loss = 1.8
    patience = 20
    triggers = 0

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}')
        train_loop(train_loader, model, optimizer, device, loss_func)
        loss = test_loop(test_loader, model, device, loss_func)

        print(f'Loss: {loss}')

        if loss < best_loss:
            best_loss = loss
            triggers = 0

            torch.save(model.state_dict(), 'trained_detection_model_custom.pth')
        else:
            triggers += 1

            if triggers >= patience:
                print(f'No improvement for {patience} epochs.  Stopping early with {best_loss}')
                break

def test_model():
    mean, std = 0.45, 0.12

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detection_model = fasterrcnn_resnet50_fpn_v2()
    in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
    detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    detection_model.load_state_dict(torch.load('trained_detection_model_cnn.pth', map_location=device))

    detection_model.eval()

    transform = A.Compose([
        A.Resize(128, 128),
        A.ToGray(num_output_channels=3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    img = cv2.imread('annotated_data/felsic_1.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transform(image=img)['image']

    torch.set_printoptions(threshold=float('inf'))

    start = time.time()

    detection_model(img.unsqueeze(0))

    print(time.time() - start)


def visualize_layer():
    model = DetectionCNN()
    model.load_state_dict(torch.load('trained_detection_model_custom.pth'))

    model.convBlock1.visualize_kernels()
    plt.show()


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#visualize_layer()
#train_model(max_epochs=150, batch_size=256, lr=1e-5)
#test_model()