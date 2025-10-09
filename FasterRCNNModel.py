import os
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
import xml.etree.ElementTree as ET
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DATA_PATH = 'annotated_data'

class DetectionDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform

        self.img_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.bmp'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.root_dir, img_filename)
        xml_path = img_path.replace('.jpg', '.xml').replace('.bmp', '.xml')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self._parse_xml(xml_path)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        labels = [1]*len(boxes)

        transformed = self.transform(image=img, bboxes=boxes, labels=labels)
        img = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['labels']

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes' : boxes,
            'labels' : labels
        }

        return img, target

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

def train_loop(dataloader, model, optimizer, device):
    model.train()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward Propagation
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        #Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, device):
    model.train()

    total_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            total_loss += loss.item()

    test_loss = total_loss / len(dataloader)

    return test_loss

def load_model(path : str = None):
    backbone = resnet_fpn_backbone('resnet18', weights='ResNet18_Weights.DEFAULT',
                                   trainable_layers=5, returned_layers=[1,2])
    anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,)),
                                       aspect_ratios=((.75, 1.25, 1.75), (.75, 1.25, 1.75), (.75, 1.25, 1.75)))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2'],
                                    output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone=backbone, num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       box_nms_thresh=.1,
                       min_size=256, max_size=256)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    if path:
        model.load_state_dict(torch.load(path))

    return model


def train_model(max_epochs : int = 10, batch_size : int = 32, lr : float = 1e-5, momentum : float = 0.9, l2 = 0):
    mean, std = 0.45, 0.12

    transform = A.Compose([
        A.Resize(256, 256),
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

    data = DetectionDataset(root_dir=f'{DATA_PATH}/', transform=transform)

    train_size = int(0.8 * len(data))
    train_data, test_data = random_split(data, [train_size, len(data) - train_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2)

    best_loss = np.inf
    patience = 200
    triggers = 0

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}')
        train_loop(train_loader, model, optimizer, device)
        loss = test_loop(test_loader, model, device)

        print(f'Loss: {loss}')

        if loss < best_loss:
            best_loss = loss
            triggers = 0

            torch.save(model.state_dict(), 'FasterrCNN_256_256.pth')
        else:
            triggers += 1

            if triggers >= patience:
                print(f'No improvement for {patience} epochs.  Stopping early with {best_loss}')
                break
