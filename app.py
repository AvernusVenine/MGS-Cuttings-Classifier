import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from Stats import threshold
from torch.nn import CrossEntropyLoss, Linear
from torchvision.models import resnet50
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PyQt6.QtWidgets import QWidget, QMenuBar, QMainWindow, QApplication, QToolBar, QComboBox, QGraphicsPixmapItem, \
    QGraphicsScene, QGraphicsView, QDoubleSpinBox, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import QSize, Qt, QObject, QEvent, QTimer, pyqtSignal, QThread, QRectF
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QImage, QPixmap, QPainter, QFont, QPen, QColor
from pygrabber.dshow_graph import FilterGraph
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, ssd300_vgg16, _utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
from torchvision.models.detection.ssd import SSDClassificationHead

from DetectionModel import DetectionCNN
import dataset
import ImagePreprocessor
from dataset import CuttingLabel

class GrainCountInfo(QWidget):
    # TODO: Fill out as more grain types are added
    grain_counts = {
        CuttingLabel.SHALE : 0,
        CuttingLabel.FELSIC : 0,
    }

    def __init__(self):
        super().__init__()
        self.update_counts({})
        self.setFixedWidth(220)

    def update_counts(self, update : dict):
        for key in update.keys():
            self.grain_counts[key] += update[key]

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)

        label = QLabel('Grain Counts')
        label.setStyleSheet("""
                                    QLabel {
                                        color: black;
                                        font-size: 16px;
                                        font-weight: bold;
                                        padding: 4px;
                                    }
                                """)
        layout.addWidget(label)

        for key, value in self.grain_counts.items():
            if value == 0:
                continue

            label = QLabel(f'{key} | {value}')
            label.setStyleSheet("""
                            QLabel {
                                color: black;
                                font-size: 12px;
                                font-weight: bold;
                                padding: 2px;
                            }
                        """)
            layout.addWidget(label)

        self.setLayout(layout)

    def reset_counts(self):
        for key in self.grain_counts.keys():
            self.grain_counts[key] = 0

class CameraWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    model_done = pyqtSignal(dict)
    reset_count = pyqtSignal()

    def __init__(self, camera_idx : int, detection_model, classifier_model, device):
        super().__init__()
        self.camera_idx = camera_idx
        self.detection_model = detection_model
        self.classifier_model = classifier_model
        self.device = device
        self.threshold = 0.3


    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap.isOpened():
            print('Failed to open camera!')
            return

        # Main Camera Loop
        while True:
            _, frame = self.cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, w * ch, QImage.Format.Format_RGB888)

            painter = QPainter(qt_image)
            painter.setFont(QFont('Times New Roman', 16))

            pen = QPen(QColor('Red'))
            pen.setWidth(2)
            painter.setPen(pen)

            # Returns as score, xmin, ymin, xmax, ymax
            detections = self.model_detect(qt_image.copy())

            resize_w, resize_h = 128, 128 #TODO: Dont forget to resize with model changes
            scale_x = w / resize_w
            scale_y = h / resize_h

            grain_dict = {}

            for detection in detections:
                if detection[0][0] > self.threshold:
                    x1, y1, x2, y2 = detection[0][1:5]

                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y

                    #label, confidence = self.model_classify(qt_image, x1, y1, x2, y2)

                    painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    #painter.drawText(int(x1), int(y2 + 20), f"{label} | {float(confidence):.2f}")

            painter.end()

            self.frame_ready.emit(qt_image)


    def model_classify(self, image : QImage, x1, y1, x2, y2):
        mean, std = (0.4435, 0.4484, 0.4092), (0.1288, 0.1286, 0.1236)

        image = image.convertToFormat(QImage.Format.Format_RGB888)
        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

        cropped = arr[int(y1):int(y2), int(x1):int(x2)]

        transform = A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        X = transform(image=cropped)['image']
        X = X.to(self.device)

        self.classifier_model.eval()
        with torch.no_grad():
            prediction = self.classifier_model(X.unsqueeze(0))

        label = prediction.argmax(dim=1).item()
        confidence = F.softmax(prediction, dim=1)[0, label].item()

        return dataset.CLASS_MAP[label], confidence

    def model_detect(self, image : QImage):
        mean, std = 0.45, 0.12

        image = image.convertToFormat(QImage.Format.Format_RGB888)
        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

        transform = A.Compose([
            A.Resize(128, 128),
            A.ToGray(num_output_channels=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        X = transform(image=arr)['image']

        if X.shape[0] != 1:
            X = X[:1, :, :]

        X = X.unsqueeze(0)
        X = X.to(self.device)

        self.detection_model.eval()
        with torch.no_grad():
            prediction = self.detection_model(X)

        return prediction

class AppWindow(QMainWindow):

    capture_signal = pyqtSignal()

    def __init__(self, detection_model, classifier_model, device):
        super().__init__()

        self.detection_model = detection_model
        self.classifier_model = classifier_model
        self.device = device

        self.setWindowTitle('ResNet Grain Classifier')

        # Graphics and Camera Feed
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.device_box = QComboBox()

        for cam in FilterGraph().get_input_devices():
            self.device_box.addItem(cam)

        self.device_box.setCurrentIndex(-1)
        self.device_box.currentIndexChanged.connect(self.device_index_changed)
        self.setCentralWidget(self.device_box)

        self.camera_thread = None
        self.camera_worker = None
        self.count_info = None

    def device_index_changed(self, idx):
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(idx, self.detection_model, self.classifier_model, self.device)
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.start_camera)
        self.camera_worker.frame_ready.connect(self.update_frame)

        self.camera_thread.start()

        central_widget = QWidget()
        layout = QHBoxLayout()

        self.count_info = GrainCountInfo()
        self.camera_worker.model_done.connect(self.count_info.update_counts)
        self.camera_worker.reset_count.connect(self.count_info.reset_counts)

        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.count_info, stretch=0)

        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)
        self.resize(960 + 220, 540) #TODO: Change the width to correspond with info box width

    def update_frame(self, image : QImage):
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.view.fitInView(self.pixmap_item)

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detection_model = DetectionCNN()
    detection_model.load_state_dict(torch.load('trained_detection_model_custom.pth', map_location=device))

    detection_model.to(device)
    detection_model.eval()

    classifier_model = resnet50()
    in_features = classifier_model.fc.in_features
    classifier_model.fc = Linear(in_features, 2) # TODO: Change to number of final classes
    classifier_model.load_state_dict(torch.load('trained_classification_model.pth', map_location=device))
    classifier_model.to(device)
    classifier_model.eval()

    app = QApplication(sys.argv)
    window = AppWindow(detection_model, classifier_model, device)
    window.show()
    sys.exit(app.exec())

run()