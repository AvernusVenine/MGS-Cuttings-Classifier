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
from PyQt6.QtCore import QSize, Qt, QObject, QEvent, QTimer, pyqtSignal, QThread, QRectF, QCoreApplication
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QImage, QPixmap, QPainter, QFont, QPen, QColor
from pygrabber.dshow_graph import FilterGraph
import torch.nn.functional as F

import ClassificationModel
import FasterRCNNModel
from DetectionModel import DetectionCNN
import dataset
import ImagePreprocessor
from dataset import GrainLabel

class DetectionWorker(QObject):
    boxes_ready = pyqtSignal(list)
    freeze_model = pyqtSignal()

    def __init__(self, cap, model, device):
        super().__init__()
        self.cap = cap
        self.model = model
        self.device = device

        self.threshold = .5
        self.frozen = False

        self.freeze_model.connect(self.freeze)

    def freeze(self):
        self.frozen = ~self.frozen

    def start(self):
        while True:
            if self.frozen:
                return

            _, frame = self.cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, w * ch, QImage.Format.Format_RGB888)

            resize_w, resize_h = 256, 256
            scale_x = w / resize_w
            scale_y = h / resize_h

            detections = self.model_detect(qt_image)

            scores = detections[0]['scores'].cpu().numpy().tolist()
            boxes = detections[0]['boxes'].cpu().numpy().tolist() # [x1, y1, x2, y2]

            grains = []

            for box, score in zip(boxes, scores):
                if score > self.threshold:
                    x1, y1, x2, y2 = box[0:4]

                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y

                    grains.append([int(x1), int(y1), int(x2), int(y2)])

            self.boxes_ready.emit(grains)
            QCoreApplication.processEvents()

    def model_detect(self, image : QImage):
        mean, std = 0.45, 0.12

        image = image.convertToFormat(QImage.Format.Format_RGB888)
        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

        transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        X = transform(image=arr)['image']

        X = X.unsqueeze(0)
        X = X.to(self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)

        return prediction

class CameraWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    cropped_images_ready = pyqtSignal(list)

    def __init__(self, camera_idx : int, model, device):
        super().__init__()
        self.camera_idx = camera_idx
        self.model = model
        self.device = device

        self.boxes = []

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap.isOpened():
            print('Failed to open camera!')
            print(self.camera_idx)
            return

        self.detect_thread = QThread()
        self.detect_worker = DetectionWorker(self.cap, self.model, self.device)
        self.detect_worker.boxes_ready.connect(self.update_boxes)
        self.detect_worker.moveToThread(self.detect_thread)

        self.detect_thread.started.connect(self.detect_worker.start)
        self.detect_thread.start()

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

            cropped_images = []

            for box in self.boxes:
                x1, y1, x2, y2 = box

                cropped = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                height, width = cropped.shape[:2]

                cropped_image = QImage(cropped.data, width, height, width * 3, QImage.Format.Format_RGB888)

                cropped_images.append(cropped_image.copy())

                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            painter.end()

            self.frame_ready.emit(qt_image)
            self.cropped_images_ready.emit(cropped_images)
            QCoreApplication.processEvents()


    def update_boxes(self, boxes):
        self.boxes = boxes

class AppWindow(QMainWindow):

    def __init__(self, model, device):
        super().__init__()

        self.model = model
        self.device = device

        self.grain_images = []
        self.grain_index = 0

        self.setWindowTitle('MGS Grain Annotator')

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

    def device_index_changed(self, idx):
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(idx, self.model, self.device)
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.start_camera)
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.cropped_images_ready.connect(self.update_grain_images)

        self.camera_thread.start()

        central_widget = QWidget()
        layout = QHBoxLayout()

        grain_layout = QVBoxLayout()

        self.grain_index_label = QLabel(f'Grain {self.grain_index}')

        self.grains_scene = QGraphicsScene(self)
        self.grains_view = QGraphicsView(self.grains_scene, self)
        self.grains_pixmap = QGraphicsPixmapItem()
        self.grains_scene.addItem(self.grains_pixmap)

        grain_layout.addWidget(self.grain_index_label)
        grain_layout.addWidget(self.grains_view)

        layout.addWidget(self.view, stretch=4)
        layout.addLayout(grain_layout, stretch=1)

        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)
        self.resize(960 + 240, 540)

    def change_grain_index(self, delta : int):
        if len(self.grain_images) == 0:
            self.grains_pixmap.setPixmap(QPixmap())
            self.grain_index_label.setText(f'Grain {self.grain_index}')
            return

        new_idx = delta + self.grain_index

        if new_idx >= len(self.grain_images):
            new_idx = 0
        elif new_idx < 0:
            new_idx = len(self.grain_images) - 1

        self.grains_pixmap.setPixmap(QPixmap.fromImage(self.grain_images[new_idx]))
        self.grain_index = new_idx
        self.grain_index_label.setText(f'Grain {self.grain_index}')

    def update_grain_images(self, grains):
        self.grain_images = grains
        self.change_grain_index(0)

    def update_frame(self, image : QImage):
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.view.fitInView(self.pixmap_item)

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detection_model = FasterRCNNModel.load_model(path='Trained Models/FasterRCNN_256_256.pth')
    detection_model.to(device)
    detection_model.eval()

    app = QApplication(sys.argv)
    window = AppWindow(detection_model, device)
    window.show()
    sys.exit(app.exec())

run()