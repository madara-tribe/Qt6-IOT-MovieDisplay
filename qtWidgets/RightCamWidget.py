from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from qtWidgets.utils import onnx_predict, torch_predict
import cv2, os, sys
import numpy as np
import time

class RightCamWidget(QWidget):
    def __init__(self, parent, video_path, G_BA, G_AB, ort_session):
        super().__init__(parent=parent)
        
        self.parent = parent
        if video_path:
            self.capture = cv2.VideoCapture(video_path)
        else:
            self.capture = cv2.VideoCapture(0)
        self.size = 256
        self.video_size = QSize(self.size*2, self.size*2)
        self.TIMEOUT = 1
        self.cur_fps = 0
        self.G_BA = G_BA
        self.G_AB = G_AB
        self.ort_session = ort_session
        self.text = 'Pytorch'
        self.old_timestamp = time.time()
        self.setup_ui()
        self.setup_camera()
        self.plot_fps(initial=True)
        
    def setup_ui(self):
        """Initialize widgets.
        """
        
        self.title = QLabel('hourse => zebra')
        self.title.setStyleSheet('font-family: Times New Roman; font-size: 40px;color: black;')
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)
        
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.title)
        self.main_layout.addWidget(self.image_label)
        # select model type
        self.modeltype()
        
        # plot bar
        self.main_bar_layout.addLayout(self.button_layout)
        self.main_bar_layout.addLayout(self.bar_layout1)
        self.main_bar_layout.addLayout(self.bar_layout2)
        self.main_bar_layout.addLayout(self.bar_layout3)
        
        self.main_layout.addLayout(self.main_bar_layout)
        self.setLayout(self.main_layout)

    def setup_camera(self):
        """Initialize camera.
        """
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        fps = (time.time() - self.old_timestamp) / self.TIMEOUT
        if (time.time() - self.old_timestamp) > self.TIMEOUT:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.size, self.size))
            
            start = time.time()
            if self.text == 'ONNX':
                pred_frame = onnx_predict(frame.copy(), self.ort_session)
            elif self.text == 'Pytorch':
                pred_frame = torch_predict(frame.copy(), self.G_AB)
            self.predict_time = np.round((time.time() - start), decimals=5)
            
            image = QImage(pred_frame, pred_frame.shape[1], pred_frame.shape[0],
                            pred_frame.strides[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.old_timestamp = time.time()
            
            self.cur_fps = np.round(fps, decimals=3)
            self.plot_fps(initial=None)
            
    def return_cap(self):
        return self.capture
        
    def modeltype(self):
        self.main_bar_layout = QVBoxLayout()
        
        self.button_layout = QHBoxLayout()
        self.types = QButtonGroup()
        title = QLabel('Model Type')
        title.setStyleSheet('color: blue;')
        self.button_layout.addWidget(title)
        for type in ['ONNX','Pytorch']:
            botan = QRadioButton(type)
            self.button_layout.addWidget(botan)
            self.types.addButton(botan)

        self.types.buttonClicked.connect(self.select_type)
    
    
        # Predicted time bar
        self.bar_layout1 = QHBoxLayout()
        self.title1 = QLabel('Predicted time')
        self.predictbar = QLabel('', self)
        self.predictbar.setStyleSheet('background-color: gray;')

        self.bar_layout1.addWidget(self.title1)
        self.bar_layout1.addWidget(self.predictbar)

        # FPS bar
        self.bar_layout2 = QHBoxLayout()
        self.title2 = QLabel('Constant FPS')
        self.fps = QLabel('', self)
        self.fps.setStyleSheet('background-color: gray;')

        self.bar_layout2.addWidget(self.title2)
        self.bar_layout2.addWidget(self.fps)
        
        # model type bar
        self.bar_layout3 = QHBoxLayout()
        self.title3 = QLabel('Model type')
        self.model_type = QLabel('', self)
        self.model_type.setStyleSheet('background-color: gray;')

        self.bar_layout3.addWidget(self.title3)
        self.bar_layout3.addWidget(self.model_type)
        
        
    def select_type(self, botan):
        self.text = botan.text()
        self.model_type.setText(str(self.text))
        self.model_type.setStyleSheet('font-family: Times New Roman; font-size: 15px; color: black; background-color: azure')
    
    def plot_fps(self, initial=None):
        if initial:
            self.model_type.setText(self.text)
            self.model_type.setStyleSheet('font-family: Times New Roman; font-size: 15px; color: black; background-color: azure')
        else:
            self.fps.setText(str(self.cur_fps))
            self.fps.setStyleSheet('font-family: Times New Roman; font-size: 15px; color: black; background-color: azure')
            
            self.predictbar.setText(str(self.predict_time*1000)+"[ms]")
            self.predictbar.setStyleSheet('font-family: Times New Roman; font-size: 15px; color: black; background-color: azure')
