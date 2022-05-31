from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap

import matplotlib.patches as patches
import scipy.spatial.distance
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 
import random as rand

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
import joblib

import PIL
import cv2

file_open_flag = False  # define global flag
detected = np.array

def extract_features(img, model = "yuv"):
    if model == "hsv": 
        ABC_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif model == "hls":
        ABC_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    else:
        ABC_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    hogA, hogA_img = hog(ABC_img[:, :, 0], 
                         orientations = 11, 
                         pixels_per_cell = (16,16), 
                         cells_per_block = (2,2), 
                         transform_sqrt = True, 
                         visualize = True, 
                         feature_vector = False)
    hogB, hogB_img = hog(ABC_img[:, :, 1],
                         orientations = 11,
                         pixels_per_cell = (16,16),
                         cells_per_block = (2,2),
                         transform_sqrt = True, 
                         visualize = True,
                         feature_vector = False)
    hogC, hogC_img = hog(ABC_img[:, :, 2],
                         orientations = 11,
                         pixels_per_cell = (16,16),
                         cells_per_block = (2,2),
                         transform_sqrt = True, 
                         visualize = True,
                         feature_vector = False)
    y_end = img.shape[1] // 16 - 1
    x_end = img.shape[0] // 16 - 1
    x_start, y_start = 0, 0
    hogA = hogA[y_start: y_end, x_start: x_end].ravel()
    hogB = hogB[y_start: y_end, x_start: x_end].ravel()
    hogC = hogC[y_start: y_end, x_start: x_end].ravel()
    hog_features = np.hstack((hogA, hogB, hogC))
    
    return hog_features

def slide_extract(image, model = "yuv"):
    windowSize = (80, 80)
    step = 10
    if model == "hsv": 
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif model == "hls":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    coords, features = [], []
    hIm,wIm = image.shape[:2] 
    for w1, w2 in zip(range(0, wIm-windowSize[0], step),range(windowSize[0], wIm, step)):
        for h1, h2 in zip(range(0, hIm-windowSize[1], step),range(windowSize[1], hIm, step)):
            window = img[h1:h2, w1:w2]
            features_of_window = extract_features(window, model)
            coords.append((w1, w2, h1, h2))
            features.append(features_of_window)
    return (coords, np.asarray(features))

class Heatmap():
    def __init__(self, image):
        self.mask = np.zeros(image.shape[:2])
    def increase_value(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] + 30
    def construct(self):
        scaler = MinMaxScaler()
        self.mask = scaler.fit_transform(self.mask)
        self.mask = np.asarray(self.mask * 255).astype(np.uint8)
        self.mask = cv2.inRange(self.mask, 170, 255)
        
        return self.mask
    
def detect(image, model = "yuv"):
    coords, features = slide_extract(image, model)
    if model == "hsv": 
        scaler = joblib.load(os.path.dirname(__file__) + '/models/scaler_hsv.pkl')
        svc = joblib.load(os.path.dirname(__file__) + '/models/svc_hsv.pkl')
    elif model == "hls":
        scaler = joblib.load(os.path.dirname(__file__) + '/models/scaler_hls.pkl')
        svc = joblib.load(os.path.dirname(__file__) + '/models/svc_hls.pkl')
    else:
        scaler = joblib.load(os.path.dirname(__file__) + '/models/scaler_yuv.pkl')
        svc = joblib.load(os.path.dirname(__file__) + '/models/svc_yuv.pkl')

    features = scaler.transform(features)
    heatmap = Heatmap(image)
    for i in range(len(features)):
        decision = svc.predict([features[i]])
        if decision[0] == 1:
            heatmap.increase_value(coords[i])
    mask = heatmap.construct()
    contour, _ = cv2.findContours(mask, 1, 2)[:2]
    for c in contour:
        if (cv2.contourArea(c) < 30 * 30) or (cv2.contourArea(c) > 140 * 140):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255), 2)
    return image

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(511, 312)
        Form.setStyleSheet("QWidget#Form{\n"
"background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:0.990148, y2:0.971591, stop:0 rgba(106, 199, 199, 255), stop:1 rgba(255, 255, 255, 255));}")
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 0, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 0, 1, 1, 1)
        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("YUV")
        self.comboBox.addItem("HLS")
        self.comboBox.addItem("HSV")
        self.gridLayout.addWidget(self.comboBox, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 3, 1, 1)
        self.pushButton_load = QtWidgets.QPushButton(Form)
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 2, 0, 1, 2)
        self.pushButton_save = QtWidgets.QPushButton(Form)
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 2, 2, 1, 2)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 2)
        self.label_1 = QtWidgets.QLabel(Form)
        self.label_1.setText("")
        self.label_1.setObjectName("label_1")
        self.gridLayout.addWidget(self.label_1, 1, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "SVM ship detection"))
        self.pushButton_3.setText(_translate("Form", "Developers"))
        self.pushButton_4.setText(_translate("Form", "Help"))
        self.label_3.setText(_translate("Form", ""))
        self.pushButton_load.setText(_translate("Form", "Load"))
        self.pushButton_load.clicked.connect(self.open_image)
        self.pushButton_save.setText(_translate("Form", "Save"))
        self.pushButton_save.clicked.connect(self.save_image)
        self.pushButton_3.clicked.connect(self.pop_up_2)
        self.pushButton_4.clicked.connect(self.pop_up)

    def open_image(self):
        fname = QFileDialog.getOpenFileName()
        try:
            self.pixmap = QPixmap(fname[0])
            global file_open_flag
            file_open_flag = True
            t_start = time.time()
            global detected
            color_space = str(self.comboBox.currentText()).lower()
            detected = detect(np.asarray(PIL.Image.open(fname[0])), color_space)
            time_taken = np.round(time.time() - t_start, 2)
            self.label_3.setText("Time taken: " + str(time_taken) + 's')
            self.label_1.setPixmap(self.pixmap)
            cv2.imwrite(os.path.dirname(__file__) + "temp_image.png", cv2.cvtColor(detected, cv2.COLOR_RGB2BGR))
            self.pixmap = QPixmap(os.path.dirname(__file__) +  "temp_image.png")
            self.label_2.setPixmap(self.pixmap)
        except:
            self.pop_up_5()
        
    def save_image(self):
        if file_open_flag:
            saved_file_name, _ = QFileDialog.getSaveFileName()
            try:
                cv2.imwrite(saved_file_name, cv2.cvtColor(detected, cv2.COLOR_RGB2BGR))
            except:
                self.pop_up_3()
        else:
            self.pop_up_4()
        
    def pop_up(self):
        pop = QMessageBox()
        pop.setWindowTitle("Help")
        pop.setText("Step-by-step guide on using the application:\n1.Choose one of the color spaces (YUV, HLS, HSV) that is used when analyzing the image\n2.Choose and upload an image by pressing #Load button (the image will appear in the middle left part of the application)\n3.The algorithm starts working and produces a new image with highlited ships (the image will appear in the middle right part of the application)\n4.The time (in seconds) of the algorithm execution can be seen in the upper right corner of the application\n5.Download the new image on your device by pressing #Save button")
        pop.exec_()
        
    def pop_up_2(self):
        pop2 = QMessageBox()
        pop2.setWindowTitle("Developers")
        pop2.setText("Kulikov Artyom Valerievich\nAlmasyan Sanasar Bagdasarovich\nTyuplyaev Nikita Alekseevich\nYakunin Ivan Vadimovich")
        pop2.exec_()
    
    def pop_up_3(self):
        pop3 = QMessageBox()
        pop3.setWindowTitle("Error")
        pop3.setText("Error has occured. Either saving is cancelled or file extension is wrong (.png or .jpg is required)")
        pop3.exec_()
        
    def pop_up_4(self):
        pop4 = QMessageBox()
        pop4.setWindowTitle("Error")
        pop4.setText("Error has occured. Upload an image before saving the result.")
        pop4.exec_()
        
    def pop_up_5(self):
        pop5 = QMessageBox()
        pop5.setWindowTitle("Error")
        pop5.setText("Error has occured. Image uploading has been cancelled.")
        pop5.exec_()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
