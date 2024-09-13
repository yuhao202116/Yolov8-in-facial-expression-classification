import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
                             QGroupBox, QDialog, QApplication, QLabel, QTextEdit)
import sys
from ultralytics import YOLO


import webbrowser


url1 = "https://www.bilibili.com/video/BV1AM4y1M71p/?spm_id_from=333.337.search-card.all.click"
url2 = "https://www.bilibili.com/video/BV1Ty4y1N7tr/?spm_id_from=333.337.search-card.all.click"
url3 = "https://www.bilibili.com/video/BV1ps4y1d73V/?spm_id_from=333.788.top_right_bar_window_custom_collection.content.click&vd_source=503112d3fd9385ce4ac33a2cb744500d"
def open_web(url):
    webbrowser.open(url)


def if_sad():
    open_web(url1)


def if_angry():
    open_web(url2)


def if_neural():
    open_web(url3)





def predict(frame):
    # Convert frame to the format expected by the YOLO model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model = YOLO("./runs/classify/train5(minset_30epoch)/weights/best.pt")
    results = model(frame_rgb)
    return results




class UI(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Emokiller')
        self.resize(800, 600)
        self.setStyleSheet("""
            color: #333333;
            font-size: 10pt;
            font-family: "黑体";
        """)
        self.cam_img = None

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)  # Read-only mode
        self.text_edit.setFixedSize(600, 600)

        # Set up video capture
        self.cap = cv2.VideoCapture(0)

        # Initialize UI components
        self.initUI()

        # Set up a timer to update the frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 milliseconds


        # self.driver = webdriver.Chrome()

        self.current_expression = None

    def initUI(self):
        self.label = QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 600, 600))
        self.label.setScaledContents(True)

        self.button1 = QPushButton('做事', self)
        self.button1.clicked.connect(self.on_click1)
        self.button2 = QPushButton('开始识别', self)
        self.button2.clicked.connect(self.on_click2)

        box1 = QGroupBox("实时图像")
        layout_box1 = QHBoxLayout()
        layout_box1.addWidget(self.label)
        layout_box1.addSpacing(30)
        layout_box1.addWidget(self.text_edit)
        box1.setLayout(layout_box1)

        box2 = QGroupBox("功能组1")
        layout_box2 = QHBoxLayout()
        layout_box2.addWidget(self.button1)
        layout_box2.addWidget(self.button2)
        box2.setLayout(layout_box2)

        container = QVBoxLayout()
        container.addWidget(box1)
        container.addWidget(box2)

        central_widget = QWidget()
        central_widget.setLayout(container)

        self.setLayout(container)

    def update_frame(self):
        ret, frame = self.cap.read()

        if ret:
         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         image = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
         pixmap = QPixmap.fromImage(image)
         self.label.setPixmap(pixmap.scaled(450, 450))



    def on_click1(self):
        self.text_edit.append("doing something!")
        self.react()




    def on_click2(self):
        # Start or stop recognition
        self.text_edit.append("clicked! Start recognition...")
        ret, frame = self.cap.read()
        results = predict(frame)[0]


        boxes = results.boxes  # Boxes object for bounding box outputs
        masks = results.masks  # Masks object for segmentation masks outputs
        keypoints = results.keypoints  # Keypoints object for pose outputs
        probs = results.probs  # Probs object for classification outputs
        obb = results.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk

        class_names = results.names
        top1 = probs.top1
        self.text_edit.append(f"预测表情为: {class_names[top1]}")
        self.current_expression = class_names[top1]


    def react(self):
        et = self.current_expression
        if et == 'neutral':
            self.text_edit.append("监测到neutral，是时候该学习了！")
            if_neural()

        if et == 'sad':
            self.text_edit.append("监测到sad，是时候该看点开心的了")
            if_sad()

        if et == 'angery':
            self.text_edit.append("监测到angery,平复下心情吧")
            if_angry()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = UI()
    mainWin.show()
    sys.exit(app.exec_())
