import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtGui import QFont, QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import qdarkstyle
from ultralytics import YOLO




class CalibrationThread(QThread):
    frame_ready = pyqtSignal(QImage)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = True
        self.model = YOLO("models/colab_new_datav5.pt")

        fs = cv2.FileStorage("./calibration_data.yaml", cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("K").mat()
        self.dist_coeffs = fs.getNode("D").mat()
        fs.release()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.finished.emit()
            return

        while self.running:
            ret, frame = cap.read()
            if not ret or not self.running:
                break

            h, w = frame.shape[:2]
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

            
            results_orig = self.model(frame, verbose=False)[0]
            results_undist = self.model(undistorted, verbose=False)[0]

            annotated_orig = results_orig.plot()
            annotated_undist = results_undist.plot()
            combined = np.hstack((annotated_orig, annotated_undist))

            rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                              rgb_image.strides[0], QImage.Format.Format_RGB888)
            
            
            if not self.running:
                break
            self.frame_ready.emit(qt_image)

        
        if cap.isOpened():
            cap.release()

        
        del frame, undistorted, annotated_orig, annotated_undist, combined
        self.finished.emit()

    def stop(self):
        
        self.running = False

class KalmanThread(QThread):
    frame_ready = pyqtSignal(QImage)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = True
        self.model = YOLO("models/colab_new_datav5.pt")

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.buffer_x, self.buffer_y = [], []

    def predict(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return int(predicted[0,0]), int(predicted[1,0])

    def smooth(self, x, y, size=5):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_x) > size:
            self.buffer_x.pop(0)
            self.buffer_y.pop(0)
        return int(np.mean(self.buffer_x)), int(np.mean(self.buffer_y))

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in [0, 1]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                px, py = self.predict(cx, cy)
                sx, sy = self.smooth(px, py)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(frame, (sx, sy), 5, (255, 0, 0), -1)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                              rgb_image.strides[0], QImage.Format.Format_RGB888)
            self.frame_ready.emit(qt_image)

        cap.release()
        self.finished.emit()

    def stop(self):
        self.running = False




class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection Application")
        self.setGeometry(200, 100, 1200, 700)
        self.setStyleSheet("background-color: #121212; color: white;")
        self.active_thread = None
        self.active_mode = None
        self.initUI()

    def initUI(self):
        title = QLabel("📷 Object Detection and Calibration Application")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #00bfff; margin-bottom: 20px;")

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setFixedSize(800, 600)
        self.camera_label.setStyleSheet("background-color: #222; border: 2px dashed #00bfff;")
        self.set_placeholder("Camera Output Will Appear Here")

        self.btn_calibration = QPushButton("🔧 Calibration View")
        self.btn_yolo = QPushButton("🧠 Kalman + YOLO Detection")
        self.btn_stop = QPushButton("⏹ Stop")

        for btn in [self.btn_calibration, self.btn_yolo, self.btn_stop]:
            btn.setFixedHeight(50)
            btn.setFont(QFont("Arial", 12))

        self.update_button_styles()

        self.status_label = QLabel("Status: Waiting")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setStyleSheet("color: #bbbbbb; margin-top: 10px;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_calibration)
        left_layout.addWidget(self.btn_yolo)
        left_layout.addWidget(self.btn_stop)
        left_layout.addStretch()
        left_layout.addWidget(self.status_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.camera_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(main_layout)
        self.setLayout(layout)

        
        self.btn_calibration.clicked.connect(self.start_calibration)
        self.btn_yolo.clicked.connect(self.start_kalman)
        self.btn_stop.clicked.connect(self.stop_all)

   
    def update_button_styles(self):
        base_style = """
            QPushButton {{
                background-color: {bg};
                color: white;
                border-radius: 10px;
                border: 2px solid #00bfff;
            }}
            QPushButton:hover {{
                background-color: {hover};
                color: black;
            }}
        """
        if self.active_mode == "calibration":
            self.btn_calibration.setEnabled(False)
            self.btn_yolo.setEnabled(False)
            self.btn_calibration.setStyleSheet(base_style.format(bg="#00bfff", hover="#00bfff"))
            self.btn_yolo.setStyleSheet(base_style.format(bg="#333", hover="#333"))
        elif self.active_mode == "kalman":
            self.btn_yolo.setEnabled(False)
            self.btn_calibration.setEnabled(False)
            self.btn_yolo.setStyleSheet(base_style.format(bg="#00bfff", hover="#00bfff"))
            self.btn_calibration.setStyleSheet(base_style.format(bg="#333", hover="#333"))
        else:
            for btn in [self.btn_calibration, self.btn_yolo]:
                btn.setEnabled(True)
                btn.setStyleSheet(base_style.format(bg="#1e1e1e", hover="#00bfff"))
            self.btn_stop.setStyleSheet(base_style.format(bg="#1e1e1e", hover="#00bfff"))

    def set_placeholder(self, text="Camera Stopped"):
        placeholder = QPixmap(self.camera_label.size())
        placeholder.fill(Qt.GlobalColor.transparent)
        painter = QPainter(placeholder)
        painter.fillRect(placeholder.rect(), QColor("#222"))
        painter.setPen(QColor("#00bfff"))
        painter.setFont(QFont("Arial", 14))
        painter.drawText(placeholder.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
        self.camera_label.setPixmap(placeholder)

    def update_frame(self, qt_image):
        scaled_image = qt_image.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))

  
    def start_calibration(self):
        self.stop_all()
        self.status_label.setText("Status: Running Calibration + YOLO...")
        self.active_thread = CalibrationThread()
        self.active_thread.frame_ready.connect(self.update_frame)
        self.active_thread.finished.connect(self.on_thread_finished)
        self.active_thread.start()
        self.active_mode = "calibration"
        self.update_button_styles()

    def start_kalman(self):
        self.stop_all()
        self.status_label.setText("Status: Running Kalman + YOLO...")
        self.active_thread = KalmanThread()
        self.active_thread.frame_ready.connect(self.update_frame)
        self.active_thread.finished.connect(self.on_thread_finished)
        self.active_thread.start()
        self.active_mode = "kalman"
        self.update_button_styles()

    def stop_all(self):
        
        if self.active_thread:
            self.active_thread.stop()
            self.active_thread.wait(500)
            self.active_thread = None
        self.active_mode = None
        self.status_label.setText("Status: Waiting")
        self.set_placeholder("Camera Stopped")
        self.update_button_styles()

    def on_thread_finished(self):
        
        self.active_thread = None
        self.active_mode = None
        self.status_label.setText("Status: Waiting")
        self.set_placeholder("Camera Stopped")
        self.update_button_styles()




def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
