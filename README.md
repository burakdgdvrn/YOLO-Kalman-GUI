# 📷 YOLOv8 & Kalman Filter Object Tracking GUI

This project is a real-time object detection and tracking application developed with *PyQt6. It integrates **YOLOv8* for accurate detection and *Kalman Filter* for smooth trajectory tracking. It also includes a camera calibration tool to correct lens distortion.

## 🚀 Features

* *GUI:* Modern and user-friendly interface powered by PyQt6 and QDarkStyle.
* *Dual Modes:*
    * *Calibration Mode:* Corrects camera distortion in real-time using calibration_data.yaml.
    * *Kalman + YOLO Mode:* Detects objects (Classes 0 & 1) and predicts their movement to reduce noise.
* *Multithreading:* Uses QThread to prevent the interface from freezing during heavy image processing.

## 🛠️ Installation

1.  *Clone the repository:*
    bash
    git clone [https://github.com/burakdgdvrn/YOLO-Kalman-GUI](https://github.com/burakdgdvrn/YOLO-Kalman-GUI)
    cd REPO_NAME
    

2.  *Install dependencies:*
    bash
    pip install -r requirements.txt
    

3.  *Setup Files:*
    * Place your trained model (colab_new_datav5.pt) inside the models/ folder.
    * Ensure calibration_data.yaml is in the main directory.
     

### Detection & Tracking Mode

![e55c5190-955f-48ba-9daa-2c584c15f4b5](https://github.com/user-attachments/assets/b4a636b0-4cf2-4b4d-9678-f642eb4c3613)

![d6c8d34e-976a-4b60-b000-5d397576a206](https://github.com/user-attachments/assets/8641189c-f91f-4089-8501-72fe402eb0bc)

![297515ff-9d04-48f8-8974-ed93911ca8df](https://github.com/user-attachments/assets/897b4849-e7ae-4cd4-a7c5-0982d429971f)

## ▶️ Usage


Run the main application script:


```bash  
python main.py
```

Calibration View: Displays the original vs. undistorted frame.

Kalman + YOLO: Shows the detection bounding box (Green), raw center (Red dot), and Kalman-smoothed center (Blue dot).

## 👥 Developers
Burak Dağdeviren -Çağrı Demir 

Built with Python, OpenCV, and Ultralytics YOLO.
