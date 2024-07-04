**Project Title**

OBJECT_DETECTION_USING_YOLO

**Project Description**

This project demonstrates object detection using the YOLOv8n model in a Streamlit application. Users can upload a video file, and the application will perform real-time object detection on the video frames, displaying the detected objects with bounding boxes and class labels. The project leverages the COCO-pretrained YOLOv8n model for accurate and efficient object detection.

**Table of Contents**

Project Goals
Usage Instructions
Code Explanation
Credits


**Project Goals**

To provide a user-friendly interface for object detection using YOLOv8n.
To enable users to upload video files and perform real-time object detection.
To display the detected objects with bounding boxes and class labels on the video frames.

**Usage Instructions**

Run the Streamlit application:
streamlit run app.py
Upload a video file:
Open the Streamlit app in your browser.
Use the sidebar to upload a video file (supported formats: mp4, avi, mov, mkv).
View object detection results:
The uploaded video will be processed frame by frame.
Detected objects will be displayed with bounding boxes and class labels.
The processed video frames will be shown in real-time in the app.

**Code Explanation**

Imports and Model Loading:
import streamlit as st
import cv2
import numpy as np
import tempfile
import random
from ultralytics import YOLO
Generate Random Colours for Class List:
Random colours are generated for each class to visually differentiate between detected objects.

Streamlit App:
The Streamlit app is created to upload video files, perform object detection, and display the results in real-time.

**Credits**

This project uses the following libraries:

Streamlit for creating the web app.
OpenCV for image and video processing.
NumPy for numerical operations.
Ultralytics YOLO for the object detection model.
advanced features, such as tracking, handling multiple video formats, and improving the UI.
Feel free to fork this repository, contribute, or report any issues. Thank you for checking out this project!
