import streamlit as st
import cv2
import numpy as np
import tempfile
import random
from ultralytics import YOLO

# Load the COCO-pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Load class names from coco.txt
with open(r"C:\Users\Harsha\Desktop\FULL STACK DATA SCIENCE\YOLO\MY YOLO\coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Streamlit app
st.title("Object Detection using YOLO")

# Sidebar for user input
st.sidebar.write("Upload a video file to perform object detection")
uploaded_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frame_wid = 640
    frame_hyt = 480

    if not cap.isOpened():
        st.error("Cannot open video")
    else:
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("End of video")
                break

            # Predict on image
            detect_params = model.predict(source=[frame], conf=0.45, save=False)

            # Convert tensor array to numpy
            DP = detect_params[0].numpy()

            if len(DP) != 0:
                for i in range(len(detect_params[0])):
                    boxes = detect_params[0].boxes
                    box = boxes[i]  # returns one box
                    clsID = box.cls.numpy()[0]
                    conf = box.conf.numpy()[0]
                    bb = box.xyxy.numpy()[0]

                    cv2.rectangle(
                        frame,
                        (int(bb[0]), int(bb[1])),
                        (int(bb[2]), int(bb[3])),
                        detection_colors[int(clsID)],
                        3,
                    )

                    # Display class name and confidence
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(
                        frame,
                        class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                        (int(bb[0]), int(bb[1]) - 10),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )

            # Resize frame for display
            frame = cv2.resize(frame, (frame_wid, frame_hyt))

            # Display the resulting frame
            stframe.image(frame, channels="BGR")

    cap.release()
