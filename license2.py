import cv2
import csv
import time
import easyocr
import re
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk

# Load YOLO models
vehicle_model = YOLO("yolov8m.pt")  # General vehicle detection (cars, trucks, buses, motorcycles)
plate_model = YOLO(r"c:/Users/ragha/Downloads/license-plate-finetune-v1x.onnx")  # License plate detection model

# Vehicle classes according to COCO dataset IDs for car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Confidence thresholds
VEHICLE_CONF_THRESHOLD = 0.5
PLATE_CONF_THRESHOLD = 0.5

# Initialize OCR reader for English
reader = easyocr.Reader(['en'], gpu=False)

# CSV log setup
LOG_FILE = "license_plate_log.csv"
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Plate_Number"])

# Video file or camera input
VIDEO_PATH = r"D:\vehicle_detection\videos\demo.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Tkinter GUI window setup
root = tk.Tk()
root.title("License Plate Detection (High Accuracy)")
label = tk.Label(root)
label.pack()

def process_frame():
    global cap

    try:
        ret, frame = cap.read()

        # Restart video when finished
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            root.after(10, process_frame)
            return

        # Vehicle detection in frame
        vehicle_results = vehicle_model(frame)

        for r in vehicle_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]

                if cls in vehicle_classes and conf > VEHICLE_CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_crop = frame[y1:y2, x1:x2]

                    # License plate detection within vehicle crop
                    plate_results = plate_model(vehicle_crop)

                    for p in plate_results:
                        for pb in p.boxes:
                            pconf = pb.conf[0]
                            if pconf > PLATE_CONF_THRESHOLD:
                                px1, py1, px2, py2 = map(int, pb.xyxy[0])

                                # Adjust plate box to original frame coordinates
                                px1 += x1
                                py1 += y1
                                px2 += x1
                                py2 += y1

                                plate_crop = frame[py1:py2, px1:px2]

                                # OCR on cropped plate image
                                ocr_results = reader.readtext(plate_crop)
                                plate_text = ""

                                if ocr_results:
                                    # Choose the OCR result with the highest confidence
                                    plate_text = max(ocr_results, key=lambda x: x[2])[1]
                                    # Filter to alphanumeric only
                                    plate_text = re.sub(r'[^A-Za-z0-9]', '', plate_text)

                                    if plate_text:  # Only log if text is not empty
                                        # Append to CSV log
                                        with open(LOG_FILE, "a", newline="") as f:
                                            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), plate_text])

                                        print("LICENSE PLATE:", plate_text)

                                # Draw bounding box and label on frame
                                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                                cv2.putText(frame, plate_text, (px1, py1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Draw bounding box for vehicle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert frame to RGB and display in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        label.imgtk = img
        label.configure(image=img)

        root.after(10, process_frame)

    except Exception as e:
        print(f"Error processing frame: {e}")
        root.after(10, process_frame)

process_frame()
root.mainloop()

cap.release()
