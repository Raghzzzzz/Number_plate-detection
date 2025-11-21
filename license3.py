import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re

# Load YOLOv8 model (custom-trained for Indian license plates)
model = YOLO(r"c:/Users/ragha/Downloads/license-plate-finetune-v1x.onnx")  # Replace with your model path

# Initialize EasyOCR (English)
reader = easyocr.Reader(['en'])

# Load image
img_path = r"D:\license\WhatsApp Image 2025-11-21 at 11.40.56_dc01ed20.jpg"  # Replace with your image path
image = cv2.imread(img_path)

if image is None:
    print(f"Error: Image not found at '{img_path}'")
    exit()

# Run YOLOv8 detection
results = model(image)

# Prepare Indian plate format (allows optional space/hyphen)
pattern = r'^[A-Z]{2}[ -]?\d{2}[ -]?[A-Z]{1,2}[ -]?\d{4}$'

any_plate_found = False

# Post-process detections
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue
        ocr_results = reader.readtext(plate_img)
        for (bbox, text, conf) in ocr_results:
            candidate = text.replace(" ", "").replace("-", "").upper()
            # Format with optional space/hyphen for validation
            if re.match(pattern, candidate):
                print("Detected Number Plate:", text.strip())
                any_plate_found = True

if not any_plate_found:
    print("No valid Indian license plate detected.")
