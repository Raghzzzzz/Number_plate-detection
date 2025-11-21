import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re

# Load YOLOv8 model (custom-trained for Indian license plates)
model = YOLO(r"c:/Users/ragha/Downloads/license-plate-finetune-v1x.onnx")  # Your ONNX model path

# Initialize EasyOCR (English)
reader = easyocr.Reader(['en'])

# Load image
img_path = r"WhatsApp Image 2025-11-21 at 11.10.19_77654b85.jpg"
image = cv2.imread(img_path)

if image is None:
    print(f"Error: Image not found at '{img_path}'")
    exit()

# Run YOLOv8 detection
results = model(image)

# Indian plate pattern (very descriptive, allows for three/four sections)
pattern = r'^([A-Z]{2}[ -]?\d{2}[ -]?[A-Z]{1,2}[ -]?\d{1,4})$'
# Accepts cases like TN01B0000, TN 01 B 0000, TN01 BA 1234, TN 01 BA 1234, etc.

any_plate_found = False

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue
        ocr_results = reader.readtext(plate_img)
        # Combine all lines of text found in this plate crop:
        all_lines = " ".join([item[1] for item in ocr_results])
        all_lines_flat = all_lines.replace("\n", " ").replace("  ", " ").replace("-", "").strip().upper()
        candidate = all_lines_flat.replace(" ", "")
        # Now try the pattern
        if re.match(pattern, candidate):
            print("Detected Number Plate (combined lines):", all_lines_flat)
            any_plate_found = True
        else:
            # Optionally print the raw OCR lines for manual inspection:
            print("OCR lines for region:", [item[1] for item in ocr_results])
            print("Candidate tested:", candidate)

if not any_plate_found:
    print("No valid Indian license plate detected (including 2-line formats).")
