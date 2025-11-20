📘 README – License Plate Detection System (YOLO + EasyOCR + Tkinter GUI)
🚀 Overview

This project is a real-time license plate detection and recognition system built using:

YOLOv8m → Vehicle detection

Fine-tuned License Plate Model (.onnx) → Plate detection

EasyOCR → Plate text recognition

Tkinter → GUI display

OpenCV → Video processing

The system reads frames from a video file or webcam, detects vehicles, finds license plates inside them, extracts plate text using OCR, and displays the results in a live GUI.

Detected license plate numbers are also saved to a CSV file along with timestamps.

🎯 Features
✔ Vehicle Detection

Detects the following using YOLOv8m:

Cars

Motorcycles

Buses

Trucks

✔ License Plate Detection

Detects plates inside each vehicle using a fine-tuned ONNX license plate detection model.

✔ OCR Recognition

Reads license plate text using EasyOCR with filtering for:

Highest-confidence result

Alphanumeric cleanup (removes symbols/noise)

✔ GUI Display

Shows:

Live video feed

Vehicle bounding boxes (green)

Plate bounding boxes (yellow)

Recognized plate text

✔ Logging

Detected plates are logged to:  license_plate_log.csv

With:

Timestamp

Plate number

📂 Project Structure
project/
│
├── main.py                       # Main detection script
├── license_plate_log.csv         # Log file (auto-created)
├── yolov8m.pt                    # YOLO vehicle model
├── license-plate-finetune.onnx   # Plate detection model
└── videos/
    └── demo.mp4                  # Input video

🛠 Requirements

Install required packages:

pip install ultralytics
pip install easyocr
pip install opencv-python
pip install pillow

Note:
EasyOCR may take time to initialize on first use.

⚙️ Configuration
1. Update your model paths:
vehicle_model = YOLO("yolov8m.pt")
plate_model = YOLO("c:/.../license-plate-finetune-v1x.onnx")

2. Update your video path:
VIDEO_PATH = r"D:\vehicle_detection\videos\demo.mp4"

▶️ How to Run

Run the script:

python main.py


The GUI window will open and:

Display each frame

Draw bounding boxes

Show detected plate text live

Save plate numbers to license_plate_log.csv

📌 CSV Log Output Example

license_plate_log.csv:

Timestamp,Plate_Number
2025-01-18 15:22:45,KL55AB1234
2025-01-18 15:22:48,KA03MN9087

🧠 How It Works

Vehicle Detection
YOLOv8 detects vehicles → returns bounding boxes.

Plate Detection inside Vehicle
A fine-tuned ONNX model detects license plates only inside that crop.

OCR on Plate Crop
EasyOCR reads text → highest confidence result chosen.

Filtering
Regex removes:

Spaces

Special characters

Non-alphanumeric symbols

Visualization
Tkinter continuously displays updated frames.

❗ Known Limitations

OCR accuracy depends on plate clarity.

Low-resolution or motion-blurred frames may reduce detection accuracy.

Some country plates may require additional OCR post-processing.

📌 Future Enhancements (Optional)

Add DeepSORT tracking for stable plate IDs.

Add database storage of vehicle entries.

Add automatic screenshot saving of detected plates.

Add lane-wise counting and vehicle classification.

    
