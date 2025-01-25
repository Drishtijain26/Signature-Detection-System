from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 model

# Train model
model.train(data='data.yaml', epochs=50, imgsz=640)
