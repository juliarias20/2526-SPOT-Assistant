from ultralytics import YOLO

# Load pretrained YOLOv8 nano (downloads once, cached after)
model = YOLO("yolov8n.pt")

# Fine-tune on custom dataset
results = model.train(
    data = "data.yaml",
    epochs = 100,
    imgsz = 640,
    batch = 32, # increase if using multiple GPUs
    device = 0, # GPU index (0 = first GPU)
    project = "models",
    name = "yolo-custom",
    exist_ok= True,
)

print("Training complete.")
print(f"Best weights: models/yolo-custom/weights/best.pt")