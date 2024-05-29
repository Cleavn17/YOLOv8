from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8s.yaml')

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data='dataset.yaml', epochs=50)
