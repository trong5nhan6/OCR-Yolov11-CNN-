from ultralytics import YOLO

# prompt: download yolo11m.pt
# wget -O yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt


# Load a model
model = YOLO('yolo11m.pt')
yolo_yaml_path = 'data\yolo_data\data.yml'

results = model.train(
    data=yolo_yaml_path,
    epochs=100,
    imgsz=640,
    cache=True,
    patience=20,
    plots=True
)
