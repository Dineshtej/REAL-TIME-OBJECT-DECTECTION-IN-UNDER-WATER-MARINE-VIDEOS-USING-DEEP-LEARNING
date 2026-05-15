from ultralytics import YOLO

model = YOLO("D:/downloads/underwater/aquarium_project/runs/detect/runs/aquarium_final/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
