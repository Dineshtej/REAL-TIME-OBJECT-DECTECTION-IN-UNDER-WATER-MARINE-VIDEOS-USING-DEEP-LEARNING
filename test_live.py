from ultralytics import YOLO

model = YOLO("D:/downloads/underwater/aquarium_project/runs/detect/train3/weights/best.pt")
model.predict(source=0, conf=0.4, show=True)
