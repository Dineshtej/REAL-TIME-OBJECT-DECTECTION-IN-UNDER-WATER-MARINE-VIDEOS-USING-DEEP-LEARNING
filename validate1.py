from ultralytics import YOLO

model = YOLO("D:/downloads/underwater/aquarium_project/runs/detect/runs/aquarium_final/weights/best.pt")

model.val(
    data="data.yaml",
    imgsz=640,
    conf=0.25,
    iou=0.45,
    batch=8,
    plots=True,
    save_json=True
)
