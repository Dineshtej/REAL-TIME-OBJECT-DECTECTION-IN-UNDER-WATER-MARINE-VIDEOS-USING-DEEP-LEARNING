from ultralytics import YOLO

model = YOLO("D:/downloads/underwater/aquarium_project/runs/detect/runs/aquarium_final/weights/best.pt")

model.predict(
    source="D:\downloads\WhatsApp Video 2026-02-03 at 10.53.06 AM.mp4",
    conf=0.6,
    classes=[0,1,2,3,4,5,6],
    save=True
)
