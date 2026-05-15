from ultralytics import YOLO
model = YOLO(r"C:\Users\dines\Downloads\underwater\aquarium_project\runs\detect\runs\aquarium_final\weights\best.pt")

results = model.predict(
    source=0,              # webcam
    stream=True,            # 🔥 VERY IMPORTANT (prevents RAM crash)
    conf=0.65,              # 🔥 filters false detections
    iou=0.5,
    classes=[0,1,2,3,4,5,6],# 🔥 ONLY aquarium classes
    device="cpu",
    show=True
)

for r in results:
    pass