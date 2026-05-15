from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=0.001,
        patience=8,
        device="cpu",
        workers=2,
        project="runs",
        name="aquarium_yolo",
        pretrained=True
    )

if __name__ == "__main__":
    main()
