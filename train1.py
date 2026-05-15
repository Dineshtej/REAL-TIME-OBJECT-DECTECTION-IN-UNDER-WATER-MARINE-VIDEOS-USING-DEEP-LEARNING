from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # stable & fast (CPU friendly)

    model.train(
        data="data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=15, #early stopping
        device="cpu",
        workers=2,
        project="runs",
        name="aquarium_final",
        exist_ok=False,
        seed=42,
        deterministic=True,

        # Augmentations (SAFE for small dataset)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,

        plots=True,
        verbose=True
    )

if __name__ == "__main__":
    main()
