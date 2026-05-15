from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("D:/downloads/underwater/aquarium_project/runs/detect/train3/weights/best.pt")

KNOWN_CLASSES = list(model.names.values())
CONF_THRESHOLD = 0.70

def detect(source):
    results = model.predict(
        source=source,
        stream=True,
        conf=0.25,          # keep low here, filter manually
        device="cpu"
    )

    for r in results:
        frame = r.orig_img

        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ---- UNKNOWN HANDLING ----
                if conf < CONF_THRESHOLD:
                    label = "OTHER"
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.imshow("Aquarium Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
