import pandas as pd
import matplotlib.pyplot as plt

# Load results.csv
df = pd.read_csv("D:/downloads/underwater/aquarium_project/runs/detect/train3/results.csv")

# Epochs
epochs = df.index + 1

# Plot Accuracy Metrics
plt.figure(figsize=(10, 6))
plt.plot(epochs, df["metrics/precision(B)"], label="Precision")
plt.plot(epochs, df["metrics/recall(B)"], label="Recall")
plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5")
plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("YOLOv8 Accuracy Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
