import os

BASE = "D:/downloads/underwater/aquarium_pretrain"

for split in ["train", "valid", "test"]:
    imgs = os.listdir(f"{BASE}/{split}/images")
    labels = os.listdir(f"{BASE}/{split}/labels")
    print(f"{split}: images={len(imgs)}, labels={len(labels)}")
