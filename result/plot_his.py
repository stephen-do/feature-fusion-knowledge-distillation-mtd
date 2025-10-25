import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
from tqdm import tqdm
from collections import defaultdict
import os
import seaborn as sns

# --- Path ---
image_dir = "../dataset/FLIR_Aligned/images_thermal_train/data"
ann_path = "../dataset/FLIR_Aligned/meta/thermal/flir_train.json"

# --- COCO ---
coco = COCO(ann_path)
cats = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

# --- Storage ---
pixel_values = {
    "R": defaultdict(list),
    "G": defaultdict(list),
    "B": defaultdict(list)
}

# --- Read annotations ---
for ann_id in tqdm(coco.getAnnIds()):
    ann = coco.loadAnns(ann_id)[0]
    cat_id = ann["category_id"]
    image_id = ann["image_id"]
    image_info = coco.loadImgs([image_id])[0]

    img_path = os.path.join(image_dir, image_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y, w, h = map(int, ann["bbox"])
    crop = rgb_img[y:y + h, x:x + w]  # (H, W, 3)

    R, G, B = cv2.split(crop)
    for channel, matrix in zip(["R", "G", "B"], [R, G, B]):
        norm = matrix.astype(np.float32) / 255.0
        pixel_values[channel][cat_id].extend(norm.flatten())

# --- Plot ---
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

channel_names = ["R", "G", "B"]
channel_titles = ["Red Channel", "Green Channel", "Blue Channel"]
colors = {
    "Person": "C1",   # orange
    "Car": "C0",      # blue
    "Bicycle": "C2"   # green
}

for idx, (channel, ax) in enumerate(zip(channel_names, axes)):
    for cat_id, pixels in pixel_values[channel].items():
        cat_name = cat_id_to_name[cat_id]
        ax.hist(pixels, bins=256, alpha=0.6, label=cat_name,
                density=True, color=colors.get(cat_name, "gray"))
    ax.set_title(channel_titles[idx], fontsize=14, fontweight='bold')
    ax.set_xlabel("Normalized Pixel Intensity", fontsize=12)
    if idx == 0:
        ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)

plt.suptitle("RGB Pixel Intensity Distribution per Class", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("rgb_pixel_intensity_channels.png", dpi=300, bbox_inches="tight")
plt.show()
