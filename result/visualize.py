# import json
# import os
# import glob
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
#
# # Base directories
# base_dir = "wandb/run-20250508_091816-jq7r73c3/files/media"
# image_dir = os.path.join(base_dir, "images")
# json_dir = os.path.join(base_dir, "metadata/boxes2D")
# rgb_indices = [50, 252, 562, 921]
#
# class_colors = {
#     1: 'red',
#     2: 'blue',
#     3: 'green',
#     4: 'orange',
#     5: 'purple'
# }
#
# def find_lightest_boxes2D(prefix, json_dir):
#     candidate_files = glob.glob(os.path.join(json_dir, f"{prefix}_*.boxes2D.json"))
#     if not candidate_files:
#         return None
#     lightest_file = min(candidate_files, key=os.path.getsize)
#     return lightest_file
#
# fig, axs = plt.subplots(2, len(rgb_indices), figsize=(20, 8))
#
# for col, rgb_idx in enumerate(rgb_indices):
#     for row, modality in enumerate(['rgb', 'thermal']):
#         frame_idx = rgb_idx if modality == 'rgb' else rgb_idx - 1
#         prefix = f"{modality}_{frame_idx}"
#
#         image_files = glob.glob(os.path.join(image_dir, f"{prefix}*.png"))
#         json_file = find_lightest_boxes2D(prefix, json_dir)
#
#         ax = axs[row][col]
#
#         if not image_files or not json_file:
#             ax.axis('off')
#             print(f"Không tìm thấy file cho {prefix}")
#             continue
#
#         image_path = image_files[0]
#         json_path = json_file
#
#         image = Image.open(image_path)
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#
#         ax.imshow(image)
#         ax.set_title(f"{modality.upper()} {frame_idx}", fontsize=10)
#
#         for box in data["box_data"]:
#             pos = box["position"]
#             caption = box["box_caption"]
#             class_id = box["class_id"]
#             x = pos["minX"]
#             y = pos["minY"]
#             width = pos["maxX"] - pos["minX"]
#             height = pos["maxY"] - pos["minY"]
#
#             color = class_colors.get(class_id, 'gray')
#             rect = patches.Rectangle((x, y), width, height, linewidth=1.0,
#                                      edgecolor=color, facecolor='none')
#             ax.add_patch(rect)
#             ax.text(x, y - 5, caption, color='white', fontsize=8,
#                     bbox=dict(facecolor=color, alpha=0.8, pad=1))
#
#         ax.axis('off')
#
# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('teacher.png')
img2 = cv2.imread('student.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for ax, img, label in zip(axs, [img1, img2], ['(a) Teacher Predictions', '(b) Student Predictions']):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.5, -0.1, label, transform=ax.transAxes,
            fontsize=15, ha='center', va='top')

plt.tight_layout()
plt.savefig('prediction.png')
plt.show()
