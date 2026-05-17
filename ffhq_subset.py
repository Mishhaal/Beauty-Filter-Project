import os
import shutil
import random

SRC_DIR = "datasets/ffhq_full"
DST_DIR = "datasets/ffhq_small"
NUM_IMAGES = 200

os.makedirs(DST_DIR, exist_ok=True)

# Collect only image files
images = [
    f for f in os.listdir(SRC_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if len(images) < NUM_IMAGES:
    raise ValueError("Not enough images in source folder")

# Randomly sample
selected = random.sample(images, NUM_IMAGES)

for fname in selected:
    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, fname)
    shutil.copy(src_path, dst_path)

print(f"✅ Random FFHQ subset created: {NUM_IMAGES} images")
print(f"📁 Location: {DST_DIR}")
