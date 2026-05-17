import os
import time
from app import apply_beauty_filter

DATASET_DIR = "datasets/ffhq_small"

times = []
count = 0

for img in os.listdir(DATASET_DIR):
    if not img.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(DATASET_DIR, img)

    start = time.time()
    apply_beauty_filter(path)
    end = time.time()

    times.append(end - start)
    count += 1

print("Images processed:", count)
print("Average time per image:", sum(times) / len(times))
print("Total time:", sum(times))
