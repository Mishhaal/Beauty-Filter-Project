import os
from app import apply_beauty_filter, PROCESSED_IMAGES_FOLDER

INPUT_DIR = os.path.join("datasets", "ffhq_small")

# Make sure output folder exists
os.makedirs(PROCESSED_IMAGES_FOLDER, exist_ok=True)

count = 0
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    out_path = apply_beauty_filter(in_path)

    if out_path:
        count += 1
        print("✅", fname, "->", out_path)
    else:
        print("❌ Failed:", fname)

print(f"\nDone. Processed {count} images.")
