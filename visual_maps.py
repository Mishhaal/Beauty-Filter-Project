import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# PATHS (change these)
# -----------------------------
ORIGINAL_PATH = "datasets/ffhq_small/11464.png"
PROCESSED_PATH = "processed_images/processed_0d8b81a5c0324773aa8472ee8c9d8917.jpg"

# -----------------------------
# Load images
# -----------------------------
orig = cv2.imread(ORIGINAL_PATH)
proc = cv2.imread(PROCESSED_PATH)

if orig is None or proc is None:
    raise RuntimeError("Could not load images. Check paths.")

# Resize if needed
if orig.shape != proc.shape:
    proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))

# Convert to grayscale
orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1️⃣ HEATMAP (absolute difference)
# -----------------------------
diff_abs = np.abs(proc_gray.astype(np.int16) - orig_gray.astype(np.int16))
heatmap = cv2.applyColorMap(
    np.clip(diff_abs, 0, 255).astype(np.uint8),
    cv2.COLORMAP_JET
)

cv2.imwrite("heatmap.png", heatmap)

# -----------------------------
# 2️⃣ CORRECTION MAP (signed difference)
# -----------------------------
diff_signed = proc_gray.astype(np.int16) - orig_gray.astype(np.int16)

# shift -255..255 → 0..255 for visualization
corr_map = np.clip(diff_signed + 128, 0, 255).astype(np.uint8)
corr_color = cv2.applyColorMap(corr_map, cv2.COLORMAP_TWILIGHT)

cv2.imwrite("correction_map.png", corr_color)

# -----------------------------
# 3️⃣ 3D SURFACE PLOT (difference)
# -----------------------------
# Downsample for speed
small = diff_abs[::4, ::4]
h, w = small.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, small, cmap='viridis', linewidth=0)
ax.set_title("3D Difference Surface Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Difference Intensity")

plt.tight_layout()
plt.savefig("difference_3d_plot.png", dpi=200)
plt.close()

print("✅ Generated:")
print(" - heatmap.png")
print(" - correction_map.png")
print(" - difference_3d_plot.png")
