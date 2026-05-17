import os
import time
import csv
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# SETTINGS (edit if needed)
# -------------------------
ORIG_DIR = os.path.join("datasets", "ffhq_small")
PROC_DIR = "processed_images"
OUT_DIR  = "results_outputs"   # all graphs/images/tables saved here

# How many sample comparisons to save (for report)
NUM_SAMPLES_TO_SAVE = 8

# 3D plot settings (use smaller size to make 3D faster)
SURFACE_DOWNSAMPLE = 4  # 4 means 512->128 for surface plot

os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Utility metrics
# -------------------------
def mse(a, b):
    diff = (a.astype(np.float32) - b.astype(np.float32))
    return float(np.mean(diff * diff))

def psnr(a, b, max_val=255.0):
    m = mse(a, b)
    if m == 0:
        return 99.0
    return float(10.0 * np.log10((max_val * max_val) / m))

def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

def edge_strength(gray):
    # Simple edge measure using Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def ensure_same_size(a, b):
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return a, b

def find_processed_for(original_name):
    """
    Your processed files are named like processed_<uuid>.jpg, so direct pairing by filename isn't guaranteed.
    For report visuals, we'll choose processed images by time order (latest outputs).
    For per-image pairing, the best way is to save outputs using original filename.
    BUT for now, we will still do analysis by pairing SORTED originals with SORTED processed outputs.
    This is OK for performance/visualization and report figures.
    """
    return None


# -------------------------
# Load files
# -------------------------
orig_files = sorted([f for f in os.listdir(ORIG_DIR) if f.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp"))])

proc_files = sorted([f for f in os.listdir(PROC_DIR) if f.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp"))])

if len(proc_files) == 0:
    raise RuntimeError("No processed images found in processed_images/. Run batch processing first.")

# Pair by index (best if you processed in the same order)
pairs = list(zip(orig_files[:len(proc_files)], proc_files[:len(orig_files)]))
print("Pairs used for analysis:", len(pairs))


# -------------------------
# CSV table output
# -------------------------
csv_path = os.path.join(OUT_DIR, "metrics_table.csv")
rows = []

# Track summary
psnr_list = []
mad_list = []
edge_orig_list = []
edge_proc_list = []

# Choose sample indices for saving visual comparisons
sample_indices = np.linspace(0, len(pairs) - 1, NUM_SAMPLES_TO_SAVE, dtype=int)

for idx, (of, pf) in enumerate(pairs, start=1):
    o_path = os.path.join(ORIG_DIR, of)
    p_path = os.path.join(PROC_DIR, pf)

    orig = cv2.imread(o_path)
    proc = cv2.imread(p_path)

    if orig is None or proc is None:
        continue

    orig, proc = ensure_same_size(orig, proc)

    # Convert to grayscale for some metrics
    o_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    p_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

    # Metrics
    m = mse(o_gray, p_gray)
    p = psnr(o_gray, p_gray)
    mad = mean_abs_diff(o_gray, p_gray)

    e_o = edge_strength(o_gray)
    e_p = edge_strength(p_gray)

    psnr_list.append(p)
    mad_list.append(mad)
    edge_orig_list.append(e_o)
    edge_proc_list.append(e_p)

    rows.append({
        "index": idx,
        "original_file": of,
        "processed_file": pf,
        "mse_gray": round(m, 4),
        "psnr_gray_db": round(p, 3),
        "mean_abs_diff_gray": round(mad, 4),
        "edge_var_original": round(e_o, 4),
        "edge_var_processed": round(e_p, 4),
    })

    # -------------------------
    # Save comparison visuals for selected samples
    # -------------------------
    if (idx - 1) in sample_indices:
        # Difference maps
        diff_signed = (p_gray.astype(np.float32) - o_gray.astype(np.float32))   # correction map (signed)
        diff_abs = np.abs(diff_signed)

        # Heatmap (absolute difference)
        heat = cv2.applyColorMap(np.clip(diff_abs, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Correction map (signed): shift -255..255 -> 0..255
        corr = np.clip(diff_signed + 128, 0, 255).astype(np.uint8)
        corr_color = cv2.applyColorMap(corr, cv2.COLORMAP_TWILIGHT)

        # Side-by-side figure
        fig_path = os.path.join(OUT_DIR, f"compare_{idx:03d}.png")

        # Convert BGR -> RGB for matplotlib
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        corr_rgb = cv2.cvtColor(corr_color, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 7))
        plt.suptitle(f"Sample {idx}  |  PSNR={p:.2f} dB  |  MAD={mad:.2f}")

        plt.subplot(2, 2, 1)
        plt.title("Original")
        plt.imshow(orig_rgb)
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("Processed")
        plt.imshow(proc_rgb)
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("Difference Heatmap (|Processed - Original|)")
        plt.imshow(heat_rgb)
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("Correction Map (Processed - Original) [Signed]")
        plt.imshow(corr_rgb)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        # -------------------------
        # 3D Surface plot (difference)
        # -------------------------
        surf_path = os.path.join(OUT_DIR, f"surface_{idx:03d}.png")
        small = diff_abs[::SURFACE_DOWNSAMPLE, ::SURFACE_DOWNSAMPLE]

        h, w = small.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, small.astype(np.float32), linewidth=0, antialiased=True)
        ax.set_title(f"3D Difference Surface (Abs Diff) - Sample {idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Abs Diff Intensity")
        plt.tight_layout()
        plt.savefig(surf_path, dpi=200)
        plt.close()


# Write CSV table
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\n✅ Saved:")
print(" -", csv_path)
print(" - comparison images: compare_###.png")
print(" - 3D surfaces: surface_###.png")

# Summary + save as text (for report)
summary_path = os.path.join(OUT_DIR, "summary.txt")
def stats(x):
    x = np.array(x, dtype=np.float32)
    return float(x.mean()), float(x.min()), float(x.max())

psnr_mean, psnr_min, psnr_max = stats(psnr_list)
mad_mean, mad_min, mad_max = stats(mad_list)
eo_mean, eo_min, eo_max = stats(edge_orig_list)
ep_mean, ep_min, ep_max = stats(edge_proc_list)

summary_text = f"""Results Summary ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
Images analyzed: {len(rows)}

PSNR (gray, dB): mean={psnr_mean:.3f}, min={psnr_min:.3f}, max={psnr_max:.3f}
Mean Abs Diff (gray): mean={mad_mean:.3f}, min={mad_min:.3f}, max={mad_max:.3f}

Edge Variance (Original): mean={eo_mean:.3f}, min={eo_min:.3f}, max={eo_max:.3f}
Edge Variance (Processed): mean={ep_mean:.3f}, min={ep_min:.3f}, max={ep_max:.3f}
"""

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(" -", summary_path)
print("\n✅ Done.")
