from flask import Flask, render_template, request, send_file
import os
import uuid
import cv2
import numpy as np
from PIL import Image, ImageFilter
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -----------------------------
# Paths (ABSOLUTE = fewer errors)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_IMAGES_FOLDER = os.path.join(BASE_DIR, "processed_images")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        print("---- /upload called ----")
        print("request.files keys:", list(request.files.keys()))

        if "file" not in request.files:
            return "No file field found. Your HTML input name must be 'file'.", 400

        f = request.files["file"]
        print("filename:", f.filename)
        print("content_type:", f.content_type)

        if not f or f.filename.strip() == "":
            return "No file selected.", 400

        safe_name = secure_filename(f.filename)
        ext = os.path.splitext(safe_name)[1].lower()
        print("ext:", ext)

        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if ext not in allowed:
            return f"Unsupported type {ext}. Allowed: {allowed}", 400

        upload_name = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_name)
        f.save(upload_path)

        print("saved to:", upload_path)
        print("saved exists?:", os.path.exists(upload_path), "size:", os.path.getsize(upload_path))

        processed_path = apply_beauty_filter(upload_path)
        print("processed_path:", processed_path)

        if not processed_path:
            return "Filter failed: apply_beauty_filter returned None. Check terminal logs above.", 500

        if not os.path.exists(processed_path):
            return f"Filter failed: output file not found at {processed_path}", 500

        return send_file(processed_path, as_attachment=True, download_name="processed_" + safe_name)

    except Exception as e:
        import traceback
        print("UPLOAD ERROR:", repr(e))
        traceback.print_exc()
        return f"Server error: {e}", 500



# -----------------------------
# Skin Detection
# -----------------------------
def detect_skin(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


# -----------------------------
# Smooth Skin
# -----------------------------
def smooth_skin(image_bgr):
    return cv2.bilateralFilter(image_bgr, d=15, sigmaColor=75, sigmaSpace=75)


# -----------------------------
# Sharpen Image
# -----------------------------
def unsharp(image_bgr):
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    sharp = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return cv2.cvtColor(np.array(sharp), cv2.COLOR_RGB2BGR)


# -----------------------------
# Blend (smooth skin only)
# -----------------------------
def blend_images(original_bgr, smooth_bgr, mask_gray):
    mask_3 = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    original_f = original_bgr.astype(np.float32)
    smooth_f = smooth_bgr.astype(np.float32)

    blended = smooth_f * mask_3 + original_f * (1.0 - mask_3)
    return np.clip(blended, 0, 255).astype(np.uint8)
    # Final Image=(Smooth×Mask)+(Original×(1−Mask))


# -----------------------------
# Main Function for Beauty Filter
# -----------------------------
def apply_beauty_filter(path):
    # Read image (important: OpenCV fails silently by returning None)
    img = cv2.imread(path)
    print("cv2.imread returned:", "OK" if img is not None else "None", "for path:", path)

 

    mask = detect_skin(img)
    smooth = smooth_skin(img)
    sharp = unsharp(img)
    final = blend_images(sharp, smooth, mask)

    # Unique output name
    out_name = f"processed_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(PROCESSED_IMAGES_FOLDER, out_name)

    ok = cv2.imwrite(out_path, final)
    if not ok:
        print("apply_beauty_filter: cv2.imwrite failed for:", out_path)
        return None

    return out_path


if __name__ == "__main__":
    app.run(debug=True)
