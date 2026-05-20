import os
import sys
import cv2
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ----- CONFIG -----
BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "datasets" / "friends_dataset"
CROP_DIR  = BASE_DIR / "datasets" / "friends_cropped"
IMG_SIZE  = 48   # FER2013 standard size
PADDING   = 20   # Extra pixels around detected face box

# All emotions the system supports (add more here to extend later)
ALL_EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise", "fear", "disgust"]


def load_face_detector():
    """Load MTCNN detector (already in your requirements.txt)."""
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        print("[√] MTCNN face detector loaded")
        return detector
    except ImportError:
        print("[X] MTCNN not found. Run: pip install mtcnn")
        raise


def crop_and_save(detector, src_path: Path, dst_path: Path) -> int:
    """Detect faces in one image, save each face crop. Returns count of faces saved."""
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        print(f"  [!] Could not read: {src_path.name}")
        return 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if not results:
        print(f"  [!] No face found in: {src_path.name}")
        return 0

    saved = 0
    h, w = img_bgr.shape[:2]

    for i, face_data in enumerate(results):
        confidence = face_data["confidence"]
        if confidence < 0.85:
            continue  # Skip low-confidence detections

        x, y, fw, fh = face_data["box"]

        # Apply padding while staying within image boundaries
        x1 = max(0, x - PADDING)
        y1 = max(0, y - PADDING)
        x2 = min(w, x + fw + PADDING)
        y2 = min(h, y + fh + PADDING)

        face_crop = img_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Convert to grayscale + resize to FER2013 standard
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))

        # Name: original_stem + face_index (handles multiple faces per photo)
        stem = src_path.stem
        out_name = f"{stem}_face{i}.jpg" if len(results) > 1 else f"{stem}.jpg"
        out_file = dst_path / out_name
        cv2.imwrite(str(out_file), face_resized)
        saved += 1

    return saved


def process_emotion(detector, emotion: str) -> tuple[int, int]:
    """Process all raw images for one emotion. Returns (total_input, total_saved)."""
    src_dir = RAW_DIR / emotion
    dst_dir = CROP_DIR / emotion

    if not src_dir.exists():
        return 0, 0

    image_files = [
        f for f in src_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if not image_files:
        print(f"  Folder exists but no images found for: {emotion}. Skipping.")
        return 0, 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing [{emotion}]: {len(image_files)} raw image(s)...")

    total_saved = 0
    for img_path in image_files:
        count = crop_and_save(detector, img_path, dst_dir)
        if count > 0:
            print(f"  [√] {img_path.name}  ->  {count} face(s) saved")
        total_saved += count

    return len(image_files), total_saved


def main():
    print("=" * 60)
    print("  STEP 1 — Auto-Crop Friends' Faces")
    print("=" * 60)
    print(f"  Raw photos from : {RAW_DIR.resolve()}")
    print(f"  Saving crops to : {CROP_DIR.resolve()}")
    print("=" * 60)

    detector = load_face_detector()
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    grand_input  = 0
    grand_output = 0

    for emotion in ALL_EMOTIONS:
        n_in, n_out = process_emotion(detector, emotion)
        grand_input  += n_in
        grand_output += n_out

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total raw images  : {grand_input}")
    print(f"  Total face crops  : {grand_output}")

    if grand_output == 0:
        print("\n[!] No faces were saved!")
        print("   -> Make sure you dropped photos into the emotion folders.")
        print(f"   -> Raw photos go in: {RAW_DIR.resolve()}")
    else:
        print("\n[√] Done!  Now run:")
        print("   venv\\Scripts\\python.exe scripts\\step2_train_custom_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
