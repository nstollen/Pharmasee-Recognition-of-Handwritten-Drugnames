import cv2
import numpy as np


def preprocess_image(image_path, target_w=128, target_h=32):
    # Use numpy to read file to handle special character filenames on Windows
    with open(image_path, "rb") as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    # Bilateral filter denoising
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Aspect-ratio-preserving resize
    h, w = img.shape
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img   = cv2.resize(img, (new_w, new_h))

    # Zero-pad to fixed size
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    x_offset = (target_w - new_w) // 2
    y_offset  = (target_h - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

    # Normalize to [0, 1]
    padded = padded.astype(np.float32) / 255.0

    # Add channel dimension (H, W, C)
    padded = np.expand_dims(padded, axis=-1)

    return padded


def preprocess_batch(image_paths, target_w=128, target_h=32):
    batch = []
    for path in image_paths:
        try:
            img = preprocess_image(path, target_w, target_h)
            batch.append(img)
        except ValueError as e:
            print(f"Skipping: {e}")
    return np.array(batch)