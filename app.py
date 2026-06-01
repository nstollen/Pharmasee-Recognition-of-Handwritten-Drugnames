import os
import io
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

from src.model       import CRNN
from src.fuzzy_match import load_rxnorm_list
from src.inference   import load_model, validate_rxnorm
from src.train       import DEVICE

app = Flask(__name__)
CORS(app)

# ── Load model and data once at startup ───────────────────────────────
print("Loading model...")
MODEL, LABEL_CLASSES = load_model(
    weights_path="models/phase2_weights.pt",
    labels_path="models/label_classes.json"
)

print("Loading RxNorm list...")
RXNORM_LIST = load_rxnorm_list()

# Load optimal threshold
THRESHOLD = 0.001
threshold_path = "models/optimal_threshold.txt"
if os.path.exists(threshold_path):
    with open(threshold_path) as f:
        THRESHOLD = float(f.read().strip())

REAL_WORLD_THRESHOLD = 0.50  # reject anything below 50%

print(f"Model ready. Threshold: {THRESHOLD:.6f}")
print(f"Classes: {len(LABEL_CLASSES)}")
print(f"RxNorm list: {len(RXNORM_LIST)} drugs")


def preprocess_from_bytes(image_bytes, target_w=224, target_h=224):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not decode image")

    # Identical to src/preprocess.py used during training
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    h, w  = img.shape
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img   = cv2.resize(img, (new_w, new_h))

    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    x_off  = (target_w - new_w) // 2
    y_off  = (target_h - new_h) // 2
    padded[y_off:y_off+new_h, x_off:x_off+new_w] = img

    padded = padded.astype(np.float32) / 255.0
    padded = np.expand_dims(padded, axis=-1)
    return padded


def clean_drug_name(name):
    import re
    name = re.sub(r'^\([a-z]\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*', '-', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read and preprocess image
        image_bytes = file.read()
        img         = preprocess_from_bytes(image_bytes)

        # Convert to tensor
        img_t = torch.tensor(img, dtype=torch.float32)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # Model inference
        with torch.no_grad():
            logits = MODEL(img_t)
            probs  = torch.softmax(logits, dim=1)[0]

        # Top-5 predictions
        top_probs, top_idx = torch.topk(probs, 5)
        top_preds = [
            {
                "name" : LABEL_CLASSES[idx.item()],
                "prob" : round(top_probs[i].item(), 6)
            }
            for i, idx in enumerate(top_idx)
        ]

        best_drug  = top_preds[0]["name"]
        best_score = top_preds[0]["prob"]

        # If top prediction confidence is low, flag as uncertain
        # and return all candidates for pharmacist review
        if best_score < THRESHOLD:
            return jsonify({
                "status"           : "UNCERTAIN",
                "predicted_drug"   : best_drug,
                "confidence_score" : round(best_score, 6),
                "confidence_level" : "LOW",
                "threshold_status" : "BELOW THRESHOLD",
                "rxnorm_status"    : "UNCONFIRMED",
                "rxcui"            : None,
                "canonical_name"   : "Review candidates below",
                "top_candidates"   : top_preds,
                "threshold_used"   : round(THRESHOLD, 6),
                "note"             : "Low confidence — please review top candidates"
            })

        # Threshold check
        threshold_status = "ACCEPTED" if best_score >= THRESHOLD else "UNCERTAIN"

        # RxNorm validation
        rxnorm_status = "UNCONFIRMED"
        rxcui         = None
        canonical     = best_drug

        if threshold_status == "ACCEPTED":
            cleaned_name = clean_drug_name(best_drug)
            confirmed, rxcui_result = validate_rxnorm(cleaned_name)
            if confirmed:
                rxnorm_status = "CONFIRMED"
                rxcui         = rxcui_result
                canonical     = best_drug

        return jsonify({
            "status"           : "ACCEPTED" if rxnorm_status == "CONFIRMED" else threshold_status,
            "predicted_drug"   : best_drug,
            "confidence_score" : round(best_score, 6),
            "confidence_level" : "HIGH" if best_score >= 0.70 else "MEDIUM" if best_score >= 0.40 else "LOW",
            "threshold_status" : threshold_status,
            "rxnorm_status"    : rxnorm_status,
            "rxcui"            : rxcui,
            "canonical_name"   : canonical,
            "top_candidates"   : top_preds,
            "threshold_used"   : round(THRESHOLD, 6)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"       : "ok",
        "model_loaded" : True,
        "classes"      : len(LABEL_CLASSES),
        "threshold"    : round(THRESHOLD, 6),
        "device"       : str(DEVICE)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)