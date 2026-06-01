import os
import json
import torch
import numpy as np
import pandas as pd

from src.preprocess  import preprocess_image
from src.model       import CRNN
from src.inference   import load_model
from src.train       import DEVICE, MEDICINE_DIR


def tune_threshold():
    print("\n── Threshold Tuning on Validation Set ──────────")

    model, label_classes = load_model()

    img_dir  = os.path.join(MEDICINE_DIR, "images", "validate")
    csv_path = os.path.join(MEDICINE_DIR, "validate_clean_final.csv")
    df       = pd.read_csv(csv_path)

    all_scores  = []
    all_correct = []

    for _, row in df.iterrows():
        filename = str(row["FILENAME"]).strip()
        truth    = str(row["IDENTITY"]).strip().lower()
        fpath    = os.path.join(img_dir, filename)

        if not os.path.exists(fpath):
            continue

        try:
            img   = preprocess_image(fpath, target_w=224, target_h=224)
            img_t = torch.tensor(img, dtype=torch.float32)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_t)
                probs  = torch.softmax(logits, dim=1)[0]

            top_prob, top_idx = torch.topk(probs, 1)
            pred  = label_classes[top_idx[0].item()].lower()
            score = top_prob[0].item()

            all_scores.append(score)
            all_correct.append(pred.strip() == truth.strip())

        except Exception as e:
            continue

    print(f"Processed: {len(all_scores)} validation images")
    print(f"\nScore statistics:")
    print(f"  Min  : {min(all_scores):.6f}")
    print(f"  Max  : {max(all_scores):.6f}")
    print(f"  Mean : {sum(all_scores)/len(all_scores):.6f}")
    print(f"  Median: {sorted(all_scores)[len(all_scores)//2]:.6f}")

    # Sweep thresholds based on actual score range
    max_score = max(all_scores)
    thresholds = np.linspace(0.001, max_score * 0.95, 20)

    best_f1        = 0.0
    best_threshold = thresholds[0]
    best_precision = 0.0
    best_recall    = 0.0

    print(f"\n{'Threshold':>12} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Accepted':>10}")
    print("-" * 58)

    for t in thresholds:
        tp = fp = fn = 0
        accepted = 0
        for score, correct in zip(all_scores, all_correct):
            if score >= t:
                accepted += 1
                if correct: tp += 1
                else:       fp += 1
            else:
                if correct: fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        marker = " ← best" if f1 > best_f1 else ""
        print(f"{t:>12.6f} {precision:>10.4f} {recall:>10.4f} "
              f"{f1:>10.4f} {accepted:>10}{marker}")

        if f1 > best_f1:
            best_f1        = f1
            best_threshold = t
            best_precision = precision
            best_recall    = recall

    print(f"\n── Optimal Threshold: {best_threshold:.6f} ──────────")
    print(f"  Precision : {best_precision:.4f}")
    print(f"  Recall    : {best_recall:.4f}")
    print(f"  F1-Score  : {best_f1:.4f}")

    with open("models/optimal_threshold.txt", "w") as f:
        f.write(str(best_threshold))
    print(f"Saved to: models/optimal_threshold.txt")

    return best_threshold


if __name__ == "__main__":
    tune_threshold()