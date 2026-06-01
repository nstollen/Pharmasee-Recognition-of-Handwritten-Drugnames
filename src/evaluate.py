import os
import json
import torch
import numpy as np
import pandas as pd

from src.preprocess  import preprocess_image
from src.model       import CRNN
from src.fuzzy_match import load_rxnorm_list, fuzzy_match
from src.inference   import load_model, validate_rxnorm
from src.train       import DEVICE, MEDICINE_DIR


def cer(pred, truth):
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    d = np.zeros((len(pred)+1, len(truth)+1), dtype=int)
    for i in range(len(pred)+1): d[i][0] = i
    for j in range(len(truth)+1): d[0][j] = j
    for i in range(1, len(pred)+1):
        for j in range(1, len(truth)+1):
            cost = 0 if pred[i-1] == truth[j-1] else 1
            d[i][j] = min(
                d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost
            )
    return d[len(pred)][len(truth)] / len(truth)


def evaluate(weights_path="models/phase2_weights.pt",
             labels_path="models/label_classes.json",
             threshold=None):

    print(f"\n── Evaluation ──────────────────────────────────")
    print(f"Weights  : {weights_path}")
    print(f"Device   : {DEVICE}")

    model, label_classes = load_model(weights_path, labels_path)
    rxnorm_list          = load_rxnorm_list()

    # Load optimal threshold
    if threshold is None:
        threshold_path = "models/optimal_threshold.txt"
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                threshold = float(f.read().strip())
            print(f"Threshold: {threshold:.6f} (from tuning)")
        else:
            threshold = 0.001
            print(f"Threshold: {threshold} (default)")

    csv_path = os.path.join(MEDICINE_DIR, "test_clean_final.csv")
    img_dir  = os.path.join(MEDICINE_DIR, "images", "test")
    df       = pd.read_csv(csv_path)

    print(f"Test samples: {len(df)}")

    cer_scores, exact_matches      = [], []
    raw_exact_matches              = []
    tp = fp = fn                   = 0
    rxnorm_confirmed_count         = 0
    rxnorm_unconfirmed_count       = 0
    total   = 0
    skipped = 0

    for _, row in df.iterrows():
        filename = str(row["FILENAME"]).strip()
        truth    = str(row["IDENTITY"]).strip().lower()
        fpath    = os.path.join(img_dir, filename)

        if not os.path.exists(fpath):
            skipped += 1
            continue

        try:
            img   = preprocess_image(fpath, target_w=224, target_h=224)
            img_t = torch.tensor(img, dtype=torch.float32)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_t)
                probs  = torch.softmax(logits, dim=1)[0]

            top_probs, top_idx = torch.topk(probs, 5)
            pred_text   = label_classes[top_idx[0].item()].lower()
            model_score = top_probs[0].item()

            # Track raw model accuracy
            raw_exact = pred_text.strip() == truth.strip()
            raw_exact_matches.append(raw_exact)

            # Fuzzy match — only apply if high confidence match exists
            matched, match_score, _ = fuzzy_match(
                pred_text, rxnorm_list, threshold=0.0, top_n=1
            )

            # RxNorm validation tracking
            rxnorm_confirmed = False
            if matched and match_score > 0.85:
                confirmed, rxcui = validate_rxnorm(matched)
                if confirmed:
                    rxnorm_confirmed = True
                    rxnorm_confirmed_count += 1
                else:
                    rxnorm_unconfirmed_count += 1

            # Use raw model prediction for final metrics
            final_pred = pred_text

            c     = cer(final_pred, truth)
            exact = final_pred.strip() == truth.strip()
            cer_scores.append(c)
            exact_matches.append(exact)

            # Threshold-based precision/recall
            if model_score >= threshold:
                if exact: tp += 1
                else:     fp += 1
            else:
                if exact: fn += 1

            total += 1

        except Exception as e:
            print(f"Error on {filename}: {e}")

    print(f"Processed : {total}")
    print(f"Skipped   : {skipped}")

    if total == 0:
        print("ERROR: No images processed.")
        return

    avg_cer   = np.mean(cer_scores) * 100
    ema       = np.mean(exact_matches) * 100
    raw_ema   = np.mean(raw_exact_matches) * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n── Results ({total} test images) ────────────────")
    print(f"  CER                     : {avg_cer:.2f}%")
    print(f"  Exact Match Acc (raw)   : {raw_ema:.2f}%")
    print(f"  Exact Match Acc (final) : {ema:.2f}%")
    print(f"  Precision               : {precision:.4f}")
    print(f"  Recall                  : {recall:.4f}")
    print(f"  F1-Score                : {f1:.4f}")
    print(f"\n── RxNorm Validation Summary ────────────────────")
    print(f"  Confirmed by RxNorm     : {rxnorm_confirmed_count}")
    print(f"  Unconfirmed by RxNorm   : {rxnorm_unconfirmed_count}")
    print(f"  Below fuzzy threshold   : "
          f"{total - rxnorm_confirmed_count - rxnorm_unconfirmed_count}")

    return {
        "cer"       : avg_cer,
        "ema"       : ema,
        "raw_ema"   : raw_ema,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "rxnorm_confirmed"   : rxnorm_confirmed_count,
        "rxnorm_unconfirmed" : rxnorm_unconfirmed_count,
        "total"     : total
    }


if __name__ == "__main__":
    evaluate()