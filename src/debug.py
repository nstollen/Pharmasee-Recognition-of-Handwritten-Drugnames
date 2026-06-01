import os
import json
import torch
import pandas as pd

from src.preprocess  import preprocess_image
from src.model       import CRNN
from src.fuzzy_match import load_rxnorm_list, fuzzy_match
from src.inference   import load_model
from src.train       import DEVICE, MEDICINE_DIR


def debug():
    model, label_classes = load_model()
    rxnorm_list          = load_rxnorm_list()

    csv_path = os.path.join(MEDICINE_DIR, "validate_clean_final.csv")
    img_dir  = os.path.join(MEDICINE_DIR, "images", "validate")
    df       = pd.read_csv(csv_path).head(20)

    print(f"\n{'Truth':<40} {'Predicted':<40} {'Score':>8} {'Correct':>8}")
    print("-" * 100)

    correct = 0
    total   = 0

    for _, row in df.iterrows():
        filename = str(row["FILENAME"]).strip()
        truth    = str(row["IDENTITY"]).strip().lower()
        fpath    = os.path.join(img_dir, filename)

        if not os.path.exists(fpath):
            continue

        try:
            img   = preprocess_image(fpath)
            img_t = torch.tensor(img, dtype=torch.float32)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_t)
                probs  = torch.softmax(logits, dim=1)[0]

            top_prob, top_idx = torch.topk(probs, 1)
            pred  = label_classes[top_idx[0].item()].lower()
            score = top_prob[0].item()
            match = "✓" if pred == truth else "✗"
            if pred == truth:
                correct += 1
            total += 1

            print(f"{truth:<40} {pred:<40} {score:>8.4f} {match:>8}")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\nAccuracy on sample: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    debug()