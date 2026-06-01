import json
import torch
import numpy as np
import requests as req_lib

from src.preprocess  import preprocess_image
from src.model       import CRNN
from src.fuzzy_match import fuzzy_match, load_rxnorm_list
from src.train       import DEVICE

RXNORM_API_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"


def load_model(weights_path="models/phase2_weights.pt",
               labels_path="models/label_classes.json"):
    with open(labels_path, "r") as f:
        label_classes = json.load(f)
    model = CRNN(num_classes=len(label_classes)).to(DEVICE)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE,
                   weights_only=True)
    )
    model.eval()
    return model, label_classes


def validate_rxnorm(drug_name):
    try:
        response = req_lib.get(
            RXNORM_API_URL,
            params={"name": drug_name, "allsrc": 0},
            timeout=10
        )
        data   = response.json()
        rxcuis = data.get("idGroup", {}).get("rxnormId", [])
        if rxcuis:
            return True, rxcuis[0]
    except Exception as e:
        print(f"RxNorm API error: {e}")
    return False, None


def run_inference(image_path, model, label_classes,
                  rxnorm_list, threshold=0.65, top_n=5):

    # 1 — Preprocess
    img   = preprocess_image(image_path, target_w=224, target_h=224)
    img_t = torch.tensor(img, dtype=torch.float32)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 2 — Model prediction
    with torch.no_grad():
        logits = model(img_t)
        probs  = torch.softmax(logits, dim=1)[0]

    # 3 — Get top-N predictions
    top_probs, top_indices = torch.topk(probs, min(top_n, len(label_classes)))
    top_preds = [
        (label_classes[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    best_drug  = top_preds[0][0]
    best_score = top_preds[0][1]

    # 4 — Fuzzy match best prediction against RxNorm list
    matched, match_score, candidates = fuzzy_match(
        best_drug, rxnorm_list, threshold=0.0, top_n=5
    )
    if matched:
        best_drug  = matched
        best_score = match_score

    # 5 — Apply confidence threshold
    if best_score < threshold:
        return _build_result(
            best_drug, best_score, "UNCERTAIN",
            "BELOW THRESHOLD", "UNCONFIRMED",
            None, top_preds
        )

    # 6 — RxNorm validation
    confirmed, rxcui = validate_rxnorm(best_drug)
    status        = "ACCEPTED"   if confirmed else "UNCERTAIN"
    rxnorm_status = "CONFIRMED"  if confirmed else "UNCONFIRMED"

    return _build_result(
        best_drug, best_score, status,
        "ACCEPTED", rxnorm_status, rxcui, top_preds
    )


def _build_result(drug, score, status, threshold_status,
                  rxnorm_status, rxcui, candidates):
    return {
        "status"           : status,
        "predicted_drug"   : drug,
        "confidence_score" : round(float(score), 4),
        "threshold_status" : threshold_status,
        "rxnorm_status"    : rxnorm_status,
        "rxcui"            : rxcui,
        "top_candidates"   : candidates
    }