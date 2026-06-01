import json
from rapidfuzz import fuzz
import jellyfish


def load_rxnorm_list(path="datasets/rxnorm_drug_list.json"):
    with open(path, "r") as f:
        return json.load(f)


def jaro_winkler_score(s1, s2):
    return jellyfish.jaro_winkler_similarity(
        s1.lower().strip(),
        s2.lower().strip()
    )


def fuzzy_match(ocr_text, rxnorm_list, threshold=0.65, top_n=5):
    if not ocr_text or not ocr_text.strip():
        return None, 0.0, []

    candidates = []

    for drug in rxnorm_list:
        rf_score  = fuzz.token_set_ratio(
            ocr_text.lower(), drug.lower()
        ) / 100.0
        jw_score  = jaro_winkler_score(ocr_text, drug)
        combined  = (rf_score + jw_score) / 2.0

        if combined >= threshold:
            candidates.append((drug, round(combined, 4)))

    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates:
        return None, 0.0, []

    best_drug, best_score = candidates[0]
    return best_drug, best_score, candidates[:top_n]