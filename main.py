import sys
from src.inference   import load_model, run_inference
from src.fuzzy_match import load_rxnorm_list


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
        return

    image_path  = sys.argv[1]
    print("Loading model...")
    model       = load_model("models/phase2_weights.pt")

    print("Loading RxNorm list...")
    rxnorm_list = load_rxnorm_list()

    print(f"Running inference on: {image_path}\n")
    result = run_inference(image_path, model, rxnorm_list, threshold=0.65)

    print("── Result ──────────────────────────────────────")
    print(f"  Status           : {result['status']}")
    print(f"  Predicted drug   : {result['predicted_drug']}")
    print(f"  Confidence score : {result['confidence_score']}")
    print(f"  Threshold status : {result['threshold_status']}")
    print(f"  RxNorm status    : {result['rxnorm_status']}")
    print(f"  RxCUI            : {result['rxcui']}")
    print(f"\n  Top candidates:")
    for drug, score in result["top_candidates"]:
        print(f"    {drug:<35} {score:.4f}")


if __name__ == "__main__":
    main()