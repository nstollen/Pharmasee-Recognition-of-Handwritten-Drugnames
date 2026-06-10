# PHARMASEE

A deep learning-based medicine recognition system that identifies pharmaceutical drugs from handwritten prescriptions using optical character recognition (OCR) and RxNorm validation.

## Features

- **Handwriting Recognition**: Identifies drug names from prescription images using a CRNN (Convolutional Recurrent Neural Network) model
- **RxNorm Validation**: Validates recognized drugs against the RxNorm pharmaceutical database
- **Confidence Scoring**: Provides confidence levels and threshold-based decision support
- **Fuzzy Matching**: Handles spelling variations and partial matches
- **Flask REST API**: Easy integration with external applications
- **Web Interface**: Interactive HTML-based UI for testing predictions
- **Multi-Phase Training**: Two-phase training pipeline for improved accuracy

## Project Structure

```
PHARMASEE/
├── app.py                          # Flask API server
├── main.py                         # CLI interface for single image inference
├── pharmaseen_webapp.html          # Web interface
├── models/                         # Pre-trained model weights and labels
│   ├── phase1_weights.pt          # Phase 1 trained weights
│   ├── phase2_weights.pt          # Phase 2 trained weights (recommended)
│   ├── label_classes.json         # Drug class labels
│   └── optimal_threshold.txt      # Confidence threshold
├── src/                           # Core modules
│   ├── model.py                   # CRNN architecture definition
│   ├── train.py                   # Training pipeline
│   ├── inference.py               # Inference utilities
│   ├── preprocess.py              # Image preprocessing
│   ├── fuzzy_match.py             # RxNorm fuzzy matching
│   ├── evaluate.py                # Model evaluation
│   ├── tune_threshold.py          # Threshold optimization
│   ├── debug.py                   # Debugging utilities
│   └── decode.py                  # Output decoding
├── datasets/                      # Training data
│   └── 2Doctor's Prescription/   # Handwritten medicine dataset
└── .venv/                        # Python virtual environment
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda
- GPU (optional, but recommended) - CUDA-enabled PyTorch

### Setup

1. **Clone or download the repository**
   ```bash
   cd PHARMASEE
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install flask flask-cors torch torchvision pillow opencv-python numpy
   ```

4. **Verify setup**
   ```bash
   python app.py
   ```
   The server should start on `http://0.0.0.0:5000`

## Usage

### Option 1: CLI Interface

Run inference on a single image:

```bash
python main.py path/to/prescription/image.png
```

**Output:**
```
── Result ──────────────────────────────────────
  Status           : ACCEPTED
  Predicted drug   : Amoxicillin
  Confidence score : 0.856432
  Threshold status : ACCEPTED
  RxNorm status    : CONFIRMED
  RxCUI            : 723496
  
  Top candidates:
    Amoxicillin                         0.8564
    Ampicillin                          0.0892
    Benzyl Penicillin                   0.0321
    ...
```

### Option 2: Flask REST API

Start the API server:

```bash
python app.py
```

#### Health Check
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": 180,
  "threshold": 0.001,
  "device": "cuda" or "cpu"
}
```

#### Predict
```bash
curl -X POST -F "image=@path/to/image.png" http://localhost:5000/predict
```

**Response:**
```json
{
  "status": "ACCEPTED",
  "predicted_drug": "Amoxicillin",
  "confidence_score": 0.856432,
  "confidence_level": "HIGH",
  "threshold_status": "ACCEPTED",
  "rxnorm_status": "CONFIRMED",
  "rxcui": "723496",
  "canonical_name": "Amoxicillin",
  "top_candidates": [
    {"name": "Amoxicillin", "prob": 0.856432},
    {"name": "Ampicillin", "prob": 0.089234},
    {"name": "Benzyl Penicillin", "prob": 0.032105}
  ],
  "threshold_used": 0.001
}
```

### Option 3: Web Interface

Open `pharmaseen_webapp.html` in a web browser for an interactive UI to test predictions.

## Model Details

### Architecture
- **Model Type**: CRNN (Convolutional Recurrent Neural Network)
- **Backbone**: Convolutional layers for feature extraction
- **Sequence Layer**: LSTM for temporal sequence processing
- **Output**: Softmax classification across 180+ drug classes

### Training
- **Phase 1**: Initial training on handwritten medicine dataset
- **Phase 2**: Fine-tuning with additional optimization

### Input Requirements
- **Image Format**: Grayscale or color images (auto-converted to grayscale)
- **Size**: Images are resized to 224×224 with padding
- **Quality**: Handwritten text clarity affects accuracy

### Preprocessing Pipeline
1. Bilateral filtering for noise reduction
2. Aspect ratio-preserving resize
3. Center padding to 224×224
4. Normalization (0-1 range)

## Key Functions

### `app.py` - Flask Server

- `preprocess_from_bytes()`: Convert image bytes to normalized tensor
- `clean_drug_name()`: Normalize drug name strings
- `/predict`: Main prediction endpoint
- `/health`: Health check endpoint

### `main.py` - CLI Interface

- `main()`: Entry point for command-line inference

### `src/inference.py`

- `load_model()`: Load pre-trained weights and labels
- `validate_rxnorm()`: Check drug against RxNorm database
- `run_inference()`: Execute full inference pipeline

### `src/fuzzy_match.py`

- `load_rxnorm_list()`: Load RxNorm drug database
- Fuzzy matching for handling spelling variations

## Configuration

### Threshold Settings

The model uses confidence thresholds to determine prediction reliability:

- **THRESHOLD** (default: 0.001): Minimum confidence for consideration
- **REAL_WORLD_THRESHOLD** (default: 0.50): Minimum for acceptance in production

Edit these in `app.py` or read from `models/optimal_threshold.txt`

### Model Files

Ensure these files exist in the `models/` directory:

- `phase2_weights.pt` - Pre-trained model weights
- `label_classes.json` - Drug class mapping
- `optimal_threshold.txt` - Optimal confidence threshold (optional)

## Status Codes

The API returns different status codes for predictions:

| Status | Meaning |
|--------|---------|
| `ACCEPTED` | High confidence match + RxNorm confirmed |
| `UNCERTAIN` | Below confidence threshold, needs review |
| `CONFIRMED` | RxNorm validation successful |
| `UNCONFIRMED` | Drug name not in RxNorm database |

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Ensure `phase2_weights.pt` exists in the `models/` directory

### Issue: "No image uploaded" (API)
**Solution**: Use the `-F "image=@path/to/file"` flag with curl

### Issue: Low confidence scores
**Solution**: 
- Ensure image quality is high
- Check that handwriting is clear
- Verify image format is supported (PNG, JPG)

### Issue: CUDA out of memory
**Solution**: The model automatically falls back to CPU if CUDA is unavailable

## Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **opencv-python**: Image preprocessing
- **pillow**: Image handling
- **flask**: Web framework
- **flask-cors**: Cross-origin support
- **numpy**: Numerical computing

## Performance

- **Inference Time**: ~100-200ms per image (GPU)
- **Model Size**: ~44MB (phase2_weights.pt)
- **Memory Usage**: ~2GB (with model loaded)
- **Classes Supported**: 180+ pharmaceuticals

## Future Enhancements

- [ ] Support for printed text recognition
- [ ] Multi-drug detection in single prescription
- [ ] Extended drug database coverage
- [ ] Fine-tuning on additional datasets
- [ ] Mobile app integration

## License

This project is provided as-is for research and educational purposes.

## Authors

Gabriel James M. Gulmatico
Edwin Carl Jr. demonteverde
Genesis B. Makabenta

---

For issues, questions, or contributions, please refer to the project documentation or contact the development team.
