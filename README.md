# PlantAI · Apple-leaf disease explorer

A local, mobile-friendly FastAPI + HTML/CSS/JavaScript prototype for your trained TensorFlow EfficientNetB0 model. No build step, external fonts, analytics, sample predictions, or trained weights are included. If the model is absent or fails validation, the interface stays usable for photo selection but analysis is disabled; the API returns 503 instead of inventing scores.

## Start here

```sh
git clone https://github.com/Seangoh97/plantai-apple-leaf-detection.git
cd plantai-apple-leaf-detection
```

**Ready to run as an application; your trained model is required for predictions.** Copy your checkpoint into `models/plantai_effnetb0_stage1.keras`, then follow the installation commands below. This repository does not contain the checkpoint stored in your Colab/Drive and does not claim validated diagnostic accuracy.

To explore the interface without installing TensorFlow, install `requirements-web.txt` instead of `requirements.txt` and run the same server command. Analysis will stay disabled until TensorFlow and the real model are available. There is no mock-prediction mode.

## 1. Get your model from Colab / Drive

Your earlier training chat used this checkpoint name:

`plantai_effnetb0_stage1.keras`

Download that **full trained model** from Drive and copy it into this project's `models/` folder. Do not use an ImageNet-only backbone, a weights-only file, or a new randomly initialized model.

If working in Colab, mount Drive, find the actual checkpoint, and download it:

```python
from google.colab import drive, files
from pathlib import Path
import tensorflow as tf
import keras

drive.mount('/content/drive')
print('TensorFlow:', tf.__version__, 'Keras:', keras.__version__)
# Replace this with the actual saved checkpoint path in your Drive.
checkpoint = Path('/content/drive/MyDrive/YOUR_PROJECT/models/plantai_effnetb0_stage1.keras')
assert checkpoint.is_file(), 'Find your real trained checkpoint first'
files.download(str(checkpoint))
```

Also download `plantai_stage1_thresholds.csv` if you previously saved it. It must belong to this same model checkpoint; thresholds tuned for another checkpoint may be inappropriate. Nothing in this project connects to or changes your Drive automatically.

## 2. Install and run locally

Use **Python 3.11 or 3.12**, with a TensorFlow wheel available for your operating system. The Python version bundled with macOS may be too old. CPU inference is enough; a GPU is optional.

macOS / Linux, from the cloned repository (or extracted `plantai` directory):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Choose a photo, or use “Take a photo” on a supported phone, then select **Analyze leaf**. The same server serves the frontend and API, so no separate frontend server or CORS setup is needed. Stop it with Ctrl+C.

The default model path is relative to the project, so launching from another folder will still find the model. Run commands from the project directory so Python can import `app`.

### Custom model and threshold paths

Set environment variables **before** starting the server. No `.env` file loader is included.

```sh
export MODEL_PATH="/absolute/path/to/your/model.keras"
export THRESHOLDS_PATH="/absolute/path/to/plantai_stage1_thresholds.csv"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

PowerShell uses `$env:MODEL_PATH = 'C:\path\model.keras'` and `$env:THRESHOLDS_PATH = 'C:\path\thresholds.csv'`.

When no `THRESHOLDS_PATH` is set, all six labels use **0.50**, explicitly disclosed in the UI and API as untuned defaults. An explicitly configured missing/malformed threshold file prevents model readiness; it never silently falls back. Restart the server after changing model files or settings.

## 3. Model contract — verify this before trusting outputs

This matches the model definition visible in your earlier chat:

| Setting | Expected value |
| --- | --- |
| Architecture | Full saved EfficientNetB0 classification model |
| Input | One float32 RGB tensor, channels last; original size 224 × 224 |
| Resize | TensorFlow bilinear resize to the saved model's fixed spatial dimensions |
| Pixel range | 0–255; **no extra division by 255** |
| Output | One vector of six independent **sigmoid probabilities** |
| Class order | `healthy`, `scab`, `frog_eye_leaf_spot`, `rust`, `powdery_mildew`, `complex` |

Do not reorder class names to make the UI look nicer. Keras cannot tell us the semantic meaning of output positions; the server can check shapes and final sigmoid activation, but **cannot automatically verify class order or internal preprocessing**. Compare against your actual training code. A changed training pipeline requires corresponding changes in `app/inference.py`.

Standard Keras EfficientNet includes its own rescaling, and expects float pixels in 0–255: [TensorFlow EfficientNetB0 documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0). This prototype matches the earlier `tf.image.resize` pipeline and calls the model with `training=False` so augmentation and dropout are inactive. Phone photos are EXIF-oriented; transparent images are composited onto white. PIL and TensorFlow JPEG decoding can differ slightly; compare a known validation image with Colab before field use.

The loader uses `compile=False, safe_mode=True`; it does not need your training optimizer or loss. Only load your own trusted model. If custom layers are required, register their trusted implementations in the loading code. Do not bypass safe mode to suppress an error. See [Keras loading documentation](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/).

### Optional saved thresholds

The CSV from your earlier notebook is supported directly, including its unused `best_f1` column:

```text
condition,best_threshold,best_f1
```

There must be exactly six rows, one per class. `best_threshold` must be a finite number strictly between 0 and 1. CSV row order is irrelevant; names are matched exactly. JSON is also supported as a direct object mapping the six class names to numeric thresholds. No example numeric thresholds are supplied because none were recovered from your training results.

If `best_thresholds` still exists in your Colab session, export the actual values:

```python
import json
from google.colab import files
from pathlib import Path

expected = ['healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 'powdery_mildew', 'complex']
assert set(best_thresholds) == set(expected)
Path('thresholds.json').write_text(json.dumps({c: float(best_thresholds[c]) for c in expected}, indent=2))
files.download('thresholds.json')
```

## Phone use

Put your phone and computer on the same trusted Wi-Fi. Start with:

```sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://YOUR_COMPUTER_LAN_IP:8000` on the phone and allow the computer's firewall prompt if needed. `127.0.0.1` on your phone points to the phone, not the computer. “Take a photo” uses the native file-input camera picker (`capture="environment"`); browser support varies and some browsers offer a file picker instead. This app does not use a continuous camera stream or `getUserMedia`. Use **Choose photo** as a fallback. HEIC/HEIF must be converted to JPEG; automatic iPhone conversion varies by browser. Do not expose this unauthenticated development server to the public internet.

## Results and uncertainty

Scores are model-estimated probabilities, not validated diagnostic confidence. They are not normalized with softmax and may sum above 100%. Every class shows its own decision threshold and whether the score meets it (`>=`). More than one label may be active.

The app recommends review when thresholds are defaults, any score is within 0.10 of its threshold, no label passes, healthy and symptom labels conflict, or complex symptoms pass. These are transparent **heuristics**, not a calibrated uncertainty estimate or an out-of-distribution detector. Even when no heuristic flag triggers, the result can be wrong. A non-leaf photo may receive high scores. Healthy does not rule out disease; “complex” does not identify a specific cause. The app gives no pesticide or treatment instructions.

## API

- `GET /api/health`: readiness, message, class order, input size and threshold source. Returns HTTP 200 even when not ready; check `ready`.
- `POST /api/predict`: multipart field `file`. Returns summary, six `classes` with `probability`, `threshold`, `detected`, `near_threshold`, uncertainty reasons, and safety text.
- `GET /docs`: interactive FastAPI API documentation (Swagger UI assets require internet access; the main PlantAI interface uses local assets only).

```sh
curl -F "file=@/absolute/path/to/leaf.jpg" http://127.0.0.1:8000/api/predict
```

Errors use `{"detail": "..."}`: 413 for byte/pixel limits, 415 for unsupported formats or animation, 422 for invalid/empty/tiny images, 503 for unavailable model or inference failure. Validation reads actual image content rather than trusting extensions or MIME headers. JPEG, PNG, and single-frame WebP are supported; minimum 32 pixels on either axis, maximum 20 megapixels, maximum 10 MiB. The browser also validates common mistakes before upload.

Photos are held in memory for processing and not written to application storage or sent to third parties. Multipart parsing may spool uploads to temporary disk before application validation; request state is closed afterward. No persistence or history is implemented. Run one worker for one loaded model; predictions are serialized to limit simultaneous model calls. This is a local prototype: public deployment needs authentication, rate/concurrency limits, HTTPS and a reverse proxy with an upload-body limit (multipart parsing happens before endpoint validation).

## Troubleshooting

- **Model not found:** confirm the filename and real path. A Drive browser URL is not a local path. Copy/download the file first.
- **Model cannot load:** inspect terminal logs. Match TensorFlow **and Keras** versions to the Colab environment printed earlier; pin them in `requirements.txt`. A newer runtime is not guaranteed to load every older checkpoint.
- **Final sigmoid / shape error:** check that you downloaded the full six-label model, not the backbone. This prototype deliberately rejects softmax/logit/multi-output models.
- **Odd results:** compare one validation photo's six scores locally and in Colab. Check output class order, preprocessing, checkpoint and paired thresholds. Do not assess quality using a random web photo alone.
- **No TensorFlow wheel:** use a supported 64-bit Python/OS combination. For legacy Intel Macs or older systems, a matching older TensorFlow runtime or Linux environment may be necessary.
- **Camera unavailable:** use a JPEG photo from the gallery. Camera picker behavior belongs to the browser/OS.

## Tests and project layout

GitHub Actions runs application/API and decision-logic tests on pushes and pull requests. These tests do not require a trained model or TensorFlow; they do not measure real-model accuracy. To run that same lightweight check locally:

```sh
python -m pip install -r requirements-web.txt -r requirements-test.txt
python -m pytest -q
```


```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Tests cover missing-model behavior, invalid uploads and limits, image orientation/transparency, threshold validation, multi-label logic, uncertainty flags, invalid model outputs and API success/failure paths. API tests inject clearly identified test doubles; these are not an app prediction mode and do not demonstrate trained-model accuracy.

```text
app/main.py             FastAPI routes, image validation, static files
app/inference.py        Safe model loading, matching preprocessing, label decisions
app/static/             Responsive HTML, CSS, vanilla JavaScript
models/                 Your checkpoint and optional thresholds (not included)
tests/test_app.py       Unit and API regression tests
requirements.txt        Full inference runtime dependencies
requirements-web.txt    Interface/API dependencies without TensorFlow
.github/workflows/      Automated application test workflow
requirements-test.txt   Test dependencies
VALIDATION.md           What was verified for this delivery
```

**Next validation with your model:** confirm one known-image score vector matches Colab; then evaluate per-class precision, recall, F1 and calibration on held-out data. No accuracy or performance claims are made here. The actual checkpoint was not available during development.
