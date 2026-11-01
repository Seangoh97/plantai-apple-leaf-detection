# Delivery validation

Verified on 1 November 2026:

- **22 tests passed** (`python -m pytest -q`): upload decoding/limits, missing-model behavior, threshold loading and validation, multi-label decisions, review heuristics, invalid outputs, API response contract, and inference-failure handling.
- Live local FastAPI server returned HTTP 200 for `/`, `/api/health`, frontend CSS/JavaScript, `/docs`, and `/openapi.json`.
- Browser opened the actual app and showed the model-missing message, available photo buttons, and disabled analysis action. Desktop visual inspection completed. Captured browser console contained no warnings or errors.
- Python source syntax checks passed.

Test environment: macOS arm64, Python 3.9.6, FastAPI 0.128.8, Starlette 0.49.3, NumPy 2.0.2, Pillow 11.3.0, pytest 8.4.2, httpx 0.28.1, uvicorn 0.39.0. Setup instructions recommend Python 3.11/3.12 for current TensorFlow compatibility.

## Verification limits

The user's trained `.keras` checkpoint and thresholds were not attached. TensorFlow loading, actual model inference, Colab/local numerical parity, accuracy, calibration and inference latency have **not** been verified. API success-path tests use explicit test doubles; the app has no mock prediction mode. TensorFlow was not installed in this verification environment.

Responsive breakpoints and native camera file input are implemented. Physical-phone layout, browser camera picker behavior, and HEIC conversion were not tested on a device. The saved preview shows the desktop missing-model state, not a prediction.

Before using model outputs: connect the actual checkpoint, verify the six output names/order and original preprocessing, compare scores for a known validation image with Colab, and evaluate held-out performance. Retain validation thresholds paired with the checkpoint they were tuned against.

## Repository preparation — 2 November 2026

Added clone-and-run instructions, a web-only dependency set for exploring the interface without TensorFlow, and a GitHub Actions workflow for the existing API/decision-logic tests. All 22 tests passed again locally. This does not change model behavior or remove the requirement for a trained checkpoint.
