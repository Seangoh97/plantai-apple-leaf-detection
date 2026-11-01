"""Single-output, six-label sigmoid inference. Never creates fallback predictions."""
import csv
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 'powdery_mildew', 'complex')
LABELS = ('Healthy', 'Apple scab', 'Frog-eye leaf spot', 'Rust', 'Powdery mildew', 'Complex symptoms')
SAFETY = ('Research prototype, not a confirmed diagnosis. Scores may be uncalibrated. '
          'This model cannot verify that a photo contains an apple leaf or detect every disease. '
          'Consult a local plant specialist before treatment; do not select pesticides from this result.')


def load_thresholds(path):
    if not path:
        return np.full(6, 0.5), 'default_0.5'
    path = Path(path)
    if not path.is_file():
        raise ValueError('Configured threshold file was not found. Check THRESHOLDS_PATH and restart.')
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 6 or len({r['condition'] for r in rows}) != 6:
            raise ValueError('Threshold CSV must contain exactly one row per class.')
        data = {r['condition']: float(r['best_threshold']) for r in rows}
    else:
        data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) != set(CLASSES):
        raise ValueError('Threshold keys must exactly match the six class names.')
    if any(isinstance(data[c], bool) for c in CLASSES):
        raise ValueError('Thresholds must be numbers, not booleans.')
    values = np.array([float(data[c]) for c in CLASSES])
    if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
        raise ValueError('Thresholds must be finite and strictly between 0 and 1.')
    return values, 'saved'


def describe(probabilities, thresholds, source):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected six finite sigmoid probabilities between 0 and 1.')
    active = p >= thresholds
    near = np.abs(p - thresholds) <= 0.10
    reasons = []
    if source == 'default_0.5':
        reasons.append('Default 0.50 thresholds are in use; validation-tuned thresholds were not supplied.')
    if near.any():
        reasons.append('One or more scores are within 0.10 of their decision threshold.')
    if not active.any():
        reasons.append('No label crossed its threshold. This does not establish that the leaf is healthy.')
    if active[0] and active[1:].any():
        reasons.append('Healthy and symptom labels crossed their thresholds together; expert review is needed.')
    if active[5]:
        reasons.append('Complex symptoms were flagged; this label does not identify a specific cause.')
    status = 'review_needed' if reasons else 'signals_detected'
    selected = [LABELS[i] for i in range(6) if active[i]]
    return {
        'status': status,
        'summary': 'Possible signals: ' + ', '.join(selected) if selected else 'Inconclusive — no label met its threshold',
        'classes': [dict(key=c, label=LABELS[i], probability=float(p[i]),
                         threshold=float(thresholds[i]), detected=bool(active[i]),
                         near_threshold=bool(near[i])) for i, c in enumerate(CLASSES)],
        'threshold_source': source,
        'uncertainty': {'review_recommended': bool(reasons), 'reasons': reasons,
                        'method': 'Heuristic threshold proximity and label consistency; not calibrated uncertainty or an out-of-distribution detector.'},
        'safety': SAFETY,
    }


class Predictor:
    def __init__(self):
        self.model = None
        self.error = 'Model not loaded.'
        self.lock = threading.Lock()
        self.thresholds = None
        self.source = None
        self.tf = None

    def load(self):
        try:
            path = Path(os.getenv('MODEL_PATH', str(ROOT / 'models/plantai_effnetb0_stage1.keras')))
            if not path.is_file():
                raise FileNotFoundError('Model file not found. Add your trained .keras file and restart the server.')
            if path.suffix != '.keras':
                raise ValueError('MODEL_PATH must point to a .keras file.')
            self.thresholds, self.source = load_thresholds(os.getenv('THRESHOLDS_PATH'))
            import tensorflow as tf
            model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
            shape = model.input_shape
            if isinstance(shape, list) or len(shape) != 4 or shape[0] not in (None, 1) or shape[-1] != 3:
                raise ValueError('Expected one channels-last RGB input with variable or single-image batch.')
            if shape[1] is None or shape[2] is None or not (32 <= shape[1] <= 2048 and 32 <= shape[2] <= 2048):
                raise ValueError('Expected fixed spatial dimensions between 32 and 2048.')
            if isinstance(model.output_shape, list) or tuple(model.output_shape[1:]) != (6,):
                raise ValueError('Expected one six-label output.')
            if getattr(getattr(model.layers[-1], 'activation', None), '__name__', '') != 'sigmoid':
                raise ValueError('Expected a final sigmoid layer, matching the original multi-label model.')
            self.tf, self.model = tf, model
            self.size = (int(shape[1]), int(shape[2]))
            self.error = None
        except Exception as exc:
            logging.exception('PlantAI model initialization failed')
            # Do not return host file paths or TensorFlow stack traces through the API.
            self.error = str(exc) if isinstance(exc, (FileNotFoundError, ValueError)) else 'Model could not load. Check server logs and TensorFlow/Keras compatibility.'
            self.model = None

    def predict(self, image):
        with self.lock:
            # Match training: RGB, float32 0..255, TF bilinear resize, no external /255.
            x = self.tf.convert_to_tensor(np.asarray(image), dtype=self.tf.float32)
            x = self.tf.image.resize(x, self.size, method='bilinear')
            output = self.model(x[None, ...], training=False).numpy()
            if output.shape != (1, 6):
                raise ValueError('Unexpected model output shape.')
            return describe(output[0], self.thresholds, self.source)
