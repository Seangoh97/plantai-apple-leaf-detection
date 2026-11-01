"""Unit/API tests use explicit test doubles; they do not validate trained-model accuracy."""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.inference import CLASSES, Predictor, describe, load_thresholds


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.predictor, 'load', lambda: None)
    monkeypatch.setattr(main.predictor, 'model', None)
    monkeypatch.setattr(main.predictor, 'error', 'Model missing. No prediction available.')
    with TestClient(main.app) as client:
        yield client


def image_bytes(mode='RGB', fmt='PNG', size=(64, 64)):
    out = io.BytesIO()
    Image.new(mode, size, 'white').save(out, format=fmt)
    return out.getvalue()


def test_no_model_is_explicit(client):
    assert client.get('/').status_code == 200
    assert client.get('/api/health').json()['ready'] is False
    response = client.post('/api/predict', files={'file': ('leaf.png', image_bytes(), 'image/png')})
    assert response.status_code == 503
    assert 'classes' not in response.json()


@pytest.mark.parametrize('data,status', [(b'', 422), (b'not an image', 422), (b'x' * (main.MAX_BYTES + 1), 413)])
def test_invalid_uploads(client, data, status):
    assert client.post('/api/predict', files={'file': ('leaf.jpg', data, 'image/jpeg')}).status_code == status


def test_content_not_extension_and_limits(client):
    # A misleading MIME type cannot turn text into an image.
    assert client.post('/api/predict', files={'file': ('leaf.png', b'hello', 'image/png')}).status_code == 422
    assert client.post('/api/predict', files={'file': ('leaf.gif', image_bytes(fmt='GIF'), 'image/gif')}).status_code == 415
    assert client.post('/api/predict', files={'file': ('tiny.png', image_bytes(size=(8, 8)), 'image/png')}).status_code == 422


def test_decode_alpha_and_orientation():
    assert main.decode_image(image_bytes('RGBA')).mode == 'RGB'
    im = Image.new('RGB', (80, 40), 'white')
    exif = im.getexif(); exif[274] = 6
    out = io.BytesIO(); im.save(out, format='JPEG', exif=exif)
    assert main.decode_image(out.getvalue()).size == (40, 80)


def test_pixel_bomb(client, monkeypatch):
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 1000)
    assert client.post('/api/predict', files={'file': ('leaf.png', image_bytes(), 'image/png')}).status_code == 413


def test_saved_threshold_csv(tmp_path):
    file = tmp_path / 'thresholds.csv'
    file.write_text('condition,best_threshold,best_f1\n' + '\n'.join(f'{c},0.3,0.8' for c in reversed(CLASSES)))
    values, source = load_thresholds(file)
    assert np.allclose(values, 0.3) and source == 'saved'


@pytest.mark.parametrize('value', [float('nan'), 0, 1, True, -0.1])
def test_invalid_thresholds(tmp_path, value):
    file = tmp_path / 'thresholds.json'
    file.write_text(json.dumps(dict.fromkeys(CLASSES, value)))
    with pytest.raises(ValueError): load_thresholds(file)


def test_incomplete_thresholds(tmp_path):
    file = tmp_path / 'thresholds.json'; file.write_text('{"healthy": 0.5}')
    with pytest.raises(ValueError): load_thresholds(file)


def test_multi_label_and_conflict():
    result = describe([.9, .85, .05, .8, .04, .03], np.full(6, .5), 'saved')
    assert sum(r['detected'] for r in result['classes']) == 3
    assert result['status'] == 'review_needed'
    assert any('together' in r for r in result['uncertainty']['reasons'])


def test_no_signal_and_borderline():
    result = describe([.1] * 6, np.full(6, .5), 'saved')
    assert 'Inconclusive' in result['summary']
    result = describe([.49] * 6, np.full(6, .5), 'default_0.5')
    assert all(r['near_threshold'] for r in result['classes'])
    assert result['threshold_source'] == 'default_0.5'


@pytest.mark.parametrize('p', [[.2] * 5, [float('nan')] * 6, [1.1] * 6])
def test_reject_invalid_model_outputs(p):
    with pytest.raises(ValueError): describe(p, np.full(6, .5), 'saved')


def test_prediction_contract(client, monkeypatch):
    monkeypatch.setattr(main.predictor, 'model', object())
    # Explicit unit-test fixture, never exposed as an application mode.
    monkeypatch.setattr(main.predictor, 'predict', lambda image: describe([.1, .9, .1, .1, .1, .1], np.full(6, .5), 'saved'))
    data = client.post('/api/predict', files={'file': ('leaf.png', image_bytes(), 'image/png')}).json()
    assert len(data['classes']) == 6 and data['classes'][1]['detected']
    assert data['status'] == 'signals_detected'


def test_failed_inference_returns_no_scores(client, monkeypatch):
    monkeypatch.setattr(main.predictor, 'model', object())
    def fail(image): raise ValueError('bad output')
    monkeypatch.setattr(main.predictor, 'predict', fail)
    response = client.post('/api/predict', files={'file': ('leaf.png', image_bytes(), 'image/png')})
    assert response.status_code == 503 and 'classes' not in response.json()


def test_missing_model_load(monkeypatch, tmp_path):
    monkeypatch.setenv('MODEL_PATH', str(tmp_path / 'missing.keras'))
    predictor = Predictor(); predictor.load()
    assert predictor.model is None and 'not found' in predictor.error
