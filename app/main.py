import io
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .inference import CLASSES, Predictor

MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
STATIC = Path(__file__).parent / 'static'
predictor = Predictor()


@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(predictor.load)
    yield


app = FastAPI(title='PlantAI', version='0.1.0', lifespan=lifespan, redoc_url=None)
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.middleware('http')
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    if request.url.path == '/docs':
        # FastAPI Swagger UI uses its documented CDN and inline initialization.
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; img-src 'self' https://fastapi.tiangolo.com data:; object-src 'none'; frame-ancestors 'none'"
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/')
def home():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/health')
def health():
    return {'ready': predictor.model is not None, 'message': predictor.error or 'Trained model connected.',
            'classes': CLASSES, 'threshold_source': predictor.source,
            'input_size': getattr(predictor, 'size', None), 'max_upload_bytes': MAX_BYTES}


def decode_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as check:
                if check.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise HTTPException(415, 'Use a JPEG, PNG, or WebP photo. Convert HEIC images to JPEG first.')
                if getattr(check, 'n_frames', 1) != 1:
                    raise HTTPException(415, 'Animated images are not supported.')
                if min(check.size) < 32:
                    raise HTTPException(422, 'Image must be at least 32 × 32 pixels.')
                check.verify()
            with Image.open(io.BytesIO(data)) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                    rgba = image.convert('RGBA')
                    background = Image.new('RGBA', rgba.size, 'white')
                    image = Image.alpha_composite(background, rgba)
                return image.convert('RGB')
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, 'Image exceeds the 20 megapixel limit. Resize it and try again.')
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(422, 'This file could not be decoded as a valid image.')


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Photo must be 10 MB or smaller.')
    if not data:
        raise HTTPException(422, 'Please select a photo.')
    image = await run_in_threadpool(decode_image, data)
    if predictor.model is None:
        raise HTTPException(503, predictor.error or 'Model unavailable. No prediction was made.')
    try:
        return await run_in_threadpool(predictor.predict, image)
    except Exception:
        logging.exception('Inference failed')
        raise HTTPException(503, 'The model could not produce a valid result. No prediction is available; check server logs.')
