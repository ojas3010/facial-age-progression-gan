from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import traceback
from PIL import Image, ImageOps
import sys
import os

# Ensure ml_core is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.inference import AgeProgressor

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
# The client-declared content type is only a label: without restricting the
# decoders, PIL would decode any format it knows (TIFF, BMP, GIF, EPS...)
# sent under an allowed type.
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
# The byte cap does not bound decode memory: a ~140 KB PNG can declare a
# 144-megapixel canvas. 50 MP clears any real photo that fits in 5 MB.
MAX_IMAGE_PIXELS = 50_000_000

progressor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Singleton model initialization: load weights exactly once at server boot
    global progressor
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "ml_core", "models", "latest-G.ckpt")
        print(f"Looking for model at: {model_path}")
        progressor = AgeProgressor(model_path=model_path if os.path.exists(model_path) else None)
    except Exception as e:
        print(f"Error initializing AgeProgressor: {e}")
        progressor = None
    yield
    progressor = None


app = FastAPI(title="Age Progression GAN API", lifespan=lifespan)

# Setup CORS to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",  # Vite preview server
        "http://127.0.0.1:4173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def decode_upload(contents: bytes) -> Image.Image:
    """Decode uploaded bytes into an upright RGB image, or raise a 400."""
    try:
        image = Image.open(io.BytesIO(contents), formats=ALLOWED_IMAGE_FORMATS)
    except (OSError, Image.DecompressionBombError):
        # UnidentifiedImageError (unknown or disallowed format) is an OSError
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # open() only parsed the header; check dimensions before decoding pixels
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions too large. Maximum is {MAX_IMAGE_PIXELS // 1_000_000} megapixels."
        )

    try:
        # Phone cameras record rotation as an EXIF tag instead of rotating the
        # pixels. Browsers apply it in the preview, so the model must too, or
        # it receives a sideways face.
        ImageOps.exif_transpose(image, in_place=True)
        return image.convert("RGB")
    except OSError:
        # Truncated or corrupt pixel data behind a valid header
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


@app.get("/")
def read_root():
    return {
        "message": "Age Progression GAN API is running.",
        # False: no trained checkpoint was loaded and output will be noise
        "model_loaded": bool(progressor and progressor.weights_loaded),
    }


# Plain `def`, not `async def`: FastAPI runs sync endpoints in a worker
# thread. Decoding and GAN inference are CPU-bound and would otherwise block
# the event loop, stalling every other request (health checks included).
@app.post("/api/progress_age")
def progress_age(
    file: UploadFile = File(...),
    target_age_group: int = Form(...)
):
    if not progressor:
        raise HTTPException(status_code=500, detail="Model not initialized.")

    # Validate MIME type before touching the payload
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP."
        )

    # Validate target age domain against the allowed schema (0-5)
    if not 0 <= target_age_group < progressor.c_dim:
        raise HTTPException(
            status_code=422,
            detail=f"target_age_group must be an integer between 0 and {progressor.c_dim - 1}."
        )

    # Read at most limit+1 bytes so an oversized upload is never copied into
    # memory whole. (Starlette has already spooled the request body to a temp
    # file by this point; a hard cap on request size belongs in the reverse
    # proxy, e.g. nginx client_max_body_size.)
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    image = decode_upload(contents)

    try:
        # Process the image
        output_image = progressor.progress_age(image, target_age_group)

        # Convert output image to base64. Quality 95: the default 75 adds
        # visible block artifacts at 128x128.
        buffered = io.BytesIO()
        output_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"status": "success", "image_base64": img_str}

    except Exception:
        # Log the details server-side; never echo internals to the client
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Image processing failed.")


if __name__ == "__main__":
    import uvicorn

    # Loopback only: the API is unauthenticated. If port 8000 is taken,
    # uvicorn exits with "address already in use" - stop that process yourself.
    uvicorn.run(app, host="127.0.0.1", port=8000)
