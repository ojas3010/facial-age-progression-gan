from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image, UnidentifiedImageError
import sys
import os

# Ensure ml_core is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.inference import AgeProgressor

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Age Progression GAN API is running."}


@app.post("/api/progress_age")
async def progress_age(
    file: UploadFile = File(...),
    target_age_group: int = Form(...)
):
    if not progressor:
        return JSONResponse(status_code=500, content={"message": "Model not initialized."})

    # Validate MIME type before touching the payload
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP."
        )

    # Validate target age domain against the allowed schema (0-5)
    if not 0 <= target_age_group <= 5:
        raise HTTPException(
            status_code=422,
            detail="target_age_group must be an integer between 0 and 5."
        )

    # Read the uploaded image into memory, enforcing the payload size limit
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        # Process the image
        output_image = progressor.progress_age(image, target_age_group)

        # Convert output image to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"status": "success", "image_base64": img_str}

    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


if __name__ == "__main__":
    import uvicorn
    import socket
    import subprocess

    def kill_process_on_port(port):
        """Kill any process using the specified port."""
        try:
            if sys.platform == "win32":
                # Windows
                result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            try:
                                subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True)
                                print(f"Killed process {pid} using port {port}")
                                return True
                            except OSError:
                                pass
            else:
                # Unix-like systems
                result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            subprocess.run(['kill', '-9', pid], capture_output=True)
                            print(f"Killed process {pid} using port {port}")
                        except OSError:
                            pass
                    return True
        except Exception as e:
            print(f"Warning: Could not check/kill process on port {port}: {e}")
        return False

    def is_port_in_use(port):
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    # Check if port 8000 is in use and try to free it
    if is_port_in_use(8000):
        print("Port 8000 is in use. Attempting to free it...")
        if kill_process_on_port(8000):
            # Wait a moment for the port to be freed
            import time
            time.sleep(1)
            if is_port_in_use(8000):
                print("Warning: Port 8000 is still in use after attempting to free it.")
            else:
                print("Successfully freed port 8000.")
        else:
            print("Could not automatically free port 8000. Please manually kill the process using it.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
