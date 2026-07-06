from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image
import sys
import os

# Ensure ml_core is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.inference import AgeProgressor

app = FastAPI(title="Age Progression GAN API")

# Setup CORS to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AgeProgressor model
try:
    # Get the project root directory (parent of backend)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "ml_core", "models", "latest-G.ckpt")
    print(f"Looking for model at: {model_path}")
    progressor = AgeProgressor(model_path=model_path if os.path.exists(model_path) else None)
except Exception as e:
    print(f"Error initializing AgeProgressor: {e}")
    progressor = None

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
    
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Ensure age group is within bounds (0 to 5)
        target_age_group = max(0, min(5, int(target_age_group)))
        
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
    import sys
    
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
                            except:
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
                        except:
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
