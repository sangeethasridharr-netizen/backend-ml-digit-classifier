# app.py
import os
import io
import base64
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model_train import train_model

app = FastAPI(title="MNIST Digit Classifier")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = 'model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Training model now...")
        train_model()
    
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")

class PredictionInput(BaseModel):
    image: str  # Base64 encoded image string

@app.post("/predict")
async def predict(data: PredictionInput):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        # 1. Decode base64 image
        if ',' in data.image:
            # Handle data:image/png;base64,...
            base64_data = data.image.split(',')[1]
        else:
            base64_data = data.image
            
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 2. Preprocess image
        # Convert to grayscale
        image = image.convert('L')
        # Resize to 28x28 (MNIST size)
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Invert if necessary (MNIST is white digits on black background)
        # If the input is black drawing on white canvas, we need to invert
        # We can check mean value to decide
        if np.mean(image_array) > 0.5:
            image_array = 1.0 - image_array
            
        # Flatten to 1D array (784 features)
        image_flat = image_array.reshape(1, -1)
        
        # 3. Make prediction
        prediction = model.predict(image_flat)[0]
        
        # Get confidence scores
        probabilities = model.predict_proba(image_flat)[0]
        confidence = float(np.max(probabilities))
        
        return {
            "prediction": str(prediction),
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

# Serve static files from 'dist' directory
if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if os.path.exists(os.path.join("dist", "index.html")):
        return FileResponse(os.path.join("dist", "index.html"))
    return {"detail": "Frontend not built yet. Run 'npm run build'"}

if __name__ == "__main__":
    import uvicorn
    # Use port 3000 as per infrastructure requirements
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
