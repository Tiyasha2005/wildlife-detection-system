"""
Wildlife Detection System - FastAPI Version for Vercel
Deploy to Vercel for free permanent hosting
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import base64
import os

# Initialize FastAPI app
app = FastAPI(
    title="🐾 Wildlife Detection System",
    description="Multi-task AI for wildlife species detection and activity classification"
)

# Load models (cached)
models_loaded = False
custom_model = None
coco_model = None

def load_models():
    global models_loaded, custom_model, coco_model
    if not models_loaded:
        try:
            custom_model = YOLO('./weights/wildlife_custom.pt')
            coco_model = YOLO('yolov8x.pt')
            models_loaded = True
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            models_loaded = True  # Mark as attempted to avoid retry loops

# Load models on startup
load_models()

def analyze_habitat_context(frame, x1, y1, x2, y2):
    """Simple habitat analysis"""
    margin = 50
    h, w = frame.shape[:2]
    
    x1_env = max(0, x1 - margin)
    y1_env = max(0, y1 - margin)
    x2_env = min(w, x2 + margin)
    y2_env = min(h, y2 + margin)
    
    region = frame[y1_env:y2_env, x1_env:x2_env]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Count green pixels (forest)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_ratio = np.sum(green_mask > 0) / green_mask.size
    
    return 'forest' if green_ratio > 0.3 else 'pasture'

def smart_species_mapping(cls, conf, x1, y1, x2, y2, frame_shape, frame):
    """Smart species mapping"""
    species_map = {
        21: 'bear',
        19: 'deer',     # cow -> deer in forest
        18: 'deer',     # horse -> deer in forest
        17: 'fox',      # dog -> fox in forest
        16: 'bird',
        14: 'bird',
        15: 'wild cat', # cat -> wild cat in forest
        20: 'elephant',
        22: 'zebra',
        23: 'giraffe',
    }
    
    base_species = species_map.get(cls, f'unknown_{cls}')
    return base_species

def detect_wildlife_api(image_data, conf_threshold=0.25, enhance_image=True):
    """Wildlife detection for API"""
    
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Enhancement
        if enhance_image:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Detect with both models
        all_detections = []
        
        # Custom model
        if custom_model:
            results = custom_model(frame, conf=max(0.01, conf_threshold), verbose=False)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        species = ['deer', 'squirrel', 'rabbit', 'fox', 'wolf', 'bear'][cls] if cls < 6 else f'wildlife_{cls}'
                        
                        all_detections.append({
                            'species': species,
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2],
                            'source': 'custom'
                        })
        
        # COCO model
        if coco_model:
            results = coco_model(frame, conf=max(0.01, conf_threshold), verbose=False)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        species = smart_species_mapping(cls, conf, x1, y1, x2, y2, frame.shape, frame)
                        
                        all_detections.append({
                            'species': species,
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2],
                            'source': 'coco'
                        })
        
        # Draw results on image
        annotated = frame.copy()
        
        for det in all_detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            species = det['species']
            
            # Draw bounding box
            color = (0, 255, 0) if det['source'] == 'custom' else (255, 165, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{species} {conf:.2f} ({det["source"]})'
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated)
        img_str = base64.b64encode(buffer).decode()
        
        return {
            'success': True,
            'detections': all_detections,
            'annotated_image': img_str,
            'total_detections': len(all_detections)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'detections': []
        }

# HTML interface
HTML_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐾 Wildlife Detection System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: #16a34a; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #16a34a; }
        .results { margin-top: 30px; }
        .detection-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .detection-table th, .detection-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .detection-table th { background-color: #f2f2f2; }
        .result-image { max-width: 100%; height: auto; }
        .loading { text-align: center; color: #666; }
        button { background-color: #16a34a; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background-color: #15803d; }
        .confidence-slider { width: 100%; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐾 Wildlife Detection & Activity Analysis System</h1>
        <p>Multi-task AI model for wildlife species detection and habitat-aware classification</p>
    </div>
    
    <div class="upload-area">
        <input type="file" id="imageInput" accept="image/*" style="display: none;">
        <label for="imageInput">
            <h3>📸 Click to Upload Wildlife Image</h3>
            <p>or drag and drop image here</p>
        </label>
    </div>
    
    <div>
        <label for="confidenceSlider">Detection Confidence: <span id="confidenceValue">0.25</span></label>
        <input type="range" id="confidenceSlider" class="confidence-slider" min="0.1" max="0.5" step="0.05" value="0.25">
        <br>
        <input type="checkbox" id="enhanceImage" checked> Enhance Forest Images
    </div>
    
    <button onclick="detectWildlife()">🔍 Detect Wildlife</button>
    
    <div id="loading" class="loading" style="display: none;">
        <p>🔄 Analyzing image...</p>
    </div>
    
    <div id="results" class="results"></div>
    
    <script>
        const imageInput = document.getElementById('imageInput');
        const confidenceSlider = document.getElementById('confidenceSlider');
        const confidenceValue = document.getElementById('confidenceValue');
        const enhanceCheckbox = document.getElementById('enhanceImage');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        let uploadedImage = null;
        
        confidenceSlider.addEventListener('input', (e) => {
            confidenceValue.textContent = e.target.value;
        });
        
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                uploadedImage = file;
            }
        });
        
        async function detectWildlife() {
            if (!uploadedImage) {
                alert('Please upload an image first');
                return;
            }
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            const formData = new FormData();
            formData.append('image', uploadedImage);
            formData.append('confidence', confidenceSlider.value);
            formData.append('enhance', enhanceCheckbox.checked);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result);
                } else {
                    results.innerHTML = `<p style="color: red;">❌ Error: ${result.error}</p>`;
                }
            } catch (error) {
                results.innerHTML = `<p style="color: red;">❌ Network error: ${error.message}</p>`;
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function displayResults(result) {
            let html = '';
            
            // Display annotated image
            html += `<h3>🎯 Detection Results</h3>`;
            html += `<img src="data:image/jpeg;base64,${result.annotated_image}" class="result-image" alt="Detection results">`;
            
            // Display detection summary
            html += `<h3>📊 Detection Summary (${result.total_detections} animals found)</h3>`;
            
            if (result.detections.length > 0) {
                html += '<table class="detection-table">';
                html += '<tr><th>#</th><th>Species</th><th>Confidence</th><th>Source</th></tr>';
                
                result.detections.forEach((detection, index) => {
                    html += `<tr>
                        <td>${index + 1}</td>
                        <td>${detection.species}</td>
                        <td>${(detection.confidence * 100).toFixed(1)}%</td>
                        <td>${detection.source}</td>
                    </tr>`;
                });
                
                html += '</table>';
            } else {
                html += '<p>No wildlife detected in this image.</p>';
            }
            
            results.innerHTML = html;
        }
        
        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#16a34a';
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#ccc';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#ccc';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadedImage = files[0];
                imageInput.files = files;
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_INTERFACE

@app.post("/detect")
async def detect_wildlife_endpoint(
    image: UploadFile = File(...),
    confidence: float = 0.25,
    enhance: bool = True
):
    """Detect wildlife in uploaded image"""
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image data
    image_data = await image.read()
    
    # Perform detection
    result = detect_wildlife_api(image_data, confidence, enhance)
    
    if not result['success']:
        raise HTTPException(status_code=500, detail=result['error'])
    
    return JSONResponse(result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "custom_model": custom_model is not None,
        "coco_model": coco_model is not None
    }

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
