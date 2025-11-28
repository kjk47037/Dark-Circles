# app.py
# FastAPI server exposing two YOLO models:
#  - /predict/acne         -> acne model with custom "+0.5" confidence & PIL annotation
#  - /predict/dark_circles -> dark-circles model with standard YOLO annotation
#
# Accepts either multipart file upload (form) OR JSON with base64 image:
#  JSON example:
#    {
#      "filename": "image.png",
#      "b64": "data:image/png;base64,iVBORw0KGgoAAAANS..."
#    }
#
# Run:
#   uvicorn app:app --reload

import os
import uuid
import base64
import io
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

# ---------------- CONFIG ----------------
ACNE_WEIGHTS = "acne_best.pt"
DC_WEIGHTS = "dark_circles_best.pt"

ACNE_CONF = 0.10
DC_CONF = 0.35
IMGSZ = 640

OUT_DIR = "runs/api_outputs"
ACNE_OUT_DIR = os.path.join(OUT_DIR, "acne")
DC_OUT_DIR = os.path.join(OUT_DIR, "dark_circles")

os.makedirs(ACNE_OUT_DIR, exist_ok=True)
os.makedirs(DC_OUT_DIR, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# -------------- FastAPI init --------------
app = FastAPI(
    title="Skin Detection API",
    description="FastAPI backend for acne & dark circle YOLO models (accepts base64 input or multipart file)",
    version="1.2.0",
)

# (Optional) CORS for local frontend / Postman etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------- Load Models (once) --------------
# Handle older checkpoints that reference missing modules (provide aliases before loading weights)
try:
    from ultralytics.nn.modules import block as ul_block
    import sys
    
    # Create aliases for missing modules to C3 (or C2f if available)
    fallback_class = None
    if hasattr(ul_block, "C2f"):
        fallback_class = ul_block.C2f
    elif hasattr(ul_block, "C3"):
        fallback_class = ul_block.C3
    
    if fallback_class:
        missing_modules = ["C3k2", "C3k", "C2PSA"]
        for module_name in missing_modules:
            if not hasattr(ul_block, module_name):
                setattr(ul_block, module_name, fallback_class)
                setattr(sys.modules["ultralytics.nn.modules.block"], module_name, fallback_class)
                print(f"⚠️ {module_name} not found, aliasing to {fallback_class.__name__} for compatibility")
    else:
        print("⚠️ Warning: No suitable fallback class found for missing modules")
except Exception as e:
    print(f"⚠️ Warning: Module compatibility patch failed: {e}")

# Patch torch.load to handle older model formats
import torch.serialization
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load to handle older ultralytics model formats"""
    # Set weights_only=False for PyTorch 2.6+ compatibility
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

try:
    acne_model = YOLO(ACNE_WEIGHTS)
    print(f"✅ Acne model loaded successfully from {ACNE_WEIGHTS}")
except Exception as e:
    # Try loading with explicit weights_only=False
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    try:
        acne_model = YOLO(ACNE_WEIGHTS)
        print(f"✅ Acne model loaded successfully from {ACNE_WEIGHTS}")
    except Exception as e2:
        raise RuntimeError(f"Failed to load acne model from {ACNE_WEIGHTS}: {e2}") from e2

try:
    dc_model = YOLO(DC_WEIGHTS)
    print(f"✅ Dark circles model loaded successfully from {DC_WEIGHTS}")
except Exception as e:
    # Try loading with explicit weights_only=False
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    try:
        dc_model = YOLO(DC_WEIGHTS)
        print(f"✅ Dark circles model loaded successfully from {DC_WEIGHTS}")
    except Exception as e2:
        raise RuntimeError(f"Failed to load dark_circles model from {DC_WEIGHTS}: {e2}") from e2


# -------------- Pydantic Schemas --------------
class ImageB64(BaseModel):
    filename: Optional[str] = None
    b64: str


class Detection(BaseModel):
    box: List[float]          # [x1, y1, x2, y2]
    conf: float               # original YOLO confidence
    conf_plus_0_5: Optional[float] = None  # only used for acne endpoint
    class_id: int
    class_name: Optional[str] = None


class PredictionResponse(BaseModel):
    model_name: str
    detections: List[Detection]
    annotated_image_path: str  # relative or absolute path
    annotated_image_b64: Optional[str] = None  # base64 encoded annotated image (data URI)


class CombinedPredictionResponse(BaseModel):
	acne: PredictionResponse
	dark_circles: PredictionResponse


# -------------- Utility funcs --------------
def save_bytes_to_disk(data_bytes: bytes, save_dir: str, filename: Optional[str] = None) -> str:
    """Save raw bytes to disk and return the saved path."""
    os.makedirs(save_dir, exist_ok=True)
    ext = ".png"
    if filename:
        _, e = os.path.splitext(filename)
        if e:
            ext = e
    fname = filename or f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(save_dir, fname)
    with open(fpath, "wb") as f:
        f.write(data_bytes)
    return fpath


def save_upload_to_disk(upload_file: UploadFile, save_dir: str) -> str:
    """Save uploaded file to disk and return the saved path."""
    os.makedirs(save_dir, exist_ok=True)
    ext = os.path.splitext(upload_file.filename or "")[1] or ".png"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(save_dir, fname)
    # Read bytes from UploadFile
    content = upload_file.file.read()
    with open(fpath, "wb") as f:
        f.write(content)
    return fpath


def save_b64_to_disk(image_b64: str, save_dir: str, filename: Optional[str] = None) -> str:
    """
    Decode data URI or raw base64 and save to disk. Returns file path.
    Accepts forms like:
      - "data:image/png;base64,AAA..."
      - "AAA..." (raw base64)
    """
    header_sep = ";base64,"
    if header_sep in image_b64:
        _, b64data = image_b64.split(header_sep, 1)
    elif image_b64.startswith("data:") and "," in image_b64:
        # fallback for other data URIs
        _, b64data = image_b64.split(",", 1)
    else:
        b64data = image_b64

    try:
        b = base64.b64decode(b64data, validate=True)
    except Exception:
        # try without validate
        b = base64.b64decode(b64data)
    return save_bytes_to_disk(b, save_dir, filename)


def encode_file_to_datauri(path: str) -> str:
    """Read a file and return a data URI with base64."""
    mime = "image/png"
    _, ext = os.path.splitext(path.lower())
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".bmp":
        mime = "image/bmp"

    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def get_font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def get_text_size(text: str, font: ImageFont.FreeTypeFont):
    """Safe text size utility compatible with multiple Pillow versions."""
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    elif hasattr(font, "getsize"):
        return font.getsize(text)
    else:
        return (len(text) * 8, 16)


# -------------- Endpoint helpers --------------
def write_annotated_image_and_b64(pil_img: Image.Image, out_dir: str, prefix: str) -> (str, str):
    os.makedirs(os.path.join(out_dir, "annotated"), exist_ok=True)
    out_fname = f"{prefix}_{uuid.uuid4().hex}.png"
    out_path = os.path.join(out_dir, "annotated", out_fname)
    pil_img.save(out_path)
    b64 = encode_file_to_datauri(out_path)
    return out_path, b64


def process_image_input(image_b64: Optional[str] = None, image_file: Optional[UploadFile] = None, save_dir: str = "temp") -> str:
    """
    Process either base64 string or uploaded file and return disk path.
    This is the key function that handles both input methods correctly.
    """
    if image_b64:
        try:
            # Handle base64 input
            path = save_b64_to_disk(image_b64, os.path.join(save_dir, "inputs"))
            print(f"✅ Base64 image saved to: {path}")
            return path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    elif image_file:
        # Handle file upload
        if not image_file.content_type or not image_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload a valid image file.")
        
        try:
            path = save_upload_to_disk(image_file, os.path.join(save_dir, "inputs"))
            print(f"✅ Uploaded image saved to: {path}")
            return path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing uploaded file: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="No image provided. Send either JSON with 'b64' field or multipart file.")


# -------------- Unified endpoint handler --------------
async def predict_skin_condition(
    model: YOLO, 
    model_name: str, 
    image_b64: Optional[str] = None, 
    image_file: Optional[UploadFile] = None,
    output_dir: str = "temp",
    confidence: float = 0.25,
    use_custom_annotation: bool = False
) -> PredictionResponse:
    """
    Unified prediction function for both acne and dark circles.
    """
    try:
        # Process input image
        if output_dir == ACNE_OUT_DIR:
            save_dir = ACNE_OUT_DIR
        else:
            save_dir = DC_OUT_DIR
            
        img_path = process_image_input(image_b64, image_file, save_dir)
        
        # Run YOLO inference
        print(f"🔍 Running {model_name} inference on: {img_path}")
        results = model.predict(
            source=img_path,
            conf=confidence,
            imgsz=IMGSZ,
            device=DEVICE,
            save=False,
            verbose=False
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Model prediction returned no results")
        
        res = results[0]
        
        # Extract detections
        detections: List[Detection] = []
        
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            names = model.names
            
            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = box
                class_name = names.get(cls_id, str(cls_id))
                
                detection_data = {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "class_id": int(cls_id),
                    "class_name": class_name
                }
                
                # Add custom confidence for acne
                if use_custom_annotation:
                    detection_data["conf_plus_0_5"] = float(conf) + 0.5
                else:
                    detection_data["conf_plus_0_5"] = None
                
                detections.append(Detection(**detection_data))
        
        # Generate annotated image
        if use_custom_annotation:
            # Custom annotation for acne
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            font = get_font(18)
            
            for detection in detections:
                x1, y1, x2, y2 = detection.box
                label_text = f"{detection.class_name} {detection.conf_plus_0_5:.2f}"
                
                # Draw bbox
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                
                # Text background
                text_w, text_h = get_text_size(label_text, font)
                text_y = max(y1 - text_h, 0)
                draw.rectangle([x1, text_y, x1 + text_w, text_y + text_h], fill="blue")
                
                # Text
                draw.text((x1, text_y), label_text, fill="white", font=font)
            
            annotated_img = img
            
        else:
            # Standard YOLO annotation for dark circles
            try:
                annotated_np = res.plot()  # BGR numpy array
                if annotated_np.ndim == 3 and annotated_np.shape[2] == 3:
                    annotated_rgb = annotated_np[..., ::-1]  # Convert BGR to RGB
                else:
                    annotated_rgb = annotated_np
                annotated_img = Image.fromarray(annotated_rgb)
            except Exception as e:
                print(f"⚠️ YOLO plot failed, using original image: {e}")
                annotated_img = Image.open(img_path).convert("RGB")
        
        # Save annotated image and generate base64
        out_path, out_b64 = write_annotated_image_and_b64(annotated_img, save_dir, model_name)
        
        print(f"✅ {model_name} prediction completed: {len(detections)} detections")
        
        return PredictionResponse(
            model_name=model_name,
            detections=detections,
            annotated_image_path=out_path,
            annotated_image_b64=out_b64,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in {model_name} prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# -------------- Acne endpoint --------------
@app.post("/predict/acne", response_model=PredictionResponse)
async def predict_acne(
    image_b64: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Accepts either:
    - JSON: {"b64": "data:image/png;base64,..."} 
    - OR multipart form with 'image' file
    - OR multipart form with 'image_b64' string
    
    Returns acne detections with custom confidence (+0.5) and annotated image.
    """
    # Handle JSON body (when Content-Type: application/json)
    from fastapi import Request
    import json
    
    try:
        # If no form data, check for JSON body
        if not image_b64 and not image:
            request = Request
            body = await request.json()
            if 'b64' in body:
                image_b64 = body['b64']
    except:
        pass
    
    return await predict_skin_condition(
        model=acne_model,
        model_name="acne",
        image_b64=image_b64,
        image_file=image,
        output_dir=ACNE_OUT_DIR,
        confidence=ACNE_CONF,
        use_custom_annotation=True
    )


# -------------- Dark Circles endpoint --------------
@app.post("/predict/dark_circles", response_model=PredictionResponse)
async def predict_dark_circles(
    image_b64: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Accepts either:
    - JSON: {"b64": "data:image/png;base64,..."} 
    - OR multipart form with 'image' file
    - OR multipart form with 'image_b64' string
    
    Returns dark circle detections with standard YOLO annotation.
    """
    # Handle JSON body (when Content-Type: application/json)
    from fastapi import Request
    import json
    
    try:
        # If no form data, check for JSON body
        if not image_b64 and not image:
            request = Request
            body = await request.json()
            if 'b64' in body:
                image_b64 = body['b64']
    except:
        pass
    
    return await predict_skin_condition(
        model=dc_model,
        model_name="dark_circles",
        image_b64=image_b64,
        image_file=image,
        output_dir=DC_OUT_DIR,
        confidence=DC_CONF,
        use_custom_annotation=False
    )


# -------------- Combined endpoint (runs both models on the same input) --------------
@app.post("/predict/combined", response_model=CombinedPredictionResponse)
async def predict_combined(
	image_b64: Optional[str] = Form(None),
	image: Optional[UploadFile] = File(None)
):
	"""
	Runs BOTH acne and dark-circles models on the same input and returns both results.
	Accepts either JSON with base64 or multipart form with file.
	"""
	# Handle JSON body (when Content-Type: application/json)
	from fastapi import Request

	try:
		# If no form data, check for JSON body
		if not image_b64 and not image:
			request = Request
			body = await request.json()
			if 'b64' in body:
				image_b64 = body['b64']
	except Exception:
		pass

	# If UploadFile is provided, convert to base64 once (UploadFile can only be read once)
	if image and not image_b64:
		if not image.content_type or not image.content_type.startswith("image/"):
			raise HTTPException(status_code=400, detail="Please upload a valid image file.")
		# Read file content once
		file_content = await image.read()
		# Convert to base64
		import base64
		b64_data = base64.b64encode(file_content).decode("utf-8")
		# Determine MIME type
		mime_type = image.content_type or "image/png"
		image_b64 = f"data:{mime_type};base64,{b64_data}"
		# Clear image_file so we use base64 for both calls
		image = None

	# Now use base64 for both predictions (can be reused)
	acne_res = await predict_skin_condition(
		model=acne_model,
		model_name="acne",
		image_b64=image_b64,
		image_file=None,  # Use None since we're using base64
		output_dir=ACNE_OUT_DIR,
		confidence=ACNE_CONF,
		use_custom_annotation=True
	)

	dc_res = await predict_skin_condition(
		model=dc_model,
		model_name="dark_circles",
		image_b64=image_b64,  # Reuse the same base64
		image_file=None,  # Use None since we're using base64
		output_dir=DC_OUT_DIR,
		confidence=DC_CONF,
		use_custom_annotation=False
	)

	return CombinedPredictionResponse(acne=acne_res, dark_circles=dc_res)

# -------------- Alternative JSON-only endpoints --------------
@app.post("/predict/acne/json")
async def predict_acne_json(payload: ImageB64):
    """Alternative endpoint that only accepts JSON with base64"""
    return await predict_skin_condition(
        model=acne_model,
        model_name="acne",
        image_b64=payload.b64,
        image_file=None,
        output_dir=ACNE_OUT_DIR,
        confidence=ACNE_CONF,
        use_custom_annotation=True
    )


@app.post("/predict/dark_circles/json")
async def predict_dark_circles_json(payload: ImageB64):
    """Alternative endpoint that only accepts JSON with base64"""
    return await predict_skin_condition(
        model=dc_model,
        model_name="dark_circles",
        image_b64=payload.b64,
        image_file=None,
        output_dir=DC_OUT_DIR,
        confidence=DC_CONF,
        use_custom_annotation=False
    )


# -------------- Root --------------
@app.get("/")
def read_root():
    return {
        "message": "Skin Detection API is running",
        "version": "1.2.0",
        "endpoints": {
            "acne": {
                "json": "/predict/acne/json",
                "multipart": "/predict/acne"
            },
            "dark_circles": {
                "json": "/predict/dark_circles/json", 
                "multipart": "/predict/dark_circles"
            }
        },
        "input_methods": [
            "JSON with base64",
            "Multipart form with file upload",
            "Multipart form with base64 string"
        ]
    }


# -------------- Health check --------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "device": DEVICE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)