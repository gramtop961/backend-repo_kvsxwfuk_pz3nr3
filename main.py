import os
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Image processing
from PIL import Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Floture Detector API is running"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect_floture(image: UploadFile = File(...)):
    """
    Simple heuristic detector for the fictional flower "floture".
    We treat images dominated by vivid reds (like spider lilies) as positive.
    Returns a confidence score in [0,1].
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        data = await image.read()
        pil = Image.open(BytesIO(data)).convert("RGB")
        # Resize for speed and normalization
        pil_small = pil.copy()
        pil_small.thumbnail((256, 256))
        arr = np.asarray(pil_small).astype(np.float32)

        # Compute red emphasis metric
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # Mask: red significantly higher than green/blue and reasonably bright
        red_dominant = (r > g * 1.2) & (r > b * 1.2) & (r > 100)
        red_ratio = float(red_dominant.mean()) if red_dominant.size > 0 else 0.0

        # Additional saturation-like cue
        max_rgb = np.maximum(np.maximum(r, g), b) + 1e-6
        saturation = ((max_rgb - np.minimum(np.minimum(r, g), b)) / max_rgb)
        sat_mean = float(saturation.mean())

        # Combine metrics into confidence
        conf = max(0.0, min(1.0, red_ratio * 2.0 * (0.5 + 0.5 * sat_mean)))

        # Threshold for declaring "floture" present
        detected = conf >= 0.35
        result = {
            "detected": detected,
            "label": "floture" if detected else "unknown",
            "confidence": round(conf, 3),
            "metrics": {
                "red_ratio": round(red_ratio, 4),
                "saturation": round(sat_mean, 4),
                "width": pil.width,
                "height": pil.height,
            },
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)[:200]}")


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
