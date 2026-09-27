from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from pathlib import Path

# Adjust path to import analytics_builder
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.analytics.analytics_builder import build_analytics_json

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    # Save uploaded file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Member 1 Analytics Pipeline
        analytics_result = build_analytics_json(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return analytics_result
        
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
