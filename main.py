from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
# Kita import fungsi dari inference_script yang udah kita buat sebelumnya
from inference_script import predict_user_category

app = FastAPI()

# Input Schema (Sesuai dengan data mentah yang dikirim Backend)
class InputData(BaseModel):
    users: list = []
    trackings: list = []
    submissions: list = []
    completions: list = []
    journeys: list = []
    tutorials: list = []
    exam_registrations: list = []
    exam_results: list = []

@app.get("/")
def home():
    return {"message": "AI Learning Insight API is Running! Send POST to /predict"}

@app.post("/predict")
def predict_endpoint(data: InputData):
    try:
        # 1. Konversi JSON Input ke Dictionary of DataFrames
        # (Format ini yang dimengerti oleh inference_script kita)
        raw_data = {k: pd.DataFrame(v) for k, v in data.dict().items()}
        
        # 2. Panggil Fungsi Prediksi Utama
        # Fungsi ini ada di inference_script.py, dia yang handle feature engineering + predict + insight message
        result = predict_user_category(raw_data)
        
        # 3. Cek Error dari fungsi prediksi
        if "error" in result:
            # Kalau model gak ke-load atau error lain
            raise HTTPException(status_code=500, detail=result["error"])
        
        if result.get("category") == "Unknown" and result.get("message") == "Data insufficient":
             # Kalau data user gak cukup (misal user baru banget daftar)
             return {
                 "user_id": 0,
                 "category": "Newcomer",
                 "insight_message": "Selamat datang! Yuk mulai belajar biar AI bisa kasih insight.",
                 "metrics": {}
             }
            
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Blok ini biar bisa dijalankan lokal pakai 'python main.py'
if __name__ == "__main__":
    import uvicorn
    # Host 0.0.0.0 biar bisa diakses dari luar container (penting buat deployment)
    uvicorn.run(app, host="0.0.0.0", port=8000)
