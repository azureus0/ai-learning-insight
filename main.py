from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
# Kita import fungsi dari inference_script yang udah kita buat sebelumnya
from inference_script import predict_user_category

app = FastAPI()

# --- INPUT SCHEMA (WAJIB SAMA PERSIS DENGAN JSON DARI BACKEND) ---
# Karena Backend ngambil dari DB, key-nya pasti nama tabel panjang.
# Kita terima dulu apa adanya di sini.
class InputData(BaseModel):
    users: list = []
    developer_journeys: list = []            # Sesuai DB
    developer_journey_tutorials: list = []   # Sesuai DB
    developer_journey_trackings: list = []   # Sesuai DB
    developer_journey_submissions: list = [] # Sesuai DB
    developer_journey_completions: list = [] # Sesuai DB
    exam_registrations: list = []            # Sesuai DB
    exam_results: list = []                  # Sesuai DB

@app.get("/")
def home():
    return {"message": "AI Learning Insight API is Running! Send POST to /predict"}

@app.post("/predict")
def predict_endpoint(data: InputData):
    try:
        # 1. Konversi JSON Input ke Dictionary of DataFrames
        # Hasilnya: {'developer_journey_trackings': DataFrame, ...}
        raw_data = {k: pd.DataFrame(v) for k, v in data.dict().items()}

        # 2. Panggil Fungsi Prediksi Utama
        # Note: Di dalam fungsi ini (tepatnya di ml_utils.py), 
        # kita udah bikin logika buat RENAME 'developer_journey_trackings' jadi 'trackings'.
        # Jadi aman!
        result = predict_user_category(raw_data)

        # 3. Cek Error dari fungsi prediksi
        if "error" in result:
            # Kalau model gak ke-load atau error lain
            raise HTTPException(status_code=500, detail=result["error"])

        # Handle kasus user baru/data kosong
        if result.get("category") == "Unknown" or result.get("category") == "Sleeping Student":
             # Opsional: Bisa di-customize handling-nya di sini
             pass

        return result

    except Exception as e:
        # Print error ke terminal log biar gampang debug kalau ada apa-apa
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Blok ini biar bisa dijalankan lokal pakai 'python main.py'
if __name__ == "__main__":
    import uvicorn
    import os

    # Ambil PORT dari Railway, kalau gak ada pake 8000 (buat lokal)
    port = int(os.getenv("PORT", 8000))

    # Host wajib 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=port)
