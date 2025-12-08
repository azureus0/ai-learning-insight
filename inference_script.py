import pandas as pd
import joblib
import os
from ml_utils import perform_feature_engineering_final

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    features_list = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
except:
    model, scaler, features_list = None, None, []

# --- FUNGSI GENERATE INSIGHT (LOGIKA KAMU) ---
def generate_insight_message(row, cluster_label):
    # Ambil data pendukung (Handle kalau kolom gak ada biar gak error)
    # Kita sesuaikan nama kolom dengan output dari ml_utils.py
    total_materi = int(row.get('total_completed_tutorials', 0)) # Di ml_utils namanya ini
    hari_aktif   = int(row.get('active_days', 0))
    nilai_kuis   = round(float(row.get('avg_submission_rating', 0)), 1)
    durasi_rata  = int(row.get('avg_tutorial_duration', 0)) # Di ml_utils namanya ini

    # 1. Kategori: FAST LEARNER
    if cluster_label == "Fast Learner":
        if nilai_kuis >= 4.5:
            return f"Luar biasa! Kamu melahap {total_materi} materi dengan sangat cepat dan nilai sempurna ({nilai_kuis}). Pertahankan speed-nya!"
        else:
            return f"Wow, ngebut banget! {total_materi} materi selesai minggu ini. Coba pelan sedikit biar nilai kuisnya makin maksimal ya."

    # 2. Kategori: REFLECTIVE LEARNER
    elif cluster_label == "Reflective Learner":
        if durasi_rata > 30: # (Sesuaikan threshold durasi menitmu)
            return f"Kamu tipe pemikir dalam. Rata-rata {durasi_rata} menit per materi menunjukkan dedikasimu. Hasilnya nilai kuis kamu oke ({nilai_kuis})!"
        else:
            return f"Pendekatan belajarmu sangat mendalam. Kualitas jawaban submission kamu menunjukkan pemahaman yang kuat."

    # 3. Kategori: CONSISTENT LEARNER
    elif cluster_label == "Consistent Learner":
        if hari_aktif >= 5:
            return f"Konsistensi juara! Kamu hadir belajar selama {hari_aktif} hari minggu ini. Disiplin adalah kunci suksesmu."
        else:
            return f"Kerja bagus! Kamu rutin menyelesaikan materi secara berkala. Terus jaga ritme belajar ini ya."

    # 4. Fallback (Unknown/Sleeping)
    else:
        return "Belum ada aktivitas signifikan minggu ini. Yuk, mulai buka satu materi ringan hari ini!"


def predict_user_category(raw_data_dict):
    if not model: return {"error": "Model not loaded"}

    # 1. Masak Data
    df_features = perform_feature_engineering_final(raw_data_dict)
    if df_features.empty: return {"category": "Unknown", "message": "Data insufficient"}

    # 2. Select Features
    for c in features_list:
        if c not in df_features.columns: df_features[c] = 0
            
    X = df_features[features_list].fillna(0)
    
    # 3. Predict Cluster
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]
    
    # 4. Mapping Label (Pastikan mapping ini sesuai hasil trainingmu!)
    # Tips: Nanti pas training, cek dulu cluster 0 itu karakternya apa, baru update dict ini.
    labels_map = {0: "Fast Learner", 1: "Reflective Learner", 2: "Consistent Learner"}
    result_label = labels_map.get(cluster, "Unknown")
    
    # 5. Generate Insight Message (Panggil fungsi di atas)
    # Kita ambil row pertama (karena prediksi per user)
    user_row = df_features.iloc[0]
    final_message = generate_insight_message(user_row, result_label)
    
    return {
        "user_id": int(df_features.index[0]),
        "category": result_label,
        "insight_message": final_message,
        "metrics": df_features[features_list].to_dict('records')[0]
    }

if __name__ == "__main__":
    print("Inference script ready.")
