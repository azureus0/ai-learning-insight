import pandas as pd
import joblib
import os
import google.generativeai as genai
from ml_utils import perform_feature_engineering_final

# --- 1. SETUP MODEL & API ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load Model K-Means & Scaler
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    features_list = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
except:
    model, scaler, features_list = None, None, []

# Setup Google Gemini (Ambil dari Environment Variable biar aman)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None
    print("Peringatan: GEMINI_API_KEY tidak ditemukan. Menggunakan pesan default.")

# --- 2. FUNGSI FALLBACK (LOGIKA LAMA - SAFETY NET) ---
def generate_fallback_message(row, cluster_label):
    """Dipanggil jika Gemini Error atau API Key tidak ada."""
    total_materi = int(row.get('total_completed_tutorials', 0))
    nilai_kuis   = round(float(row.get('avg_submission_rating', 0)), 1)
    durasi_rata  = int(row.get('avg_tutorial_duration', 0))
    hari_aktif   = int(row.get('active_days', 0))

    if cluster_label == "Fast Learner":
        if nilai_kuis >= 4.5:
            return f"Luar biasa! Melahap {total_materi} materi dengan cepat dan nilai sempurna ({nilai_kuis})."
        return f"Wow, ngebut banget! {total_materi} materi selesai. Tetap jaga ketelitian ya."
    
    elif cluster_label == "Reflective Learner":
        return f"Pendekatan belajarmu mendalam (rata-rata {durasi_rata} menit/materi). Kualitas pemahamanmu mantap!"
    
    elif cluster_label == "Consistent Learner":
        return f"Konsistensi juara! Hadir {hari_aktif} hari minggu ini. Disiplin adalah kuncimu."
    
    else:
        return "Yuk, mulai buka satu materi ringan hari ini untuk menjaga momentum!"

# --- 3. FUNGSI GEMINI (DYNAMIC INSIGHT) ---
def generate_dynamic_insight(row, cluster_label):
    # Kalau model gemini belum siap, pakai fallback
    if not gemini_model:
        return generate_fallback_message(row, cluster_label)

    # Siapkan data user yang 'human-readable'
    durasi = round(row.get('avg_tutorial_duration', 0), 1)
    revisit = round(row.get('tutorial_revisit_rate', 0), 2)
    nilai = round(row.get('avg_weighted_exam_score', 0), 1)
    aktif = int(row.get('active_days', 0))
    materi = int(row.get('total_completed_tutorials', 0))

    # Prompt Engineering (Instruksi Persona)
    prompt = f"""
    Berperanlah sebagai Mentor Pembelajaran IT yang asik, suportif, dan gaul (pake bahasa Indonesia 'aku-kamu').
    Buatlah SATU kalimat insight pendek (maksimal 20 kata) untuk user ini.

    DATA USER:
    - Status: {cluster_label}
    - Nilai: {nilai}/100
    - Durasi: {durasi} menit/materi
    - Revisit: {revisit} (Makin tinggi makin sering mengulang)
    - Keaktifan: {aktif} hari

    KONTEKS:
    - Fast Learner: Gesit, materi banyak. (Puji speed-nya).
    - Consistent Learner: Nilai bagus, rajin mengulang/revisit. (Puji kedisiplinannya).
    - Reflective Learner: Durasi lama per materi, jarang mengulang karena sekali baca langsung paham (Deep Dive). (Puji fokus & kedalamannya, JANGAN bilang dia lambat).

    Berikan komentar yang spesifik berdasarkan data di atas.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        # Kalau error (kuota abis/koneksi), panggil fallback
        return generate_fallback_message(row, cluster_label)

# --- 4. PREDICTION FUNCTION UTAMA ---
def predict_user_category(raw_data_dict):
    if not model: return {"error": "Model not loaded"}

    # Feature Engineering
    df_features = perform_feature_engineering_final(raw_data_dict)
    if df_features.empty: return {"category": "Unknown", "message": "Data insufficient"}

    # Isi kolom yang hilang dengan 0
    for c in features_list:
        if c not in df_features.columns: df_features[c] = 0
    X = df_features[features_list].fillna(0)

    # Predict
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]

    # Mapping Label (SESUAI ANALISIS TERAKHIR KITA)
    # 0: Fast (Cepat & Banyak)
    # 1: Consistent (Nilai Tinggi & Sering Revisit)
    # 2: Reflective (Durasi Lama & Deep Dive)
    labels_map = {
        0: "Fast Learner", 
        1: "Consistent Learner", 
        2: "Reflective Learner"
    }
    result_label = labels_map.get(cluster, "Unknown")

    # Generate Insight (Coba Gemini dulu, kalau gagal otomatis fallback)
    user_row = df_features.iloc[0]
    final_message = generate_dynamic_insight(user_row, result_label)

    return {
        "user_id": int(df_features.index[0]),
        "category": result_label,
        "insight_message": final_message,
        "metrics": df_features[features_list].to_dict('records')[0]
    }

if __name__ == "__main__":
    print("Inference script with Gemini & Fallback ready.")
