import pandas as pd
import joblib
import os
import google.generativeai as genai
from ml_utils import perform_feature_engineering_final

# --- 1. SETUP MODEL ML ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    features_list = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
except Exception as e:
    print(f"Error loading models: {e}")
    model, scaler, features_list = None, None, []

# --- 2. SETUP GEMINI API ---
# Ambil API Key dari Environment Variable (Set di Railway nanti)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Pake model flash biar cepet & murah
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None
    print("Warning: GEMINI_API_KEY tidak ditemukan. Mode Fallback aktif.")

# --- 3. FUNGSI "BAN SEREP" (Rule Based) ---
# Dipanggil kalau Gemini error atau API Key gak ada
def get_fallback_message(row, label):
    nilai = row.get('avg_weighted_exam_score', 0)
    aktif = row.get('active_days', 0)
    
    if label == "Fast Learner":
        return f"Wussh! Kamu ngebut banget. Pertahankan speed-nya tapi jangan lupa teliti ya!"
    elif label == "Consistent Learner":
        return f"Konsistensi juara! Hadir {int(aktif)} hari dengan nilai {int(nilai)}. Keren!"
    elif label == "Reflective Learner":
        return f"Belajarmu mendalam dan fokus. Kualitas pemahamanmu mantap!"
    else:
        return "Yuk mulai belajar hari ini biar dapet insight keren!"

# --- 4. FUNGSI UTAMA GEMINI ---
def generate_ai_insight(row, label):
    # Kalau gak ada Gemini, langsung pake ban serep
    if not gemini_model:
        return get_fallback_message(row, label)
    
    # Siapkan data buat dikirim ke AI
    data_context = {
        "status": label,
        "nilai": int(row.get('avg_weighted_exam_score', 0)),
        "durasi": int(row.get('avg_tutorial_duration', 0)),
        "revisit": round(row.get('tutorial_revisit_rate', 0), 2),
        "aktif": int(row.get('active_days', 0))
    }

    # Instruksi ke AI (Prompt)
    prompt = f"""
    Berperanlah sebagai Mentor Coding yang asik, gaul, dan suportif (pake 'aku-kamu').
    Buatlah SATU kalimat pendek (max 20 kata) untuk user ini:
    
    DATA USER: {data_context}
    
    KONTEKS:
    - Fast Learner: Cepet banget, materi banyak. (Puji kecepatannya).
    - Consistent Learner: Nilai bagus, rajin, disiplin. (Puji konsistensinya).
    - Reflective Learner: Belajar lama per sesi, mendalam (Deep Dive). (Puji fokusnya, JANGAN bilang lambat).
    
    Berikan komentar spesifik berdasarkan data angka di atas tapi jadikan narasi santai.
    """

    try:
        # Tembak ke Google
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        # Kalau gagal (misal kuota abis), pake ban serep
        return get_fallback_message(row, label)

# --- 5. FUNGSI PREDIKSI (Dipanggil main.py) ---
def predict_user_category(raw_data_dict):
    if not model: return {"error": "Model not loaded"}

    # A. Masak Data
    df_features = perform_feature_engineering_final(raw_data_dict)
    if df_features.empty: 
        return {"category": "Newcomer", "insight_message": "Mulai belajar yuk!", "metrics": {}}

    # Isi 0 kalo ada kolom yang ilang
    for c in features_list:
        if c not in df_features.columns: df_features[c] = 0
    
    # B. Prediksi K-Means
    X = df_features[features_list].fillna(0)
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]

    # C. Mapping Label (Sesuai analisis kita tadi)
    labels_map = {
        0: "Fast Learner",      # Density Tinggi, Durasi Cepat
        1: "Consistent Learner",# Nilai Tinggi, Sering Revisit
        2: "Reflective Learner" # Durasi Lama, Revisit Rendah
    }
    result_label = labels_map.get(cluster, "Unknown")

    # D. Generate Insight (Gemini / Fallback)
    user_row = df_features.iloc[0] # Ambil baris pertama (user ybs)
    final_message = generate_ai_insight(user_row, result_label)

    # E. Bungkus Output JSON
    return {
        "user_id": int(df_features.index[0]),
        "category": result_label,
        "insight_message": final_message, # <-- Ini hasil Gemini
        "metrics": df_features[features_list].to_dict('records')[0]
    }