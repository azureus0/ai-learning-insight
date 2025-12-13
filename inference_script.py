import pandas as pd
import joblib
import os
import google.generativeai as genai
from ml_utils import perform_feature_engineering_final

# --- 1. SETUP MODEL & API ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load Model
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    features_to_use = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
except Exception as e:
    print(f"Error loading models: {e}")
    model, scaler, features_to_use = None, None, []

# Setup Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IS_GEMINI_ACTIVE = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_ai = genai.GenerativeModel('gemini-2.5-flash')
        IS_GEMINI_ACTIVE = True
        print("Gemini API Active.")
    except:
        print("Gemini API Key invalid.")
else:
    print("Gemini API Key not found. Using Fallback mode.")

# Mapping Cluster (SESUAI NOTEBOOK)
CLUSTER_MAPPING = {
    0: "Reflective Learner",
    1: "Fast Learner",
    2: "Consistent Learner"
}

# --- 2. FUNGSI FALLBACK ---
def _get_fallback_message(user_row, cluster_label):
    density      = float(user_row.get('completion_density', 0))
    score        = float(user_row.get('avg_weighted_exam_score', 0))
    consistency  = float(user_row.get('consistency_score', 0))
    duration     = float(user_row.get('avg_tutorial_duration', 0))
    revisit_pct  = float(user_row.get('tutorial_revisit_rate', 0)) * 100

    if cluster_label == "Fast Learner":
        return f"Gokil! Kecepatan belajarmu tembus {density:.1f} materi/hari dengan nilai rata-rata {score:.1f}. Speed run yang berkualitas!"
    elif cluster_label == "Reflective Learner":
        return f"Kamu tipe Deep Learner. Rata-rata {duration:.0f} menit/materi dan mereview ulang {revisit_pct:.0f}% tutorial membuktikan dedikasimu!"
    elif cluster_label == "Consistent Learner":
        return f"Konsistensi juara! Skor disiplinmu mencapai {consistency*100:.0f}% ditambah kerajinanmu mengulang {revisit_pct:.0f}% materi."
    elif cluster_label == "Sleeping Student":
        return "Belum ada aktivitas signifikan. Yuk, mulai buka satu materi ringan hari ini buat pemanasan!"
    else:
        return "Tetap semangat belajar! Lanjutkan progres positifmu hari ini."

# --- 3. FUNGSI UTAMA ---
def generate_insight_message(user_row, cluster_label):
    if not IS_GEMINI_ACTIVE:
        return _get_fallback_message(user_row, cluster_label)

    context = {
        "kategori": cluster_label,
        "nilai_ujian": f"{user_row.get('avg_weighted_exam_score', 0):.1f}",
        "kepadatan_belajar": f"{user_row.get('completion_density', 0):.1f}",
        "konsistensi": f"{user_row.get('consistency_score', 0)*100:.0f}%",
        "durasi_per_materi": f"{user_row.get('avg_tutorial_duration', 0):.0f} menit",
        "tingkat_mengulang": f"{user_row.get('tutorial_revisit_rate', 0)*100:.0f}%"
    }

    prompt = f"""
    Bertindaklah sebagai "AI Learning Buddy" yang seru, antusias, dan personal (seperti gaya bahasa 'Spotify Wrapped').
    Tugasmu adalah merangkum performa belajar siswa ini dalam 3-4 kalimat yang menarik dan bikin bangga.

    DATA SISWA: {context}

    PANDUAN NARASI PER KATEGORI (WAJIB GABUNGKAN DATA):
    1. Jika 'Fast Learner':
       - Soroti kombinasi 'Speed' ({context['kepadatan_belajar']} materi/hari) DAN 'Quality' (Nilai {context['nilai_ujian']}).
       - Contoh vibe: "Gila! Kamu ngebut banget hari ini tapi nilaimu tetap fantastis!"
    2. Jika 'Reflective Learner':
       - Soroti 'Dedikasi Waktu' ({context['durasi_per_materi']}) DAN dampaknya ke 'Nilai' ({context['nilai_ujian']}).
       - Contoh vibe: "Kamu bukan sekadar belajar, tapi benar-benar mendalami. Pantas nilaimu setinggi itu!"
    3. Jika 'Consistent Learner':
       - Soroti 'Skor Konsistensi' ({context['konsistensi']}) DAN 'Revisit Rate' ({context['tingkat_mengulang']}).
       - Contoh vibe: "Disiplinmu juara! Konsistensi setinggi ini jarang dimiliki orang lain lho."
    4. Jika 'Sleeping Student':
       - Ajak dengan nada penasaran/FOMO. "Semua orang udah mulai level up, kamu kapan nyusul?"

    ATURAN TAMBAHAN:
    - Gunakan kata seru seperti "Wow", "Gokil", "Salut", "Mantap".
    - WAJIB sebutkan minimal 2 angka spesifik dari data di atas agar terasa sangat personal.

    Outputkan teks paragraf saja tanpa tanda kutip.
    """

    try:
        response = model_ai.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return _get_fallback_message(user_row, cluster_label)

# --- 4. FUNGSI PREDIKSI ---
def predict_user_category(raw_data_dict):
    if not model: return {"error": "Model not loaded properly."}

    # A. Feature Engineering (Panggil ml_utils)
    df_features = perform_feature_engineering_final(raw_data_dict)

    # Handling Data Kosong (Langsung isi 0 semua biar jadi Sleeping Student)
    if df_features.empty:
        # Buat dictionary dengan nilai 0 untuk semua fitur
        user_features_dict = {col: 0.0 for col in features_to_use}
        user_id = 0 # Default ID kalau data kosong
    else:
        # Ambil data user pertama
        user_features_dict = df_features.iloc[0].to_dict()
        user_id = int(df_features.index[0])

    # B. Cek "Sleeping Student"
    # Logic: Jika total semua nilai fitur adalah 0 (atau mendekati 0), maka dia Sleeping
    total_value = sum(user_features_dict.get(col, 0) for col in features_to_use)

    if total_value == 0:
        cluster_label = "Sleeping Student"
        cluster_id = -1
        ai_message = generate_insight_message(user_features_dict, cluster_label)

        return {
            "user_id": user_id,
            "category": cluster_label,
            "cluster_id": cluster_id,
            "insight_message": ai_message,
            "metrics": user_features_dict
        }

    # C. Predict (Jika ada data > 0)
    input_df = pd.DataFrame([user_features_dict])
    for col in features_to_use:
        if col not in input_df.columns: input_df[col] = 0

    X_input = input_df[features_to_use]
    X_scaled = scaler.transform(X_input.values)
    cluster_id = int(model.predict(X_scaled)[0])
    cluster_label = CLUSTER_MAPPING.get(cluster_id, "Unknown Learner")

    # D. Insight
    ai_message = generate_insight_message(user_features_dict, cluster_label)

    return {
        "user_id": user_id,
        "category": cluster_label,
        "cluster_id": cluster_id,
        "insight_message": ai_message,
        "metrics": user_features_dict
    }
