import pandas as pd
import numpy as np
import joblib
import gdown
import os
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from ml_utils import perform_feature_engineering_final

# 1. LOAD DATA
output_folder = 'dataset_project'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    url = 'https://drive.google.com/drive/folders/1uRI03cmYx24CzIfGmjoIB6UBwv0oTy_v?usp=sharing'
    gdown.download_folder(url=url, output=output_folder, quiet=True, use_cookies=False)

files = ['users', 'developer_journeys', 'developer_journey_tutorials', 'developer_journey_trackings',
         'developer_journey_submissions', 'developer_journey_completions', 'exam_registrations', 'exam_results']
dfs = {}
for f in files:
    try:
        path = f'{output_folder}/{f}.csv'
        if not os.path.exists(path): path = f'{output_folder}/{f if "developer" in f else "developer_journey_" + f}.csv'
        dfs[f] = pd.read_csv(f'{output_folder}/{f}.csv')
    except: dfs[f] = pd.DataFrame()

# 2. MASAK DATA
df_master = perform_feature_engineering_final(dfs)

# 3. TRAINING
print("Training KMeans...")
# List fitur final
features_final = [
    'avg_weighted_exam_score',
    'exam_duration_utilization_ratio',
    'avg_submission_revision_count',
    'avg_submission_revision_duration',
    'avg_submission_rating',
    'completion_density',
    'consistency_score',
    'tutorial_revisit_rate',
    'avg_tutorial_duration'
]

# Pastikan kolom ada
for col in features_final:
    if col not in df_master.columns: df_master[col] = 0

X = df_master[features_final].fillna(0)

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans (k=3)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)

# 4. SAVE
print("Saving Models...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(features_final, 'models/feature_list.pkl')
print("Training Selesai. Model tersimpan.")
