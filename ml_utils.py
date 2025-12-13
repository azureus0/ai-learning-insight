import pandas as pd
import numpy as np

def perform_feature_engineering_final(dfs):
    """
    FEATURE ENGINEERING SINKRON (Updated with DB Mapping)
    """

    # =========================================================
    # 0.A. MAPPING NAMA TABEL DB KE NAMA LOGIC PYTHON (BARU)
    # =========================================================
    # Ini bagian PENTING yang harus ditambah biar sinkron sama Backend/DB
    table_mapping = {
        'developer_journey_trackings': 'trackings',
        'developer_journey_submissions': 'submissions',
        'developer_journey_completions': 'completions',
        'developer_journey_tutorials': 'tutorials',
        'developer_journeys': 'journeys',
        # Tabel yang namanya sudah sama, tidak perlu di-mapping
        'users': 'users',
        'exam_registrations': 'exam_registrations',
        'exam_results': 'exam_results'
    }

    # Lakukan Rename Key di Dictionary dfs
    # Kita pakai .pop() untuk mengambil data dari key lama dan memindahkannya ke key baru
    for db_name, script_name in table_mapping.items():
        if db_name in dfs:
            dfs[script_name] = dfs.pop(db_name)

    # =========================================================
    # 0.B. PREPARATION (Code Lama Kamu Mulai dari Sini)
    # =========================================================
    
    # Pastikan tabel ada (pake nama script_name yang pendek)
    required_tables = ['users', 'exam_results', 'exam_registrations', 'trackings', 'submissions']
    for tbl in required_tables:
        if tbl not in dfs: dfs[tbl] = pd.DataFrame()

    # Helper DateTime
    def to_dt(df, cols):
        for c in cols:
            if c in df.columns: df[c] = pd.to_datetime(df[c], errors='coerce')
        return df

    # Convert DateTime
    if not dfs['exam_registrations'].empty:
        dfs['exam_registrations'] = to_dt(dfs['exam_registrations'], ['created_at', 'deadline_at', 'exam_finished_at'])
    if not dfs['trackings'].empty:
        dfs['trackings'] = to_dt(dfs['trackings'], ['last_viewed', 'first_opened_at', 'completed_at'])
    if not dfs['submissions'].empty:
        dfs['submissions'] = to_dt(dfs['submissions'], ['created_at', 'ended_review_at'])

    # --- 1. BASE DATAFRAME (Basis: User ID) ---
    if not dfs['users'].empty:
        df_master_features = dfs['users'][['id', 'created_at']].rename(columns={'id': 'user_id'})
    else:
        # Fallback kalau tabel users kosong/error
        all_ids = pd.concat([
            dfs['trackings']['developer_id'] if 'developer_id' in dfs['trackings'] else pd.Series(dtype=int),
            dfs['exam_registrations']['examinees_id'] if 'examinees_id' in dfs['exam_registrations'] else pd.Series(dtype=int)
        ]).unique()
        df_master_features = pd.DataFrame({'user_id': all_ids})

    # Inisialisasi kolom dengan 0.0 (Biar aman)
    cols_to_init = [
        'avg_weighted_exam_score', 'completion_density', 'consistency_score',
        'tutorial_revisit_rate', 'avg_tutorial_duration',
        'active_days', 'total_completed_tutorials' # Kolom bantu
    ]
    for col in cols_to_init:
        df_master_features[col] = 0.0

    # =========================================================
    # FITUR 1: avg_weighted_exam_score
    # =========================================================
    if not dfs['exam_results'].empty and not dfs['exam_registrations'].empty:
        df_res = dfs['exam_results'].copy()
        df_reg = dfs['exam_registrations'].copy()

        merged = df_res.merge(
            df_reg[['id', 'examinees_id']],
            left_on='exam_registration_id',
            right_on='id'
        )

        # Hitung bobot
        merged['weighted'] = merged['score'] * merged['total_questions']

        # Agregasi
        stats = merged.groupby('examinees_id')[['weighted', 'total_questions']].sum()
        weighted_avg = stats['weighted'] / stats['total_questions'].replace(0, 1)

        # Map ke Master
        df_master_features['avg_weighted_exam_score'] = df_master_features['user_id'].map(weighted_avg).fillna(0)

    # =========================================================
    # PERSIAPAN DATA TRACKING (Completed Only)
    # =========================================================
    df_t = pd.DataFrame()
    df_done = pd.DataFrame() # Khusus yg completed

    if not dfs['trackings'].empty:
        df_t = dfs['trackings'].copy()

        # Filter Completed/Passed
        if 'status' in df_t.columns:
            is_done = df_t['status'].astype(str).str.contains('completed|passed|1', case=False, regex=True)
            df_done = df_t[is_done & df_t['completed_at'].notna()].copy()
        else:
            df_done = df_t[df_t['completed_at'].notna()].copy()

    # =========================================================
    # FITUR 2: completion_density (Butuh active_days & total)
    # =========================================================
    if not df_done.empty:
        # A. Hitung Active Days (Owl Logic: -2 Jam)
        dates_dt = pd.to_datetime(df_done['completed_at'])
        dates_owl = (dates_dt - pd.Timedelta(hours=2)).dt.date

        # Map active_days
        count_owl = df_done.assign(d=dates_owl).groupby('developer_id')['d'].nunique()
        df_master_features['active_days'] = df_master_features['user_id'].map(count_owl).fillna(0)

        # B. Hitung Total Completed Tutorials
        total_completed = df_done.groupby('developer_id')['tutorial_id'].nunique()
        df_master_features['total_completed_tutorials'] = df_master_features['user_id'].map(total_completed).fillna(0)

        # C. Hitung Density (Total / Active Days)
        numerator = df_master_features['total_completed_tutorials']
        denominator = df_master_features['active_days'].replace(0, 1)
        df_master_features['completion_density'] = numerator / denominator

    # =========================================================
    # FITUR 3: consistency_score
    # =========================================================
    if not df_done.empty:
        # Pake data owl tadi
        df_cons = df_done.copy()
        df_cons['adj_dt'] = pd.to_datetime(df_cons['completed_at']) - pd.Timedelta(hours=2)
        df_cons['date'] = df_cons['adj_dt'].dt.date

        # Span
        span_stats = df_cons.groupby('developer_id')['date'].agg(['min', 'max'])
        span_stats['total_days'] = (pd.to_datetime(span_stats['max']) - pd.to_datetime(span_stats['min'])).dt.days + 1
        span_stats['total_weeks'] = np.ceil(span_stats['total_days'] / 7)

        # Weekly & Daily Score
        df_cons['week_uniq'] = df_cons['adj_dt'].dt.strftime('%Y-%U')
        score_weekly = df_cons.groupby('developer_id')['week_uniq'].nunique() / span_stats['total_weeks']
        score_daily = df_cons.groupby('developer_id')['date'].nunique() / span_stats['total_days']

        # Final Score (70:30)
        final_score = (0.7 * score_weekly) + (0.3 * score_daily)

        # Map ke Master
        df_master_features['consistency_score'] = df_master_features['user_id'].map(final_score).clip(0, 1).fillna(0)

    # =========================================================
    # FITUR 4: tutorial_revisit_rate
    # =========================================================
    if not df_t.empty:
        cols_rev = ['completed_at', 'last_viewed']
        # Filter Completed & Valid Dates
        if 'status' in df_t.columns:
            is_done_rev = df_t['status'].astype(str).str.contains('completed|passed|1', case=False, regex=True)
            df_rev = df_t[is_done_rev & df_t[cols_rev].notna().all(axis=1)].copy()
        else:
            df_rev = df_t[df_t[cols_rev].notna().all(axis=1)].copy()

        if not df_rev.empty:
            threshold = df_rev['completed_at'] + pd.Timedelta(minutes=10)
            df_rev['is_revisited'] = (df_rev['last_viewed'] > threshold).astype(int)

            revisit_rate = df_rev.groupby('developer_id')['is_revisited'].mean()
            df_master_features['tutorial_revisit_rate'] = df_master_features['user_id'].map(revisit_rate).fillna(0)

    # =========================================================
    # FITUR 5: avg_tutorial_duration
    # =========================================================
    if not df_t.empty:
        # Butuh first_opened_at & completed_at
        df_dur = df_t.dropna(subset=['first_opened_at', 'completed_at']).copy()

        if not df_dur.empty:
            durations = (df_dur['completed_at'] - df_dur['first_opened_at']).dt.total_seconds() / 60
            # Filter 0 < durasi <= 30
            valid_durations = durations[(durations > 0) & (durations <= 30)]

            # Map rata-rata
            avg_dur_per_user = valid_durations.groupby(df_dur.loc[valid_durations.index, 'developer_id']).mean()
            df_master_features['avg_tutorial_duration'] = df_master_features['user_id'].map(avg_dur_per_user).fillna(0)

    # --- FINAL RETURN ---
    # Set Index user_id
    df_master_features = df_master_features.set_index('user_id')

    # Return 5 fitur UTAMA saja untuk model (sisanya kayak active_days cuma pembantu)
    final_cols = [
        'avg_weighted_exam_score',
        'completion_density',
        'consistency_score',
        'tutorial_revisit_rate',
        'avg_tutorial_duration'
    ]
    return df_master_features[final_cols]
