import pandas as pd
import numpy as np
from datetime import timedelta

def perform_feature_engineering_final(dfs):
    # --- 0. PREPARATION ---
    required_tables = ['users', 'exam_results', 'exam_registrations', 'submissions', 'trackings', 'completions', 'journeys', 'tutorials']
    for tbl in required_tables:
        if tbl not in dfs: dfs[tbl] = pd.DataFrame()

    # Helper DateTime
    def to_dt(df, cols):
        for c in cols:
            if c in df.columns: df[c] = pd.to_datetime(df[c], errors='coerce')
        return df

    if not dfs['exam_registrations'].empty:
        dfs['exam_registrations'] = to_dt(dfs['exam_registrations'], ['created_at', 'deadline_at', 'exam_finished_at'])
    if not dfs['submissions'].empty:
        dfs['submissions'] = to_dt(dfs['submissions'], ['created_at', 'ended_review_at'])
    if not dfs['trackings'].empty:
        dfs['trackings'] = to_dt(dfs['trackings'], ['last_viewed', 'first_opened_at', 'completed_at'])

    # --- 1. BASE DATAFRAME (Sesuai TXT: Basis Users) ---
    if 'users' in dfs and not dfs['users'].empty:
        df_master_features = dfs['users'][['id', 'created_at']].rename(columns={'id': 'developer_id'})
    else:
        # Fallback
        all_ids = pd.concat([dfs['trackings']['developer_id'], dfs['submissions']['submitter_id']]).unique()
        df_master_features = pd.DataFrame({'developer_id': all_ids})

    # --- 2. avg_weighted_exam_score (Sesuai TXT) ---
    if not dfs['exam_results'].empty and not dfs['exam_registrations'].empty:
        df_res = dfs['exam_results'].copy()
        df_reg = dfs['exam_registrations'].copy()

        # Merge results -> registrations
        exam_merged = df_res.merge(
            df_reg[['id', 'examinees_id']],
            left_on='exam_registration_id',
            right_on='id',
            how='inner'
        )

        # Hitung weighted score per baris
        exam_merged['weighted_score'] = exam_merged['score'] * exam_merged['total_questions']

        # Agregasi per user
        user_exam_stats = exam_merged.groupby('examinees_id').agg(
            sum_weighted=('weighted_score', 'sum'),
            sum_questions=('total_questions', 'sum')
        )

        # Hitung rata-rata tertimbang
        user_exam_stats['avg_weighted_exam_score'] = (
            user_exam_stats['sum_weighted'] / user_exam_stats['sum_questions'].replace(0, 1)
        )

        # Merge ke Master
        df_master_features = df_master_features.merge(
            user_exam_stats[['avg_weighted_exam_score']],
            left_on='developer_id',
            right_index=True,
            how='left'
        )
    else:
        df_master_features['avg_weighted_exam_score'] = 0

    df_master_features['avg_weighted_exam_score'] = df_master_features['avg_weighted_exam_score'].fillna(0)

    # --- 3. exam_duration_utilization_ratio (Sesuai TXT) ---
    if not dfs['exam_registrations'].empty:
        df_reg = dfs['exam_registrations'].copy()

        # Filter finished & deadline > created
        mask_valid = df_reg['exam_finished_at'].notna() & (df_reg['deadline_at'] > df_reg['created_at'])
        df_exam_valid = df_reg[mask_valid].copy()

        if not df_exam_valid.empty:
            df_exam_valid['max_seconds'] = (df_exam_valid['deadline_at'] - df_exam_valid['created_at']).dt.total_seconds()
            df_exam_valid['used_seconds'] = (df_exam_valid['exam_finished_at'] - df_exam_valid['created_at']).dt.total_seconds()

            # Ratio & Cap
            df_exam_valid['utilization_ratio'] = (df_exam_valid['used_seconds'] / df_exam_valid['max_seconds'].replace(0, 1)).clip(lower=0.0, upper=1.0)

            user_exam_util = df_exam_valid.groupby('examinees_id')['utilization_ratio'].mean().rename('exam_duration_utilization_ratio')

            df_master_features = df_master_features.merge(
                user_exam_util,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['exam_duration_utilization_ratio'] = 0.0
    else:
        df_master_features['exam_duration_utilization_ratio'] = 0.0

    df_master_features['exam_duration_utilization_ratio'] = df_master_features['exam_duration_utilization_ratio'].fillna(0.0)

    # --- 4. avg_submission_revision_count (Sesuai TXT) ---
    if not dfs['submissions'].empty:
        df_sub = dfs['submissions'].copy()
        # Filter status != -2
        df_sub_valid = df_sub[df_sub['status'] != -2].copy()

        if not df_sub_valid.empty:
            # Hitung revisi (status == -1)
            df_sub_valid['is_revision'] = (df_sub_valid['status'] == -1).astype(int)

            project_revisions = df_sub_valid.groupby(['submitter_id', 'quiz_id']).agg(
                total_revisions=('is_revision', 'sum')
            ).reset_index()

            user_rev_count = project_revisions.groupby('submitter_id')['total_revisions'].mean().rename('avg_submission_revision_count')

            df_master_features = df_master_features.merge(
                user_rev_count,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['avg_submission_revision_count'] = 0.0
    else:
        df_master_features['avg_submission_revision_count'] = 0.0

    df_master_features['avg_submission_revision_count'] = df_master_features['avg_submission_revision_count'].fillna(0.0)

    # --- 5. avg_submission_revision_duration (Sesuai TXT) ---
    if not dfs['submissions'].empty:
        df_sub = dfs['submissions'].copy()
        df_sub_valid = df_sub[df_sub['status'] != -2].copy()

        if not df_sub_valid.empty:
            df_sub_valid = df_sub_valid.sort_values(['submitter_id', 'quiz_id', 'id'])
            # Shift ambil waktu sebelumnya
            df_sub_valid['prev_ended_review_at'] = df_sub_valid.groupby(['submitter_id', 'quiz_id'])['ended_review_at'].shift(1)

            # Hitung selisih jam
            df_sub_valid['revision_duration_hours'] = (
                df_sub_valid['created_at'] - df_sub_valid['prev_ended_review_at']
            ).dt.total_seconds() / 3600

            # Validasi: > 0 dan <= 720 jam
            mask_valid_rev = (df_sub_valid['revision_duration_hours'] > 0) & (df_sub_valid['revision_duration_hours'] <= 720)
            df_rev_clean = df_sub_valid[mask_valid_rev]

            user_rev_duration = df_rev_clean.groupby('submitter_id')['revision_duration_hours'].mean().rename('avg_submission_revision_duration')

            df_master_features = df_master_features.merge(
                user_rev_duration,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['avg_submission_revision_duration'] = 0.0
    else:
        df_master_features['avg_submission_revision_duration'] = 0.0

    df_master_features['avg_submission_revision_duration'] = df_master_features['avg_submission_revision_duration'].fillna(0.0)

    # --- 6. avg_submission_rating (Sesuai TXT) ---
    if not dfs['submissions'].empty:
        df_sub = dfs['submissions'].copy()
        df_sub_valid = df_sub[df_sub['rating'].notna()].copy()

        if not df_sub_valid.empty:
            user_rating = df_sub_valid.groupby('submitter_id')['rating'].mean().rename('avg_submission_rating')
            df_master_features = df_master_features.merge(
                user_rating,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['avg_submission_rating'] = 0.0
    else:
        df_master_features['avg_submission_rating'] = 0.0

    df_master_features['avg_submission_rating'] = df_master_features['avg_submission_rating'].fillna(0.0)

    # --- 7. active_days (Sesuai TXT: Strict Owl 2 AM) ---
    if not dfs['trackings'].empty:
        df_t = dfs['trackings'].copy()

        # Filter Completed Only
        if 'status' in df_t.columns:
            is_done = df_t['status'].astype(str).str.contains('completed|passed|1', case=False, regex=True)
            df_done = df_t[is_done].copy()
        else:
            df_done = df_t[df_t['completed_at'].notna()].copy()

        if not df_done.empty:
            # Owl Adjustment (2 Jam)
            df_done['adjusted_time'] = df_done['completed_at'] - pd.Timedelta(hours=2)
            df_done['learning_date'] = df_done['adjusted_time'].dt.date

            user_active_days = df_done.groupby('developer_id')['learning_date'].nunique().rename('active_days')

            df_master_features = df_master_features.merge(
                user_active_days,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['active_days'] = 0
    else:
        df_master_features['active_days'] = 0

    df_master_features['active_days'] = df_master_features['active_days'].fillna(0)

    # --- 8. completion_density (Sesuai TXT: Total / Active) ---
    if not dfs['trackings'].empty:
        df_t = dfs['trackings'].copy()

        if 'status' in df_t.columns:
            is_done = df_t['status'].astype(str).str.contains('completed|passed|1', case=False, regex=True)
            df_done = df_t[is_done].copy()
        else:
            df_done = df_t[df_t['completed_at'].notna()].copy()

        if not df_done.empty:
            # Numerator: Total Tutorial Selesai
            user_total_completed = df_done.groupby('developer_id')['tutorial_id'].nunique().rename('total_completed_tutorials')

            df_master_features = df_master_features.merge(
                user_total_completed,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
            df_master_features['total_completed_tutorials'] = df_master_features['total_completed_tutorials'].fillna(0)

            # Density = Total / Active Days
            df_master_features['completion_density'] = (
                df_master_features['total_completed_tutorials'] /
                df_master_features['active_days'].replace(0, 1)
            )
        else:
            df_master_features['completion_density'] = 0.0
    else:
        df_master_features['completion_density'] = 0.0

    df_master_features['completion_density'] = df_master_features['completion_density'].fillna(0.0)

    # --- 9. consistency_score (Sesuai TXT: Weighted 70:30) ---
    if not dfs['trackings'].empty:
        df_t = dfs['trackings'].copy()

        if 'status' in df_t.columns:
            is_done = df_t['status'].astype(str).str.contains('completed|passed|1', case=False, regex=True)
            df_done = df_t[is_done & df_t['completed_at'].notna()].copy()
        else:
            df_done = df_t[df_t['completed_at'].notna()].copy()

        if not df_done.empty:
            df_done['adjusted_time'] = df_done['completed_at'] - pd.Timedelta(hours=2)
            df_done['learning_date_dt'] = df_done['adjusted_time'].dt.normalize()

            # Span & Total Weeks
            user_span = df_done.groupby('developer_id')['adjusted_time'].agg(['min', 'max'])
            user_span['total_days'] = (user_span['max'] - user_span['min']).dt.days + 1
            user_span['total_weeks'] = np.ceil(user_span['total_days'] / 7)

            # Daily Count
            daily_counts = df_done.groupby('developer_id')['learning_date_dt'].nunique().rename('active_days_count')

            # Weekly Count
            df_done['week_id'] = df_done['adjusted_time'].dt.strftime('%Y-%U')
            weekly_counts = df_done.groupby('developer_id')['week_id'].nunique().rename('active_weeks')

            stats = user_span.join(daily_counts).join(weekly_counts)

            stats['score_daily'] = stats['active_days_count'] / stats['total_days']
            stats['score_weekly'] = stats['active_weeks'] / stats['total_weeks']

            stats['consistency_score'] = (0.7 * stats['score_weekly']) + (0.3 * stats['score_daily'])
            stats['consistency_score'] = stats['consistency_score'].clip(upper=1.0)

            df_master_features = df_master_features.merge(
                stats[['consistency_score']],
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['consistency_score'] = 0.0
    else:
        df_master_features['consistency_score'] = 0.0

    df_master_features['consistency_score'] = df_master_features['consistency_score'].fillna(0.0)

    # --- 10. tutorial_revisit_rate (Sesuai TXT) ---
    if not dfs['trackings'].empty:
        df_t = dfs['trackings'].copy()
        # Filter valid dates & completed
        df_t = df_t.dropna(subset=['first_opened_at', 'completed_at', 'last_viewed'])

        if 'status' in df_t.columns:
            is_done = df_t['status'].astype(str).str.contains('completed|passed|1', case=False, regex=True)
            df_done = df_t[is_done].copy()
        else:
            df_done = df_t.copy()

        if not df_done.empty:
            buffer_time = pd.Timedelta(minutes=10)
            df_done['is_revisited'] = (df_done['last_viewed'] > (df_done['completed_at'] + buffer_time)).astype(int)

            user_revisit_rate = df_done.groupby('developer_id')['is_revisited'].mean().rename('tutorial_revisit_rate')

            df_master_features = df_master_features.merge(
                user_revisit_rate,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['tutorial_revisit_rate'] = 0.0
    else:
        df_master_features['tutorial_revisit_rate'] = 0.0

    df_master_features['tutorial_revisit_rate'] = df_master_features['tutorial_revisit_rate'].fillna(0.0)

    # --- 11. avg_tutorial_duration (Sesuai TXT: Idle Filter) ---
    if not dfs['trackings'].empty:
        df_t = dfs['trackings'].copy()
        df_t = df_t.dropna(subset=['first_opened_at', 'completed_at'])

        if not df_t.empty:
            df_t['duration_minutes'] = (df_t['completed_at'] - df_t['first_opened_at']).dt.total_seconds() / 60

            # Filter Idle: > 0 dan <= 30 menit
            mask_valid_duration = (df_t['duration_minutes'] > 0) & (df_t['duration_minutes'] <= 30)
            df_clean_duration = df_t[mask_valid_duration].copy()

            user_avg_duration = df_clean_duration.groupby('developer_id')['duration_minutes'].mean().rename('avg_tutorial_duration')

            df_master_features = df_master_features.merge(
                user_avg_duration,
                left_on='developer_id',
                right_index=True,
                how='left'
            )
        else:
            df_master_features['avg_tutorial_duration'] = 0.0
    else:
        df_master_features['avg_tutorial_duration'] = 0.0

    df_master_features['avg_tutorial_duration'] = df_master_features['avg_tutorial_duration'].fillna(0.0)

    # Final Set Index
    df_master_features = df_master_features.set_index('developer_id')
    return df_master_features
