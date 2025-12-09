# AI Learning Insight API

Dokumentasi penggunaan API untuk integrasi Machine Learning Learning Insight.

## 1\. Integrasi API (Untuk Backend)

API ini digunakan untuk memprediksi kategori belajar user dan memberikan insight berdasarkan data aktivitas mereka.

  * **Base URL:** `https://ai-learning-insight-ai-learning-insight.up.railway.app`
  * **Documentation & Test:** `/docs` (Swagger UI)

### Endpoint Prediksi

  * **URL:** `/predict`
  * **Method:** `POST`
  * **Content-Type:** `application/json`

#### Cara Penggunaan

Backend melakukan query data user terkait (raw data) dari database, lalu mengirimkannya sebagai JSON body ke endpoint ini. Pastikan format tanggal dikirim sebagai String (ISO 8601 atau `YYYY-MM-DD HH:MM:SS`).

**Format Request Body (Input Lengkap):**

```json
{
  "users": [
    {
      "id": 101,
      "created_at": "2023-01-01 08:00:00"
    }
  ],
  "trackings": [
    {
      "id": 5001,
      "developer_id": 101,
      "journey_id": 10,
      "tutorial_id": 55,
      "status": "completed",
      "last_viewed": "2023-01-05 14:30:00",
      "first_opened_at": "2023-01-05 13:00:00",
      "completed_at": "2023-01-05 14:30:00"
    },
    {
      "id": 5002,
      "developer_id": 101,
      "journey_id": 10,
      "tutorial_id": 56,
      "status": "started",
      "last_viewed": "2023-01-06 09:00:00",
      "first_opened_at": "2023-01-06 08:30:00",
      "completed_at": null
    }
  ],
  "submissions": [
    {
      "id": 8001,
      "submitter_id": 101,
      "quiz_id": 200,
      "status": 3,
      "created_at": "2023-01-07 10:00:00",
      "ended_review_at": "2023-01-07 15:00:00",
      "rating": 5
    }
  ],
  "completions": [
    {
      "id": 3001,
      "user_id": 101,
      "journey_id": 10,
      "created_at": "2023-01-10 12:00:00",
      "updated_at": "2023-01-10 12:00:00",
      "enrollments_at": "['2023-01-01 08:00:00']",
      "last_enrolled_at": "2023-01-01 08:00:00",
      "study_duration": 36000
    }
  ],
  "journeys": [
    {
      "id": 10,
      "hours_to_study": 20
    }
  ],
  "tutorials": [
    {
      "id": 55,
      "developer_journey_id": 10
    },
    {
      "id": 56,
      "developer_journey_id": 10
    }
  ],
  "exam_registrations": [
    {
      "id": 9001,
      "examinees_id": 101,
      "tutorial_id": 99,
      "created_at": "2023-01-15 09:00:00",
      "deadline_at": "2023-01-15 11:00:00",
      "exam_finished_at": "2023-01-15 10:30:00"
    }
  ],
  "exam_results": [
    {
      "id": 9501,
      "exam_registration_id": 9001,
      "score": 85,
      "total_questions": 50
    }
  ]
}
```

**Format Response (Output Lengkap):**

```json
{
  "user_id": 101,
  "category": "Consistent Learner",
  "insight_message": "Kerja bagus! Kamu rutin menyelesaikan materi secara berkala. Terus jaga ritme belajar ini ya.",
  "metrics": {
    "avg_weighted_exam_score": 85,
    "exam_duration_utilization_ratio": 0.75,
    "avg_submission_revision_count": 0,
    "avg_submission_revision_duration": 0,
    "avg_submission_rating": 5,
    "completion_density": 1,
    "consistency_score": 1,
    "tutorial_revisit_rate": 0,
    "avg_tutorial_duration": 0
  }
}
```
