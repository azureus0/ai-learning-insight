# AI Learning Insight API

Dokumentasi penggunaan API untuk integrasi Machine Learning Learning Insight.

## 1. Integrasi API (Untuk Backend)

API ini digunakan untuk memprediksi kategori belajar user dan memberikan insight personal berdasarkan data aktivitas mentah (Raw Data) dari database.

* **Base URL:** `https://ai-learning-insight-ai-learning-insight.up.railway.app`
* **Documentation & Test:** `/docs` (Swagger UI)

### Endpoint Prediksi

* **URL:** `/predict`
* **Method:** `POST`
* **Content-Type:** `application/json`

#### Cara Penggunaan

Backend melakukan query data user terkait dari database, lalu mengirimkannya sebagai JSON body ke endpoint ini.

⚠️ **PENTING:** Nama key dalam JSON harus **SAMA PERSIS** dengan nama tabel di database.
* ✅ Gunakan: `developer_journey_trackings`
* ❌ Jangan gunakan: `trackings`

---

### Contoh Request & Response (Skenario: Fast Learner)

Berikut adalah contoh data user yang menyelesaikan **30 tutorial** dalam waktu singkat (hanya 2 hari aktif terpisah), dengan nilai ujian rata-rata **68.5**, menghasilkan prediksi kategori **Fast Learner**.

#### Format Request Body (Input)

```json
{
  "users": [
    { "id": 707, "created_at": "2023-01-01 08:00:00" }
  ],
  "developer_journey_trackings": [
    { "id": 8201, "developer_id": 707, "tutorial_id": 301, "status": 1, "first_opened_at": "2023-01-01 10:00:00", "completed_at": "2023-01-01 10:02:00", "last_viewed": "2023-01-01 10:02:00" },
    { "id": 8202, "developer_id": 707, "tutorial_id": 302, "status": 1, "first_opened_at": "2023-04-01 08:00:00", "completed_at": "2023-04-01 08:01:30", "last_viewed": "2023-04-01 08:01:30" },
    { "id": 8203, "developer_id": 707, "tutorial_id": 303, "status": 1, "first_opened_at": "2023-04-01 08:01:30", "completed_at": "2023-04-01 08:03:00", "last_viewed": "2023-04-01 08:03:00" },
    { "id": 8204, "developer_id": 707, "tutorial_id": 304, "status": 1, "first_opened_at": "2023-04-01 08:03:00", "completed_at": "2023-04-01 08:04:30", "last_viewed": "2023-04-01 08:04:30" },
    { "id": 8205, "developer_id": 707, "tutorial_id": 305, "status": 1, "first_opened_at": "2023-04-01 08:04:30", "completed_at": "2023-04-01 08:06:00", "last_viewed": "2023-04-01 08:06:00" },
    { "id": 8206, "developer_id": 707, "tutorial_id": 306, "status": 1, "first_opened_at": "2023-04-01 08:06:00", "completed_at": "2023-04-01 08:07:30", "last_viewed": "2023-04-01 08:07:30" },
    { "id": 8207, "developer_id": 707, "tutorial_id": 307, "status": 1, "first_opened_at": "2023-04-01 08:07:30", "completed_at": "2023-04-01 08:09:00", "last_viewed": "2023-04-01 08:09:00" },
    { "id": 8208, "developer_id": 707, "tutorial_id": 308, "status": 1, "first_opened_at": "2023-04-01 08:09:00", "completed_at": "2023-04-01 08:10:30", "last_viewed": "2023-04-01 08:10:30" },
    { "id": 8209, "developer_id": 707, "tutorial_id": 309, "status": 1, "first_opened_at": "2023-04-01 08:10:30", "completed_at": "2023-04-01 08:12:00", "last_viewed": "2023-04-01 08:12:00" },
    { "id": 8210, "developer_id": 707, "tutorial_id": 310, "status": 1, "first_opened_at": "2023-04-01 08:12:00", "completed_at": "2023-04-01 08:13:30", "last_viewed": "2023-04-01 08:13:30" },
    { "id": 8211, "developer_id": 707, "tutorial_id": 311, "status": 1, "first_opened_at": "2023-04-01 08:13:30", "completed_at": "2023-04-01 08:15:00", "last_viewed": "2023-04-01 08:15:00" },
    { "id": 8212, "developer_id": 707, "tutorial_id": 312, "status": 1, "first_opened_at": "2023-04-01 08:15:00", "completed_at": "2023-04-01 08:16:30", "last_viewed": "2023-04-01 08:16:30" },
    { "id": 8213, "developer_id": 707, "tutorial_id": 313, "status": 1, "first_opened_at": "2023-04-01 08:16:30", "completed_at": "2023-04-01 08:18:00", "last_viewed": "2023-04-01 08:18:00" },
    { "id": 8214, "developer_id": 707, "tutorial_id": 314, "status": 1, "first_opened_at": "2023-04-01 08:18:00", "completed_at": "2023-04-01 08:19:30", "last_viewed": "2023-04-01 08:19:30" },
    { "id": 8215, "developer_id": 707, "tutorial_id": 315, "status": 1, "first_opened_at": "2023-04-01 08:19:30", "completed_at": "2023-04-01 08:21:00", "last_viewed": "2023-04-01 08:21:00" },
    { "id": 8216, "developer_id": 707, "tutorial_id": 316, "status": 1, "first_opened_at": "2023-04-01 08:21:00", "completed_at": "2023-04-01 08:22:30", "last_viewed": "2023-04-01 08:22:30" },
    { "id": 8217, "developer_id": 707, "tutorial_id": 317, "status": 1, "first_opened_at": "2023-04-01 08:22:30", "completed_at": "2023-04-01 08:24:00", "last_viewed": "2023-04-01 08:24:00" },
    { "id": 8218, "developer_id": 707, "tutorial_id": 318, "status": 1, "first_opened_at": "2023-04-01 08:24:00", "completed_at": "2023-04-01 08:25:30", "last_viewed": "2023-04-01 08:25:30" },
    { "id": 8219, "developer_id": 707, "tutorial_id": 319, "status": 1, "first_opened_at": "2023-04-01 08:25:30", "completed_at": "2023-04-01 08:27:00", "last_viewed": "2023-04-01 08:27:00" },
    { "id": 8220, "developer_id": 707, "tutorial_id": 320, "status": 1, "first_opened_at": "2023-04-01 08:27:00", "completed_at": "2023-04-01 08:28:30", "last_viewed": "2023-04-01 08:28:30" },
    { "id": 8221, "developer_id": 707, "tutorial_id": 321, "status": 1, "first_opened_at": "2023-04-01 08:28:30", "completed_at": "2023-04-01 08:30:00", "last_viewed": "2023-04-01 08:30:00" },
    { "id": 8222, "developer_id": 707, "tutorial_id": 322, "status": 1, "first_opened_at": "2023-04-01 08:30:00", "completed_at": "2023-04-01 08:31:30", "last_viewed": "2023-04-01 08:31:30" },
    { "id": 8223, "developer_id": 707, "tutorial_id": 323, "status": 1, "first_opened_at": "2023-04-01 08:31:30", "completed_at": "2023-04-01 08:33:00", "last_viewed": "2023-04-01 08:33:00" },
    { "id": 8224, "developer_id": 707, "tutorial_id": 324, "status": 1, "first_opened_at": "2023-04-01 08:33:00", "completed_at": "2023-04-01 08:34:30", "last_viewed": "2023-04-01 08:34:30" },
    { "id": 8225, "developer_id": 707, "tutorial_id": 325, "status": 1, "first_opened_at": "2023-04-01 08:34:30", "completed_at": "2023-04-01 08:36:00", "last_viewed": "2023-04-01 08:36:00" },
    { "id": 8226, "developer_id": 707, "tutorial_id": 326, "status": 1, "first_opened_at": "2023-04-01 08:36:00", "completed_at": "2023-04-01 08:37:30", "last_viewed": "2023-04-01 08:37:30" },
    { "id": 8227, "developer_id": 707, "tutorial_id": 327, "status": 1, "first_opened_at": "2023-04-01 08:37:30", "completed_at": "2023-04-01 08:39:00", "last_viewed": "2023-04-01 08:39:00" },
    { "id": 8228, "developer_id": 707, "tutorial_id": 328, "status": 1, "first_opened_at": "2023-04-01 08:39:00", "completed_at": "2023-04-01 08:40:30", "last_viewed": "2023-04-01 08:40:30" },
    { "id": 8229, "developer_id": 707, "tutorial_id": 329, "status": 1, "first_opened_at": "2023-04-01 08:40:30", "completed_at": "2023-04-01 08:42:00", "last_viewed": "2023-04-01 08:42:00" },
    { "id": 8230, "developer_id": 707, "tutorial_id": 330, "status": 1, "first_opened_at": "2023-04-01 08:42:00", "completed_at": "2023-04-01 08:43:30", "last_viewed": "2023-04-01 08:43:30" }
  ],
  "exam_registrations": [
    { "id": 9601, "examinees_id": 707, "created_at": "2023-04-02 10:00:00", "deadline_at": "2023-04-02 12:00:00", "exam_finished_at": "2023-04-02 11:00:00" },
    { "id": 9602, "examinees_id": 707, "created_at": "2023-04-02 13:00:00", "deadline_at": "2023-04-02 15:00:00", "exam_finished_at": "2023-04-02 14:00:00" }
  ],
  "exam_results": [
    { "id": 9961, "exam_registration_id": 9601, "score": 68, "total_questions": 50, "is_passed": 1 },
    { "id": 9962, "exam_registration_id": 9602, "score": 69, "total_questions": 50, "is_passed": 1 }
  ]
}
````

#### Format Response Body (Output)

```json
{
  "user_id": 707,
  "category": "Fast Learner",
  "cluster_id": 1,
  "insight_message": "Wow, performa belajarmu bener-bener definisi 'Fast Learner' sejati! Kamu berhasil melibas **15.0 materi** dalam sehari dengan kecepatan luar biasa. Tapi yang lebih gokil, hasil ujianmu tetap cemerlang di angka **68.5**! Salut banget, ini kombinasi speed dan kualitas yang bikin bangga!",
  "metrics": {
    "avg_weighted_exam_score": 68.5,
    "completion_density": 15,
    "consistency_score": 0.1142857142857143,
    "tutorial_revisit_rate": 0,
    "avg_tutorial_duration": 1.5166666666666666
  }
}
```

```
```
