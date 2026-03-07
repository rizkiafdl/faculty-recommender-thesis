# faculty-recommender

Sistem Rekomendasi Dosen Pembimbing

Sistem ini membuat mapping mahasiswa ke dosen pembimbing dengan pendekatan hybrid:

- content-based berbasis `transformer embedding` + `cosine similarity` (gaya skripsi),
- rule-based booster sesuai aturan dosen,
- multilabel supervisor profiling (static + adaptive dari histori mapping),
- login/register berbasis web session untuk akses dashboard,
- manajemen kategori dosen (track/industry) langsung dari web,
- capacity balancing (target 10-12 mahasiswa per dosen),
- grouping perusahaan yang sama,
- API (`FastAPI`), web (`Flask`), database (`SQLite`), dan export Excel.

## Algoritma Content-Based (Transformer)

Pipeline content-based:

1. Gabungkan fitur teks mahasiswa (`track`, `partner/lecturer`, `position/topic`, `work schema`).
2. Bentuk dokumen profil dosen dari nama + kata kunci keahlian.
3. Encode dokumen dengan model transformer (`ModernBERT` default) menjadi embedding.
4. Hitung `cosine similarity` antar embedding mahasiswa-dosen.
5. Lanjutkan dengan rule boosting + optimasi assignment kuota.

Catatan multilabel fleksibel:

- tiap dosen punya banyak label keahlian (`research`, `finance_banking`, `apple_mobile`, dll.),
- profil dosen diperkaya otomatis dari histori `current_supervisor_code` (adaptive keywords + adaptive labels),
- rule boosting menghitung overlap label mahasiswa-dosen + bonus fleksibilitas dosen lintas topik.
- label `binus_internal_internship` dipakai untuk kasus magang internal BINUS, dengan lebih banyak dosen eligible.
- continuity-aware scoring: jika mahasiswa sudah punya `current_supervisor_code`, sistem memprioritaskan stabilitas assignment.
- mahasiswa tanpa label existing diarahkan ke dosen prioritas overflow agar tidak menggeser mapping yang sudah stabil.
- niche hard-constraint:
  - konteks `government/public sector` diprioritaskan keras ke `D6407` (Dr. Dany Eka Saputra),
  - konteks `hospital/rumah sakit` diprioritaskan keras ke `D6274` (Dr. Husni Iskandar Pohan).

Konfigurasi model:

- `EMBEDDING_MODEL_NAME` (default: `answerdotai/ModernBERT-base`)
- `EMBEDDING_FALLBACK_MODEL_NAME` (default: `sentence-transformers/all-mpnet-base-v2`)
- `EMBEDDING_DEVICE` (default: `auto`, set `cuda` / `cuda:0` untuk RTX 3070 Ti)
- `BENCHMARK_MODEL_NAMES` (default: `answerdotai/ModernBERT-base,sentence-transformers/all-mpnet-base-v2,sentence-transformers/all-MiniLM-L6-v2`)
- `CURRENT_SUPERVISOR_CONTINUITY_WEIGHT` (default: `12.0`, set `0` untuk mode cold-start murni)
- `LABELED_STUDENT_PRIORITY_WEIGHT` (default: `2.0`, bobot prioritas saat optimasi assignment untuk data berlabel)
- `CONTEXT_PRIOR_WEIGHT` (default: `0.0`, aktifkan > `0` jika ingin pakai historical context prior)

Jika model transformer gagal dimuat, sistem fallback ke TF-IDF agar proses tetap berjalan.

## Evaluasi (Ala Skripsi)

Setiap run menyimpan metrik evaluasi retrieval menggunakan label referensi `current_supervisor_code`:

- `MRR`
- `Hit@1`, `Hit@5`
- `NDCG@5`, `NDCG@10`
- `Average Rank`
- `Avg Similarity@1`
- `Precision@5`

Metrik tersedia untuk:

- `content_based` (skor transformer),
- `hybrid_score` (setelah rule + company bonus),
- `assignment_match` (match rate rekomendasi final vs mapping saat ini).

Tambahan benchmark model (ala skripsi):

- endpoint benchmark akan membandingkan beberapa model transformer di data yang sama,
- menghasilkan leaderboard `MRR`, `Hit@k`, `NDCG@k`, `Average Rank`, dan `avg_true_similarity`,
- memilih model terbaik berdasarkan `MRR`.

## Struktur Utama

- `fastapi_app.py`: API backend.
- `flask_app.py`: web UI.
- `app/recommender.py`: engine rekomendasi + optimasi kuota.
- `app/embedding.py`: transformer embedding provider.
- `app/evaluation.py`: metrik evaluasi.
- `app/services.py`: import/generate/list/summary/evaluation/export.

## Menjalankan Lokal

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn fastapi_app:app --reload --port 8000
```

Web Flask:

```bash
python flask_app.py
```

Buka:

- API: `http://127.0.0.1:8000/docs`
- Web: `http://127.0.0.1:5000`

Autentikasi web:

- pertama kali buka web, lakukan `register` akun baru,
- setelah login, UI sekarang terpisah multi-halaman:
  - `/dashboard`: KPI ringkas + latest run,
  - `/data`: import data + trigger generate run,
  - `/runs`: histori run + glossary istilah evaluasi/solver (termasuk PuLP),
  - `/runs/<run_id>`: detail evaluasi + kuota + mismatch spotlight,
  - `/runs/<run_id>/recommendations`: tabel rekomendasi dengan filter,
  - `/supervisors`: kelola kategori/keywords dosen.
- manajemen dosen mendukung:
  - dropdown searchable (ketik lalu pilih),
  - tambah dosen baru langsung dari web,
  - keyword studio berbasis chip (input satu-per-satu, warna-warni),
  - export konfigurasi ke Excel (`/supervisors/export`) untuk update train/review tahunan.

## Endpoint API

- `POST /api/import/default`
- `POST /api/import/upload`
- `POST /api/recommend/run`
- `GET /api/runs/latest`
- `GET /api/recommendations`
- `GET /api/summary`
- `GET /api/evaluation`
- `GET /api/evaluation/benchmark-models`
- `GET /api/export`

Catatan: fitur login/register dan manajemen kategori dosen saat ini tersedia di web Flask.

Contoh benchmark:

```bash
curl "http://127.0.0.1:8000/api/evaluation/benchmark-models"
```

Custom model list:

```bash
curl "http://127.0.0.1:8000/api/evaluation/benchmark-models?models=answerdotai/ModernBERT-base,sentence-transformers/all-mpnet-base-v2"
```

## Docker

Build + run:

```bash
cp .env.example .env
docker compose up --build
```

Service:

- FastAPI: `http://localhost:8000`
- Flask Web: `http://localhost:5000`

Data persist di folder `./data`:

- `recommendation.db`
- cache model transformer (`hf-cache`)

Jika Docker belum aktif, start Docker Desktop dulu sebelum menjalankan `docker compose`.
Jika ingin GPU di Docker, pastikan driver NVIDIA + NVIDIA Container Toolkit tersedia.
Untuk GPU, ubah `EMBEDDING_DEVICE=cuda` di file `.env`.

## Screenshots

Hasil capture UI ada di folder `screenshots/`:

- `screenshots/dashboard.png`
- `screenshots/supervisors.png`
- `screenshots/runs.png`

## Catatan Kuota 10-12

Jika total mahasiswa tidak feasible dengan batas keras 10-12 (contoh 171 mahasiswa, 14 dosen), sistem otomatis melakukan relaksasi kapasitas terkontrol (mis. +1 slot pada dosen prioritas) dan menyimpan catatan relaksasi pada metadata run.
