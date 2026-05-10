# Faculty Supervisor Recommender — Skripsi Rizki Afdolli

> Sistem rekomendasi dosen pembimbing magang berbasis **semantic similarity** (ModernBERT) + greedy capacity assignment.
> Binus University — Jurusan Teknik Informatika — Bandung Campus.

---

## Deskripsi Singkat

Sistem ini memetakan mahasiswa magang ke dosen pembimbing menggunakan pipeline 10 tahap. Inti algoritmanya adalah:

1. Bangun dokumen teks mahasiswa dan profil dosen
2. Encode menjadi dense embedding (ModernBERT)
3. Hitung cosine similarity matrix (N mahasiswa × M dosen)
4. Terapkan booster opsional (rule-based + company cohort)
5. Selesaikan assignment dengan greedy solver ber-constraint kapasitas

Semua logika berjalan di atas **Flask + SQLite** (single-process, file-based). Tidak ada API terpisah.

---

## 10-Stage Pipeline

```
[STAGE 1]  Excel Import        → Student DB         app/excel_io.py
[STAGE 2]  Document Building   → Text Docs          app/rules.py + datasets/map_loader.py
[STAGE 3]  Embedding           → Similarity Matrix  app/embedding.py
[STAGE 4]  Rule Boost Matrix   (toggle: ENABLE_RULE_BOOST)   app/rules.py
[STAGE 5]  Company Group Bonus (toggle: ENABLE_GROUP_BONUS)  app/recommender.py
[STAGE 6]  Score Matrix Assembly                    app/recommender.py
[STAGE 7]  Capacity Planning   (min=10, max=12)     app/recommender.py
[STAGE 8]  Assignment Solver   (Greedy)             app/recommender.py
[STAGE 9]  Evaluation          (MRR, nDCG, Hit@k)  app/evaluation.py
[STAGE 10] DB Storage → Output (SQLite + Excel)     app/database.py + app/services.py
```

---

## Tech Stack (Actual)

| Layer | Implementasi |
|-------|-------------|
| Web Framework | Flask 3.x |
| Database | SQLite (`recommendation.db`) via SQLAlchemy 2.x ORM |
| Embedding (primary) | `answerdotai/ModernBERT-base` via `sentence-transformers` |
| Embedding (fallback 1) | `sentence-transformers/all-mpnet-base-v2` |
| Embedding (fallback 2) | TF-IDF (scikit-learn) |
| Embedding (fallback 3) | Token Overlap / Jaccard |
| Assignment Solver | Greedy (in-house, `app/recommender.py`) |
| Export | Excel via openpyxl + pandas |
| Containerisasi | Docker + docker-compose |

---

## Struktur File Utama

```
flask_app.py              — Entry point web UI + semua route
app/
  config.py               — Semua env var dan konstanta
  models.py               — SQLAlchemy ORM: Supervisor, Student, RecommendationRun, Recommendation, ...
  database.py             — Engine + SessionLocal
  queries.py              — Semua raw ORM queries (dipisah dari business logic)
  services.py             — Business logic + orchestration
  recommender.py          — Pipeline utama: score assembly → capacity plan → greedy solver
  embedding.py            — EmbeddingProvider (3 backend) + LabelEmbeddingCache
  rules.py                — Document building, label detection, rule boost evaluation
  evaluation.py           — Retrieval metrics: MRR, Hit@k, nDCG@k, Precision@k
  excel_io.py             — Excel import (read_students_from_excel_*)
datasets/
  map_loader.py           — Load extra supervisor docs dari map_2026.xlsx
templates/                — Jinja2 templates (Flask)
static/style.css          — CSS
```

---

## Web UI Routes

| Path | Fungsi |
|------|--------|
| `/dashboard` | KPI summary, BI trend chart, run snapshot, pipeline config badges |
| `/data` | Import Excel mahasiswa, trigger generate run |
| `/runs` | Histori semua run |
| `/runs/<id>` | Detail run: evaluasi, kapasitas, pipeline config, mismatch spotlight |
| `/runs/<id>/recommendations` | Tabel rekomendasi per mahasiswa (filter, search) |
| `/runs/<id>/export` | Download Excel hasil run |
| `/supervisors` | Kelola profil dosen (keywords, kategori) |
| `/rules` | Rules Studio: label descriptions + affinity matrix |
| `/benchmark` | Benchmark model embedding |

---

## Cara Menjalankan Lokal

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python flask_app.py
# Buka: http://127.0.0.1:5001
```

Pertama kali buka: register akun baru, lalu login.

### Docker

```bash
cp .env.example .env               # sesuaikan jika perlu
docker compose up --build
# Web: http://localhost:5001
```

---

## Konfigurasi (Environment Variables)

### Pipeline Toggles

| Variable | Default | Efek |
|----------|---------|------|
| `ENABLE_RULE_BOOST` | `false` | Aktifkan rule boost matrix (Stage 4) — label/affinity scoring |
| `ENABLE_GROUP_BONUS` | `false` | Aktifkan company cohort bonus (Stage 5) |
| `ENABLE_EXTRA_DOCS` | `false` | Tambahkan teks historis `map_2026.xlsx` ke supervisor docs (Stage 2) |

Semua default `false` → mode **pure embedding similarity** tanpa rule layer.

### Kapasitas & Scoring

| Variable | Default | Keterangan |
|----------|---------|-----------|
| `TARGET_MIN_CAPACITY` | `10` | Minimum mahasiswa per dosen |
| `TARGET_MAX_CAPACITY` | `12` | Maximum mahasiswa per dosen |
| `SIMILARITY_WEIGHT` | `1.0` | Pengali cosine similarity di score matrix |
| `COMPANY_GROUP_BONUS` | `1.5` | Bonus nilai jika mahasiswa satu perusahaan (aktif bila ENABLE_GROUP_BONUS=true) |

### Embedding

| Variable | Default |
|----------|---------|
| `EMBEDDING_MODEL_NAME` | `answerdotai/ModernBERT-base` |
| `EMBEDDING_FALLBACK_MODEL_NAME` | `sentence-transformers/all-mpnet-base-v2` |
| `EMBEDDING_DEVICE` | `auto` (gunakan `cuda` untuk GPU) |

### Database

| Variable | Default |
|----------|---------|
| `DATABASE_URL` | `sqlite:///recommendation.db` |
| `DEFAULT_EXCEL_PATH` | `map_2026.xlsx` |

---

## Algoritma: Detail Teknis

### Document Building (Stage 2)

**Student doc:**
```
{track} {partner_lecturer} {position_topic} {work_schema}
```

**Supervisor doc:**
```
{name} {code} {keywords} {labels}
[+ extra text dari map_2026.xlsx jika ENABLE_EXTRA_DOCS=true]
```
Normalisasi: lowercase → hapus non-alphanumeric → collapse spaces.

### Embedding + Similarity (Stage 3)

```python
# N+M dokumen di-encode sekaligus (batch)
all_vecs = model.encode(student_docs + supervisor_docs, normalize_embeddings=True)
student_vecs    = all_vecs[:N]   # shape (N, D)
supervisor_vecs = all_vecs[N:]   # shape (M, D)
similarity = np.matmul(student_vecs, supervisor_vecs.T)  # shape (N, M)
```

### Greedy Solver (Stage 8)

1. **Phase 0** — `np.argmax(axis=1)`: setiap mahasiswa pilih dosen dengan skor tertinggi (unconstrained)
2. **Phase 1** — Overflow repair: pindahkan mahasiswa dari dosen overcap ke dosen yang belum penuh, dengan penalty skor terkecil
3. **Phase 2** — Underflow repair: tarik mahasiswa dari donor (> min_cap) ke dosen yang underfill

Greedy dipilih atas ILP karena skala sistem (~100–200 mahasiswa, 14 dosen) kecil sehingga hasil hampir-optimal tanpa dependency eksternal.

### Evaluasi (Stage 9)

Ground truth: kolom `current_supervisor_code` (mapping EPC aktual).

Metrik dihitung untuk **3 sumber skor**:
- `content_based` — raw cosine similarity matrix
- `hybrid_score` — score matrix setelah rule boost + group bonus
- `assignment_match` — assignment final greedy vs ground truth

| Metrik | Keterangan |
|--------|-----------|
| MRR | Mean Reciprocal Rank |
| Hit@1, Hit@5 | Apakah true supervisor masuk top-1 / top-5 |
| nDCG@5, nDCG@10 | Normalized Discounted Cumulative Gain |
| Avg Rank | Rata-rata rank true supervisor |
| Precision@5 | Proporsi top-5 yang relevan |

---

## Supervisor Priority Codes

Dosen berikut diprioritaskan pertama saat relaksasi kapasitas (overflow/underflow):

```python
CAPACITY_PRIORITY_CODES = [
    "D2211",  # Dr. Abdul Haris Rangkuti
    "D6184",  # Dr. Mochammad Haldi Widianto
    "D6826",  # Karen Etania Saputra
    "D1749",  # Dr. Johan Muliadi Kerta
]
```

---

## ⚠️ Thesis–Code Discrepancies

Ketidaksesuaian antara dokumen thesis (Notion) dan implementasi aktual.
Detail lengkap di `wiki/thesis/known-issues.md`.

### Discrepancy A — NLP Pre-processing

| | |
|---|---|
| **Thesis says** | Pre-proses NLP: stop word removal, case folding, normalization, Sastrawi stemming |
| **Code does** | Hanya: lowercase → hapus karakter non-alphanumeric → collapse whitespace (`normalize_text()` di `app/rules.py`) |
| **Location** | BAB 1 Metode Penelitian, BAB 3 |
| **Status** | Belum direkonsiliasi |

### Discrepancy B — Embedding Model Candidates

| | |
|---|---|
| **Thesis says** | Kandidat model: Qwen3 / EmbeddingGemma / Jina |
| **Code does** | Primary: `answerdotai/ModernBERT-base`, fallback bertingkat: `all-mpnet-base-v2` → TF-IDF → Jaccard |
| **Location** | BAB 2 Landasan Teori, BAB 3 |
| **Resolution path** | Gunawan (2025) membuktikan ModernBERT terbaik (MRR 0.848, Hit@5=1.0) di task serupa. Rekomendasi: update thesis ke ModernBERT + cite Gunawan |

### Discrepancy C — Database Engine

| | |
|---|---|
| **Thesis says** | PostgreSQL + pgvector (vector database) |
| **Code does** | SQLite (`recommendation.db`) — plain relational, tidak ada vector store |
| **Location** | BAB 2 Tech Stack, ERD, Deployment diagram |
| **Status** | Untuk prototype SQLite OK — tambahkan footnote justifikasi di thesis |

### Discrepancy D — Supervisor Document Source

| | |
|---|---|
| **Thesis says** | Dokumen FS dari: data historis mahasiswa + topik skripsi rekomendasi |
| **Code does** | Dokumen FS dari: `name + code + keywords + labels` (static profile) + optional extra text dari `map_2026.xlsx` |
| **Location** | BAB 3 Pembentukan Representasi FS |
| **Status** | Belum direkonsiliasi |

### Discrepancy E — Pure Semantic vs Hybrid *(Resolved)*

| | |
|---|---|
| **Thesis says** | "Sistem berbasis semantic similarity berbasis data tekstual saja" |
| **Code does** | Semantic similarity backbone + rule boost (Stage 4) + company group bonus (Stage 5) |
| **Resolution** | Resolved by design (2026-04-26): rule layer kini menggunakan semantic label detection (cosine sim antara student embedding dan label description embedding). Sistem dideskripsikan sebagai **fully-semantic hybrid**. Declare hybrid design eksplisit di BAB 3. |

---

## Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Run History
![Run History](screenshots/run-history.png)

### Supervisor Studio
![Supervisor Studio](screenshots/supervisor-studio.png)
