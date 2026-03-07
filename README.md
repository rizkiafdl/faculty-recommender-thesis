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

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Supervisor Studio
![Supervisor Studio](screenshots/supervisors.png)

### Run History
![Run History](screenshots/runs.png)

## Referensi Riset (2022-2026)

Daftar berikut dipilih untuk latar belakang skripsi, metode content-based/embedding, dan evaluasi sistem rekomendasi:

1. Roy, D. & Dutta, M. (2022). *A systematic review and research perspective on recommender systems*. Journal of Big Data, 9, 59. https://doi.org/10.1186/s40537-022-00592-5
2. Zangerle, E. & Bauer, C. (2022). *Evaluating Recommender Systems: Survey and Framework*. ACM Computing Surveys. https://doi.org/10.1145/3556536
3. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. (2022). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*. NAACL 2022. https://aclanthology.org/2022.naacl-main.272/
4. Ni, J., Qu, C., Lu, J., Dai, Z., Abrego, G. H., Ma, J., Zhao, V., Luan, Y., Hall, K., Chang, M.-W., & Yang, Y. (2022). *Large Dual Encoders Are Generalizable Retrievers*. EMNLP 2022. https://aclanthology.org/2022.emnlp-main.669/
5. Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)*. arXiv:2212.03533. https://arxiv.org/abs/2212.03533
6. Su, H., Shi, W., Kasai, J., Wang, Y., Hu, Y., Ostendorf, M., Yih, W.-t., Smith, N. A., Zettlemoyer, L., & Yu, T. (2023). *One Embedder, Any Task: Instruction-Finetuned Text Embeddings*. Findings of ACL 2023. https://aclanthology.org/2023.findings-acl.71/
7. Gao, L., Ma, X., Lin, J., & Callan, J. (2023). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. ACL 2023. https://aclanthology.org/2023.acl-long.99/
8. Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). *MTEB: Massive Text Embedding Benchmark*. EACL 2023. https://aclanthology.org/2023.eacl-main.148/
9. Bauer, C., Zangerle, E., & Said, A. (2024). *Exploring the Landscape of Recommender Systems Evaluation: Practices and Perspectives*. ACM Transactions on Recommender Systems. https://doi.org/10.1145/3629170
10. Yu, X., Zhang, J., & Yu, Z. (2024). *ConFit: Improving Resume-Job Matching using Data Augmentation and Contrastive Learning*. arXiv:2401.16349. https://arxiv.org/abs/2401.16349
11. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.137/
12. Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catanzaro, B., & Ping, W. (2024). *NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models*. arXiv:2405.17428. https://arxiv.org/abs/2405.17428
13. Warner, B., Chaffin, A., Clavie, B., Weller, O., Hallstrom, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., & Poli, I. (2024). *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)*. arXiv:2412.13663. https://arxiv.org/abs/2412.13663
14. Siafis, V., Rangoussi, M., & Psaromiligkos, Y. (2024). *Recommender Systems for Teachers: A Systematic Literature Review of Recent (2011-2023) Research*. Education Sciences, 14(7), 723. https://doi.org/10.3390/educsci14070723
15. Celik Ertugrul, D., & Bitirim, S. (2025). *Job recommender systems: a systematic literature review, applications, open issues, and challenges*. Journal of Big Data, 12, 140. https://doi.org/10.1186/s40537-025-01173-y
16. Yu, X., Xu, R., Xue, C., Zhang, J., Ma, X., & Yu, Z. (2025). *ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding and Runner-Up Hard-Negative Mining*. arXiv:2502.12361. https://arxiv.org/abs/2502.12361
17. Rosenberger, J., Wolfrum, L., Weinzierl, S., Kraus, M., & Zschech, P. (2025). *CareerBERT: Matching resumes to ESCO jobs in a shared embedding space for generic job recommendations*. Expert Systems with Applications, 275, 127043. https://doi.org/10.1016/j.eswa.2025.127043
18. Schellingerhout, R., Barile, F., & Tintarev, N. (2025). *OKRA: an Explainable, Heterogeneous, Multi-Stakeholder Job Recommender System*. arXiv:2504.07108. https://arxiv.org/abs/2504.07108
19. Enevoldsen, K., Chung, I., Kerboua, I., Kardos, M., Mathur, A., Stap, D., Gala, J., Siblini, W., Krzeminski, D., Winata, G. I., et al. (2025). *MMTEB: Massive Multilingual Text Embedding Benchmark*. arXiv:2502.13595 (accepted at ICLR 2025). https://arxiv.org/abs/2502.13595
20. Ajjam, M.-H., & Al-Raweshidy, H. S. (2026). *AI-driven semantic similarity-based job matching framework for recruitment systems*. Information Sciences, 724, 122728. https://doi.org/10.1016/j.ins.2025.122728

## Catatan Kuota 10-12

Jika total mahasiswa tidak feasible dengan batas keras 10-12 (contoh 171 mahasiswa, 14 dosen), sistem otomatis melakukan relaksasi kapasitas terkontrol (mis. +1 slot pada dosen prioritas) dan menyimpan catatan relaksasi pada metadata run.
