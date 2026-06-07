# Faculty Supervisor Recommender

Sistem rekomendasi dosen pembimbing magang — Binus University, Teknik Informatika Bandung.

Maps internship students to supervisors using semantic similarity (ModernBERT) + greedy capacity assignment.

---

## Quick Start

```bash
# 1. Clone & install
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Seed the database (run once after fresh clone or DB reset)
python seed.py --all

# 3. Start the app
python flask_app.py
# Open: http://127.0.0.1:5001
```

On first open: register a new account, then log in.

> **Note:** `python seed.py --all` runs all four seeders in order: supervisor profiles → category assignments → label descriptions → affinity matrix. Idempotent — safe to re-run.

---

## Docker

```bash
cp .env.example .env            # edit if needed
docker compose up --build
# Open: http://localhost:5001
```

The container runs `seed.py --all` before starting Flask. No manual seeding needed.

---

## Environment Variables

Copy `.env.example` to `.env` and adjust. Most defaults work out of the box.

### Pipeline toggles (all default `false` = pure embedding mode)

| Variable | Default | Effect |
|---|---|---|
| `ENABLE_RULE_BOOST` | `false` | Turn on label/affinity rule boost layer |
| `ENABLE_GROUP_BONUS` | `false` | Turn on company cohort bonus |
| `ENABLE_EXTRA_DOCS` | `false` | Add historical text from `map_2026.xlsx` to supervisor docs |

### Capacity

| Variable | Default | Notes |
|---|---|---|
| `TARGET_MIN_CAPACITY` | `10` | Min students per supervisor |
| `TARGET_MAX_CAPACITY` | `12` | Max students per supervisor |

### Embedding

| Variable | Default |
|---|---|
| `EMBEDDING_MODEL_NAME` | `answerdotai/ModernBERT-base` |
| `EMBEDDING_DEVICE` | `auto` (use `cuda` for GPU) |

### Database

| Variable | Default |
|---|---|
| `DATABASE_URL` | `sqlite:///recommendation.db` |
| `DEFAULT_EXCEL_PATH` | `map_2026.xlsx` |

---

## Seeding

Seeding is explicit — it does not run automatically when the app starts.

```bash
python seed.py --all              # seed everything (recommended order)
python seed.py --supervisors      # supervisor profiles → supervisors table
python seed.py --categories       # supervisor label assignments → must run after --supervisors
python seed.py --labels           # label descriptions
python seed.py --affinity         # affinity matrix
```

> `--supervisors` must run before `--categories` — category assignments require supervisor rows to exist.

Seed data lives in `datasets/seed_dataset/`:

| File | Role |
|---|---|
| `supervisor_profiles.py` | Source of truth for 14 supervisors — read only by `seeder.py`, never by the app pipeline |
| `label_descriptions.py` | 19 label (name, description, threshold, is_niche) definitions |
| `affinity_matrix.py` | 57 supervisor-label boost/penalty rows |
| `seeder.py` | Four seed functions: `seed_supervisors`, `seed_supervisor_categories`, `seed_label_descriptions`, `seed_affinity_matrix` |

The app pipeline reads supervisor data exclusively from the database. `supervisor_profiles.py` is only ever touched by the seeder.

---

## Batch Run

Runs the pipeline across all 18 configurations automatically (3 models × 3 toggle combos × 2 capacity variants) and exports one detailed `.xlsx` per run.

```bash
python batch_run.py                              # output → output/batch_<timestamp>/
python batch_run.py --output-dir output/my_run  # custom output folder
```

**Test matrix:**

| Dimension | Variants |
|---|---|
| Embedding model | `BAAI/bge-m3`, `Qwen3-Embedding-0.6B`, `multilingual-e5-large-instruct` |
| Toggle config | `no_group_bonus`, `no_extra_docs`, `both_off` |
| Capacity priority | `no_priority` (`[]`), `default_priority` (`CAPACITY_PRIORITY_CODES`) |

Each export is a multi-sheet `.xlsx` (`recommendations`, `rankings`, `summary`, `config`, `evaluation`). The `config` sheet records the exact overrides used for that run.

---

## Web UI

| Path | What it does |
|---|---|
| `/dashboard` | KPI summary, run snapshot, pipeline config |
| `/data` | Import student Excel, trigger a recommendation run |
| `/runs` | Run history |
| `/runs/<id>` | Run detail: evaluation metrics, capacity plan |
| `/runs/<id>/recommendations` | Per-student recommendation table |
| `/supervisors` | Manage supervisor profiles (keywords, categories) |
| `/rules` | Rules Studio: edit label descriptions + affinity matrix |
| `/benchmark` | Benchmark embedding models |

---

## Project Structure

```
flask_app.py                  — Flask entry point + all routes
seed.py                       — CLI seeder (run once after setup)

app/
  config.py                   — All env vars and constants
  models.py                   — SQLAlchemy ORM models
  schemas.py                  — In-memory pipeline dataclasses
  database.py                 — Engine + SessionLocal
  queries.py                  — All ORM queries
  services.py                 — Business logic + orchestration
  recommender.py              — Score assembly → capacity plan → greedy solver
  embedding.py                — EmbeddingProvider (3 backends) + label cache
  rules.py                    — Document building, label detection, rule boost
  evaluation.py               — Retrieval metrics: MRR, Hit@k, nDCG@k
  excel_io.py                 — Excel student import

datasets/
  map_loader.py               — Load extra supervisor docs from map_2026.xlsx
  seed_dataset/
    supervisor_profiles.py    — Static supervisor data (seeder-only, not read by pipeline)
    label_descriptions.py     — Semantic label definitions
    affinity_matrix.py        — Supervisor-label boost/penalty values
    label_terms.py            — Keyword term lists for label detection
    stopwords.py              — Profile token stopwords
    seeder.py                 — seed_supervisors, seed_supervisor_categories, seed_label_descriptions, seed_affinity_matrix

templates/                    — Jinja2 HTML templates
static/style.css
```

---

## Tech Stack

| Layer | Implementation |
|---|---|
| Web | Flask 3.x |
| Database | SQLite via SQLAlchemy 2.x |
| Embedding (primary) | `answerdotai/ModernBERT-base` via `sentence-transformers` |
| Embedding (fallback) | `all-mpnet-base-v2` → TF-IDF → Token Overlap |
| Solver | Greedy (capacity-constrained, `app/recommender.py`) |
| Export | Excel via openpyxl + pandas |
| Container | Docker + docker-compose |