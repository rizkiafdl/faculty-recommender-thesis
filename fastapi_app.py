from __future__ import annotations

import io
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.config import DEFAULT_EXCEL_PATH, DEFAULT_SHEET_NAME
from app.database import SessionLocal, get_session
from app.schemas import (
    EvaluationResponse,
    ImportResponse,
    RecommendationListResponse,
    RunResponse,
    SummaryResponse,
)
from app.services import (
    benchmark_models,
    evaluation_by_run,
    export_recommendations_excel,
    generate_and_store_recommendations,
    get_latest_run,
    import_students_from_bytes,
    import_students_from_path,
    init_db,
    list_recommendations,
    seed_supervisors,
    summary_by_supervisor,
)


app = FastAPI(title="Faculty Supervisor Recommender", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    session = SessionLocal()
    try:
        seed_supervisors(session)
    finally:
        session.close()


def _serialize_run(run) -> RunResponse:
    return RunResponse(
        run_id=run.id,
        created_at=run.created_at,
        total_students=run.total_students,
        total_supervisors=run.total_supervisors,
        target_min_capacity=run.target_min_capacity,
        target_max_capacity=run.target_max_capacity,
        capacity_relaxed=run.capacity_relaxed,
        capacity_note=run.capacity_note,
        solver_name=run.solver_name,
        solver_note=run.solver_note,
        embedding_backend=run.embedding_backend,
        embedding_model=run.embedding_model,
        objective_score=run.objective_score,
    )


@app.get("/")
def health_check():
    return {"message": "Faculty Supervisor Recommendation API is running"}


@app.post("/api/import/default", response_model=ImportResponse)
def import_default(
    sheet_name: str = Query(DEFAULT_SHEET_NAME),
    session: Session = Depends(get_session),
):
    path = Path(DEFAULT_EXCEL_PATH)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File default tidak ditemukan: {path}")
    try:
        result = import_students_from_path(session=session, excel_path=path, sheet_name=sheet_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ImportResponse(**result)


@app.post("/api/import/upload", response_model=ImportResponse)
async def import_upload(
    file: UploadFile = File(...),
    sheet_name: str = Query(DEFAULT_SHEET_NAME),
    session: Session = Depends(get_session),
):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Hanya file .xlsx yang didukung.")
    content = await file.read()
    try:
        result = import_students_from_bytes(
            session=session,
            content=content,
            filename=file.filename,
            sheet_name=sheet_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ImportResponse(**result)


@app.post("/api/recommend/run", response_model=RunResponse)
def run_recommendation(
    source: str = Query("api-run"),
    session: Session = Depends(get_session),
):
    try:
        run = generate_and_store_recommendations(session=session, input_source=source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_run(run)


@app.get("/api/runs/latest", response_model=RunResponse)
def latest_run(session: Session = Depends(get_session)):
    run = get_latest_run(session=session)
    if run is None:
        raise HTTPException(status_code=404, detail="Belum ada run rekomendasi.")
    return _serialize_run(run)


@app.get("/api/recommendations", response_model=RecommendationListResponse)
def get_recommendation_list(
    run_id: int | None = Query(default=None),
    session: Session = Depends(get_session),
):
    try:
        run, data = list_recommendations(session=session, run_id=run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RecommendationListResponse(run_id=run.id, data=data)


@app.get("/api/summary", response_model=SummaryResponse)
def get_summary(
    run_id: int | None = Query(default=None),
    session: Session = Depends(get_session),
):
    try:
        run, summary = summary_by_supervisor(session=session, run_id=run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SummaryResponse(run_id=run.id, summary=summary)


@app.get("/api/evaluation", response_model=EvaluationResponse)
def get_evaluation(
    run_id: int | None = Query(default=None),
    session: Session = Depends(get_session),
):
    try:
        run, evaluation = evaluation_by_run(session=session, run_id=run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EvaluationResponse(run_id=run.id, evaluation=evaluation)


@app.get("/api/evaluation/benchmark-models")
def benchmark_evaluation_models(
    models: str | None = Query(
        default=None,
        description="Daftar model dipisah koma. Contoh: answerdotai/ModernBERT-base,sentence-transformers/all-mpnet-base-v2",
    ),
    device: str | None = Query(default=None),
    session: Session = Depends(get_session),
):
    model_names = None
    if models:
        parsed = [value.strip() for value in models.split(",") if value.strip()]
        model_names = parsed or None
    try:
        return benchmark_models(
            session=session,
            model_names=model_names,
            device=device,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/export")
def export_excel(
    run_id: int | None = Query(default=None),
    session: Session = Depends(get_session),
):
    try:
        content, filename = export_recommendations_excel(session=session, run_id=run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.exception_handler(ValueError)
def value_error_handler(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
