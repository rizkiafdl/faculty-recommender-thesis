from __future__ import annotations

import io
import json
import os
from functools import wraps
from pathlib import Path

from flask import Flask, flash, g, redirect, render_template, request, send_file, session as web_session, url_for
from sqlalchemy import func, select

from app.config import DEFAULT_EXCEL_PATH, DEFAULT_SHEET_NAME
from app.database import SessionLocal
from app.models import RecommendationRun, Student, Supervisor
from app.services import (
    add_or_update_supervisor,
    assign_supervisor_category,
    authenticate_user,
    evaluation_by_run,
    export_recommendations_excel,
    export_supervisor_configuration_excel,
    generate_and_store_recommendations,
    get_latest_run,
    get_user_by_id,
    import_students_from_bytes,
    import_students_from_path,
    init_db,
    list_recommendations,
    list_runs,
    list_supervisor_profiles_for_web,
    register_user,
    remove_supervisor_category,
    seed_supervisors,
    summary_by_supervisor,
    update_supervisor_keywords,
)


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


def bootstrap() -> None:
    init_db()
    with SessionLocal() as session:
        seed_supervisors(session)


bootstrap()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if g.current_user is None:
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapped_view


def _safe_next(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate.startswith("/"):
        return None
    if candidate.startswith("//"):
        return None
    return candidate


def _parse_supervisor_code(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if " - " in raw:
        return raw.split(" - ", 1)[0].strip() or None
    return raw


def _metric_value(payload: dict, section: str, metric: str):
    section_data = payload.get(section) if isinstance(payload, dict) else None
    if not isinstance(section_data, dict):
        return None
    return section_data.get(metric)


def _evaluation_payload_from_run(run) -> dict[str, object]:
    raw = getattr(run, "evaluation_json", None)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_metric_row(run) -> dict[str, object]:
    payload = _evaluation_payload_from_run(run)
    return {
        "id": run.id,
        "created_at": run.created_at,
        "objective_score": run.objective_score,
        "solver_name": run.solver_name,
        "embedding_model": run.embedding_model,
        "assignment_match": _metric_value(payload, "assignment_match", "match_rate"),
        "hybrid_mrr": _metric_value(payload, "hybrid_score", "mrr"),
        "content_mrr": _metric_value(payload, "content_based", "mrr"),
    }


@app.before_request
def load_current_user() -> None:
    g.current_user = None
    user_id = web_session.get("user_id")
    if not user_id:
        return
    try:
        resolved_user_id = int(user_id)
    except (TypeError, ValueError):
        web_session.pop("user_id", None)
        return
    with SessionLocal() as session:
        g.current_user = get_user_by_id(session=session, user_id=resolved_user_id)


@app.context_processor
def inject_auth_context():
    return {
        "current_user": g.get("current_user"),
        "request_path": request.path,
    }


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.current_user is not None:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "")
        full_name = request.form.get("full_name", "")
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if password != confirm_password:
            flash("Password dan konfirmasi password tidak sama.", "error")
            return redirect(url_for("register"))
        try:
            with SessionLocal() as session:
                user = register_user(
                    session=session,
                    username=username,
                    password=password,
                    full_name=full_name,
                )
            web_session["user_id"] = user.id
            flash("Registrasi berhasil. Kamu sudah login.", "success")
            return redirect(url_for("dashboard"))
        except Exception as exc:
            flash(f"Registrasi gagal: {exc}", "error")
            return redirect(url_for("register"))
    return render_template("register.html", page_title="Create Account")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.current_user is not None:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        with SessionLocal() as session:
            user = authenticate_user(session=session, username=username, password=password)
        if user is None:
            flash("Username atau password salah.", "error")
            return redirect(url_for("login"))
        web_session["user_id"] = user.id
        next_url = _safe_next(request.args.get("next")) or url_for("dashboard")
        flash("Login berhasil.", "success")
        return redirect(next_url)
    return render_template("login.html", page_title="Welcome Back")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    web_session.pop("user_id", None)
    flash("Logout berhasil.", "success")
    return redirect(url_for("login"))


@app.route("/", methods=["GET"])
@login_required
def root():
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    with SessionLocal() as session:
        latest_run = get_latest_run(session)
        recent_runs = list_runs(session, limit=12)
        total_students = int(session.execute(select(func.count(Student.id))).scalar_one() or 0)
        active_supervisors = int(
            session.execute(
                select(func.count(Supervisor.id)).where(Supervisor.is_active.is_(True))
            ).scalar_one()
            or 0
        )
        total_runs = int(
            session.execute(select(func.count(RecommendationRun.id))).scalar_one() or 0
        )
    run_rows = [_run_metric_row(run) for run in recent_runs]
    latest_metrics = run_rows[0] if run_rows else {}

    highlight_metrics = {
        "assignment_match": latest_metrics.get("assignment_match"),
        "hybrid_mrr": latest_metrics.get("hybrid_mrr"),
        "content_mrr": latest_metrics.get("content_mrr"),
    }

    return render_template(
        "dashboard.html",
        page_title="Command Center",
        page_subtitle="Monitor run quality, capacity, and recommendation health in one glance.",
        latest_run=latest_run,
        recent_runs=recent_runs,
        run_rows=run_rows,
        total_students=total_students,
        active_supervisors=active_supervisors,
        total_runs=total_runs if total_runs else len(run_rows),
        highlight_metrics=highlight_metrics,
    )


@app.route("/data", methods=["GET"])
@login_required
def data_center():
    with SessionLocal() as session:
        latest_run = get_latest_run(session)
        total_students = int(session.execute(select(func.count(Student.id))).scalar_one() or 0)
        labeled_students = int(
            session.execute(
                select(func.count(Student.id)).where(
                    Student.current_supervisor_code.is_not(None),
                    func.trim(Student.current_supervisor_code) != "",
                )
            ).scalar_one()
            or 0
        )
        active_supervisors = int(
            session.execute(
                select(func.count(Supervisor.id)).where(Supervisor.is_active.is_(True))
            ).scalar_one()
            or 0
        )
        total_runs = int(
            session.execute(select(func.count(RecommendationRun.id))).scalar_one()
            or 0
        )

    return render_template(
        "data_center.html",
        page_title="Data Center",
        page_subtitle="Import data and trigger assignment generation pipeline.",
        default_excel_path=DEFAULT_EXCEL_PATH,
        default_sheet_name=DEFAULT_SHEET_NAME,
        latest_run=latest_run,
        total_students=total_students,
        labeled_students=labeled_students,
        active_supervisors=active_supervisors,
        total_runs=total_runs,
        default_source_exists=Path(DEFAULT_EXCEL_PATH).exists(),
    )


@app.route("/data/import-default", methods=["POST"])
@login_required
def import_default():
    path = Path(DEFAULT_EXCEL_PATH)
    if not path.exists():
        flash(f"File default tidak ditemukan: {path}", "error")
        return redirect(url_for("data_center"))

    try:
        with SessionLocal() as session:
            result = import_students_from_path(
                session=session,
                excel_path=path,
                sheet_name=DEFAULT_SHEET_NAME,
            )
        flash(
            f"Import default berhasil. inserted={result['inserted']}, "
            f"updated={result['updated']}, total={result['total']}",
            "success",
        )
    except Exception as exc:
        flash(f"Gagal import default: {exc}", "error")
    return redirect(url_for("data_center"))


@app.route("/data/import-upload", methods=["POST"])
@login_required
def import_upload():
    file = request.files.get("excel_file")
    if file is None or not file.filename:
        flash("Pilih file Excel terlebih dahulu.", "error")
        return redirect(url_for("data_center"))
    if not file.filename.lower().endswith(".xlsx"):
        flash("Hanya file .xlsx yang didukung.", "error")
        return redirect(url_for("data_center"))

    try:
        content = file.read()
        with SessionLocal() as session:
            result = import_students_from_bytes(
                session=session,
                content=content,
                filename=file.filename,
                sheet_name=DEFAULT_SHEET_NAME,
            )
        flash(
            f"Upload berhasil. inserted={result['inserted']}, "
            f"updated={result['updated']}, total={result['total']}",
            "success",
        )
    except Exception as exc:
        flash(f"Gagal upload Excel: {exc}", "error")

    return redirect(url_for("data_center"))


@app.route("/generate", methods=["POST"])
@login_required
def generate():
    next_url = _safe_next(request.form.get("next"))
    try:
        with SessionLocal() as session:
            run = generate_and_store_recommendations(session=session, input_source="flask-ui")
        flash(f"Run rekomendasi berhasil dibuat (run_id={run.id}).", "success")
        if next_url:
            return redirect(next_url)
        return redirect(url_for("run_detail", run_id=run.id))
    except Exception as exc:
        flash(f"Gagal generate rekomendasi: {exc}", "error")
        return redirect(next_url or url_for("data_center"))


@app.route("/runs", methods=["GET"])
@login_required
def runs_history():
    with SessionLocal() as session:
        runs = list_runs(session=session, limit=250)
    run_rows = [_run_metric_row(run) for run in runs]
    return render_template(
        "runs.html",
        page_title="Run History",
        page_subtitle="Track every execution and jump straight to analysis pages.",
        runs=runs,
        run_rows=run_rows,
    )


@app.route("/runs/<int:run_id>", methods=["GET"])
@login_required
def run_detail(run_id: int):
    with SessionLocal() as session:
        run, summary = summary_by_supervisor(session=session, run_id=run_id)
        _, evaluation = evaluation_by_run(session=session, run_id=run_id)
        _, recommendations = list_recommendations(session=session, run_id=run_id)

    mismatches = [
        row
        for row in recommendations
        if (row.get("current_supervisor_code") or "").strip()
        and row.get("current_supervisor_code") != row.get("recommended_supervisor_code")
    ]

    return render_template(
        "run_detail.html",
        page_title=f"Run #{run.id}",
        page_subtitle="Evaluation, capacity, and mismatch spotlight.",
        run=run,
        summary=summary,
        evaluation=evaluation,
        mismatches=mismatches[:10],
        mismatch_count=len(mismatches),
    )


@app.route("/runs/<int:run_id>/recommendations", methods=["GET"])
@login_required
def run_recommendations(run_id: int):
    query = (request.args.get("q") or "").strip().lower()
    supervisor_code_filter = (request.args.get("supervisor_code") or "").strip()
    mismatch_only = (request.args.get("mismatch") or "").strip().lower() in {"1", "true", "yes", "on"}

    with SessionLocal() as session:
        run, recommendations = list_recommendations(session=session, run_id=run_id)

    available_supervisors = sorted(
        {(row.get("recommended_supervisor_code") or "", row.get("recommended_supervisor_name") or "") for row in recommendations}
    )

    filtered = recommendations
    if supervisor_code_filter:
        filtered = [
            row
            for row in filtered
            if (row.get("recommended_supervisor_code") or "") == supervisor_code_filter
        ]

    if mismatch_only:
        filtered = [
            row
            for row in filtered
            if (row.get("current_supervisor_code") or "").strip()
            and row.get("current_supervisor_code") != row.get("recommended_supervisor_code")
        ]

    if query:
        filtered = [
            row
            for row in filtered
            if query in (row.get("student_id") or "").lower()
            or query in (row.get("student_name") or "").lower()
            or query in (row.get("partner_lecturer") or "").lower()
            or query in (row.get("position_topic") or "").lower()
            or query in (row.get("recommended_supervisor_name") or "").lower()
        ]

    return render_template(
        "recommendations.html",
        page_title=f"Run #{run.id} Recommendations",
        page_subtitle="Search and inspect assignment-level decisions.",
        run=run,
        recommendations=filtered,
        recommendation_count=len(filtered),
        total_count=len(recommendations),
        available_supervisors=available_supervisors,
        query=query,
        supervisor_code_filter=supervisor_code_filter,
        mismatch_only=mismatch_only,
    )


@app.route("/runs/<int:run_id>/export", methods=["GET"])
@login_required
def export_run(run_id: int):
    try:
        with SessionLocal() as session:
            content, filename = export_recommendations_excel(session=session, run_id=run_id)
    except Exception as exc:
        flash(f"Gagal export Excel: {exc}", "error")
        return redirect(url_for("run_detail", run_id=run_id))

    return send_file(
        io.BytesIO(content),
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/export/<int:run_id>", methods=["GET"])
@login_required
def export_legacy(run_id: int):
    return redirect(url_for("export_run", run_id=run_id))


@app.route("/supervisors", methods=["GET"])
@login_required
def supervisors_page():
    selected_supervisor_code = _parse_supervisor_code(
        request.args.get("code") or request.args.get("pick")
    )

    with SessionLocal() as session:
        supervisors = list_supervisor_profiles_for_web(session=session)
        keyword_suggestions_set: set[str] = set()
        for supervisor in supervisors:
            raw = str(supervisor.get("profile_keywords") or "")
            for token in raw.split(","):
                keyword = token.strip().lower()
                if keyword:
                    keyword_suggestions_set.add(keyword)
        keyword_suggestions = sorted(keyword_suggestions_set)

    if supervisors and not selected_supervisor_code:
        selected_supervisor_code = supervisors[0]["code"]

    selected_supervisor = next(
        (item for item in supervisors if item["code"] == selected_supervisor_code),
        None,
    )

    return render_template(
        "supervisors.html",
        page_title="Supervisor Studio",
        page_subtitle="Manage supervisor roster and semantic keywords for recommendation matching.",
        supervisors=supervisors,
        keyword_suggestions=keyword_suggestions,
        selected_supervisor=selected_supervisor,
        selected_supervisor_code=selected_supervisor_code,
    )


@app.route("/supervisors/add", methods=["POST"])
@login_required
def add_supervisor():
    supervisor_code = request.form.get("supervisor_code", "")
    supervisor_name = request.form.get("supervisor_name", "")
    profile_keywords = request.form.get("profile_keywords", "")
    try:
        with SessionLocal() as session:
            supervisor = add_or_update_supervisor(
                session=session,
                supervisor_code=supervisor_code,
                supervisor_name=supervisor_name,
                profile_keywords=profile_keywords,
            )
        flash(f"Dosen {supervisor.code} berhasil disimpan.", "success")
        return redirect(url_for("supervisors_page", code=supervisor.code))
    except Exception as exc:
        flash(f"Gagal simpan dosen: {exc}", "error")
        return redirect(url_for("supervisors_page"))


@app.route("/supervisors/category/add", methods=["POST"])
@login_required
def add_supervisor_category():
    supervisor_code = _parse_supervisor_code(request.form.get("supervisor_code")) or ""
    category_name = request.form.get("category_name", "")
    try:
        with SessionLocal() as session:
            assign_supervisor_category(
                session=session,
                supervisor_code=supervisor_code,
                category_name=category_name,
            )
        flash("Kategori dosen berhasil ditambahkan.", "success")
    except Exception as exc:
        flash(f"Gagal menambahkan kategori: {exc}", "error")
    return redirect(url_for("supervisors_page", code=supervisor_code))


@app.route("/supervisors/category/remove", methods=["POST"])
@login_required
def delete_supervisor_category():
    supervisor_code = _parse_supervisor_code(request.form.get("supervisor_code")) or ""
    category_name = request.form.get("category_name", "")
    try:
        with SessionLocal() as session:
            removed = remove_supervisor_category(
                session=session,
                supervisor_code=supervisor_code,
                category_name=category_name,
            )
        if removed:
            flash("Kategori dosen berhasil dihapus.", "success")
        else:
            flash("Kategori tidak ditemukan.", "error")
    except Exception as exc:
        flash(f"Gagal menghapus kategori: {exc}", "error")
    return redirect(url_for("supervisors_page", code=supervisor_code))


@app.route("/supervisors/keywords/update", methods=["POST"])
@login_required
def update_keywords():
    supervisor_code = _parse_supervisor_code(request.form.get("supervisor_code")) or ""
    profile_keywords = request.form.get("profile_keywords", "")
    try:
        with SessionLocal() as session:
            update_supervisor_keywords(
                session=session,
                supervisor_code=supervisor_code,
                profile_keywords=profile_keywords,
            )
        flash("Deskripsi/keywords dosen berhasil diupdate.", "success")
    except Exception as exc:
        flash(f"Gagal update deskripsi dosen: {exc}", "error")
    return redirect(url_for("supervisors_page", code=supervisor_code))


@app.route("/supervisors/export", methods=["GET"])
@login_required
def export_supervisor_config():
    try:
        with SessionLocal() as session:
            content, filename = export_supervisor_configuration_excel(session=session)
    except Exception as exc:
        flash(f"Gagal export konfigurasi dosen: {exc}", "error")
        return redirect(url_for("supervisors_page"))

    return send_file(
        io.BytesIO(content),
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/export/supervisor-config", methods=["GET"])
@login_required
def export_supervisor_config_legacy():
    return redirect(url_for("export_supervisor_config"))


@app.route("/benchmark", methods=["GET"])
@login_required
def benchmark_page():
    return render_template(
        "benchmark.html",
        page_title="Benchmark",
        page_subtitle="Evaluation metrics and model comparison.",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
