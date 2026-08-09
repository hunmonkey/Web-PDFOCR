# -*- coding: utf-8 -*-
"""
PDFOCR 웹 서버 (FastAPI)

화면 흐름:
  1. 사용자가 PDF를 올리고 설정을 고른다.        → POST /api/jobs
  2. 서버가 뒤에서 변환을 돌린다.                 → 작업 스레드 1개가 순서대로 처리
  3. 진행 상황을 화면에 실시간으로 흘려보낸다.    → GET  /api/jobs/{id}/events
  4. 다 되면 결과 파일을 내려받는다.              → GET  /api/jobs/{id}/download

용어 설명:
  · SSE(Server-Sent Events) : 서버가 브라우저에게 소식을 계속 밀어 넣는 방법.
                              새로고침 없이 진행률을 갱신할 때 쓴다.
  · 작업 스레드              : 변환은 오래 걸리므로 따로 떼어 돌린다.
                              그래야 그동안에도 화면이 멈추지 않는다.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pymupdf
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.ocr_engine import MODE_KEEP, MODE_REBUILD, OcrOptions, OcrSummary, run_ocr

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
UPLOAD_DIR = ROOT_DIR / "uploads"
OUTPUT_DIR = ROOT_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 화면에 그대로 보여 줄 상태 이름 (Rule_UIUX.txt 2-1 작업 상태 기준)
STATUS_WAITING = "대기"
STATUS_RUNNING = "실행 중"
STATUS_DONE = "성공"
STATUS_FAILED = "실패"
STATUS_CANCELED = "취소"
FINISHED_STATUSES = {STATUS_DONE, STATUS_FAILED, STATUS_CANCELED}

ALLOWED_DPI = {150, 200, 300}

# 변환은 그래픽카드를 쓰므로 한 번에 하나씩만 처리한다.
# 여러 개를 동시에 돌리면 서로 메모리를 뺏어 오히려 느려지고 불안정해진다.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")


@dataclass
class Job:
    """변환 작업 하나의 상태. 서버 메모리에만 둔다."""
    id: str
    original_name: str
    input_path: Path
    output_path: Path
    options: OcrOptions
    status: str = STATUS_WAITING
    done_pages: int = 0
    total_pages: int = 0
    message: str = "차례를 기다리는 중입니다."
    stop_requested: bool = False
    summary: Optional[OcrSummary] = None
    error: str = ""


jobs: dict[str, Job] = {}

app = FastAPI(title="PDFOCR")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ─────────────────────────────────────────────────────────────────────────────
# 실제 변환을 도는 부분 (작업 스레드에서 실행)
# ─────────────────────────────────────────────────────────────────────────────
def _process(job: Job) -> None:
    job.status = STATUS_RUNNING
    job.message = "글자를 읽는 중입니다."

    def on_progress(done: int, total: int) -> None:
        job.done_pages = done
        job.total_pages = total
        job.message = f"{total}쪽 중 {done}쪽을 읽었습니다."

    try:
        summary = run_ocr(
            job.input_path,
            job.output_path,
            job.options,
            on_progress=on_progress,
            should_stop=lambda: job.stop_requested,
        )
        job.summary = summary

        if summary.stopped:
            job.status = STATUS_CANCELED
            job.message = (
                f"사용자 요청으로 중지했습니다. "
                f"({summary.total_pages}쪽 중 {summary.done_pages}쪽까지 읽음)"
            )
        else:
            job.status = STATUS_DONE
            job.message = "변환이 완료되었습니다."

    except Exception as exc:
        job.status = STATUS_FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "변환이 실패했습니다. 아래 내용을 확인해 주세요."

    finally:
        # 올린 원본은 변환이 끝나면 남겨 둘 이유가 없다.
        job.input_path.unlink(missing_ok=True)


def _job_payload(job: Job) -> dict:
    """화면으로 보낼 진행 상황 한 덩어리."""
    percent = 0
    if job.total_pages:
        percent = round(job.done_pages / job.total_pages * 100)

    payload = {
        "id": job.id,
        "status": job.status,
        "message": job.message,
        "done_pages": job.done_pages,
        "total_pages": job.total_pages,
        "percent": percent,
        "finished": job.status in FINISHED_STATUSES,
        "error": job.error,
    }

    if job.summary is not None and job.status == STATUS_DONE:
        s = job.summary
        payload["result"] = {
            "device": s.device,
            "elapsed": round(s.elapsed, 1),
            "pages": s.total_pages,
            "text_lines": s.text_lines,
            "input_bytes": s.input_bytes,
            "output_bytes": s.output_bytes,
            # 변환 뒤 용량이 원본의 몇 %인지. 100보다 작으면 줄어든 것이다.
            "size_percent": round(s.output_bytes / s.input_bytes * 100) if s.input_bytes else 0,
            "download_name": job.output_path.name,
        }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# 화면
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# 작업 만들기
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/jobs")
async def create_job(
    file: UploadFile,
    dpi: int = Form(300),
    text_score: float = Form(0.5),
    mode: str = Form(MODE_KEEP),
) -> dict:
    if dpi not in ALLOWED_DPI:
        raise HTTPException(400, "해상도는 150, 200, 300 중에서 골라 주세요.")
    if not 0.0 <= text_score <= 1.0:
        raise HTTPException(400, "인식 기준 점수는 0과 1 사이여야 합니다.")
    if mode not in (MODE_KEEP, MODE_REBUILD):
        raise HTTPException(400, "변환 방식 값이 올바르지 않습니다.")

    original_name = Path(file.filename or "문서.pdf").name
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF 파일만 올릴 수 있습니다.")

    job_id = uuid.uuid4().hex[:12]
    input_path = UPLOAD_DIR / f"{job_id}.pdf"
    input_path.write_bytes(await file.read())

    # 진짜 PDF인지, 그리고 몇 쪽짜리인지 미리 확인한다.
    # 여기서 걸러야 변환을 시작한 뒤에 실패하는 일이 없다.
    try:
        with pymupdf.open(input_path) as doc:
            page_count = doc.page_count
            if doc.needs_pass:
                raise HTTPException(400, "암호가 걸린 PDF는 변환할 수 없습니다.")
    except HTTPException:
        input_path.unlink(missing_ok=True)
        raise
    except Exception:
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "PDF 파일을 열 수 없습니다. 파일이 손상되었는지 확인해 주세요.")

    if page_count == 0:
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "쪽이 하나도 없는 PDF입니다.")

    job = Job(
        id=job_id,
        original_name=original_name,
        input_path=input_path,
        output_path=OUTPUT_DIR / f"{Path(original_name).stem}_Searchable_{job_id}.pdf",
        options=OcrOptions(dpi=dpi, text_score=text_score, mode=mode),
        total_pages=page_count,
    )
    jobs[job_id] = job

    _worker.submit(_process, job)
    return {"id": job_id, "total_pages": page_count, "original_name": original_name}


# ─────────────────────────────────────────────────────────────────────────────
# 진행 상황 실시간 전달 (SSE)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")

    async def stream():
        last = None
        while True:
            payload = _job_payload(job)
            # 바뀐 내용이 있을 때만 보낸다. (같은 내용을 계속 보내면 낭비다)
            if payload != last:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last = payload
            if payload["finished"]:
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 중지
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")

    if job.status in FINISHED_STATUSES:
        return {"status": job.status, "message": "이미 끝난 작업입니다."}

    job.stop_requested = True
    job.message = "중지하는 중입니다. 읽고 있던 쪽까지만 마무리합니다."
    return {"status": job.status, "message": job.message}


# ─────────────────────────────────────────────────────────────────────────────
# 결과 내려받기
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}/download")
async def download(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")
    if job.status != STATUS_DONE or not job.output_path.exists():
        raise HTTPException(409, "아직 내려받을 수 있는 결과가 없습니다.")

    # 한글 파일 이름이 깨지지 않도록 UTF-8로 적어 보낸다.
    download_name = f"{Path(job.original_name).stem}_Searchable.pdf"
    return FileResponse(
        job.output_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f"attachment; filename*=UTF-8''{quote(download_name)}"
        },
    )
