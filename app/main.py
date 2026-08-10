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
import importlib.util
import json
import math
import os
import re
import shutil
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Annotated, Optional
from urllib.parse import quote

import pymupdf
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app import auth, storage
from app.ocr_engine import (
    EXPORT_FORMATS,
    MODE_KEEP,
    MODE_REBUILD,
    BrowserOcrLine,
    BrowserOcrPage,
    OcrOptions,
    OcrSummary,
    build_pdf_from_browser_results,
    document_has_text_layer,
    render_browser_page,
    run_ocr,
    write_text_export,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
# 올린 파일이 잠깐 머무는 곳. 변환이 끝나면 지운다.
# 변환 결과는 여기가 아니라 보관함(app/storage.py)에 저장한다.
UPLOAD_DIR = ROOT_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)
storage.ensure_dirs()

# 화면에 그대로 보여 줄 상태 이름 (Rule_UIUX.txt 2-1 작업 상태 기준)
STATUS_WAITING = "대기"
STATUS_RUNNING = "실행 중"
STATUS_DONE = "성공"
STATUS_FAILED = "실패"
STATUS_CANCELED = "취소"
FINISHED_STATUSES = {STATUS_DONE, STATUS_FAILED, STATUS_CANCELED}

ALLOWED_DPI = {150, 200, 300}

# 글자 파일을 내려줄 때 브라우저에 알려 줄 파일 종류
EXPORT_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

PROCESSOR_SERVER = "server"
PROCESSOR_BROWSER = "browser"
ALLOWED_PROCESSORS = {PROCESSOR_SERVER, PROCESSOR_BROWSER}

# 브라우저가 한 번에 다루는 자료가 지나치게 커져 서버 메모리를 채우지 않게 막는다.
BROWSER_PAGE_MAX_SIDE = 2000
BROWSER_MAX_PAGES = 2000
BROWSER_MAX_RESULT_BYTES = 4 * 1024 * 1024
BROWSER_MAX_LINES_PER_PAGE = 2000
BROWSER_MAX_LINES_PER_JOB = 100_000
BROWSER_MAX_TEXT_LENGTH = 1000
BROWSER_MAX_RESULT_BYTES_PER_JOB = 32 * 1024 * 1024
BROWSER_JOB_TTL_SECONDS = 30 * 60
BROWSER_JOB_CLEANUP_INTERVAL_SECONDS = 60
REQUEST_TOMBSTONE_TTL_SECONDS = BROWSER_JOB_TTL_SECONDS
REQUEST_TOMBSTONE_MAX = 4096
UPLOAD_CHUNK_SIZE = 1024 * 1024
_UPLOAD_TEMP_FILE = re.compile(r'^[0-9a-f]{32}\.pdf$')

# 변환은 그래픽카드를 쓰므로 한 번에 하나씩만 처리한다.
# 여러 개를 동시에 돌리면 서로 메모리를 뺏어 오히려 느려지고 불안정해진다.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
# PDF 페이지 그림은 메모리를 많이 쓰므로 서버 전체에서 한 장씩만 만든다.
_browser_render_semaphore = asyncio.Semaphore(1)
# 요청 연결이 끊겨도 실제 렌더링이 끝날 때까지 작업과 세마포어를 살려 둔다.
_browser_render_tasks: set[asyncio.Task] = set()


@dataclass
class Job:
    """변환 작업 하나의 상태. 서버 메모리에만 둔다."""
    id: str
    original_name: str
    input_path: Path
    options: OcrOptions
    processor: str = PROCESSOR_SERVER
    output_path: Optional[Path] = None
    result_reservation: Optional[storage.ReservedResultPath] = None
    export_paths: dict[str, Path] = field(default_factory=dict)
    export_error: str = ""
    status: str = STATUS_WAITING
    done_pages: int = 0
    total_pages: int = 0
    has_text_layer: bool = False
    message: str = "차례를 기다리는 중입니다."
    stop_requested: bool = False
    summary: Optional[OcrSummary] = None
    error: str = ""
    browser_page_images: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    browser_results: dict[int, BrowserOcrPage] = field(default_factory=dict)
    browser_result_bytes: int = 0
    browser_finishing: bool = False
    worker_future: Optional[Future] = field(default=None, repr=False, compare=False)
    started_at: float = field(default_factory=time.perf_counter)
    last_activity: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=Lock, repr=False, compare=False)


jobs: dict[str, Job] = {}
# 같은 생성 요청이 네트워크 재시도로 두 번 실행되지 않게 요청 키를 기억한다.
# 별도 표식으로 파일 복사 중과 그 복사를 취소한 상태를 구분한다.
_REQUEST_CREATING = object()
_REQUEST_CANCELED = object()
_request_jobs: dict[str, str | object] = {}
# create_job 본문이 시작되기 전에 받은 중지 표식만 만료 시간과 함께 관리한다.
_request_tombstone_times: dict[str, float] = {}
_request_jobs_lock = Lock()


def _valid_request_id(value: str) -> bool:
    return (
        len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _expire_request_tombstones_locked(now: float) -> int:
    """_request_jobs_lock 안에서 오래된 선행 중지 표식을 지운다."""
    removed = 0
    for request_id, created_at in list(_request_tombstone_times.items()):
        if _request_jobs.get(request_id) is not _REQUEST_CANCELED:
            _request_tombstone_times.pop(request_id, None)
            continue
        if now - created_at < REQUEST_TOMBSTONE_TTL_SECONDS:
            continue
        _request_tombstone_times.pop(request_id, None)
        _request_jobs.pop(request_id, None)
        removed += 1
    return removed


def _expire_request_tombstones(now: Optional[float] = None) -> int:
    current = time.monotonic() if now is None else now
    with _request_jobs_lock:
        return _expire_request_tombstones_locked(current)


def _remember_request_tombstone_locked(request_id: str) -> None:
    """선행 중지 표식을 TTL 동안, 정해진 최대 개수까지만 기억한다."""
    current = time.monotonic()
    _expire_request_tombstones_locked(current)
    _request_jobs[request_id] = _REQUEST_CANCELED
    _request_tombstone_times.setdefault(request_id, current)

    while len(_request_tombstone_times) > REQUEST_TOMBSTONE_MAX:
        oldest_request_id = next(iter(_request_tombstone_times))
        _request_tombstone_times.pop(oldest_request_id, None)
        if _request_jobs.get(oldest_request_id) is _REQUEST_CANCELED:
            _request_jobs.pop(oldest_request_id, None)


def _run_periodic_cleanup() -> None:
    _expire_stale_browser_jobs()
    _expire_request_tombstones()


async def _browser_job_cleanup_loop(stop_event: asyncio.Event) -> None:
    """오랫동안 소식이 없는 브라우저 작업을 주기적으로 정리한다."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=BROWSER_JOB_CLEANUP_INTERVAL_SECONDS,
            )
        except TimeoutError:
            # 종료 신호가 이 작업 도중 도착해도 실제 정리 스레드가 끝날 때까지
            # 기다린 뒤 루프를 빠져나간다.
            await asyncio.to_thread(_run_periodic_cleanup)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _browser_render_semaphore
    # asyncio 잠금은 처음 사용한 이벤트 루프에 묶인다. 같은 app을 새 루프로
    # 다시 시작할 때 이전 루프의 잠금을 재사용하지 않는다.
    _browser_render_semaphore = asyncio.Semaphore(1)
    # 이전 컨테이너가 강제로 종료되며 남긴 앱 전용 빈 파일만 요청을 받기 전에 정리한다.
    storage.cleanup_orphaned_results()
    _cleanup_orphaned_uploads()
    cleanup_stop = asyncio.Event()
    cleanup_task = asyncio.create_task(_browser_job_cleanup_loop(cleanup_stop))
    try:
        yield
    finally:
        cleanup_stop.set()
        # to_thread는 바깥 Task를 취소해도 실제 스레드는 계속 돈다. 취소하지 않고
        # shield로 끝까지 회수해야 다음 lifespan과 정리 작업이 겹치지 않는다.
        await asyncio.shield(cleanup_task)
        await _shutdown_active_jobs()


app = FastAPI(title="PDFOCR", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ─────────────────────────────────────────────────────────────────────────────
# 로그인 검사
# ─────────────────────────────────────────────────────────────────────────────
# 로그인하지 않아도 들어갈 수 있는 곳
PUBLIC_PATHS = {"/login", "/api/login"}
# 비밀번호를 아직 바꾸지 않았을 때도 쓸 수 있는 곳
PASSWORD_CHANGE_PATHS = {"/change-password", "/api/change-password", "/api/logout"}


@app.middleware("http")
async def require_login(request: Request, call_next):
    """모든 요청에 로그인 여부를 확인한다.

    라우터마다 하나씩 붙이면 새 주소를 추가할 때 빠뜨리기 쉬우므로,
    입구에서 한 번에 검사한다.
    """
    path = request.url.path

    # 로그인 화면의 CSS·자바스크립트는 공개하되, 용량이 큰 OCR 실행 파일과 모델은
    # 로그인한 사용자만 받게 한다. 경로 한 덩어리의 경계를 확인해 vendorish 같은
    # 다른 이름까지 잘못 막지 않는다.
    is_vendor_static = path == "/static/vendor" or path.startswith("/static/vendor/")
    is_public_static = (
        path == "/static" or path.startswith("/static/")
    ) and not is_vendor_static
    if is_public_static or path in PUBLIC_PATHS:
        return await call_next(request)

    token = request.cookies.get(auth.SESSION_COOKIE)
    if not auth.is_logged_in(token):
        if path.startswith("/api/") or is_vendor_static:
            return JSONResponse({"detail": "로그인이 필요합니다."}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    # 처음 비밀번호(admin)를 아직 바꾸지 않았다면 다른 화면을 쓰지 못하게 막는다.
    if auth.load_user().must_change_password and path not in PASSWORD_CHANGE_PATHS:
        if path.startswith("/api/"):
            return JSONResponse(
                {"detail": "비밀번호를 먼저 바꿔 주세요."}, status_code=403)
        return RedirectResponse("/change-password", status_code=303)

    return await call_next(request)


@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.post("/api/login")
async def do_login(username: str = Form(...), password: str = Form(...)):
    token = auth.login(username, password)
    if token is None:
        # 아이디가 틀렸는지 비밀번호가 틀렸는지 알려주지 않는다.
        # 알려주면 아이디를 찾아내는 데 도움이 되기 때문이다.
        raise HTTPException(401, "아이디 또는 비밀번호가 맞지 않습니다.")

    user = auth.load_user()
    response = JSONResponse({
        "must_change_password": user.must_change_password,
        "next": "/change-password" if user.must_change_password else "/",
    })
    response.set_cookie(
        auth.SESSION_COOKIE, token,
        httponly=True,          # 자바스크립트가 쿠키를 읽지 못하게 한다
        samesite="lax",         # 다른 사이트에서 넘어온 요청에는 쿠키를 보내지 않는다
        max_age=auth.SESSION_HOURS * 3600,
    )
    return response


@app.post("/api/logout")
async def do_logout(request: Request):
    auth.logout(request.cookies.get(auth.SESSION_COOKIE))
    response = JSONResponse({"next": "/login"})
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


@app.get("/change-password")
async def change_password_page(request: Request):
    return templates.TemplateResponse(request, "change_password.html")


@app.post("/api/change-password")
async def do_change_password(request: Request, new_password: str = Form(...)):
    ok, message = auth.change_password(new_password)
    if not ok:
        raise HTTPException(400, message)
    # 비밀번호를 바꾸면 지금 접속도 끊기므로 쿠키를 지운다.
    response = JSONResponse({"message": message, "next": "/login"})
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# 실제 변환을 도는 부분 (작업 스레드에서 실행)
# ─────────────────────────────────────────────────────────────────────────────
def _remove_file(path: Optional[Path]) -> None:
    """임시 파일을 지운다. 이미 없으면 그대로 둔다."""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # 파일을 읽는 짧은 순간과 중지가 겹치면 윈도우에서는 바로 못 지울 수 있다.
        # 해당 작업 스레드의 finally에서 다시 한 번 정리한다.
        pass


def _discard_result_reservation(job: Job) -> None:
    """실패하거나 중지한 작업의 예약 파일과 임시 파일을 함께 지운다."""
    with job.lock:
        reservation = job.result_reservation
        job.result_reservation = None
    if reservation is not None:
        _remove_file(reservation.part_path)
        _remove_file(reservation.final_path)


def _clear_browser_buffers_locked(job: Job) -> None:
    """job.lock을 잡은 상태에서 브라우저가 보낸 큰 자료를 놓아준다."""
    job.browser_results.clear()
    job.browser_page_images.clear()
    job.browser_result_bytes = 0


def _cleanup_orphaned_uploads() -> int:
    """비정상 종료로 남은 앱 형식의 업로드 사본만 지운다."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    with os.scandir(UPLOAD_DIR) as entries:
        for entry in entries:
            if (
                _UPLOAD_TEMP_FILE.fullmatch(entry.name) is None
                or not entry.is_file(follow_symlinks=False)
            ):
                continue
            try:
                Path(entry.path).unlink()
                removed += 1
            except OSError:
                continue
    return removed


async def _shutdown_active_jobs() -> None:
    """서버 종료 때 새 결과를 보존하면서 미완료 자료를 안전하게 정리한다."""
    snapshot = list(jobs.values())
    running_futures: list[Future] = []

    for job in snapshot:
        with job.lock:
            future = job.worker_future
            if job.status not in FINISHED_STATUSES:
                job.stop_requested = True
            if future is not None and not future.done():
                if job.status not in FINISHED_STATUSES and future.cancel():
                    job.worker_future = None
                else:
                    # 성공/실패 상태를 먼저 기록한 작업도 finally 정리가 끝날 때까지
                    # Future는 잠시 살아 있을 수 있으므로 상태와 무관하게 기다린다.
                    running_futures.append(future)

    # 요청 연결이 먼저 끊긴 렌더도 실제 스레드가 끝날 때까지 기다려야 원본을
    # 지우는 순간과 파일 읽기가 겹치지 않는다.
    while _browser_render_tasks:
        await asyncio.gather(
            *list(_browser_render_tasks),
            return_exceptions=True,
        )

    if running_futures:
        await asyncio.gather(
            *(asyncio.wrap_future(future) for future in running_futures),
            return_exceptions=True,
        )

    for job in snapshot:
        reservation = None
        with job.lock:
            if job.status != STATUS_DONE:
                job.status = STATUS_CANCELED
                job.message = "서버 종료로 작업을 정리했습니다."
                reservation = job.result_reservation
                job.result_reservation = None
                _clear_browser_buffers_locked(job)
            job.browser_finishing = False
            job.worker_future = None

        _remove_file(job.input_path)
        if reservation is not None:
            _remove_file(reservation.part_path)
            _remove_file(reservation.final_path)

    jobs.clear()
    with _request_jobs_lock:
        _request_jobs.clear()
        _request_tombstone_times.clear()


def _expire_stale_browser_jobs(now: Optional[float] = None) -> int:
    """30분 동안 활동이 없는 미완료 브라우저 작업을 취소하고 정리한다.

    주기 실행과 단위 시험이 같은 함수를 사용하도록 한 번의 순회로 분리했다.
    반환값은 이번에 정리한 작업 수다.
    """
    current = time.monotonic() if now is None else now
    expired: list[tuple[Path, Optional[storage.ReservedResultPath]]] = []

    for job in list(jobs.values()):
        with job.lock:
            if (
                job.processor != PROCESSOR_BROWSER
                or job.status in FINISHED_STATUSES
                or job.browser_finishing
                or current - job.last_activity < BROWSER_JOB_TTL_SECONDS
            ):
                continue

            job.stop_requested = True
            job.status = STATUS_CANCELED
            job.message = "30분 동안 작업 소식이 없어 자동으로 정리했습니다."
            _clear_browser_buffers_locked(job)
            reservation = job.result_reservation
            job.result_reservation = None
            expired.append((job.input_path, reservation))

    # 파일 삭제는 잠금 밖에서 한다. 느린 저장장치 때문에 다른 상태 요청까지
    # 기다리게 하지 않으며, 실행 중인 조립 작업의 finally가 한 번 더 정리한다.
    for input_path, reservation in expired:
        _remove_file(input_path)
        if reservation is not None:
            _remove_file(reservation.part_path)
            _remove_file(reservation.final_path)

    return len(expired)


def _save_text_exports(job: Job, summary: OcrSummary) -> None:
    """읽어 낸 글을 고른 형식으로 결과 PDF 옆에 저장한다.

    PDF는 이미 무사히 저장된 뒤에 부르는 함수다. 그래서 여기서 문제가 생겨도
    변환 자체를 실패로 돌리지 않고, 화면에 알리기만 한다.
    """
    if not job.options.export_formats or job.output_path is None:
        return

    written: dict[str, Path] = {}
    try:
        for export_format in job.options.export_formats:
            path = storage.reserve_sidecar_path(job.output_path, export_format)
            written[export_format] = path
            write_text_export(path, summary.pages_text, export_format)
    except Exception as exc:
        # 만들다 만 파일은 남기지 않는다. 반쪽짜리 결과를 완성본으로 오해하지 않도록.
        for path in written.values():
            _remove_file(path)
        with job.lock:
            job.export_error = f"{type(exc).__name__}: {exc}"
        return

    with job.lock:
        job.export_paths = written


def _process(job: Job) -> None:
    with job.lock:
        job.status = STATUS_RUNNING
        job.message = "글자를 읽는 중입니다."

    def on_progress(done: int, total: int) -> None:
        job.done_pages = done
        job.total_pages = total
        job.message = f"{total}쪽 중 {done}쪽을 읽었습니다."

    try:
        with job.lock:
            if job.stop_requested:
                job.status = STATUS_CANCELED
                job.message = "사용자 요청으로 중지했습니다."
                return

        # 이름 확인과 빈 예약 파일 생성을 한 동작으로 처리한다. 완성본은 같은
        # 폴더의 .part에 만든 뒤 마지막 순간에 예약 파일과 교체한다.
        reservation = storage.reserve_result_path(job.original_name)
        with job.lock:
            job.output_path = reservation.final_path
            job.result_reservation = reservation
            if job.stop_requested:
                job.status = STATUS_CANCELED
                job.message = "사용자 요청으로 중지했습니다."
                return

        summary = run_ocr(
            job.input_path,
            reservation.part_path,
            job.options,
            on_progress=on_progress,
            should_stop=lambda: job.stop_requested,
        )
        job.summary = summary

        committed = False
        with job.lock:
            if summary.stopped or job.stop_requested:
                job.status = STATUS_CANCELED
                job.message = (
                    f"사용자 요청으로 중지했습니다. "
                    f"({summary.total_pages}쪽 중 {summary.done_pages}쪽까지 읽음)"
                )
            else:
                storage.commit_reserved_result(reservation)
                job.result_reservation = None
                committed = True

        if committed:
            # 글자 파일 저장은 잠금 밖에서 한다. 그동안 진행 상황 조회가 막히지
            # 않게 하고, 완료를 알리기 전에 내려받을 파일을 모두 갖춰 두기 위해서다.
            _save_text_exports(job, summary)
            with job.lock:
                job.status = STATUS_DONE
                job.message = (
                    "변환은 끝났지만 글자 파일을 저장하지 못했습니다."
                    if job.export_error else "변환이 완료되었습니다."
                )

    except Exception as exc:
        with job.lock:
            if job.stop_requested:
                job.status = STATUS_CANCELED
                job.message = "사용자 요청으로 중지했습니다."
            else:
                job.status = STATUS_FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "변환이 실패했습니다. 아래 내용을 확인해 주세요."

    finally:
        if job.status != STATUS_DONE:
            _discard_result_reservation(job)
        with job.lock:
            job.worker_future = None
        # 올린 원본은 변환이 끝나면 남겨 둘 이유가 없다.
        _remove_file(job.input_path)


def _process_browser(job: Job) -> None:
    """브라우저가 읽어 보낸 글자를 원본 PDF에 얹는다."""
    with job.lock:
        if job.stop_requested:
            job.status = STATUS_CANCELED
            job.message = "사용자 요청으로 중지했습니다."
            pages = None
        else:
            job.status = STATUS_RUNNING
            job.message = "인식한 글자를 PDF에 넣는 중입니다."
            pages = dict(job.browser_results)

    def on_progress(done: int, total: int) -> None:
        # 글자 인식 진행률은 이미 100%이므로 숫자를 되돌리지 않고 문구만 바꾼다.
        with job.lock:
            if not job.stop_requested:
                job.message = f"{total}쪽 중 {done}쪽의 PDF를 마무리했습니다."
                job.last_activity = time.monotonic()

    try:
        if pages is None:
            return

        reservation = storage.reserve_result_path(job.original_name)
        with job.lock:
            job.output_path = reservation.final_path
            job.result_reservation = reservation
            if job.stop_requested:
                job.status = STATUS_CANCELED
                job.message = "사용자 요청으로 중지했습니다."
                return

        summary = build_pdf_from_browser_results(
            job.input_path,
            reservation.part_path,
            job.options,
            pages,
            on_progress=on_progress,
            should_stop=lambda: job.stop_requested,
        )
        # 모델 준비, 쪽별 인식, 서버 PDF 조립까지 사용자가 실제로 기다린 시간이다.
        summary.elapsed = time.perf_counter() - job.started_at
        job.summary = summary

        committed = False
        with job.lock:
            if summary.stopped or job.stop_requested:
                job.status = STATUS_CANCELED
                job.message = (
                    f"사용자 요청으로 중지했습니다. "
                    f"({summary.total_pages}쪽 중 {summary.done_pages}쪽까지 PDF를 마무리함)"
                )
            else:
                storage.commit_reserved_result(reservation)
                job.result_reservation = None
                committed = True

        if committed:
            _save_text_exports(job, summary)
            with job.lock:
                job.status = STATUS_DONE
                job.message = (
                    "변환은 끝났지만 글자 파일을 저장하지 못했습니다."
                    if job.export_error else "변환이 완료되었습니다."
                )

    except Exception as exc:
        with job.lock:
            if job.stop_requested:
                job.status = STATUS_CANCELED
                job.message = "사용자 요청으로 중지했습니다."
            else:
                job.status = STATUS_FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "변환이 실패했습니다. 아래 내용을 확인해 주세요."

    finally:
        if job.status != STATUS_DONE:
            _discard_result_reservation(job)
        with job.lock:
            _clear_browser_buffers_locked(job)
            job.browser_finishing = False
            job.worker_future = None
        _remove_file(job.input_path)


def _job_payload(job: Job) -> dict:
    """화면으로 보낼 진행 상황 한 덩어리."""
    percent = 0
    if job.total_pages:
        percent = round(job.done_pages / job.total_pages * 100)

    payload = {
        "id": job.id,
        "processor": job.processor,
        "status": job.status,
        "message": job.message,
        "done_pages": job.done_pages,
        "total_pages": job.total_pages,
        "percent": percent,
        "finished": job.status in FINISHED_STATUSES,
        "error": job.error,
    }

    if job.summary is not None and job.status == STATUS_DONE and job.output_path is not None:
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
            # 서버 어디에 저장됐는지 화면에 알려 준다.
            "saved_name": job.output_path.name,
            "saved_dir": str(storage.RESULT_DIR),
            # 함께 만든 글자 파일들
            "exports": [
                {"format": export_format, "name": path.name}
                for export_format, path in sorted(job.export_paths.items())
            ],
            "export_error": job.export_error,
            "rotated_by": s.rotated_by,
            "deskewed_pages": s.deskewed_pages,
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
# ─────────────────────────────────────────────────────────────────────────────
# 브라우저 그래픽카드 시험 (실현 가능성 확인용)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/gpu-test")
async def gpu_test_page(request: Request):
    return templates.TemplateResponse(request, "gpu_test.html")


@app.get("/api/models/{kind}")
async def get_model(kind: str):
    """인식 모델 파일을 브라우저에 내려준다.

    브라우저가 그래픽카드로 계산하려면 모델 파일 자체가 필요하다.
    파일이 크므로(수십 MB) 브라우저가 한 번 받아 두고 다시 쓰도록 캐시를 길게 준다.
    """
    from app import ocr_engine

    models_dir = Path(ocr_engine.__file__).resolve()
    try:
        import rapidocr
        models_dir = Path(rapidocr.__file__).resolve().parent / "models"
    except ImportError:
        raise HTTPException(500, "인식 모델 폴더를 찾지 못했습니다.")

    # 이름을 그대로 파일 경로에 쓰지 않는다. 정해진 것만 받는다.
    names = {
        "det": ["PP-OCRv6_det_small.onnx", "PP-OCRv6_det_tiny.onnx"],
        "rec": ["korean_PP-OCRv5_rec_mobile.onnx"],
    }
    if kind not in names:
        raise HTTPException(404, "그런 모델은 없습니다.")

    for name in names[kind]:
        path = models_dir / name
        if path.is_file():
            return FileResponse(
                path,
                media_type="application/octet-stream",
                headers={"Cache-Control": "public, max-age=604800"},   # 7일
            )
    raise HTTPException(404, "모델 파일을 찾지 못했습니다. 먼저 한 번 변환을 실행해 주세요.")


@app.get("/api/storage/files")
async def storage_files() -> dict:
    """보관함('원본' 폴더)에 있는 PDF 목록을 돌려준다."""
    files = storage.list_source_files()
    return {
        "source_dir": str(storage.SOURCE_DIR),
        "result_dir": str(storage.RESULT_DIR),
        "files": [
            {"name": f.name, "size": f.size, "modified": f.modified, "relative": f.relative}
            for f in files
        ],
    }


async def _copy_upload(upload: UploadFile, target: Path) -> None:
    """큰 업로드를 한꺼번에 메모리에 올리지 않고 조금씩 저장한다."""
    with target.open("wb") as output:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)


def _parse_export_formats(raw: str) -> tuple[str, ...]:
    """화면에서 쉼표로 이어 보낸 글자 파일 형식을 확인해 정리한다."""
    picked: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in EXPORT_FORMATS:
            raise HTTPException(400, "글자 파일 형식 값이 올바르지 않습니다.")
        if name not in picked:
            picked.append(name)

    if "docx" in picked and importlib.util.find_spec("docx") is None:
        # 변환을 다 끝낸 뒤에 실패를 알리면 시간이 아깝다. 시작 전에 막는다.
        raise HTTPException(
            400,
            "이 서버에는 docx를 만드는 부품(python-docx)이 설치되어 있지 않습니다. "
            "txt나 md를 골라 주세요.",
        )
    return tuple(picked)


def _created_job_payload(job: Job) -> dict:
    """작업 생성 응답의 기존 모양을 그대로 유지한다."""
    return {
        "id": job.id,
        "processor": job.processor,
        "total_pages": job.total_pages,
        "original_name": job.original_name,
        "has_text_layer": job.has_text_layer,
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile | None = None,
    source_path: str = Form(""),
    dpi: int = Form(300),
    text_score: float = Form(0.5),
    mode: str = Form(MODE_KEEP),
    processor: str = Form(PROCESSOR_SERVER),
    # 기본값을 Form(...)이 아니라 순수 파이썬 값으로 둔다. 이렇게 해야 이 함수를
    # 그대로 불러 시험할 때도 Form 객체가 아닌 진짜 기본값이 들어온다.
    export_formats: Annotated[str, Form()] = "",
    auto_rotate: Annotated[bool, Form()] = True,
    deskew: Annotated[bool, Form()] = False,
    request_id: Annotated[str, Form()] = "",
) -> dict:
    if dpi not in ALLOWED_DPI:
        raise HTTPException(400, "해상도는 150, 200, 300 중에서 골라 주세요.")
    if not math.isfinite(text_score) or not 0.0 <= text_score <= 1.0:
        raise HTTPException(400, "인식 기준 점수는 0과 1 사이여야 합니다.")
    if mode not in (MODE_KEEP, MODE_REBUILD):
        raise HTTPException(400, "변환 방식 값이 올바르지 않습니다.")
    if processor not in ALLOWED_PROCESSORS:
        raise HTTPException(400, "글자를 읽을 장소 값이 올바르지 않습니다.")
    if processor == PROCESSOR_BROWSER and mode != MODE_KEEP:
        raise HTTPException(400, "브라우저 처리는 원본 그대로 두기 방식만 지원합니다.")
    if processor == PROCESSOR_BROWSER:
        # 브라우저 방식은 접속한 PC가 글자를 읽으므로 서버 쪽 보정 단계를 거치지 않는다.
        # 화면에서도 잠가 두지만, 기본값이 켜짐이라 오류로 막지 않고 조용히 끈다.
        auto_rotate = False
        deskew = False
    if deskew and mode != MODE_REBUILD:
        raise HTTPException(
            400,
            "미세 기울기 보정은 '용량 줄이기' 방식에서만 쓸 수 있습니다. "
            "페이지를 다시 그려야 하기 때문입니다.",
        )

    picked_formats = _parse_export_formats(export_formats)
    if request_id and not _valid_request_id(request_id):
        raise HTTPException(400, "요청 키는 32자리 소문자 영문·숫자 형식이어야 합니다.")

    request_claimed = False
    if request_id:
        with _request_jobs_lock:
            _expire_request_tombstones_locked(time.monotonic())
            if request_id in _request_jobs:
                request_state = _request_jobs[request_id]
                if request_state is _REQUEST_CREATING:
                    raise HTTPException(425, "같은 변환 요청을 아직 만들고 있습니다.")
                if request_state is _REQUEST_CANCELED:
                    raise HTTPException(409, "이미 취소한 변환 요청입니다.")
                if isinstance(request_state, str):
                    existing_job = jobs.get(request_state)
                    if existing_job is not None:
                        return _created_job_payload(existing_job)
                # 작업 표만 사라진 비정상 상태라면 오래된 요청 키도 함께 놓아준다.
                _request_jobs.pop(request_id, None)
            _request_jobs[request_id] = _REQUEST_CREATING
            request_claimed = True

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}.pdf"
    request_registered = False

    try:
        try:
            if source_path:
                # 보관함에 있는 파일을 고른 경우.
                # 지정한 폴더 밖으로 빠져나가지 못하게 반드시 확인한다.
                try:
                    picked = storage.resolve_inside(storage.SOURCE_DIR, source_path)
                except ValueError as exc:
                    raise HTTPException(400, str(exc))
                if not picked.is_file() or picked.suffix.lower() != ".pdf":
                    raise HTTPException(404, "보관함에서 그 PDF를 찾지 못했습니다.")

                original_name = picked.name
                # 원본은 그대로 두고 사본으로 작업한다. copyfile은 파일을 나누어 복사한다.
                await asyncio.to_thread(shutil.copyfile, picked, input_path)

            elif file is not None:
                original_name = Path(file.filename or "문서.pdf").name
                if not original_name.lower().endswith(".pdf"):
                    raise HTTPException(400, "PDF 파일만 올릴 수 있습니다.")
                await _copy_upload(file, input_path)

            else:
                raise HTTPException(400, "변환할 PDF를 올리거나 보관함에서 골라 주세요.")
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(500, "PDF 파일을 작업 폴더로 복사하지 못했습니다.") from exc

        # 진짜 PDF인지, 그리고 몇 쪽짜리인지 미리 확인한다.
        # 여기서 걸러야 변환을 시작한 뒤에 실패하는 일이 없다.
        try:
            with pymupdf.open(input_path) as doc:
                page_count = doc.page_count
                if doc.needs_pass:
                    raise HTTPException(400, "암호가 걸린 PDF는 변환할 수 없습니다.")
                # 이미 글자가 들어 있는 PDF인지도 이때 함께 살펴본다. 막지는 않고
                # 화면에서 알려 주기만 한다.
                has_text_layer = document_has_text_layer(doc)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                400,
                "PDF 파일을 열 수 없습니다. 파일이 손상되었는지 확인해 주세요.",
            ) from exc

        if page_count == 0:
            raise HTTPException(400, "쪽이 하나도 없는 PDF입니다.")
        if processor == PROCESSOR_BROWSER and page_count > BROWSER_MAX_PAGES:
            raise HTTPException(
                400,
                f"브라우저 처리는 {BROWSER_MAX_PAGES}쪽 이하의 PDF만 지원합니다.",
            )

        job = Job(
            id=job_id,
            original_name=original_name,
            input_path=input_path,
            options=OcrOptions(
                dpi=dpi,
                text_score=text_score,
                mode=mode,
                export_formats=picked_formats,
                auto_rotate=auto_rotate,
                deskew=deskew,
            ),
            processor=processor,
            total_pages=page_count,
            has_text_layer=has_text_layer,
        )
        if processor == PROCESSOR_BROWSER:
            job.status = STATUS_RUNNING
            job.message = "브라우저에서 첫 쪽을 읽을 준비를 하는 중입니다."

        if request_id:
            # 파일 검사가 끝난 바로 이 지점에서 중지 표식과 작업 등록을 한 잠금으로
            # 처리한다. 그래야 중지가 두 동작 사이에 끼어 작업을 되살리지 못한다.
            with _request_jobs_lock:
                request_state = _request_jobs.get(request_id)
                if request_state is _REQUEST_CANCELED:
                    _request_jobs.pop(request_id, None)
                    _request_tombstone_times.pop(request_id, None)
                    raise HTTPException(409, "변환 시작 요청이 취소되었습니다.")
                if request_state is not _REQUEST_CREATING:
                    raise HTTPException(409, "변환 시작 요청 상태를 확인할 수 없습니다.")
                jobs[job_id] = job
                _request_jobs[request_id] = job_id
        else:
            jobs[job_id] = job

        if processor == PROCESSOR_SERVER:
            # Future 저장까지 같은 잠금에서 끝내야 대기열에 있는 서버 작업도
            # 중지 버튼으로 즉시 취소할 수 있다.
            with job.lock:
                job.worker_future = _worker.submit(_process, job)

        request_registered = True
        return _created_job_payload(job)

    finally:
        if not request_registered:
            jobs.pop(job_id, None)
            _remove_file(input_path)
            if request_claimed:
                with _request_jobs_lock:
                    request_state = _request_jobs.get(request_id)
                    if (
                        request_state is _REQUEST_CREATING
                        or request_state == job_id
                    ):
                        _request_jobs.pop(request_id, None)
                        _request_tombstone_times.pop(request_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# 브라우저 OCR용 페이지와 결과 받기
# ─────────────────────────────────────────────────────────────────────────────
def _browser_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")
    if job.processor != PROCESSOR_BROWSER:
        raise HTTPException(409, "서버에서 처리하는 작업에는 사용할 수 없는 주소입니다.")
    return job


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(400, f"{label} 값은 숫자여야 합니다.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise HTTPException(400, f"{label} 값이 너무 크거나 올바르지 않습니다.") from exc
    if not math.isfinite(number):
        raise HTTPException(400, f"{label} 값이 너무 크거나 올바르지 않습니다.")
    return number


def _validate_browser_page_result(
    payload,
    expected_image: tuple[int, int, int],
) -> BrowserOcrPage:
    """브라우저가 보낸 한 쪽 결과를 빠짐없이 검사한다."""
    if not isinstance(payload, dict):
        raise HTTPException(400, "한 쪽 인식 결과의 모양이 올바르지 않습니다.")

    render_width = payload.get("render_width")
    render_height = payload.get("render_height")
    rotation = payload.get("rotation")
    expected_width, expected_height, expected_rotation = expected_image

    if isinstance(render_width, bool) or not isinstance(render_width, int):
        raise HTTPException(400, "페이지 이미지 가로 크기가 올바르지 않습니다.")
    if isinstance(render_height, bool) or not isinstance(render_height, int):
        raise HTTPException(400, "페이지 이미지 세로 크기가 올바르지 않습니다.")
    if render_width != expected_width or render_height != expected_height:
        raise HTTPException(400, "인식한 페이지 이미지 크기가 서버에서 보낸 것과 다릅니다.")
    if isinstance(rotation, bool) or not isinstance(rotation, int):
        raise HTTPException(400, "페이지 회전값이 올바르지 않습니다.")
    if rotation != expected_rotation or rotation not in (0, 90, 180, 270):
        raise HTTPException(400, "페이지 회전값이 원본 PDF와 다릅니다.")

    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        raise HTTPException(400, "인식한 글줄 목록이 올바르지 않습니다.")
    if len(raw_lines) > BROWSER_MAX_LINES_PER_PAGE:
        raise HTTPException(
            400,
            f"한 쪽에는 글줄을 {BROWSER_MAX_LINES_PER_PAGE}개까지만 보낼 수 있습니다.",
        )

    lines: list[BrowserOcrLine] = []
    for line_index, raw_line in enumerate(raw_lines, start=1):
        label = f"{line_index}번째 글줄"
        if not isinstance(raw_line, dict):
            raise HTTPException(400, f"{label}의 모양이 올바르지 않습니다.")

        text = raw_line.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, f"{label}에 읽은 글자가 없습니다.")
        if len(text) > BROWSER_MAX_TEXT_LENGTH:
            raise HTTPException(
                400,
                f"{label}은 {BROWSER_MAX_TEXT_LENGTH}자를 넘을 수 없습니다.",
            )
        if "\x00" in text:
            raise HTTPException(400, f"{label}에 넣을 수 없는 글자가 있습니다.")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HTTPException(400, f"{label}의 글자 형식이 올바르지 않습니다.") from exc

        score = _finite_number(raw_line.get("score"), f"{label}의 점수")
        if not 0.0 <= score <= 1.0:
            raise HTTPException(400, f"{label}의 점수는 0과 1 사이여야 합니다.")

        raw_box = raw_line.get("box")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise HTTPException(400, f"{label}의 위치는 네 꼭짓점이어야 합니다.")

        box: list[tuple[float, float]] = []
        for point_index, raw_point in enumerate(raw_box, start=1):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise HTTPException(400, f"{label}의 {point_index}번째 꼭짓점이 올바르지 않습니다.")
            x = _finite_number(raw_point[0], f"{label}의 가로 좌표")
            y = _finite_number(raw_point[1], f"{label}의 세로 좌표")
            if not 0.0 <= x <= render_width or not 0.0 <= y <= render_height:
                raise HTTPException(400, f"{label}의 위치가 페이지 이미지 밖에 있습니다.")
            box.append((x, y))

        if max(point[0] for point in box) <= min(point[0] for point in box):
            raise HTTPException(400, f"{label}의 가로 길이가 0입니다.")
        if max(point[1] for point in box) <= min(point[1] for point in box):
            raise HTTPException(400, f"{label}의 세로 길이가 0입니다.")

        lines.append(BrowserOcrLine(box=tuple(box), text=text, score=score))

    return BrowserOcrPage(
        render_width=render_width,
        render_height=render_height,
        rotation=rotation,
        lines=tuple(lines),
    )


async def _read_browser_result(request: Request):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(415, "쪽 결과는 JSON 형식으로 보내 주세요.")

    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > BROWSER_MAX_RESULT_BYTES:
            raise HTTPException(413, "한 쪽 인식 결과가 너무 큽니다.")
        chunks.append(chunk)
    body = b"".join(chunks)
    try:
        return json.loads(body), size
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "쪽 결과 JSON을 읽을 수 없습니다.") from exc


def _track_browser_render_task(task: asyncio.Task) -> None:
    """요청이 취소된 뒤에도 렌더 작업이 끝날 때까지 강하게 참조한다."""
    _browser_render_tasks.add(task)

    def forget(completed: asyncio.Task) -> None:
        _browser_render_tasks.discard(completed)
        if not completed.cancelled():
            # 요청이 먼저 끊긴 뒤 생긴 예외도 회수해 비동기 경고가 남지 않게 한다.
            with suppress(Exception):
                completed.exception()

    task.add_done_callback(forget)


async def _render_browser_page_with_slot(job: Job, page_index: int):
    """실제 렌더 스레드가 끝날 때까지 페이지 렌더 자리를 소유한다."""
    async with _browser_render_semaphore:
        with job.lock:
            if job.status in FINISHED_STATUSES or job.stop_requested or job.browser_finishing:
                raise HTTPException(409, "이미 끝났거나 중지된 작업입니다.")
            job.last_activity = time.monotonic()

        return await asyncio.to_thread(
            render_browser_page,
            job.input_path,
            page_index,
            BROWSER_PAGE_MAX_SIDE,
        )


@app.get("/api/jobs/{job_id}/pages/{page_index}/image")
async def browser_page_image(job_id: str, page_index: int):
    job = _browser_job(job_id)
    if page_index < 0 or page_index >= job.total_pages:
        raise HTTPException(404, "해당 PDF 쪽을 찾을 수 없습니다.")
    with job.lock:
        if job.status in FINISHED_STATUSES or job.browser_finishing:
            raise HTTPException(409, "이미 끝났거나 마무리 중인 작업입니다.")
        job.last_activity = time.monotonic()

    # 요청이 취소돼도 child task가 실제 렌더 종료까지 세마포어를 쥔다. 다음 요청은
    # 스레드를 미리 만들지 않고 비동기 대기열에서 기다리므로 메모리 제한이 유지된다.
    render_task = asyncio.create_task(_render_browser_page_with_slot(job, page_index))
    _track_browser_render_task(render_task)
    try:
        image = await asyncio.shield(render_task)
    except HTTPException:
        raise
    except (IndexError, ValueError) as exc:
        with job.lock:
            stopped = job.status in FINISHED_STATUSES or job.stop_requested
        if stopped:
            raise HTTPException(409, "중지된 작업입니다.") from exc
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        with job.lock:
            stopped = job.status in FINISHED_STATUSES or job.stop_requested
        if stopped:
            raise HTTPException(409, "중지된 작업입니다.") from exc
        raise HTTPException(500, "PDF 쪽 이미지를 만들지 못했습니다.") from exc

    # 렌더링과 중지가 겹쳐도 취소 뒤에 이미지 정보가 다시 생기지 않도록
    # 확인과 상태 저장을 같은 잠금 안에서 처리한다.
    with job.lock:
        if job.status in FINISHED_STATUSES or job.stop_requested or job.browser_finishing:
            rejected = True
            # finish는 같은 원본으로 PDF를 조립하고 있으므로 지우면 안 된다.
            # 중지·실패일 때만 렌더링이 놓은 파일을 다시 정리한다.
            cleanup_input = (
                job.stop_requested
                or job.status in FINISHED_STATUSES
            )
        else:
            rejected = False
            cleanup_input = False
            job.browser_page_images[page_index] = (
                image.width,
                image.height,
                image.rotation,
            )
            job.message = (
                f"{job.total_pages}쪽 중 {page_index + 1}쪽을 브라우저에서 읽는 중입니다."
            )
            job.last_activity = time.monotonic()

    if rejected:
        if cleanup_input:
            _remove_file(job.input_path)
        raise HTTPException(409, "이미 끝났거나 마무리 중인 작업입니다.")

    return Response(
        content=image.content,
        media_type="image/png",
        headers={
            "Cache-Control": "private, no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "X-PDFOCR-Width": str(image.width),
            "X-PDFOCR-Height": str(image.height),
            "X-PDFOCR-Rotation": str(image.rotation),
        },
    )


@app.put("/api/jobs/{job_id}/pages/{page_index}")
async def save_browser_page(job_id: str, page_index: int, request: Request) -> dict:
    job = _browser_job(job_id)
    if page_index < 0 or page_index >= job.total_pages:
        raise HTTPException(404, "해당 PDF 쪽을 찾을 수 없습니다.")

    payload, raw_size = await _read_browser_result(request)
    with job.lock:
        if job.status in FINISHED_STATUSES or job.stop_requested:
            raise HTTPException(409, "이미 끝났거나 중지된 작업입니다.")

        expected_image = job.browser_page_images.get(page_index)
        if expected_image is None:
            raise HTTPException(409, "먼저 이 PDF 쪽 이미지를 받아 주세요.")

        result = _validate_browser_page_result(payload, expected_image)

        # 응답만 유실되어 같은 PUT이 다시 온 경우에는 이미 성공한 요청으로 본다.
        # 내용이 달라진 재전송은 어느 것이 맞는지 결정할 수 없으므로 거부한다.
        if page_index in job.browser_results:
            if job.browser_results[page_index] == result:
                job.last_activity = time.monotonic()
                return _job_payload(job)
            raise HTTPException(409, "이미 받은 PDF 쪽과 다른 결과가 다시 왔습니다.")

        if job.browser_finishing:
            raise HTTPException(409, "PDF를 마무리하기 시작해 결과를 더 받을 수 없습니다.")
        if page_index != len(job.browser_results):
            raise HTTPException(409, "PDF 쪽 결과를 앞쪽부터 차례대로 보내 주세요.")

        received_lines = sum(len(page.lines) for page in job.browser_results.values())
        if received_lines + len(result.lines) > BROWSER_MAX_LINES_PER_JOB:
            raise HTTPException(
                400,
                f"한 작업에는 글줄을 {BROWSER_MAX_LINES_PER_JOB}개까지만 보낼 수 있습니다.",
            )
        if job.browser_result_bytes + raw_size > BROWSER_MAX_RESULT_BYTES_PER_JOB:
            raise HTTPException(
                413,
                "한 작업에서 받은 인식 결과가 32MB를 넘었습니다.",
            )
        job.browser_results[page_index] = result
        job.browser_result_bytes += raw_size
        job.done_pages = len(job.browser_results)
        job.message = f"{job.total_pages}쪽 중 {job.done_pages}쪽을 읽었습니다."
        job.last_activity = time.monotonic()

    return _job_payload(job)


@app.post("/api/jobs/{job_id}/finish")
async def finish_browser_job(job_id: str) -> dict:
    job = _browser_job(job_id)
    with job.lock:
        if job.status == STATUS_DONE:
            return _job_payload(job)
        if job.status in (STATUS_FAILED, STATUS_CANCELED) or job.stop_requested:
            raise HTTPException(409, "이미 끝났거나 중지된 작업입니다.")
        if job.browser_finishing:
            job.last_activity = time.monotonic()
            return _job_payload(job)
        if len(job.browser_results) != job.total_pages:
            raise HTTPException(
                409,
                f"전체 {job.total_pages}쪽 중 {len(job.browser_results)}쪽 결과만 받았습니다.",
            )

        job.browser_finishing = True
        job.message = "인식한 글자를 PDF에 넣을 차례를 기다리는 중입니다."
        job.last_activity = time.monotonic()
        # Future 저장까지 같은 잠금에서 끝내야 바로 뒤의 중지가 대기 작업을
        # 놓치지 않는다. 작업 스레드는 이 잠금이 풀린 뒤 상태를 읽기 시작한다.
        try:
            job.worker_future = _worker.submit(_process_browser, job)
        except Exception:
            job.browser_finishing = False
            raise
    return _job_payload(job)


# ─────────────────────────────────────────────────────────────────────────────
# 진행 상황 실시간 전달 (SSE)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """연결이 잠시 끊긴 뒤에도 현재 작업 상태를 다시 확인할 수 있게 한다."""
    job = jobs.get(job_id)
    if job is None:
        # 생성 POST의 응답 자체가 유실되면 화면은 서버가 만든 job_id를 아직 모른다.
        # 이때에만 요청 키를 같은 조회 주소의 별칭으로 받아 실제 작업을 찾아 준다.
        with _request_jobs_lock:
            _expire_request_tombstones_locked(time.monotonic())
            request_state = _request_jobs.get(job_id)
        if request_state is _REQUEST_CREATING:
            raise HTTPException(425, "같은 변환 요청을 아직 만들고 있습니다.")
        if request_state is _REQUEST_CANCELED:
            raise HTTPException(404, "취소된 변환 시작 요청입니다.")
        if isinstance(request_state, str):
            job = jobs.get(request_state)
    if job is None:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")
    return _job_payload(job)


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
        # 생성 응답을 받기 전에도 화면이 알고 있는 request_id로 중지할 수 있다.
        # 복사 중이라면 표식만 남기고, 복사가 끝난 create_job이 파일을 정리한다.
        with _request_jobs_lock:
            _expire_request_tombstones_locked(time.monotonic())
            request_state = _request_jobs.get(job_id)
            if request_state is _REQUEST_CREATING:
                _request_jobs[job_id] = _REQUEST_CANCELED
                return {
                    "status": STATUS_CANCELED,
                    "message": "변환 시작 요청을 취소했습니다.",
                }
            if request_state is _REQUEST_CANCELED:
                return {
                    "status": STATUS_CANCELED,
                    "message": "이미 취소한 변환 시작 요청입니다.",
                }
            if isinstance(request_state, str):
                job = jobs.get(request_state)
            elif request_state is None and _valid_request_id(job_id):
                # 큰 multipart 본문을 FastAPI가 아직 읽는 중이면 create_job이
                # 시작되기 전에도 stop이 먼저 도착할 수 있다. 이 tombstone을
                # 남겨 두면 뒤늦은 create가 작업을 되살리지 못한다.
                _remember_request_tombstone_locked(job_id)
                return {
                    "status": STATUS_CANCELED,
                    "message": "변환 시작 요청을 취소했습니다.",
                }
        if job is None:
            raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")

    cleanup_now = False
    with job.lock:
        if job.status in FINISHED_STATUSES:
            return {"status": job.status, "message": "이미 끝난 작업입니다."}

        job.stop_requested = True
        if job.processor == PROCESSOR_BROWSER:
            future = job.worker_future
            can_finish_now = not job.browser_finishing
            if job.browser_finishing and future is not None and future.cancel():
                # 아직 작업 스레드가 시작하지 않았다면 기다릴 이유가 없다.
                can_finish_now = True
                job.browser_finishing = False
                job.worker_future = None

            if can_finish_now:
                job.status = STATUS_CANCELED
                job.message = "사용자 요청으로 중지했습니다."
                _clear_browser_buffers_locked(job)
                cleanup_now = True
            else:
                job.message = "중지하는 중입니다. 읽고 있던 쪽까지만 마무리합니다."
        else:
            future = job.worker_future
            if future is not None and future.cancel():
                job.worker_future = None
                job.status = STATUS_CANCELED
                job.message = "사용자 요청으로 중지했습니다."
                cleanup_now = True
            else:
                # 이미 실행 중인 서버 OCR은 현재 쪽까지만 안전하게 마무리한다.
                job.message = "중지하는 중입니다. 읽고 있던 쪽까지만 마무리합니다."

    if cleanup_now:
        _remove_file(job.input_path)
    return {"status": job.status, "message": job.message}


# ─────────────────────────────────────────────────────────────────────────────
# 결과 내려받기
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}/download")
async def download(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")
    if job.status != STATUS_DONE or job.output_path is None or not job.output_path.exists():
        raise HTTPException(409, "아직 내려받을 수 있는 결과가 없습니다.")

    # 한글 파일 이름이 깨지지 않도록 UTF-8로 적어 보낸다.
    download_name = f"{Path(job.original_name).stem}_Searchable.pdf"
    return FileResponse(
        job.output_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f"attachment; filename*=UTF-8''{quote(download_name)}",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/jobs/{job_id}/download/{export_format}")
async def download_text(job_id: str, export_format: str):
    """함께 만든 글자 파일(txt / md / docx)을 내려준다."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")
    if export_format not in EXPORT_FORMATS:
        raise HTTPException(404, "그런 형식의 글자 파일은 만들지 않습니다.")

    path = job.export_paths.get(export_format)
    if job.status != STATUS_DONE or path is None or not path.exists():
        raise HTTPException(409, "아직 내려받을 수 있는 글자 파일이 없습니다.")

    download_name = f"{Path(job.original_name).stem}_Searchable.{export_format}"
    return FileResponse(
        path,
        media_type=EXPORT_MEDIA_TYPES[export_format],
        headers={
            "Content-Disposition":
                f"attachment; filename*=UTF-8''{quote(download_name)}",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
