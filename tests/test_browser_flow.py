# -*- coding: utf-8 -*-
"""브라우저 OCR 작업 흐름 회귀 시험.

외부 시험 도구나 실제 OCR 모델은 쓰지 않는다. 작은 임시 PDF를 만들고 FastAPI
엔드포인트 함수를 직접 호출해, 브라우저 결과 수신과 PDF 조립만 빠르게 확인한다.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from unittest import mock

import pymupdf
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from starlette.requests import Request

from app import main, storage
from app.ocr_engine import (
    MODE_KEEP,
    BrowserOcrLine,
    BrowserOcrPage,
    OcrOptions,
    OcrSummary,
    build_pdf_from_browser_results,
    render_browser_page,
)


def _pdf_bytes(rotations: tuple[int, ...]) -> bytes:
    """지정한 회전값으로 작은 시험 PDF를 메모리에서 만든다."""
    document = pymupdf.open()
    try:
        for rotation in rotations:
            page = document.new_page(width=200, height=100)
            page.set_rotation(rotation)
        return document.tobytes()
    finally:
        document.close()


def _json_request(payload) -> Request:
    """TestClient 없이 JSON 요청 하나를 만든다."""
    body = json.dumps(payload, ensure_ascii=False, allow_nan=True).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "PUT",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1234),
        "server": ("test", 80),
    }
    return Request(scope, receive)


def _path_request(path: str) -> Request:
    """로그인 미들웨어 경계 시험용 GET 요청을 만든다."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("test", 1234),
        "server": ("test", 443),
    }
    return Request(scope)


def _payload_for(
    image_info: tuple[int, int, int],
    text: str,
    *,
    score: float = 0.99,
) -> dict:
    """페이지 안쪽에 충분히 넓은 정상 글상자를 만든다."""
    width, height, rotation = image_info
    x0, x1 = width * 0.05, width * 0.95
    y0, y1 = height * 0.40, height * 0.50
    return {
        "render_width": width,
        "render_height": height,
        "rotation": rotation,
        "lines": [{
            "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "text": text,
            "score": score,
        }],
    }


class BrowserJobFlowTests(unittest.IsolatedAsyncioTestCase):
    """브라우저 작업 API의 상태 변화와 임시 파일 정리를 확인한다."""

    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        upload_dir = root / "uploads"
        source_dir = root / "source"
        result_dir = root / "result"
        for directory in (upload_dir, source_dir, result_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.patchers = [
            mock.patch.object(main, "UPLOAD_DIR", upload_dir),
            mock.patch.object(storage, "SOURCE_DIR", source_dir),
            mock.patch.object(storage, "RESULT_DIR", result_dir),
            mock.patch.object(main._worker, "submit", side_effect=self._capture_submit),
            mock.patch.object(main, "_browser_render_semaphore", asyncio.Semaphore(1)),
            mock.patch.object(main, "_browser_render_tasks", set()),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        main.jobs.clear()
        with main._request_jobs_lock:
            main._request_jobs.clear()
            main._request_tombstone_times.clear()
        self.submitted: list[tuple[object, tuple, dict]] = []

    async def asyncTearDown(self) -> None:
        main.jobs.clear()
        with main._request_jobs_lock:
            main._request_jobs.clear()
            main._request_tombstone_times.clear()

    def _capture_submit(self, function, *args, **kwargs) -> Future:
        """뒤 작업을 바로 실행하지 않고 시험이 정한 시점까지 보관한다."""
        self.submitted.append((function, args, kwargs))
        future = Future()
        future.set_result(None)
        return future

    def _run_next_worker_job(self) -> None:
        function, args, kwargs = self.submitted.pop(0)
        function(*args, **kwargs)

    async def _create_browser_job(
        self,
        name: str = "sample.pdf",
        rotations: tuple[int, ...] = (0,),
        request_id: str = "",
    ) -> main.Job:
        upload = UploadFile(BytesIO(_pdf_bytes(rotations)), filename=name)
        try:
            response = await main.create_job(
                file=upload,
                source_path="",
                dpi=150,
                text_score=0.5,
                mode=MODE_KEEP,
                processor=main.PROCESSOR_BROWSER,
                request_id=request_id,
            )
        finally:
            await upload.close()
        return main.jobs[response["id"]]

    async def _prepare_page(self, job: main.Job, page_index: int, text: str) -> dict:
        image_response = await main.browser_page_image(job.id, page_index)
        self.assertEqual(image_response.media_type, "image/png")
        self.assertEqual(image_response.headers["cache-control"], "private, no-store")
        self.assertTrue(image_response.body.startswith(b"\x89PNG\r\n\x1a\n"))

        image_info = job.browser_page_images[page_index]
        payload = _payload_for(image_info, text)
        response = await main.save_browser_page(
            job.id,
            page_index,
            _json_request(payload),
        )
        return response

    async def test_normal_browser_flow_builds_searchable_pdf(self) -> None:
        job = await self._create_browser_job(rotations=(0, 0))

        first = await self._prepare_page(job, 0, "PAGE_ZERO")
        self.assertEqual(first["done_pages"], 1)
        self.assertEqual(first["percent"], 50)

        await main.browser_page_image(job.id, 1)
        second_payload = _payload_for(job.browser_page_images[1], "PAGE_ONE")
        low_score_line = copy.deepcopy(second_payload["lines"][0])
        low_score_line["text"] = "LOW_SCORE_MUST_NOT_EXIST"
        low_score_line["score"] = 0.1
        second_payload["lines"].append(low_score_line)
        second = await main.save_browser_page(job.id, 1, _json_request(second_payload))
        self.assertEqual(second["done_pages"], 2)
        self.assertEqual(second["percent"], 100)

        finish = await main.finish_browser_job(job.id)
        self.assertTrue(job.browser_finishing)
        self.assertEqual(finish["done_pages"], 2)
        self.assertIsNone(job.output_path)
        self.assertEqual(len(self.submitted), 1)

        duplicate_finish = await main.finish_browser_job(job.id)
        self.assertEqual(duplicate_finish["status"], main.STATUS_RUNNING)
        self.assertEqual(len(self.submitted), 1)

        await asyncio.to_thread(self._run_next_worker_job)

        self.assertEqual(job.status, main.STATUS_DONE)
        self.assertIsNotNone(job.output_path)
        self.assertTrue(job.output_path.is_file())
        self.assertEqual(list(job.output_path.parent.glob("*.part")), [])
        self.assertFalse(job.input_path.exists())
        self.assertEqual(job.browser_results, {})
        self.assertEqual(job.browser_page_images, {})
        self.assertEqual(job.browser_result_bytes, 0)
        self.assertEqual(job.summary.device, "WebGPU")
        self.assertEqual(job.summary.text_lines, 2)
        self.assertGreater(job.summary.elapsed, 0)

        with pymupdf.open(job.output_path) as document:
            self.assertIn("PAGE_ZERO", document[0].get_text())
            self.assertIn("PAGE_ONE", document[1].get_text())
            self.assertNotIn("LOW_SCORE_MUST_NOT_EXIST", document[1].get_text())

        finished_again = await main.finish_browser_job(job.id)
        self.assertEqual(finished_again["status"], main.STATUS_DONE)
        self.assertEqual(len(self.submitted), 0)

    async def test_duplicate_page_result_is_idempotent_but_changes_are_rejected(self) -> None:
        job = await self._create_browser_job()
        await main.browser_page_image(job.id, 0)
        payload = _payload_for(job.browser_page_images[0], "ONCE_ONLY")

        first = await main.save_browser_page(job.id, 0, _json_request(payload))
        received_bytes = job.browser_result_bytes
        duplicate = await main.save_browser_page(job.id, 0, _json_request(payload))

        self.assertEqual(first["done_pages"], 1)
        self.assertEqual(duplicate["done_pages"], 1)
        self.assertEqual(job.browser_result_bytes, received_bytes)

        changed = copy.deepcopy(payload)
        changed["lines"][0]["text"] = "DIFFERENT_TEXT"
        with self.assertRaises(HTTPException) as caught:
            await main.save_browser_page(job.id, 0, _json_request(changed))

        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(job.done_pages, 1)
        self.assertEqual(len(job.browser_results), 1)
        self.assertEqual(job.browser_results[0].lines[0].text, "ONCE_ONLY")

    async def test_create_request_id_returns_the_existing_job(self) -> None:
        request_id = "a" * 32
        job = await self._create_browser_job(request_id=request_id)

        duplicate = await main.create_job(
            file=None,
            source_path="",
            dpi=150,
            text_score=0.5,
            mode=MODE_KEEP,
            processor=main.PROCESSOR_BROWSER,
            request_id=request_id,
        )

        self.assertEqual(duplicate["id"], job.id)
        self.assertEqual(len(main.jobs), 1)
        recovered = await main.get_job_status(request_id)
        self.assertEqual(recovered["id"], job.id)

    async def test_invalid_create_request_id_is_rejected(self) -> None:
        for request_id in ("short", "A" * 32, "g" * 32):
            with self.subTest(request_id=request_id):
                with self.assertRaises(HTTPException) as caught:
                    await main.create_job(
                        file=None,
                        source_path="",
                        dpi=150,
                        text_score=0.5,
                        mode=MODE_KEEP,
                        processor=main.PROCESSOR_BROWSER,
                        request_id=request_id,
                    )
                self.assertEqual(caught.exception.status_code, 400)

    async def test_duplicate_create_while_upload_is_waiting_returns_425(self) -> None:
        request_id = "b" * 32
        upload = UploadFile(BytesIO(_pdf_bytes((0,))), filename="waiting.pdf")
        upload_started = asyncio.Event()
        release_upload = asyncio.Event()
        original_copy = main._copy_upload

        async def paused_copy(source, target):
            upload_started.set()
            await release_upload.wait()
            await original_copy(source, target)

        try:
            with mock.patch.object(main, "_copy_upload", side_effect=paused_copy):
                first = asyncio.create_task(main.create_job(
                    file=upload,
                    source_path="",
                    dpi=150,
                    text_score=0.5,
                    mode=MODE_KEEP,
                    processor=main.PROCESSOR_BROWSER,
                    request_id=request_id,
                ))
                await upload_started.wait()
                with self.assertRaises(HTTPException) as caught:
                    await main.create_job(
                        file=None,
                        source_path="",
                        dpi=150,
                        text_score=0.5,
                        mode=MODE_KEEP,
                        processor=main.PROCESSOR_BROWSER,
                        request_id=request_id,
                    )
                self.assertEqual(caught.exception.status_code, 425)
                release_upload.set()
                created = await first
        finally:
            release_upload.set()
            await upload.close()

        self.assertIn(created["id"], main.jobs)

    async def test_stop_request_id_while_upload_waits_prevents_job_creation(self) -> None:
        request_id = "c" * 32
        upload = UploadFile(BytesIO(_pdf_bytes((0,))), filename="cancel-waiting.pdf")
        upload_started = asyncio.Event()
        release_upload = asyncio.Event()
        original_copy = main._copy_upload
        first = None

        async def paused_copy(source, target):
            upload_started.set()
            await release_upload.wait()
            await original_copy(source, target)

        try:
            with mock.patch.object(main, "_copy_upload", side_effect=paused_copy):
                first = asyncio.create_task(main.create_job(
                    file=upload,
                    source_path="",
                    dpi=150,
                    text_score=0.5,
                    mode=MODE_KEEP,
                    processor=main.PROCESSOR_BROWSER,
                    request_id=request_id,
                ))
                await upload_started.wait()

                stopped = await main.stop_job(request_id)
                self.assertEqual(stopped["status"], main.STATUS_CANCELED)

                release_upload.set()
                with self.assertRaises(HTTPException) as caught:
                    await first
                self.assertEqual(caught.exception.status_code, 409)
        finally:
            release_upload.set()
            if first is not None and not first.done():
                await asyncio.gather(first, return_exceptions=True)
            await upload.close()

        self.assertEqual(main.jobs, {})
        self.assertEqual(list(main.UPLOAD_DIR.iterdir()), [])
        with main._request_jobs_lock:
            self.assertNotIn(request_id, main._request_jobs)

    async def test_stop_request_id_before_create_leaves_cancel_tombstone(self) -> None:
        request_id = "9" * 32

        stopped = await main.stop_job(request_id)
        self.assertEqual(stopped["status"], main.STATUS_CANCELED)

        upload = UploadFile(BytesIO(_pdf_bytes((0,))), filename="late-create.pdf")
        try:
            with self.assertRaises(HTTPException) as caught:
                await main.create_job(
                    file=upload,
                    source_path="",
                    dpi=150,
                    text_score=0.5,
                    mode=MODE_KEEP,
                    processor=main.PROCESSOR_BROWSER,
                    request_id=request_id,
                )
        finally:
            await upload.close()

        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(main.jobs, {})
        with main._request_jobs_lock:
            self.assertIs(main._request_jobs[request_id], main._REQUEST_CANCELED)

    async def test_precreate_cancel_tombstones_are_bounded_and_expire(self) -> None:
        request_ids = ("1" * 32, "2" * 32, "3" * 32)
        with mock.patch.object(main, "REQUEST_TOMBSTONE_MAX", 2):
            for request_id in request_ids:
                stopped = await main.stop_job(request_id)
                self.assertEqual(stopped["status"], main.STATUS_CANCELED)

        with main._request_jobs_lock:
            self.assertNotIn(request_ids[0], main._request_jobs)
            self.assertEqual(
                set(main._request_tombstone_times),
                set(request_ids[1:]),
            )
            expires_after = (
                max(main._request_tombstone_times.values())
                + main.REQUEST_TOMBSTONE_TTL_SECONDS
            )

        removed = main._expire_request_tombstones(now=expires_after)

        self.assertEqual(removed, 2)
        with main._request_jobs_lock:
            self.assertEqual(main._request_tombstone_times, {})
            self.assertNotIn(request_ids[1], main._request_jobs)
            self.assertNotIn(request_ids[2], main._request_jobs)

    async def test_stop_mapped_request_id_stops_the_existing_job(self) -> None:
        request_id = "d" * 32
        job = await self._create_browser_job(request_id=request_id)

        stopped = await main.stop_job(request_id)

        self.assertEqual(stopped["status"], main.STATUS_CANCELED)
        self.assertEqual(job.status, main.STATUS_CANCELED)
        self.assertFalse(job.input_path.exists())
        with main._request_jobs_lock:
            self.assertEqual(main._request_jobs[request_id], job.id)

    async def test_get_job_status_returns_payload_or_404(self) -> None:
        job = await self._create_browser_job()
        payload = await main.get_job_status(job.id)
        self.assertEqual(payload["id"], job.id)
        self.assertEqual(payload["status"], main.STATUS_RUNNING)

        with self.assertRaises(HTTPException) as caught:
            await main.get_job_status("missing-job")
        self.assertEqual(caught.exception.status_code, 404)

    async def test_page_index_and_delivery_order_are_enforced(self) -> None:
        job = await self._create_browser_job(rotations=(0, 0))

        for invalid_index in (-1, 2):
            with self.subTest(index=invalid_index):
                with self.assertRaises(HTTPException) as caught:
                    await main.browser_page_image(job.id, invalid_index)
                self.assertEqual(caught.exception.status_code, 404)

        # 이미지를 받기 전에 결과만 보내는 경우.
        unseen_payload = _payload_for((100, 50, 0), "NO_IMAGE_FIRST")
        with self.assertRaises(HTTPException) as caught:
            await main.save_browser_page(job.id, 0, _json_request(unseen_payload))
        self.assertEqual(caught.exception.status_code, 409)

        # 두 번째 쪽 이미지를 먼저 받는 것은 괜찮지만 결과는 0쪽부터 와야 한다.
        await main.browser_page_image(job.id, 1)
        page_one_payload = _payload_for(job.browser_page_images[1], "OUT_OF_ORDER")
        with self.assertRaises(HTTPException) as caught:
            await main.save_browser_page(job.id, 1, _json_request(page_one_payload))
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(job.browser_results, {})

    async def test_result_byte_limit_is_cumulative_and_cancel_resets_it(self) -> None:
        job = await self._create_browser_job(rotations=(0, 0))
        await main.browser_page_image(job.id, 0)
        await main.browser_page_image(job.id, 1)
        first_payload = _payload_for(job.browser_page_images[0], "FIRST_BYTES")
        second_payload = _payload_for(job.browser_page_images[1], "SECOND_BYTES")
        first_size = len(json.dumps(first_payload, ensure_ascii=False).encode("utf-8"))
        second_size = len(json.dumps(second_payload, ensure_ascii=False).encode("utf-8"))

        with mock.patch.object(
            main,
            "BROWSER_MAX_RESULT_BYTES_PER_JOB",
            first_size + second_size - 1,
        ):
            await main.save_browser_page(job.id, 0, _json_request(first_payload))
            with self.assertRaises(HTTPException) as caught:
                await main.save_browser_page(job.id, 1, _json_request(second_payload))

        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(job.browser_result_bytes, first_size)
        await main.stop_job(job.id)
        self.assertEqual(job.browser_result_bytes, 0)

    async def test_stale_browser_job_is_expired_and_large_buffers_are_cleared(self) -> None:
        job = await self._create_browser_job()
        await self._prepare_page(job, 0, "EXPIRE_ME")
        self.assertGreater(job.browser_result_bytes, 0)
        reservation = storage.reserve_result_path("expiring.pdf")
        job.output_path = reservation.final_path
        job.result_reservation = reservation
        job.last_activity = 100.0

        removed = []
        original_remove = main._remove_file

        def tracked_remove(path):
            removed.append(path)
            original_remove(path)

        with mock.patch.object(main, "_remove_file", side_effect=tracked_remove):
            expired = main._expire_stale_browser_jobs(
                now=100.0 + main.BROWSER_JOB_TTL_SECONDS,
            )
            # 작업 스레드가 늦게 finally에 들어와도 TTL이 가져간 파일을 다시
            # 지우지 않아야 한다.
            main._discard_result_reservation(job)

        self.assertEqual(expired, 1)
        self.assertEqual(job.status, main.STATUS_CANCELED)
        self.assertTrue(job.stop_requested)
        self.assertFalse(job.input_path.exists())
        self.assertEqual(job.browser_results, {})
        self.assertEqual(job.browser_page_images, {})
        self.assertEqual(job.browser_result_bytes, 0)
        self.assertIsNone(job.result_reservation)
        self.assertEqual(removed.count(reservation.final_path), 1)
        self.assertEqual(removed.count(reservation.part_path), 1)

    async def test_finishing_browser_job_is_not_expired(self) -> None:
        job = await self._create_browser_job()
        job.browser_finishing = True
        job.last_activity = 100.0

        expired = main._expire_stale_browser_jobs(
            now=100.0 + main.BROWSER_JOB_TTL_SECONDS,
        )

        self.assertEqual(expired, 0)
        self.assertFalse(job.stop_requested)
        self.assertTrue(job.input_path.exists())

    async def test_finish_during_page_render_does_not_delete_input(self) -> None:
        job = await self._create_browser_job()
        await self._prepare_page(job, 0, "READY_TO_FINISH")
        entered = threading.Event()
        release = threading.Event()
        original_renderer = main.render_browser_page

        def paused_renderer(*args, **kwargs):
            entered.set()
            release.wait(timeout=2)
            return original_renderer(*args, **kwargs)

        with mock.patch.object(main, "render_browser_page", side_effect=paused_renderer):
            render_task = asyncio.create_task(main.browser_page_image(job.id, 0))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            await main.finish_browser_job(job.id)
            release.set()
            with self.assertRaises(HTTPException) as caught:
                await render_task

        self.assertEqual(caught.exception.status_code, 409)
        self.assertTrue(job.browser_finishing)
        self.assertTrue(job.input_path.exists())

    async def test_queued_browser_finish_can_be_canceled_immediately(self) -> None:
        job = await self._create_browser_job()
        await self._prepare_page(job, 0, "CANCEL_QUEUED_FINISH")
        pending = Future()

        with mock.patch.object(main._worker, "submit", return_value=pending):
            await main.finish_browser_job(job.id)
            stopped = await main.stop_job(job.id)

        self.assertTrue(pending.cancelled())
        self.assertEqual(stopped["status"], main.STATUS_CANCELED)
        self.assertFalse(job.browser_finishing)
        self.assertIsNone(job.worker_future)
        self.assertEqual(job.browser_results, {})
        self.assertEqual(job.browser_result_bytes, 0)
        self.assertFalse(job.input_path.exists())

    async def test_page_rendering_is_serialized_across_browser_jobs(self) -> None:
        first_job = await self._create_browser_job(name="first.pdf")
        second_job = await self._create_browser_job(name="second.pdf")
        original_renderer = main.render_browser_page
        guard = threading.Lock()
        active = 0
        maximum_active = 0

        def measured_renderer(*args, **kwargs):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.05)
                return original_renderer(*args, **kwargs)
            finally:
                with guard:
                    active -= 1

        with mock.patch.object(main, "render_browser_page", side_effect=measured_renderer):
            await asyncio.gather(
                main.browser_page_image(first_job.id, 0),
                main.browser_page_image(second_job.id, 0),
            )

        self.assertEqual(maximum_active, 1)

    async def test_canceled_render_request_keeps_slot_until_thread_finishes(self) -> None:
        first_job = await self._create_browser_job(name="first-canceled.pdf")
        second_job = await self._create_browser_job(name="second-after-cancel.pdf")
        original_renderer = main.render_browser_page
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        guard = threading.Lock()
        active = 0
        maximum_active = 0

        def blocking_renderer(input_path, *args, **kwargs):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if Path(input_path) == first_job.input_path:
                    first_started.set()
                    if not release_first.wait(timeout=5):
                        raise RuntimeError("첫 렌더 시험 대기가 끝나지 않았습니다.")
                else:
                    second_started.set()
                return original_renderer(input_path, *args, **kwargs)
            finally:
                with guard:
                    active -= 1

        first_request = None
        second_request = None
        with mock.patch.object(main, "render_browser_page", side_effect=blocking_renderer):
            try:
                first_request = asyncio.create_task(
                    main.browser_page_image(first_job.id, 0)
                )
                started = await asyncio.to_thread(first_started.wait, 2)
                self.assertTrue(started)

                first_request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first_request
                self.assertEqual(len(main._browser_render_tasks), 1)

                second_request = asyncio.create_task(
                    main.browser_page_image(second_job.id, 0)
                )
                await asyncio.sleep(0.1)

                self.assertFalse(second_started.is_set())
                self.assertEqual(len(main._browser_render_tasks), 2)
                with guard:
                    self.assertEqual(active, 1)
                    self.assertEqual(maximum_active, 1)

                release_first.set()
                response = await asyncio.wait_for(second_request, timeout=5)
                self.assertEqual(response.media_type, "image/png")
            finally:
                release_first.set()
                requests = [
                    request
                    for request in (first_request, second_request)
                    if request is not None
                ]
                if requests:
                    await asyncio.gather(*requests, return_exceptions=True)
                if main._browser_render_tasks:
                    await asyncio.gather(
                        *list(main._browser_render_tasks),
                        return_exceptions=True,
                    )
                await asyncio.sleep(0)

        self.assertTrue(second_started.is_set())
        self.assertEqual(maximum_active, 1)
        self.assertEqual(main._browser_render_tasks, set())

    async def test_browser_cancel_is_immediate_and_cleans_temporary_data(self) -> None:
        job = await self._create_browser_job()
        await self._prepare_page(job, 0, "CANCEL_ME")
        input_path = job.input_path

        stopped = await main.stop_job(job.id)

        self.assertEqual(stopped["status"], main.STATUS_CANCELED)
        self.assertEqual(job.status, main.STATUS_CANCELED)
        self.assertTrue(job.stop_requested)
        self.assertFalse(input_path.exists())
        self.assertEqual(job.browser_results, {})
        self.assertEqual(job.browser_page_images, {})
        self.assertIsNone(job.output_path)
        self.assertEqual(self.submitted, [])
        self.assertTrue(main._job_payload(job)["finished"])

        with self.assertRaises(HTTPException) as caught:
            await main.finish_browser_job(job.id)
        self.assertEqual(caught.exception.status_code, 409)

    async def test_shutdown_cleans_active_jobs_and_preserves_completed_result(self) -> None:
        active = await self._create_browser_job(
            name="active.pdf",
            request_id="d" * 32,
        )
        await self._prepare_page(active, 0, "SHUTDOWN_ACTIVE")

        done_input = main.UPLOAD_DIR / ("e" * 32 + ".pdf")
        done_input.write_bytes(_pdf_bytes((0,)))
        done_output = storage.RESULT_DIR / "done_Searchable.pdf"
        done_output.write_bytes(b"completed-result")
        completed = main.Job(
            id="finished-job",
            original_name="done.pdf",
            input_path=done_input,
            options=OcrOptions(dpi=150, text_score=0.5, mode=MODE_KEEP),
            output_path=done_output,
            status=main.STATUS_DONE,
        )
        main.jobs[completed.id] = completed

        await main._shutdown_active_jobs()

        self.assertEqual(active.status, main.STATUS_CANCELED)
        self.assertTrue(active.stop_requested)
        self.assertEqual(active.browser_results, {})
        self.assertFalse(active.input_path.exists())
        self.assertFalse(done_input.exists())
        self.assertEqual(done_output.read_bytes(), b"completed-result")
        self.assertEqual(main.jobs, {})
        with main._request_jobs_lock:
            self.assertEqual(main._request_jobs, {})

    async def test_shutdown_waits_for_live_future_after_done_status(self) -> None:
        input_path = main.UPLOAD_DIR / ("7" * 32 + ".pdf")
        input_path.write_bytes(_pdf_bytes((0,)))
        output_path = storage.RESULT_DIR / "finished_Searchable.pdf"
        output_path.write_bytes(b"completed-result")
        running_future = Future()
        self.assertTrue(running_future.set_running_or_notify_cancel())
        completed = main.Job(
            id="finished-with-live-future",
            original_name="finished.pdf",
            input_path=input_path,
            options=OcrOptions(dpi=150, text_score=0.5, mode=MODE_KEEP),
            output_path=output_path,
            status=main.STATUS_DONE,
            worker_future=running_future,
        )
        main.jobs[completed.id] = completed

        shutdown_task = asyncio.create_task(main._shutdown_active_jobs())
        await asyncio.sleep(0)
        self.assertFalse(shutdown_task.done())

        running_future.set_result(None)
        await asyncio.wait_for(shutdown_task, timeout=1)

        self.assertEqual(output_path.read_bytes(), b"completed-result")
        self.assertFalse(input_path.exists())
        self.assertIsNone(completed.worker_future)
        self.assertEqual(main.jobs, {})

    async def test_lifespan_waits_for_running_periodic_cleanup(self) -> None:
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        cleanup_finished = threading.Event()

        def blocking_cleanup(now=None):
            cleanup_started.set()
            release_cleanup.wait(timeout=2)
            cleanup_finished.set()
            return 0

        manager = main.lifespan(main.app)
        entered = False
        exit_task = None
        try:
            with (
                mock.patch.object(main, "BROWSER_JOB_CLEANUP_INTERVAL_SECONDS", 0),
                mock.patch.object(main, "_expire_stale_browser_jobs", blocking_cleanup),
            ):
                await manager.__aenter__()
                entered = True
                started = await asyncio.to_thread(cleanup_started.wait, 1)
                self.assertTrue(started)

                exit_task = asyncio.create_task(manager.__aexit__(None, None, None))
                await asyncio.sleep(0.05)
                self.assertFalse(exit_task.done())

                release_cleanup.set()
                await asyncio.wait_for(exit_task, timeout=1)
                self.assertTrue(cleanup_finished.is_set())
        finally:
            release_cleanup.set()
            if exit_task is not None and not exit_task.done():
                await asyncio.gather(exit_task, return_exceptions=True)
            elif entered and exit_task is None:
                await manager.__aexit__(None, None, None)

    async def test_startup_upload_cleanup_uses_exact_app_file_name(self) -> None:
        orphan = main.UPLOAD_DIR / ("f" * 32 + ".pdf")
        unrelated = main.UPLOAD_DIR / "manual.pdf"
        matching_directory = main.UPLOAD_DIR / ("1" * 32 + ".pdf")
        orphan.write_bytes(b"temporary")
        unrelated.write_bytes(b"keep")
        matching_directory.mkdir()

        removed = main._cleanup_orphaned_uploads()

        self.assertEqual(removed, 1)
        self.assertFalse(orphan.exists())
        self.assertEqual(unrelated.read_bytes(), b"keep")
        self.assertTrue(matching_directory.is_dir())

    async def test_same_name_jobs_choose_output_name_when_worker_runs(self) -> None:
        first_job = await self._create_browser_job(name="same.pdf")
        second_job = await self._create_browser_job(name="same.pdf")
        await self._prepare_page(first_job, 0, "FIRST_JOB_TEXT")
        await self._prepare_page(second_job, 0, "SECOND_JOB_TEXT")

        await main.finish_browser_job(first_job.id)
        await main.finish_browser_job(second_job.id)
        self.assertIsNone(first_job.output_path)
        self.assertIsNone(second_job.output_path)
        self.assertEqual(len(self.submitted), 2)

        await asyncio.to_thread(self._run_next_worker_job)
        await asyncio.to_thread(self._run_next_worker_job)

        self.assertEqual(first_job.status, main.STATUS_DONE)
        self.assertEqual(second_job.status, main.STATUS_DONE)
        self.assertNotEqual(first_job.output_path, second_job.output_path)
        self.assertEqual(first_job.output_path.name, "same_Searchable.pdf")
        self.assertEqual(second_job.output_path.name, "same_Searchable(2).pdf")
        self.assertTrue(first_job.output_path.is_file())
        self.assertTrue(second_job.output_path.is_file())

        with pymupdf.open(first_job.output_path) as first_document:
            self.assertIn("FIRST_JOB_TEXT", first_document[0].get_text())
            self.assertNotIn("SECOND_JOB_TEXT", first_document[0].get_text())
        with pymupdf.open(second_job.output_path) as second_document:
            self.assertIn("SECOND_JOB_TEXT", second_document[0].get_text())
            self.assertNotIn("FIRST_JOB_TEXT", second_document[0].get_text())

    async def test_server_processor_default_keeps_existing_worker_flow(self) -> None:
        processor_parameter = inspect.signature(main.create_job).parameters["processor"]
        self.assertEqual(processor_parameter.default.default, main.PROCESSOR_SERVER)

        upload = UploadFile(BytesIO(_pdf_bytes((0,))), filename="server.pdf")
        try:
            response = await main.create_job(
                file=upload,
                source_path="",
                dpi=150,
                text_score=0.5,
                mode=MODE_KEEP,
                processor=main.PROCESSOR_SERVER,
            )
        finally:
            await upload.close()

        job = main.jobs[response["id"]]
        self.assertEqual(response["processor"], main.PROCESSOR_SERVER)
        self.assertEqual(job.processor, main.PROCESSOR_SERVER)
        self.assertIsNone(job.output_path)
        self.assertEqual(len(self.submitted), 1)

        def fake_server_ocr(input_path, output_path, options, **kwargs):
            Path(output_path).write_bytes(Path(input_path).read_bytes())
            return OcrSummary(
                total_pages=1,
                done_pages=1,
                input_bytes=Path(input_path).stat().st_size,
                output_bytes=Path(output_path).stat().st_size,
                device="CPU",
            )

        with mock.patch.object(main, "run_ocr", side_effect=fake_server_ocr):
            await asyncio.to_thread(self._run_next_worker_job)

        self.assertEqual(job.status, main.STATUS_DONE)
        self.assertEqual(job.output_path.name, "server_Searchable.pdf")
        self.assertTrue(job.output_path.is_file())
        self.assertFalse(job.input_path.exists())
        self.assertIsNone(job.result_reservation)
        self.assertEqual(list(job.output_path.parent.glob("*.part")), [])

    async def test_queued_server_job_can_be_canceled_immediately(self) -> None:
        upload = UploadFile(BytesIO(_pdf_bytes((0,))), filename="queued-server.pdf")
        pending = Future()
        try:
            with mock.patch.object(main._worker, "submit", return_value=pending):
                response = await main.create_job(
                    file=upload,
                    source_path="",
                    dpi=150,
                    text_score=0.5,
                    mode=MODE_KEEP,
                    processor=main.PROCESSOR_SERVER,
                )
        finally:
            await upload.close()

        job = main.jobs[response["id"]]
        stopped = await main.stop_job(job.id)

        self.assertTrue(pending.cancelled())
        self.assertEqual(stopped["status"], main.STATUS_CANCELED)
        self.assertEqual(job.status, main.STATUS_CANCELED)
        self.assertIsNone(job.worker_future)
        self.assertFalse(job.input_path.exists())

    async def test_server_failure_removes_reserved_name_and_partial_file(self) -> None:
        upload = UploadFile(BytesIO(_pdf_bytes((0,))), filename="failure.pdf")
        try:
            response = await main.create_job(
                file=upload,
                source_path="",
                dpi=150,
                text_score=0.5,
                mode=MODE_KEEP,
                processor=main.PROCESSOR_SERVER,
            )
        finally:
            await upload.close()

        job = main.jobs[response["id"]]

        def failing_server_ocr(input_path, output_path, options, **kwargs):
            Path(output_path).write_bytes(b"partial")
            raise RuntimeError("intentional test failure")

        with mock.patch.object(main, "run_ocr", side_effect=failing_server_ocr):
            await asyncio.to_thread(self._run_next_worker_job)

        self.assertEqual(job.status, main.STATUS_FAILED)
        self.assertIsNone(job.result_reservation)
        self.assertFalse(job.output_path.exists())
        self.assertEqual(list(storage.RESULT_DIR.glob("*.part")), [])


class MiddlewareBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_vendor_static_requires_login(self) -> None:
        marker = object()

        async def call_next(_request):
            return marker

        with mock.patch.object(main.auth, "is_logged_in", return_value=False):
            public_css = await main.require_login(
                _path_request("/static/style.css"),
                call_next,
            )
            similar_name = await main.require_login(
                _path_request("/static/vendorish/file.js"),
                call_next,
            )
            protected_model = await main.require_login(
                _path_request("/static/vendor/model.tar"),
                call_next,
            )

        self.assertIs(public_css, marker)
        self.assertIs(similar_name, marker)
        self.assertEqual(protected_model.status_code, 401)

        user = mock.Mock(must_change_password=False)
        with (
            mock.patch.object(main.auth, "is_logged_in", return_value=True),
            mock.patch.object(main.auth, "load_user", return_value=user),
        ):
            logged_in_model = await main.require_login(
                _path_request("/static/vendor/model.tar"),
                call_next,
            )
        self.assertIs(logged_in_model, marker)


class StorageReservationTests(unittest.TestCase):
    def test_lifespan_recreates_render_semaphore_for_a_new_event_loop(self) -> None:
        seen_semaphores = []

        async def exercise_lifespan() -> None:
            async with main.lifespan(main.app):
                semaphore = main._browser_render_semaphore
                seen_semaphores.append(semaphore)
                await semaphore.acquire()

                async def wait_for_slot() -> None:
                    await semaphore.acquire()
                    semaphore.release()

                waiter = asyncio.create_task(wait_for_slot())
                await asyncio.sleep(0)
                semaphore.release()
                await waiter

        with (
            mock.patch.object(main, "_browser_render_semaphore", asyncio.Semaphore(1)),
            mock.patch.object(storage, "cleanup_orphaned_results", return_value=0),
            mock.patch.object(main, "_cleanup_orphaned_uploads", return_value=0),
            mock.patch.object(main, "_shutdown_active_jobs", new=mock.AsyncMock()),
        ):
            asyncio.run(exercise_lifespan())
            asyncio.run(exercise_lifespan())

        self.assertEqual(len(seen_semaphores), 2)
        self.assertIsNot(seen_semaphores[0], seen_semaphores[1])

    def test_result_names_are_reserved_before_the_pdf_is_committed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            result_dir = root / "result"
            source_dir.mkdir()
            result_dir.mkdir()
            with (
                mock.patch.object(storage, "SOURCE_DIR", source_dir),
                mock.patch.object(storage, "RESULT_DIR", result_dir),
            ):
                first = storage.reserve_result_path("same.pdf")
                second = storage.reserve_result_path("same.pdf")

                self.assertEqual(first.final_path.name, "same_Searchable.pdf")
                self.assertEqual(second.final_path.name, "same_Searchable(2).pdf")
                self.assertTrue(first.final_path.is_file())
                self.assertTrue(first.part_path.is_file())
                self.assertEqual(first.final_path.parent, first.part_path.parent)

                first.part_path.write_bytes(b"finished-pdf")
                storage.commit_reserved_result(first)
                self.assertEqual(first.final_path.read_bytes(), b"finished-pdf")
                self.assertFalse(first.part_path.exists())

                second.part_path.unlink()
                second.final_path.unlink()
                self.assertEqual(list(result_dir.glob("*.part")), [])

    def test_startup_cleanup_removes_only_app_owned_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            complete = result_dir / "keep_Searchable.pdf"
            empty_reservation = result_dir / "empty_Searchable.pdf"
            app_part = result_dir / (".parted_Searchable.pdf." + "a" * 32 + ".part")
            unrelated_part = result_dir / ".manual.part"
            unrelated_empty = result_dir / "manual.pdf"
            matching_directory = result_dir / "folder_Searchable.pdf"
            foreign_first = result_dir / "foreign_Searchable(1).pdf"
            foreign_leading_zero_part = result_dir / (
                ".foreign_Searchable(01).pdf." + "b" * 32 + ".part"
            )
            generated_second = result_dir / "generated_Searchable(2).pdf"
            generated_tenth_part = result_dir / (
                ".generated_Searchable(10).pdf." + "c" * 32 + ".part"
            )

            complete.write_bytes(b"real-pdf")
            empty_reservation.write_bytes(b"")
            app_part.write_bytes(b"partial-pdf")
            unrelated_part.write_bytes(b"keep")
            unrelated_empty.write_bytes(b"")
            matching_directory.mkdir()
            foreign_first.write_bytes(b"")
            foreign_leading_zero_part.write_bytes(b"foreign-part")
            generated_second.write_bytes(b"")
            generated_tenth_part.write_bytes(b"generated-part")

            with mock.patch.object(storage, "RESULT_DIR", result_dir):
                first_removed = storage.cleanup_orphaned_results()
                second_removed = storage.cleanup_orphaned_results()

            self.assertEqual(first_removed, 4)
            self.assertEqual(second_removed, 0)
            self.assertEqual(complete.read_bytes(), b"real-pdf")
            self.assertTrue(unrelated_part.exists())
            self.assertTrue(unrelated_empty.exists())
            self.assertTrue(matching_directory.is_dir())
            self.assertTrue(foreign_first.exists())
            self.assertTrue(foreign_leading_zero_part.exists())
            self.assertFalse(empty_reservation.exists())
            self.assertFalse(app_part.exists())
            self.assertFalse(generated_second.exists())
            self.assertFalse(generated_tenth_part.exists())


class BrowserValidationTests(unittest.TestCase):
    """서버 경계에서 숫자·좌표·글자 제한을 빠짐없이 확인한다."""

    def test_invalid_numbers_coordinates_pages_and_text_are_rejected(self) -> None:
        expected = (100, 200, 0)
        valid = _payload_for(expected, "VALID_TEXT")
        cases: list[tuple[str, object]] = []

        payload = copy.deepcopy(valid)
        payload["lines"][0]["score"] = math.nan
        cases.append(("not-finite-score", payload))

        payload = copy.deepcopy(valid)
        payload["lines"][0]["box"][0][0] = math.inf
        cases.append(("not-finite-coordinate", payload))

        payload = copy.deepcopy(valid)
        payload["lines"][0]["box"][0][0] = -0.1
        cases.append(("negative-coordinate", payload))

        payload = copy.deepcopy(valid)
        payload["lines"][0]["box"][2][1] = expected[1] + 0.1
        cases.append(("coordinate-outside-page", payload))

        payload = copy.deepcopy(valid)
        for point in payload["lines"][0]["box"]:
            point[0] = 10
        cases.append(("zero-width-box", payload))

        payload = copy.deepcopy(valid)
        payload["render_width"] += 1
        cases.append(("wrong-page-width", payload))

        payload = copy.deepcopy(valid)
        payload["rotation"] = 90
        cases.append(("wrong-page-rotation", payload))

        payload = copy.deepcopy(valid)
        payload["lines"][0]["text"] = "   "
        cases.append(("blank-text", payload))

        payload = copy.deepcopy(valid)
        payload["lines"][0]["text"] = "가" * (main.BROWSER_MAX_TEXT_LENGTH + 1)
        cases.append(("too-long-text", payload))

        payload = copy.deepcopy(valid)
        payload["lines"][0]["text"] = "널\x00문자"
        cases.append(("nul-in-text", payload))

        payload = copy.deepcopy(valid)
        payload["lines"] = payload["lines"] * (main.BROWSER_MAX_LINES_PER_PAGE + 1)
        cases.append(("too-many-lines", payload))

        for name, invalid in cases:
            with self.subTest(case=name):
                with self.assertRaises(HTTPException) as caught:
                    main._validate_browser_page_result(invalid, expected)
                self.assertEqual(caught.exception.status_code, 400)

        parsed = main._validate_browser_page_result(valid, expected)
        self.assertIsInstance(parsed, BrowserOcrPage)
        self.assertEqual(parsed.lines[0].text, "VALID_TEXT")


class BrowserPdfRotationTests(unittest.TestCase):
    """회전된 PDF에서도 검색 글자의 화면상 위치가 맞는지 확인한다."""

    def test_text_position_matches_for_all_right_angle_rotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rotation in (0, 90, 180, 270):
                with self.subTest(rotation=rotation):
                    input_path = root / f"input-{rotation}.pdf"
                    output_path = root / f"output-{rotation}.pdf"
                    input_path.write_bytes(_pdf_bytes((rotation,)))

                    image = render_browser_page(input_path, 0, max_side=400)
                    self.assertTrue(image.content.startswith(b"\x89PNG\r\n\x1a\n"))
                    self.assertLessEqual(max(image.width, image.height), 400)
                    self.assertEqual(image.rotation, rotation)

                    x0, x1 = image.width * 0.05, image.width * 0.95
                    y0, y1 = image.height * 0.40, image.height * 0.50
                    text = "ROTATION_POSITION_TEST_1234567890"
                    line = BrowserOcrLine(
                        box=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                        text=text,
                        score=0.99,
                    )
                    page_result = BrowserOcrPage(
                        render_width=image.width,
                        render_height=image.height,
                        rotation=image.rotation,
                        lines=(line,),
                    )
                    summary = build_pdf_from_browser_results(
                        input_path,
                        output_path,
                        OcrOptions(dpi=150, text_score=0.5, mode=MODE_KEEP, use_gpu=False),
                        {0: page_result},
                    )

                    self.assertEqual(summary.done_pages, 1)
                    self.assertEqual(summary.text_lines, 1)
                    with pymupdf.open(output_path) as document:
                        page = document[0]
                        self.assertEqual(page.rotation, rotation)
                        found = page.search_for(text)
                        self.assertTrue(found)

                        actual = pymupdf.Rect(found[0])
                        for rectangle in found[1:]:
                            actual |= rectangle
                        if rotation:
                            actual = actual * page.rotation_matrix

                        expected = pymupdf.Rect(
                            page.rect.x0 + x0 * page.rect.width / image.width,
                            page.rect.y0 + y0 * page.rect.height / image.height,
                            page.rect.x0 + x1 * page.rect.width / image.width,
                            page.rect.y0 + y1 * page.rect.height / image.height,
                        )
                        # 긴 시험 글자는 상자 가로폭에 맞게 줄어든다. 이전 회전 계산은
                        # 90도에서 너비와 높이를 바꿔 써 실제 폭이 크게 짧아졌다.
                        self.assertAlmostEqual(actual.x0, expected.x0, delta=0.5)
                        self.assertAlmostEqual(actual.x1, expected.x1, delta=0.5)
                        self.assertGreaterEqual(actual.y0, expected.y0 - 0.5)
                        self.assertLessEqual(actual.y1, expected.y1 + 0.5)

    def test_finish_stop_is_checked_between_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "two-pages.pdf"
            output_path = root / "must-not-exist.pdf"
            input_path.write_bytes(_pdf_bytes((0, 0)))

            image_zero = render_browser_page(input_path, 0, max_side=400)
            image_one = render_browser_page(input_path, 1, max_side=400)

            def page_result(image, text):
                payload = _payload_for((image.width, image.height, image.rotation), text)
                return main._validate_browser_page_result(
                    payload,
                    (image.width, image.height, image.rotation),
                )

            pages = {
                0: page_result(image_zero, "FIRST_PAGE"),
                1: page_result(image_one, "SECOND_PAGE"),
            }
            stop = False

            def on_progress(done, total):
                nonlocal stop
                stop = done == 1

            summary = build_pdf_from_browser_results(
                input_path,
                output_path,
                OcrOptions(dpi=150, text_score=0.5, mode=MODE_KEEP, use_gpu=False),
                pages,
                on_progress=on_progress,
                should_stop=lambda: stop,
            )

            self.assertTrue(summary.stopped)
            self.assertEqual(summary.done_pages, 1)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
