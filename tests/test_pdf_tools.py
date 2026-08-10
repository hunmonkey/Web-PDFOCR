# -*- coding: utf-8 -*-
"""PDF 다루기 기능 회귀 시험.

실제 OCR 모델은 쓰지 않는다. 모델을 부르면 시험 한 번에 수십 초가 걸려
회귀 시험으로 쓸 수 없기 때문이다. 대신 인식 결과 모양만 흉내 낸 가짜 결과로
글자 정렬·기울기 측정·방향 판정 규칙을 확인하고, 파일 저장과 API 검증은
실제 코드를 그대로 부른다.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pymupdf
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app import main, storage
from app.ocr_engine import (
    EXPORT_DOCX,
    EXPORT_MD,
    EXPORT_TXT,
    MODE_KEEP,
    MODE_REBUILD,
    OcrOptions,
    _page_text_from_boxes,
    _reading_signal,
    _render_export,
    detect_rotation,
    document_has_text_layer,
    estimate_skew,
    run_ocr,
    write_text_export,
)


def _horizontal_box(x0, y0, width, height):
    """가로로 누운 글상자 하나를 만든다."""
    return [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)]


def _fake_result(boxes, texts, scores):
    """RapidOCR 결과와 같은 모양의 가짜 결과를 만든다."""
    return SimpleNamespace(boxes=boxes, txts=texts, scores=scores)


class TextLayerDetectionTests(unittest.TestCase):
    """이미 글자가 들어 있는 PDF를 알아보는지 확인한다."""

    def test_글자가_있는_PDF는_찾아낸다(self) -> None:
        document = pymupdf.open()
        try:
            page = document.new_page()
            page.insert_text((72, 100), "이 문서에는 글자가 충분히 들어 있습니다.",
                             fontname="korea", fontsize=14)
            self.assertTrue(document_has_text_layer(document))
        finally:
            document.close()

    def test_글자가_없는_PDF는_아니라고_한다(self) -> None:
        document = pymupdf.open()
        try:
            document.new_page()
            self.assertFalse(document_has_text_layer(document))
        finally:
            document.close()

    def test_쪽번호_수준의_짧은_글자는_글자층으로_보지_않는다(self) -> None:
        # 스캔본에도 쪽 번호 정도는 들어 있을 수 있다. 그것까지 글자층으로 보면
        # 멀쩡한 스캔 파일마다 경고가 떠 버린다.
        document = pymupdf.open()
        try:
            page = document.new_page()
            page.insert_text((72, 100), "12", fontname="korea", fontsize=10)
            self.assertFalse(document_has_text_layer(document))
        finally:
            document.close()


class LineOrderTests(unittest.TestCase):
    """읽은 글줄을 사람이 읽는 순서로 늘어놓는지 확인한다."""

    def test_뒤죽박죽_들어와도_위에서_아래_왼쪽에서_오른쪽으로_정렬한다(self) -> None:
        items = [
            (300.0, 100.0, 20.0, "둘째"),
            (100.0, 100.0, 20.0, "첫째"),
            (100.0, 160.0, 20.0, "아랫줄"),
        ]
        self.assertEqual(_page_text_from_boxes(items), "첫째 둘째\n아랫줄")

    def test_조금_어긋난_높이는_같은_줄로_묶는다(self) -> None:
        # 같은 줄이라도 글상자 위치가 몇 픽셀씩 어긋나서 잡힌다.
        items = [
            (100.0, 100.0, 20.0, "왼쪽"),
            (300.0, 106.0, 20.0, "오른쪽"),
        ]
        self.assertEqual(_page_text_from_boxes(items), "왼쪽 오른쪽")

    def test_줄_간격만큼_떨어지면_다른_줄로_나눈다(self) -> None:
        items = [
            (100.0, 100.0, 20.0, "윗줄"),
            (100.0, 130.0, 20.0, "아랫줄"),
        ]
        self.assertEqual(_page_text_from_boxes(items), "윗줄\n아랫줄")

    def test_읽은_글자가_없으면_빈_글이_된다(self) -> None:
        self.assertEqual(_page_text_from_boxes([]), "")


class ExportFileTests(unittest.TestCase):
    """읽은 글을 파일로 저장하는 부분을 확인한다."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_txt에는_쪽_구분선이_들어간다(self) -> None:
        rendered = _render_export(["첫 쪽", "둘째 쪽"], EXPORT_TXT)
        self.assertIn("1쪽", rendered)
        self.assertIn("2쪽", rendered)
        self.assertIn("둘째 쪽", rendered)

    def test_md에는_쪽마다_제목이_붙는다(self) -> None:
        rendered = _render_export(["첫 쪽", ""], EXPORT_MD)
        self.assertIn("## 1쪽", rendered)
        self.assertIn("## 2쪽", rendered)

    def test_세_형식_모두_실제로_저장된다(self) -> None:
        for export_format in (EXPORT_TXT, EXPORT_MD, EXPORT_DOCX):
            with self.subTest(형식=export_format):
                path = self.root / f"시험.{export_format}"
                write_text_export(path, ["첫 쪽", "둘째 쪽"], export_format)
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_빈_쪽이_섞여도_docx를_만든다(self) -> None:
        path = self.root / "빈쪽.docx"
        write_text_export(path, ["", "글 있음"], EXPORT_DOCX)
        self.assertTrue(path.is_file())

    def test_같은_이름이_이미_있으면_순번을_붙인다(self) -> None:
        pdf_path = self.root / "문서_Searchable.pdf"
        pdf_path.write_bytes(b"")
        first = storage.reserve_sidecar_path(pdf_path, "txt")
        second = storage.reserve_sidecar_path(pdf_path, "txt")
        self.assertEqual(first.name, "문서_Searchable.txt")
        self.assertEqual(second.name, "문서_Searchable(2).txt")
        self.assertNotEqual(first, second)


class SkewMeasureTests(unittest.TestCase):
    """글줄이 몇 도 기울었는지 재는 부분을 확인한다."""

    def _tilted(self, degrees: float, count: int = 10):
        radian = math.radians(degrees)
        boxes = []
        for index in range(count):
            x0, y0 = 100.0, 100.0 + index * 40
            x1 = x0 + 400.0 * math.cos(radian)
            y1 = y0 + 400.0 * math.sin(radian)
            boxes.append([(x0, y0), (x1, y1), (x1, y1 + 20), (x0, y0 + 20)])
        return _fake_result(boxes, ["글"] * count, [0.9] * count)

    def test_기울인_만큼_재어낸다(self) -> None:
        for wanted in (0.0, 2.0, -3.5):
            with self.subTest(각도=wanted):
                self.assertAlmostEqual(estimate_skew(self._tilted(wanted)), wanted, places=2)

    def test_글줄이_너무_적으면_재지_않는다(self) -> None:
        self.assertEqual(estimate_skew(self._tilted(5.0, count=2)), 0.0)

    def test_말도_안_되게_큰_각도는_무시한다(self) -> None:
        # 90도로 누운 문서는 이 함수가 아니라 방향 보정이 맡는다.
        self.assertEqual(estimate_skew(self._tilted(45.0)), 0.0)


class ReadingSignalTests(unittest.TestCase):
    """방향을 가릴 때 쓰는 점수 계산을 확인한다."""

    def test_세로로_선_글줄은_점수에_넣지_않는다(self) -> None:
        # 글자를 읽는 모델은 세로로 선 글줄도 눕혀서 읽어 낸다. 잘 읽혔는지만 보면
        # 옆으로 누운 문서를 똑바로 선 문서로 잘못 볼 수 있다.
        누운글상자 = [(100, 100), (120, 100), (120, 500), (100, 500)]
        result = _fake_result([누운글상자], ["세로글줄입니다"], [0.99])
        self.assertEqual(_reading_signal(result, 0.5), 0.0)

    def test_가로로_누운_글줄만_점수가_된다(self) -> None:
        result = _fake_result([_horizontal_box(100, 100, 400, 20)], ["가로글줄"], [0.99])
        self.assertGreater(_reading_signal(result, 0.5), 0.0)

    def test_기준_점수보다_낮으면_빼고_센다(self) -> None:
        result = _fake_result([_horizontal_box(100, 100, 400, 20)], ["흐린글자"], [0.2])
        self.assertEqual(_reading_signal(result, 0.5), 0.0)

    def test_길고_확신하는_글줄일수록_점수가_높다(self) -> None:
        짧은쪽 = _fake_result([_horizontal_box(100, 100, 400, 20)], ["가"], [0.9])
        긴쪽 = _fake_result([_horizontal_box(100, 100, 400, 20)], ["가나다라마"], [0.9])
        self.assertGreater(_reading_signal(긴쪽, 0.5), _reading_signal(짧은쪽, 0.5))


class RotationDecisionTests(unittest.TestCase):
    """어느 방향으로 돌릴지 정하는 규칙을 확인한다.

    실제 모델 대신, 미리 정한 각도에서만 글자가 잘 읽히는 가짜 엔진을 쓴다.
    """

    def setUp(self) -> None:
        self.document = pymupdf.open()
        self.document.new_page(width=400, height=600)
        self.addCleanup(self.document.close)

    def _engine_for(self, 잘읽히는각도: int, 글자수: int = 20):
        """정해 둔 회전값일 때만 글자를 잘 읽어 내는 가짜 엔진을 만든다."""
        page = self.document[0]

        def engine(_image):
            if page.rotation == 잘읽히는각도:
                return _fake_result(
                    [_horizontal_box(10, 10, 300, 20)], ["가" * 글자수], [0.95],
                )
            return _fake_result([], [], [])

        return engine

    def test_똑바로_읽히면_돌리지_않는다(self) -> None:
        self.assertEqual(detect_rotation(self._engine_for(0), self.document, 0.5), 0)

    def test_거꾸로_된_문서는_180도로_판정한다(self) -> None:
        self.assertEqual(detect_rotation(self._engine_for(180), self.document, 0.5), 180)

    def test_옆으로_누운_문서는_90도로_판정한다(self) -> None:
        self.assertEqual(detect_rotation(self._engine_for(90), self.document, 0.5), 90)

    def test_아무_방향으로도_안_읽히면_돌리지_않는다(self) -> None:
        def 백지엔진(_image):
            return _fake_result([], [], [])

        self.assertEqual(detect_rotation(백지엔진, self.document, 0.5), 0)

    def test_판정_뒤에_원래_회전값을_되돌려_놓는다(self) -> None:
        self.document[0].set_rotation(90)
        detect_rotation(self._engine_for(270), self.document, 0.5)
        self.assertEqual(self.document[0].rotation, 90)

    def test_글자가_너무_적으면_판단하지_않는다(self) -> None:
        # 도장 하나만 읽힌 정도로 문서 전체를 돌려 버리면 안 된다.
        self.assertEqual(
            detect_rotation(self._engine_for(180, 글자수=1), self.document, 0.5), 0
        )


class DeskewGuardTests(unittest.TestCase):
    """미세 기울기 보정을 쓸 수 있는 조건을 확인한다."""

    def test_원본_유지_방식에서는_거부한다(self) -> None:
        # 기울기를 되돌리려면 페이지를 다시 그려야 해서 무손실 보존과 함께 쓸 수 없다.
        with self.assertRaises(ValueError):
            run_ocr(
                "없어도된다.pdf", "결과.pdf",
                OcrOptions(mode=MODE_KEEP, deskew=True),
            )


class CreateJobOptionTests(unittest.IsolatedAsyncioTestCase):
    """새 설정값이 API 경계에서 제대로 걸러지는지 확인한다."""

    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        upload_dir = root / "uploads"
        result_dir = root / "result"
        for directory in (upload_dir, result_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.patchers = [
            mock.patch.object(main, "UPLOAD_DIR", upload_dir),
            mock.patch.object(storage, "RESULT_DIR", result_dir),
            mock.patch.object(main._worker, "submit", side_effect=self._skip_submit),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        main.jobs.clear()
        self.addCleanup(main.jobs.clear)

    def _skip_submit(self, function, *args, **kwargs):
        """실제 변환은 돌리지 않는다. 설정값이 어떻게 담기는지만 본다."""
        from concurrent.futures import Future

        future: Future = Future()
        future.set_result(None)
        return future

    def _pdf(self, with_text: bool = False) -> bytes:
        document = pymupdf.open()
        try:
            # 글자가 종이 밖으로 밀려나면 PDF에 담기지 않으므로 넉넉한 크기로 만든다.
            page = document.new_page(width=595, height=842)
            if with_text:
                page.insert_text((20, 100), "이 문서에는 글자가 이미 들어 있습니다.",
                                 fontname="korea", fontsize=12)
            return document.tobytes()
        finally:
            document.close()

    async def _create(self, **kwargs):
        upload = UploadFile(BytesIO(kwargs.pop("body", self._pdf())), filename="시험.pdf")
        options = {
            "source_path": "", "dpi": 150, "text_score": 0.5,
            "mode": MODE_KEEP, "processor": main.PROCESSOR_SERVER, "request_id": "",
        }
        options.update(kwargs)
        try:
            return await main.create_job(file=upload, **options)
        finally:
            await upload.close()

    async def test_글자가_들어_있는_PDF는_알려_준다(self) -> None:
        response = await self._create(body=self._pdf(with_text=True))
        self.assertTrue(response["has_text_layer"])

    async def test_스캔본은_글자층이_없다고_알려_준다(self) -> None:
        response = await self._create()
        self.assertFalse(response["has_text_layer"])

    async def test_고른_형식이_작업_설정에_담긴다(self) -> None:
        response = await self._create(export_formats="txt, md")
        self.assertEqual(main.jobs[response["id"]].options.export_formats, ("txt", "md"))

    async def test_같은_형식을_두_번_보내도_한_번만_담는다(self) -> None:
        response = await self._create(export_formats="txt,txt")
        self.assertEqual(main.jobs[response["id"]].options.export_formats, ("txt",))

    async def test_모르는_형식은_거부한다(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            await self._create(export_formats="hwp")
        self.assertEqual(caught.exception.status_code, 400)

    async def test_원본_유지에_기울기_보정을_켜면_거부한다(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            await self._create(mode=MODE_KEEP, deskew=True)
        self.assertEqual(caught.exception.status_code, 400)

    async def test_용량_줄이기에서는_기울기_보정을_받아들인다(self) -> None:
        response = await self._create(mode=MODE_REBUILD, deskew=True)
        self.assertTrue(main.jobs[response["id"]].options.deskew)

    async def test_브라우저_방식에서는_보정을_조용히_끈다(self) -> None:
        # 화면에서도 잠가 두지만, 기본값이 켜짐이라 오류로 막으면 쓸 수 없게 된다.
        response = await self._create(
            processor=main.PROCESSOR_BROWSER, auto_rotate=True,
        )
        options = main.jobs[response["id"]].options
        self.assertFalse(options.auto_rotate)
        self.assertFalse(options.deskew)


if __name__ == "__main__":
    unittest.main()
