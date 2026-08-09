# -*- coding: utf-8 -*-
"""
PDF OCR 코어 로직 (RapidOCR + PyMuPDF, GPU 가속)

하는 일:
  1. PDF 페이지를 지정한 해상도(DPI)의 이미지로 그려낸다.
  2. 그 이미지에서 한글/영문 글자와 위치를 찾아낸다. (RapidOCR)
  3. 찾아낸 글자를 '눈에 보이지 않는 글자'로 PDF에 얹는다.
     → 화면에는 원래 스캔 이미지 그대로 보이지만 Ctrl+F 검색이 된다.

용어 설명:
  · DPI      : 1인치를 몇 개의 점으로 그릴지. 숫자가 클수록 또렷하지만 무겁다.
  · 투명 글자 : PDF에는 '글자를 그리지 않고 자리만 차지시키는' 방법이 있다.
               (render_mode=3) 검색용 글자를 넣을 때 쓴다.
  · CUDA     : NVIDIA 그래픽카드로 계산을 대신 시키는 기술.
"""

from __future__ import annotations

import gc
import os
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CUDA 라이브러리(DLL) 경로 준비
# ─────────────────────────────────────────────────────────────────────────────
def prepare_cuda_dll_path() -> list:
    """pip으로 설치한 CUDA/cuDNN DLL 폴더를 Windows 검색 경로에 등록한다.

    NVIDIA CUDA 툴킷을 따로 설치하지 않고 pip 패키지(nvidia-*-cu12)로만 받으면
    DLL들이 site-packages/nvidia/*/bin 안에 흩어져 있다.

    onnxruntime의 preload_dlls()만으로는 부족하다. cuDNN 본체가 실행 도중
    자기 하위 DLL(cudnn_engines_tensor_ir64_9.dll 등)을 다시 불러오는데,
    그때는 파이썬이 등록한 경로가 아니라 시스템 PATH를 뒤지기 때문이다.
    이 처리를 빠뜨리면 오류 없이 조용히 CPU로 넘어가 5배 이상 느려진다.
    """
    nvidia_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not nvidia_root.is_dir():
        return []

    bin_dirs = sorted(p for p in nvidia_root.glob("*/bin") if p.is_dir())
    for bin_dir in bin_dirs:
        os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = os.pathsep.join(
        [str(p) for p in bin_dirs] + [os.environ.get("PATH", "")]
    )
    return bin_dirs


# onnxruntime을 불러오기 '전에' 경로를 잡아야 한다. (rapidocr도 내부에서 씀)
CUDA_DLL_DIRS = prepare_cuda_dll_path()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pymupdf  # noqa: E402

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

# 환경 점검은 onnxruntime을 불러온 뒤에 가져온다. (안에서 설치된 부품을 살펴보기 때문)
from app.system_check import (  # noqa: E402
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_DIRECTML,
    active_backend,
)


# ─────────────────────────────────────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────────────────────────────────────
MODE_KEEP = "keep"        # 원본 페이지를 그대로 두고 투명 글자만 얹는다
MODE_REBUILD = "rebuild"  # 선택한 DPI 이미지로 페이지를 다시 만든다 (용량 축소)

# PyMuPDF에 내장된 한국어 글꼴. 글꼴 파일을 PDF에 넣지 않아도 한글이 검색된다.
# (맑은 고딕을 넣으면 파일 하나당 7MB 넘게 늘어나므로 쓰지 않는다.)
PDF_FONT = "korea"

# 페이지를 다시 만들 때 쓰는 JPEG 품질. 85면 눈으로 차이를 느끼기 어렵다.
REBUILD_JPEG_QUALITY = 85

# 몇 페이지마다 메모리를 정리할지.
# 300dpi A4 한 장의 이미지만 26MB 남짓인데, OCR 과정에서 만들어지는 중간 이미지까지
# 합치면 한 쪽당 90MB 가까이 잡아먹는다. 이것들이 서로 얽혀 있어 파이썬이 곧바로
# 놓아주지 못하므로 주기적으로 정리해 준다.
# 20쪽마다 정리했을 때 최대 3.1GB까지 치솟아서 5쪽으로 줄였다.
GC_INTERVAL = 5


@dataclass
class OcrOptions:
    """변환 설정. 웹 화면에서 받은 값이 그대로 들어온다."""
    dpi: int = 300
    text_score: float = 0.5      # 이 점수보다 낮게 인식된 글자는 버린다
    mode: str = MODE_KEEP
    use_gpu: bool = True


@dataclass
class OcrSummary:
    """변환 결과 요약. 화면의 결과 카드에 그대로 쓰인다."""
    total_pages: int = 0
    done_pages: int = 0
    text_lines: int = 0          # 실제로 넣은 글줄 수
    elapsed: float = 0.0         # 걸린 시간(초)
    input_bytes: int = 0
    output_bytes: int = 0
    device: str = BACKEND_CPU    # 실제로 사용한 장치 (CUDA / DirectML / CPU)
    stopped: bool = False        # 사용자가 중지했는지


# ─────────────────────────────────────────────────────────────────────────────
# OCR 엔진 준비
# ─────────────────────────────────────────────────────────────────────────────
# 엔진을 만들 때 모델을 읽어 들이므로 몇 초가 걸린다. 한 번 만들어 두고 계속 쓴다.
_engine_cache: dict = {}


def _build_engine(backend: str):
    """정해진 실행 방식에 맞춰 RapidOCR 엔진을 만든다.

    검출 모델(글자 위치를 찾는 모델)은 실행 방식에 따라 크기를 달리 쓴다.
    측정해 보니 그래픽 가속을 쓸 때는 모델을 키워도 속도가 거의 그대로였지만
    (0.31~0.34초), CPU에서는 tiny 0.97초 / small 1.75초 / medium 12.2초로
    차이가 매우 컸다. 정확도는 다섯 모델 모두 같았다.
    그래서 가속이 되면 small, CPU면 tiny를 쓴다.
    """
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    on_cpu = backend == BACKEND_CPU

    params = {
        # ── 글자 위치를 찾는 모델 ──
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": ModelType.TINY if on_cpu else ModelType.SMALL,
        # v6 검출 모델은 파일 이름이 multi_ 로 시작하지만
        # 설정에는 'multi'가 아니라 'ch'를 넣어야 한다. (multi는 거부됨)
        "Det.lang_type": LangDet.CH,

        # ── 글자를 읽는 모델 ──
        # 기본값은 중국어 모델이라 한글이 제대로 나오지 않는다.
        # 이 항목들은 문자열이 아니라 rapidocr가 정한 Enum 값을 넣어야 한다.
        # 한국어 인식 모델은 mobile 한 종류뿐이라 고를 것이 없다.
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,   # 한국어 모델은 v5까지 제공된다
        "Rec.model_type": ModelType.MOBILE,

        # 글줄이 거꾸로 뒤집혀 있는지 판단하는 기능(cls)을 끈다.
        # 켜 두면 똑바로 된 한글 줄을 거꾸로라고 잘못 판단해 글자를 망가뜨린다.
        # (시험 문서 5줄 중 2줄이 "대한민국 서울특별시..." → "SZ드연무위을위위터"처럼
        #  깨졌고, 끄자 5줄 모두 0.97 이상으로 정확히 읽혔다.)
        # 대신 위아래가 뒤집힌 채 스캔된 문서는 글자를 읽지 못한다.
        # 그런 문서는 올리기 전에 바로 세워서 올려야 한다.
        "Global.use_cls": False,
        # 점수가 낮은 글자를 버리는 일은 이 코드에서 직접 하므로
        # 엔진 쪽 기준은 낮춰 두고, 설정을 바꿔도 엔진을 다시 만들지 않게 한다.
        "Global.text_score": 0.0,
        "Global.log_level": "warning",
    }

    if backend == BACKEND_CUDA:
        params["EngineConfig.onnxruntime.use_cuda"] = True
        params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = 0
    elif backend == BACKEND_DIRECTML:
        # DirectML은 윈도우가 제공하는 공용 가속 방식이라
        # AMD·Intel 그래픽에서도 쓸 수 있다.
        params["EngineConfig.onnxruntime.use_dml"] = True
    else:
        # CPU로 처리할 때는 이 PC의 처리 갈래 수만큼 나눠 쓴다.
        threads = os.cpu_count() or 0
        if threads:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads

    return RapidOCR(params=params)


def _detect_device(engine) -> str:
    """엔진이 실제로 어느 장치에서 도는지 확인한다.

    설정만 가속으로 해 두고 실제로는 CPU로 도는 경우가 있어서,
    세션에 등록된 실행 장치를 직접 들여다본다.
    글자를 찾는 부분과 읽는 부분 중 하나라도 CPU면 CPU로 본다.
    """
    found = set()
    for part_name in ("text_det", "text_rec"):
        part = getattr(engine, part_name, None)
        session = getattr(getattr(part, "session", None), "session", None)
        if session is None or not hasattr(session, "get_providers"):
            continue
        provider = session.get_providers()[0]
        if "CUDA" in provider:
            found.add(BACKEND_CUDA)
        elif "Dml" in provider or "DML" in provider:
            found.add(BACKEND_DIRECTML)
        else:
            found.add(BACKEND_CPU)

    if not found or BACKEND_CPU in found:
        return BACKEND_CPU
    return BACKEND_CUDA if BACKEND_CUDA in found else BACKEND_DIRECTML


def get_engine(use_gpu: bool):
    """엔진을 돌려준다.

    이 PC에 설치된 부품을 보고 CUDA / DirectML / CPU 중 알맞은 것을 스스로 고른다.
    가속을 쓸 수 없으면 CPU로 조용히 넘어간다.

    PDFOCR_FORCE_CPU 값을 1로 두면 그래픽카드가 있어도 CPU만 쓴다.
    낮은 사양 그래픽카드에서는 CPU가 오히려 빠른 경우가 있어서 준비해 둔 장치다.
    """
    if os.environ.get("PDFOCR_FORCE_CPU") == "1":
        backend = BACKEND_CPU
    else:
        backend = active_backend() if use_gpu else BACKEND_CPU

    if backend not in _engine_cache:
        try:
            engine = _build_engine(backend)
        except Exception:
            # 그래픽카드 쪽 준비가 안 된 경우. CPU로 다시 만든다.
            if backend == BACKEND_CPU:
                raise
            backend = BACKEND_CPU
            engine = _engine_cache.get(backend) or _build_engine(backend)
        _engine_cache[backend] = engine

    engine = _engine_cache[backend]
    return engine, _detect_device(engine)


# ─────────────────────────────────────────────────────────────────────────────
# 페이지 이미지 → OCR 입력용 배열
# ─────────────────────────────────────────────────────────────────────────────
def _pixmap_to_array(pix) -> np.ndarray:
    """PyMuPDF가 그린 이미지를 OCR이 읽을 수 있는 형태로 바꾼다.

    PDF 쪽은 색을 RGB 순서로 담고 OCR 쪽은 BGR 순서를 기대하므로 순서를 뒤집는다.
    이미지 한 줄의 실제 길이(stride)가 가로 픽셀 수보다 클 수 있어 그만큼 잘라낸다.
    """
    buffer = np.frombuffer(pix.samples, dtype=np.uint8)
    rows = buffer.reshape(pix.height, pix.stride)
    image = rows[:, : pix.width * pix.n].reshape(pix.height, pix.width, pix.n)
    return image[:, :, ::-1]


# ─────────────────────────────────────────────────────────────────────────────
# 투명 글자 얹기
# ─────────────────────────────────────────────────────────────────────────────
def _insert_invisible_text(page, box, text: str, scale: float, rotate: int) -> None:
    """인식한 글자 하나를 원래 위치에 투명하게 얹는다.

    box는 이미지 위의 픽셀 좌표라서, PDF가 쓰는 단위(포인트)로 되돌려야 한다.
    1포인트 = 1/72인치이므로 'DPI ÷ 72'로 나눈 값이 배율(scale)이다.
    """
    xs = [float(point[0]) / scale for point in box]
    ys = [float(point[1]) / scale for point in box]
    rect = pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))

    # 페이지가 돌아가 있는 문서라면, 돌리기 전 좌표로 되돌려 놓아야 위치가 맞는다.
    if rotate:
        rect = rect * page.derotation_matrix

    if rect.height <= 0 or rect.width <= 0:
        return

    # 글자 크기를 글상자 높이에 맞춘 뒤, 가로로 삐져나오면 줄인다.
    fontsize = rect.height * 0.85
    text_width = pymupdf.get_text_length(text, fontname=PDF_FONT, fontsize=fontsize)
    if text_width > rect.width and text_width > 0:
        fontsize *= rect.width / text_width

    # 글자를 놓는 기준점은 글상자의 아래쪽 선이다. 살짝 띄워야 자연스럽다.
    origin = pymupdf.Point(rect.x0, rect.y1 - rect.height * 0.15)

    page.insert_text(
        origin,
        text,
        fontname=PDF_FONT,
        fontsize=fontsize,
        render_mode=3,   # 3 = 그리지 않음(투명). 검색·복사에는 그대로 잡힌다.
        rotate=rotate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 본 처리
# ─────────────────────────────────────────────────────────────────────────────
def run_ocr(
    input_path: str | Path,
    output_path: str | Path,
    options: OcrOptions,
    on_progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> OcrSummary:
    """스캔 PDF를 검색 가능한 PDF로 바꾼다.

    on_progress(끝난 페이지 수, 전체 페이지 수) : 진행 상황을 알리는 함수
    should_stop() -> bool                      : True를 돌려주면 즉시 멈춘다

    중지한 경우에는 결과 파일을 만들지 않는다. 절반만 처리된 파일을 완성본으로
    착각하는 일이 없도록 하기 위해서다.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    started = time.perf_counter()

    engine, device = get_engine(options.use_gpu)
    summary = OcrSummary(device=device, input_bytes=input_path.stat().st_size)

    rebuild = options.mode == MODE_REBUILD
    scale = options.dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)

    source = pymupdf.open(input_path)
    # 페이지를 다시 만드는 방식일 때만 빈 문서를 새로 준비한다.
    target = pymupdf.open() if rebuild else source

    try:
        summary.total_pages = source.page_count

        for index in range(source.page_count):
            if should_stop is not None and should_stop():
                summary.stopped = True
                break

            src_page = source[index]
            pix = src_page.get_pixmap(matrix=matrix)

            image = _pixmap_to_array(pix)
            ocr_result = engine(image)
            image = None  # 큰 이미지는 다 쓴 즉시 놓아준다

            if rebuild:
                # 이미지를 JPEG로 다시 담아 용량을 줄이고, 그 크기에 맞춘 페이지를 만든다.
                page = target.new_page(width=pix.width / scale, height=pix.height / scale)
                page.insert_image(
                    page.rect,
                    stream=pix.tobytes("jpg", jpg_quality=REBUILD_JPEG_QUALITY),
                )
                rotate = 0
            else:
                page = src_page
                rotate = src_page.rotation

            boxes = getattr(ocr_result, "boxes", None)
            texts = getattr(ocr_result, "txts", None)
            scores = getattr(ocr_result, "scores", None)

            if boxes is not None and texts is not None:
                for box, text, score in zip(boxes, texts, scores):
                    if score < options.text_score or not text.strip():
                        continue
                    _insert_invisible_text(page, box, text, scale, rotate)
                    summary.text_lines += 1

            pix = None
            ocr_result = None
            summary.done_pages = index + 1

            if on_progress is not None:
                on_progress(summary.done_pages, summary.total_pages)

            # 페이지 이미지가 커서 메모리에 쌓이므로 주기적으로 정리한다.
            if summary.done_pages % GC_INTERVAL == 0:
                gc.collect()

        if not summary.stopped:
            # garbage=4 : 안 쓰는 조각을 최대한 정리, deflate=True : 내용을 압축
            target.save(str(output_path), garbage=4, deflate=True)
            summary.output_bytes = output_path.stat().st_size

    finally:
        if rebuild:
            target.close()
        source.close()
        gc.collect()

    summary.elapsed = time.perf_counter() - started
    return summary
