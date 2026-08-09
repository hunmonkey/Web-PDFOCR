# -*- coding: utf-8 -*-
"""
GPU(CUDA) 실동작 및 한글 OCR 검증 스크립트

이 스크립트는 "설치가 됐다"가 아니라 "실제로 GPU에서 한글이 읽힌다"까지 확인한다.
RTX 3050(Ampere) PC와 RTX 5070 Ti(Blackwell) PC 양쪽에서 각각 한 번씩 실행해
같은 결과가 나오는지 보는 용도다.

실행:
    .venv\\Scripts\\python.exe scripts\\check_gpu.py
"""

import sys
import time
from pathlib import Path

# 이 스크립트는 프로젝트 폴더 밖(scripts/)에 있으므로 app 폴더를 찾을 수 있게 알려 준다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# CUDA 경로 준비와 onnxruntime 불러오기는 app/ocr_engine.py가 이미 알맞은 순서로 처리한다.
# 같은 코드를 두 군데 두면 한쪽만 고치는 실수가 생기므로 그대로 가져다 쓴다.
from app.ocr_engine import CUDA_DLL_DIRS, ort

_cuda_dirs = CUDA_DLL_DIRS

print("=" * 60)
print("1. onnxruntime 기본 정보")
print("=" * 60)
print(f"  버전            : {ort.__version__}")
print(f"  인식된 장치     : {ort.get_device()}")          # GPU 또는 CPU
print(f"  사용 가능 목록  : {ort.get_available_providers()}")
print(f"  CUDA DLL 폴더   : {len(_cuda_dirs)}곳 등록됨")

# get_device()가 GPU라고 해도 실제 추론까지 성공한다는 보장은 없다.
# (예: GPU 세대가 너무 최신이라 해당 CUDA 빌드에 맞는 커널이 없는 경우)
# 그래서 아래에서 실제 모델을 돌려 본다.


# ── 2단계: 한글이 들어간 시험용 이미지 만들기 ────────────────────────────────
def make_test_image(out_path: Path) -> Path:
    """PyMuPDF로 한글 문장이 담긴 PNG 이미지를 만든다.

    별도 샘플 파일을 준비하지 않아도 검증할 수 있도록, 윈도우 기본 글꼴인
    맑은 고딕으로 문장을 그린 뒤 300dpi 이미지로 저장한다.
    """
    import pymupdf

    font_file = Path("C:/Windows/Fonts/malgun.ttf")
    if not font_file.exists():
        print("  [경고] 맑은 고딕 글꼴을 찾지 못해 영문으로만 시험합니다.")

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=250)  # A4 너비, 낮은 높이

    if font_file.exists():
        page.insert_font(fontname="malgun", fontfile=str(font_file))
        page.insert_text((40, 80), "한글 인식 시험 문장입니다.",
                         fontname="malgun", fontsize=24)
        page.insert_text((40, 130), "대한민국 서울특별시 2026년",
                         fontname="malgun", fontsize=24)
    page.insert_text((40, 180), "Searchable PDF Test 12345", fontsize=24)

    # 300dpi로 이미지화 (기본 72dpi 대비 300/72 배 확대)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72))
    pix.save(str(out_path))
    doc.close()
    return out_path


# ── 3단계: RapidOCR을 CUDA 모드로 초기화하고 실제로 돌려 보기 ────────────────
def run_ocr(use_cuda: bool, img_path: Path):
    """실제 프로그램과 똑같은 설정으로 OCR을 한 번 수행하고 결과를 돌려준다.

    설정을 여기서 따로 적지 않고 app/ocr_engine.py의 것을 그대로 쓴다.
    그래야 "검증은 통과했는데 실제 변환은 다르게 동작"하는 일이 생기지 않는다.
    """
    from app.ocr_engine import get_engine

    engine, _ = get_engine(use_gpu=use_cuda)

    # 실제로 어떤 장치에서 도는지 확인한다.
    # 세션 목록의 맨 앞이 CUDAExecutionProvider여야 GPU로 도는 것이다.
    providers = {}
    for name in ("text_det", "text_cls", "text_rec"):
        part = getattr(engine, name, None)
        session = getattr(getattr(part, "session", None), "session", None)
        if session is None:
            session = getattr(part, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            providers[name] = session.get_providers()[0]

    # 첫 호출은 모델을 GPU에 올리는 준비 시간이 포함되므로 두 번 재서
    # 두 번째(실제 처리 속도)를 기록한다.
    engine(str(img_path))
    started = time.perf_counter()
    result = engine(str(img_path))
    elapsed = time.perf_counter() - started

    return providers, result, elapsed


def main() -> int:
    tmp_dir = Path(__file__).resolve().parent
    img_path = make_test_image(tmp_dir / "_gpu_test.png")

    print()
    print("=" * 60)
    print("2. 시험용 이미지 생성")
    print("=" * 60)
    print(f"  경로 : {img_path}")

    print()
    print("=" * 60)
    print("3. GPU(CUDA) 모드 OCR 시험")
    print("=" * 60)
    print("  ※ 처음 실행 시 인식 모델을 내려받느라 시간이 걸릴 수 있습니다.")

    try:
        providers, result, elapsed = run_ocr(use_cuda=True, img_path=img_path)
    except Exception as exc:
        print(f"  [실패] GPU 모드 실행 중 오류: {type(exc).__name__}: {exc}")
        return 1

    for part, provider in providers.items():
        mark = "GPU" if "CUDA" in provider else "CPU"
        print(f"  {part:9s} → {provider}  [{mark}]")

    on_gpu = all("CUDA" in p for p in providers.values())
    print(f"  처리 시간 : {elapsed:.3f}초")
    print(f"  판정      : {'GPU 가속 정상' if on_gpu else 'GPU 가속 실패 (CPU로 처리됨)'}")

    print()
    print("=" * 60)
    print("4. 한글 인식 결과")
    print("=" * 60)
    texts = list(result.txts) if result.txts else []
    if not texts:
        print("  [실패] 인식된 글자가 없습니다.")
        return 1
    for text, score in zip(texts, result.scores):
        print(f"  ({score:.2f}) {text}")

    has_hangul = any(any("가" <= ch <= "힣" for ch in t) for t in texts)
    print()
    print(f"  한글 포함 여부 : {'정상' if has_hangul else '한글 미검출'}")

    print()
    print("=" * 60)
    print("최종 판정")
    print("=" * 60)
    if on_gpu and has_hangul:
        print("  성공 - GPU 가속과 한글 인식이 모두 정상입니다.")
        return 0
    if has_hangul:
        print("  주의 - 한글 인식은 되지만 CPU로 처리되고 있습니다.")
        print("         그래픽 드라이버 버전과 CUDA 설치 상태를 확인해 주세요.")
        return 2
    print("  실패 - 위 출력 내용을 확인해 주세요.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
