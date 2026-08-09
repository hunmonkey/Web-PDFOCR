# -*- coding: utf-8 -*-
"""
실행 환경 진단 및 한글 OCR 확인 도구

"설치가 됐다"가 아니라 "이 PC에서 실제로 한글이 읽힌다"까지 확인한다.
PC를 바꿀 때마다 한 번씩 돌려 보는 용도다.

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
from app.ocr_engine import BACKEND_CPU, CUDA_DLL_DIRS, ort
from app.system_check import collect, describe, recommend_package

_cuda_dirs = CUDA_DLL_DIRS

print("=" * 60)
print("1. 이 PC의 상태")
print("=" * 60)
_info = collect()
for _line in describe(_info):
    print("  " + _line)

_recommended, _reason = recommend_package(_info)
if _recommended.split("[")[0].split("==")[0] not in _info.installed_package:
    print()
    print("  [알림] 지금 설치된 부품이 이 PC에 가장 알맞은 것이 아닐 수 있습니다.")
    print(f"         {_reason}")
    print(f"         권장: {_recommended}")
    print(r"         바꾸려면: .venv\Scripts\python.exe scripts\setup.py")

print()
print("=" * 60)
print("2. 실행 부품 정보")
print("=" * 60)
print(f"  버전            : {ort.__version__}")
print(f"  사용 가능 목록  : {ort.get_available_providers()}")
print(f"  CUDA DLL 폴더   : {len(_cuda_dirs)}곳 등록됨")

# 목록에 이름이 있다고 해서 실제 계산까지 성공한다는 보장은 없다.
# (예: 그래픽카드가 너무 최신이라 맞는 계산 방식이 들어 있지 않은 경우)
# 그래서 아래에서 실제 모델을 돌려 본다.


# ── 2단계: 한글이 들어간 시험용 이미지 만들기 ────────────────────────────────
def make_test_image(out_path: Path) -> Path:
    """PyMuPDF로 한글 문장이 담긴 PNG 이미지를 만든다.

    글꼴은 PyMuPDF에 들어 있는 한국어 글꼴(korea)을 쓴다.
    윈도우 글꼴(맑은 고딕)을 쓰면 도커 컨테이너처럼 그 글꼴이 없는 곳에서
    한글이 그려지지 않아 "한글 미검출"로 잘못 판정된다.
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=250)  # A4 너비, 낮은 높이

    page.insert_text((40, 80), "한글 인식 시험 문장입니다.",
                     fontname="korea", fontsize=24)
    page.insert_text((40, 130), "대한민국 서울특별시 2026년",
                     fontname="korea", fontsize=24)
    page.insert_text((40, 180), "Searchable PDF Test 12345", fontsize=24)

    # 300dpi로 이미지화 (기본 72dpi 대비 300/72 배 확대)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72))
    pix.save(str(out_path))
    doc.close()
    return out_path


# ── 실제로 돌려 보기 ─────────────────────────────────────────────────────────
def run_ocr(img_path: Path):
    """실제 프로그램과 똑같은 설정으로 OCR을 한 번 수행하고 결과를 돌려준다.

    설정을 여기서 따로 적지 않고 app/ocr_engine.py의 것을 그대로 쓴다.
    그래야 "확인은 통과했는데 실제 변환은 다르게 동작"하는 일이 생기지 않는다.
    """
    from app.ocr_engine import get_engine

    engine, backend = get_engine(use_gpu=True)

    # 부분별로 어느 장치에 올라갔는지 들여다본다.
    providers = {}
    for name in ("text_det", "text_rec"):
        part = getattr(engine, name, None)
        session = getattr(getattr(part, "session", None), "session", None)
        if session is not None and hasattr(session, "get_providers"):
            providers[name] = session.get_providers()[0]

    # 첫 호출에는 모델을 장치에 올리는 준비 시간이 섞이므로 두 번 재서
    # 두 번째(실제 처리 속도)를 기록한다.
    engine(str(img_path))
    started = time.perf_counter()
    result = engine(str(img_path))
    elapsed = time.perf_counter() - started

    return backend, providers, result, elapsed


def main() -> int:
    tmp_dir = Path(__file__).resolve().parent
    img_path = make_test_image(tmp_dir / "_gpu_test.png")

    print()
    print("=" * 60)
    print("3. 한글 인식 시험")
    print("=" * 60)
    print("  ※ 처음 실행 시 인식 모델을 내려받느라 시간이 걸릴 수 있습니다.")

    try:
        backend, providers, result, elapsed = run_ocr(img_path=img_path)
    except Exception as exc:
        print(f"  [실패] 실행 중 오류: {type(exc).__name__}: {exc}")
        return 1

    for part, provider in providers.items():
        korean = {"text_det": "글자 위치 찾기", "text_rec": "글자 읽기"}.get(part, part)
        print(f"  {korean:12s} → {provider}")

    on_gpu = backend != BACKEND_CPU
    print(f"  처리 시간   : {elapsed:.3f}초")
    print(f"  실행 방식   : {backend}")
    if on_gpu:
        print("  판정        : 그래픽 가속 정상")
    else:
        print("  판정        : CPU로 처리 중 (동작은 하지만 느립니다)")

    print()
    print("=" * 60)
    print("4. 읽어 낸 글자")
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
        print(f"  성공 - {backend} 가속과 한글 인식이 모두 정상입니다.")
        return 0
    if has_hangul:
        old_gpu = next((g for g in _info.gpus if g.too_old_for_cuda), None)
        if old_gpu is not None:
            # 이 경우는 고칠 방법이 없다. 괜한 삽질을 하지 않도록 분명히 알려 준다.
            print("  정상 - 한글 인식이 잘 되고 있습니다. CPU로 처리합니다.")
            print(f"         {old_gpu.name}은(는) 세대 번호가 {old_gpu.compute_capability}로 낮아")
            print("         그래픽 가속을 쓸 수 없습니다. 드라이버를 올려도 달라지지 않습니다.")
            print("         그래픽 가속을 쓰려면 세대 번호 7.0 이상 카드가 필요합니다.")
            return 0

        print("  주의 - 한글 인식은 되지만 CPU로 처리되고 있습니다.")
        print("         동작에는 문제가 없지만 그래픽 가속보다 느립니다.")
        print("         그래픽카드가 있는 PC라면 아래를 확인해 보세요.")
        print("           1) NVIDIA 그래픽 드라이버를 최신으로 올린다")
        print(r"           2) .venv\Scripts\python.exe scripts\setup.py 를 다시 실행한다")
        return 2
    print("  실패 - 위 출력 내용을 확인해 주세요.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
