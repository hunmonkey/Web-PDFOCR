# -*- coding: utf-8 -*-
"""
PDFOCR 설치 도우미

이 PC의 그래픽카드, 윈도우 버전, CPU를 살펴보고 알맞은 부품을 골라 설치한다.

왜 이런 도우미가 필요한가:
  글자를 읽는 실행 부품은 세 종류가 있는데(NVIDIA용 / AMD·Intel용 / CPU용),
  셋 다 이름이 'onnxruntime'으로 같아서 한 PC에 하나만 깔 수 있다.
  잘못 고르면 그래픽카드가 있어도 못 쓰거나, 아예 실행이 안 된다.

사용법:
    .venv\\Scripts\\python.exe scripts\\setup.py           (알아서 고름)
    .venv\\Scripts\\python.exe scripts\\setup.py --check   (살펴보기만 하고 설치는 안 함)
    .venv\\Scripts\\python.exe scripts\\setup.py --backend cpu
        (자동 판단이 마음에 들지 않을 때 직접 지정: cuda / directml / cpu)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.system_check import (  # noqa: E402
    PACKAGE_CPU,
    PACKAGE_CUDA,
    PACKAGE_DIRECTML,
    collect,
    describe,
    recommend_package,
)

# 직접 지정할 때 쓰는 이름표
BACKEND_PACKAGES = {
    "cuda": PACKAGE_CUDA,
    "directml": PACKAGE_DIRECTML,
    "cpu": PACKAGE_CPU,
}

# 서로 부딪히는 부품들. 새로 깔기 전에 모두 지운다.
CONFLICTING = ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"]


def run_pip(*args: str) -> int:
    """이 가상환경의 pip을 실행한다."""
    command = [sys.executable, "-m", "pip", *args]
    print(f"\n$ {' '.join(command[2:])}")
    return subprocess.run(command).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="PDFOCR 설치 도우미")
    parser.add_argument("--check", action="store_true",
                        help="살펴보기만 하고 설치하지 않습니다.")
    parser.add_argument("--backend", choices=sorted(BACKEND_PACKAGES),
                        help="자동 판단 대신 직접 고릅니다.")
    options = parser.parse_args()

    print("=" * 64)
    print(" 이 PC를 살펴보는 중입니다")
    print("=" * 64)
    info = collect()
    for line in describe(info):
        print("  " + line)

    print()
    print("=" * 64)
    print(" 판단 결과")
    print("=" * 64)

    if options.backend:
        package = BACKEND_PACKAGES[options.backend]
        print(f"  직접 지정하신 방식으로 설치합니다: {options.backend}")
    else:
        package, reason = recommend_package(info)
        print(f"  {reason}")

    print(f"  설치할 부품: {package}")

    if options.check:
        print("\n  (--check 이므로 실제로 설치하지는 않았습니다.)")
        return 0

    print()
    print("=" * 64)
    print(" 설치를 시작합니다")
    print("=" * 64)
    print("  내려받는 양이 많아 몇 분 걸릴 수 있습니다.")

    # 1) 공통 패키지
    if run_pip("install", "-r", str(ROOT / "requirements.txt")) != 0:
        print("\n공통 패키지 설치에 실패했습니다. 위 내용을 확인해 주세요.")
        return 1

    # 2) 서로 부딪히는 실행 부품을 먼저 지운다.
    #    (이름이 같아 겹쳐 깔면 그래픽카드를 못 잡는 일이 생긴다)
    installed = [name for name in CONFLICTING if _is_installed(name)]
    if installed:
        print(f"\n기존 실행 부품을 정리합니다: {', '.join(installed)}")
        run_pip("uninstall", "-y", *installed)

    # 3) 이 PC에 맞는 실행 부품
    if run_pip("install", package) != 0:
        print("\n실행 부품 설치에 실패했습니다. 위 내용을 확인해 주세요.")
        return 1

    # 4) 제대로 깔렸는지 확인
    print()
    print("=" * 64)
    print(" 설치 결과 확인")
    print("=" * 64)
    return _verify()


def _is_installed(name: str) -> bool:
    from importlib.metadata import PackageNotFoundError, version
    try:
        version(name)
        return True
    except PackageNotFoundError:
        return False


def _verify() -> int:
    """설치가 끝난 뒤 실제로 어떤 방식이 잡히는지 확인한다.

    이 프로세스는 설치 전 상태로 이미 파이썬을 띄웠기 때문에,
    새 파이썬을 하나 더 띄워서 확인해야 정확하다.
    """
    # 확인 결과는 새 파이썬이 끝날 때의 번호로 알려 준다.
    #   0 = 그래픽 가속 사용 / 2 = CPU 사용 / 그 밖 = 확인 실패
    # 출력을 글자로 받아서 뒤지지 않고 번호로 판단하는 이유는,
    # 출력을 가로채는 방식이 상황에 따라 빈 값으로 오는 일이 있었기 때문이다.
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from app.ocr_engine import CUDA_DLL_DIRS;"
        "from app.system_check import BACKEND_CPU, active_backend, collect;"
        "info = collect();"
        "backend = active_backend();"
        "print('  설치된 부품 :', info.installed_package);"
        "print('  실제 실행방식:', backend);"
        "sys.exit(2 if backend == BACKEND_CPU else 0)"
    ) % str(ROOT)

    # 자식 쪽 출력을 그대로 화면에 흘려보낸다.
    # 그 전에 여기까지 찍은 글을 먼저 내보내야 두 출력이 뒤섞이지 않는다.
    sys.stdout.flush()
    done = subprocess.run([sys.executable, "-c", code])
    sys.stdout.flush()

    if done.returncode == 0:
        print("\n  그래픽 가속을 쓸 준비가 되었습니다.")
    elif done.returncode == 2:
        print("\n  CPU로 처리합니다. 동작에는 문제가 없지만 그래픽 가속보다 느립니다.")
    else:
        print("\n  확인에 실패했습니다. 위 내용을 확인해 주세요.")
        return 1

    print("\n다음 명령으로 실제 글자 인식까지 확인해 보세요.")
    print(r"    .venv\Scripts\python.exe scripts\check_gpu.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
