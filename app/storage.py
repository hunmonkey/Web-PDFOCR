# -*- coding: utf-8 -*-
"""
서버 보관 폴더(GRM-BOX) 다루기

운영서버의 /mnt/user/GRM-BOX/PDFOCR 아래를 쓴다.

    PDFOCR/
    ├── 원본/       ← 스캔한 PDF를 여기에 넣어 두면 화면에서 골라 변환할 수 있다
    └── 변환완료/   ← 변환이 끝난 PDF가 여기에 쌓인다

★ 가장 중요한 것: 경로를 벗어나지 못하게 막는 일
  화면에서 넘어온 파일 이름을 그대로 믿고 열면, "../../etc/passwd" 같은 값을 넣어
  서버의 아무 파일이나 읽어 갈 수 있다. 그래서 넘어온 경로가 정말 이 폴더 안인지
  매번 확인한다. (resolve_inside 함수)
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 도커에서는 환경값으로 지정한다. 지정이 없으면 개발 PC의 작업 폴더를 쓴다.
STORAGE_ROOT = Path(os.environ.get("PDFOCR_STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage")))

SOURCE_DIR_NAME = "원본"
RESULT_DIR_NAME = "변환완료"

SOURCE_DIR = STORAGE_ROOT / SOURCE_DIR_NAME
RESULT_DIR = STORAGE_ROOT / RESULT_DIR_NAME

# 파일 이름에 쓰면 곤란한 글자들 (윈도우/리눅스 공통으로 안전하게)
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESULT_INDEX = r'(?:\((?:[2-9]|[1-9]\d+)\))?'
_RESULT_FILE = re.compile(rf'^.+_Searchable{_RESULT_INDEX}\.pdf$')
_RESULT_PART_FILE = re.compile(
    rf'^\.(?P<final>.+_Searchable{_RESULT_INDEX}\.pdf)\.[0-9a-f]{{32}}\.part$'
)


@dataclass
class StoredFile:
    name: str
    size: int
    modified: str      # 사람이 읽는 형태
    relative: str      # 폴더 안에서의 상대 경로


@dataclass(frozen=True)
class ReservedResultPath:
    """결과 이름과 실제 저장에 쓰는 같은 폴더의 임시 파일."""

    final_path: Path
    part_path: Path


def ensure_dirs() -> None:
    """보관 폴더가 없으면 만든다."""
    for path in (SOURCE_DIR, RESULT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_inside(base: Path, relative: str) -> Path:
    """base 폴더 안의 경로로 바꿔 준다. 밖으로 나가면 거부한다.

    resolve()는 '..' 이나 바로가기(심볼릭 링크)를 모두 풀어 실제 경로로 만든다.
    그 결과가 base 안에 있는지 확인해야 진짜로 막을 수 있다.
    글자만 보고 '..'를 걸러내는 방식은 우회당하기 쉬워 쓰지 않는다.
    """
    if not relative:
        raise ValueError("파일을 지정해 주세요.")

    base_real = base.resolve()
    target = (base_real / relative).resolve()

    if target != base_real and base_real not in target.parents:
        raise ValueError("허용되지 않은 경로입니다.")
    return target


def list_source_files() -> list[StoredFile]:
    """'원본' 폴더 안의 PDF 목록을 최근 것부터 돌려준다."""
    ensure_dirs()
    base = SOURCE_DIR.resolve()
    found: list[StoredFile] = []

    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append(StoredFile(
            name=path.name,
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            relative=path.relative_to(base).as_posix(),
        ))

    found.sort(key=lambda item: item.modified, reverse=True)
    return found


def safe_file_name(name: str) -> str:
    """파일 이름에서 위험한 글자를 걸러낸다."""
    cleaned = _UNSAFE_CHARS.sub("_", name).strip().strip(".")
    return cleaned or "문서"


def _result_candidate(original_name: str, index: int) -> Path:
    """원본 이름과 순번으로 결과 파일 후보를 만든다."""
    stem = safe_file_name(Path(original_name).stem)
    suffix = "" if index == 1 else f"({index})"
    return RESULT_DIR / f"{stem}_Searchable{suffix}.pdf"


def result_path_for(original_name: str) -> Path:
    """비어 있는 결과 파일 이름을 조회한다.

    기존 호출과의 호환을 위해 남겨 둔 조회 함수다. 실제 변환 작업은 조회와 예약이
    한 동작으로 이뤄지는 :func:`reserve_result_path`를 사용해야 한다.
    """
    ensure_dirs()
    index = 1
    candidate = _result_candidate(original_name, index)

    while candidate.exists():
        index += 1
        candidate = _result_candidate(original_name, index)
    return candidate


def reserve_result_path(original_name: str) -> ReservedResultPath:
    """기존 파일을 덮어쓰지 않도록 결과 이름을 원자적으로 예약한다.

    ``exists()`` 확인만으로는 확인 직후 다른 작업이 같은 이름을 만들 수 있다.
    ``O_CREAT | O_EXCL``은 이름 확인과 빈 예약 파일 생성을 한 동작으로 처리한다.
    완성 PDF는 같은 폴더의 고유 ``.part`` 파일에 쓴 뒤 예약 파일과 교체한다.
    """
    ensure_dirs()
    create_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    create_flags |= getattr(os, "O_BINARY", 0)
    index = 1

    while True:
        final_path = _result_candidate(original_name, index)
        try:
            descriptor = os.open(final_path, create_flags, 0o666)
        except FileExistsError:
            index += 1
            continue

        os.close(descriptor)
        part_path = final_path.with_name(
            f".{final_path.name}.{uuid.uuid4().hex}.part"
        )
        try:
            part_descriptor = os.open(part_path, create_flags, 0o666)
        except Exception:
            final_path.unlink(missing_ok=True)
            raise
        os.close(part_descriptor)
        return ReservedResultPath(final_path=final_path, part_path=part_path)


def reserve_sidecar_path(final_pdf_path: Path, extension: str) -> Path:
    """결과 PDF 옆에 둘 텍스트 파일 이름을 겹치지 않게 예약한다.

    결과 PDF 이름은 이미 겹치지 않게 정해져 있지만, 그 PDF만 지우고 텍스트 파일은
    남겨 둔 경우가 있을 수 있다. 남의 파일을 말없이 덮어쓰지 않도록 여기서도
    ``O_CREAT | O_EXCL``로 확인과 생성을 한 동작으로 처리한다.
    """
    create_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    create_flags |= getattr(os, "O_BINARY", 0)
    stem = final_pdf_path.stem
    index = 1

    while True:
        suffix = "" if index == 1 else f"({index})"
        candidate = final_pdf_path.with_name(f"{stem}{suffix}.{extension}")
        try:
            descriptor = os.open(candidate, create_flags, 0o666)
        except FileExistsError:
            index += 1
            continue
        os.close(descriptor)
        return candidate


def commit_reserved_result(reservation: ReservedResultPath) -> None:
    """완성된 임시 PDF를 예약해 둔 최종 이름으로 원자적으로 교체한다."""
    os.replace(reservation.part_path, reservation.final_path)


def cleanup_orphaned_results() -> int:
    """비정상 종료 뒤 남은 앱 전용 임시 결과만 지운다.

    내용이 있는 PDF, 바로가기, 하위 폴더, 이름 모양이 다른 파일은 건드리지
    않는다. 운영 앱은 한 프로세스로 실행된다는 현재 Docker 구성을 전제로 한다.
    """
    ensure_dirs()
    removed = 0
    with os.scandir(RESULT_DIR) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue

            remove = _RESULT_PART_FILE.fullmatch(entry.name) is not None
            if not remove and _RESULT_FILE.fullmatch(entry.name) is not None:
                try:
                    remove = entry.stat(follow_symlinks=False).st_size == 0
                except OSError:
                    continue
            if not remove:
                continue

            try:
                Path(entry.path).unlink()
                removed += 1
            except OSError:
                # 잠긴 파일은 다음 시작 때 다시 확인한다. 정상 결과를 넓게 지우는
                # 것보다 한 번 보존하는 편이 안전하다.
                continue
    return removed
