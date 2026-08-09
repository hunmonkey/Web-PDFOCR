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


@dataclass
class StoredFile:
    name: str
    size: int
    modified: str      # 사람이 읽는 형태
    relative: str      # 폴더 안에서의 상대 경로


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


def result_path_for(original_name: str) -> Path:
    """결과 파일을 저장할 자리를 정한다.

    같은 이름이 이미 있으면 뒤에 번호를 붙여 기존 파일을 덮어쓰지 않는다.
    """
    ensure_dirs()
    stem = safe_file_name(Path(original_name).stem)
    candidate = RESULT_DIR / f"{stem}_Searchable.pdf"

    index = 2
    while candidate.exists():
        candidate = RESULT_DIR / f"{stem}_Searchable({index}).pdf"
        index += 1
    return candidate
