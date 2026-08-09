# -*- coding: utf-8 -*-
"""
로그인 처리

이 프로그램은 인터넷에 열려 있으므로(pdf.jaehoon.org) 로그인한 사람만 쓸 수 있어야 한다.

비밀번호를 다루는 방식:
  · 비밀번호를 그대로 저장하지 않는다. 되돌릴 수 없는 값(해시)으로 바꿔 저장한다.
    파일이 새어 나가도 원래 비밀번호를 알아낼 수 없게 하기 위해서다.
  · scrypt라는 방식을 쓴다. 일부러 계산을 무겁게 만들어,
    비밀번호를 하나씩 대입해 보는 공격을 느리게 만든다.
  · 맞는지 비교할 때는 hmac.compare_digest를 쓴다. 글자를 앞에서부터 하나씩 비교하면
    걸린 시간으로 정답을 추측할 수 있어서, 시간이 항상 같게 비교하는 함수를 쓴다.

처음 비밀번호는 admin/admin이지만, 첫 로그인 때 반드시 바꾸도록 막아 둔다.
admin/admin은 자동 공격 도구가 가장 먼저 시도하는 값이라 그대로 두면 위험하다.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 설정을 담아 두는 곳. 도커에서는 바깥 폴더와 연결해 두어야
# 컨테이너를 새로 만들어도 바꾼 비밀번호가 남는다.
CONFIG_DIR = Path(os.environ.get("PDFOCR_CONFIG_DIR", str(ROOT / "config")))
USERS_FILE = CONFIG_DIR / "users.json"

DEFAULT_USERNAME = os.environ.get("PDFOCR_ADMIN_USER", "admin")
DEFAULT_PASSWORD = os.environ.get("PDFOCR_ADMIN_PASSWORD", "admin")

SESSION_COOKIE = "pdfocr_session"
SESSION_HOURS = 12

# 비밀번호를 바꿀 때 요구하는 최소 길이
MIN_PASSWORD_LENGTH = 8


@dataclass
class User:
    username: str
    password_hash: str
    must_change_password: bool


# ─────────────────────────────────────────────────────────────────────────────
# 비밀번호 다루기
# ─────────────────────────────────────────────────────────────────────────────
def hash_password(password: str, salt: str | None = None) -> str:
    """비밀번호를 되돌릴 수 없는 값으로 바꾼다.

    salt는 사람마다 다른 소금 같은 값이다. 같은 비밀번호라도 결과가 달라지므로,
    미리 계산해 둔 표로 비밀번호를 알아내는 공격을 막을 수 있다.
    """
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        n=2 ** 14, r=8, p=1, dklen=32,
    )
    return f"scrypt${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """입력한 비밀번호가 저장된 것과 같은지 확인한다."""
    try:
        method, salt, _ = stored.split("$", 2)
    except ValueError:
        return False
    if method != "scrypt":
        return False
    # 시간이 항상 같게 비교해서, 걸린 시간으로 정답을 짐작하지 못하게 한다.
    return hmac.compare_digest(hash_password(password, salt), stored)


# ─────────────────────────────────────────────────────────────────────────────
# 사용자 정보 파일
# ─────────────────────────────────────────────────────────────────────────────
def load_user() -> User:
    """저장된 사용자를 읽는다. 없으면 처음 계정을 만들어 준다."""
    if not USERS_FILE.exists():
        return _create_default_user()

    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        return User(
            username=data["username"],
            password_hash=data["password_hash"],
            must_change_password=bool(data.get("must_change_password", False)),
        )
    except (json.JSONDecodeError, KeyError, OSError):
        # 파일이 깨졌으면 처음 상태로 되돌린다.
        return _create_default_user()


def _create_default_user() -> User:
    user = User(
        username=DEFAULT_USERNAME,
        password_hash=hash_password(DEFAULT_PASSWORD),
        # 처음 비밀번호는 누구나 아는 값이므로 반드시 바꾸게 한다.
        must_change_password=True,
    )
    save_user(user)
    return user


def save_user(user: User) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(
        json.dumps({
            "username": user.username,
            "password_hash": user.password_hash,
            "must_change_password": user.must_change_password,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def change_password(new_password: str) -> tuple[bool, str]:
    """비밀번호를 바꾼다. (성공 여부, 안내 문구)"""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return False, f"비밀번호는 {MIN_PASSWORD_LENGTH}자 이상이어야 합니다."
    if new_password == DEFAULT_PASSWORD:
        return False, "처음 비밀번호와 다른 값으로 정해 주세요."

    user = load_user()
    user.password_hash = hash_password(new_password)
    user.must_change_password = False
    save_user(user)
    _sessions.clear()   # 비밀번호를 바꾸면 기존 접속을 모두 끊는다.
    return True, "비밀번호를 바꿨습니다. 다시 로그인해 주세요."


# ─────────────────────────────────────────────────────────────────────────────
# 접속 상태(세션)
# ─────────────────────────────────────────────────────────────────────────────
# 서버가 기억하는 접속 목록. 서버를 다시 켜면 모두 지워져 다시 로그인해야 한다.
_sessions: dict[str, float] = {}


def login(username: str, password: str) -> str | None:
    """아이디와 비밀번호가 맞으면 접속 표를 하나 만들어 준다."""
    user = load_user()
    # 아이디가 틀려도 비밀번호 확인 과정을 거쳐, 걸린 시간 차이로
    # 아이디가 있는지 없는지 알아내지 못하게 한다.
    name_ok = hmac.compare_digest(username, user.username)
    password_ok = verify_password(password, user.password_hash)
    if not (name_ok and password_ok):
        return None

    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + SESSION_HOURS * 3600
    return token


def is_logged_in(token: str | None) -> bool:
    """접속 표가 아직 쓸 수 있는지 확인한다."""
    if not token:
        return False
    expires_at = _sessions.get(token)
    if expires_at is None:
        return False
    if expires_at < time.time():
        _sessions.pop(token, None)
        return False
    return True


def logout(token: str | None) -> None:
    if token:
        _sessions.pop(token, None)
