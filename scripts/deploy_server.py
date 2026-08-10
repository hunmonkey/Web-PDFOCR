# -*- coding: utf-8 -*-
"""
PDFOCR 운영서버 배포 도우미

운영서버(Unraid, HWAN-IN)에 접속해 최신 코드를 받아 컨테이너를 다시 올린다.
사람이 SSH 창을 열고 명령을 하나씩 치던 일을 한 줄로 끝내기 위한 도구다.

접속에 필요한 값은 모두 루트의 Rule_PW.txt에서 읽는다.
그 파일은 .gitignore에 들어 있어 저장소에 올라가지 않는다.
이 파일에는 비밀번호나 토큰을 절대 적지 않는다.

사용법:
    .venv\\Scripts\\python.exe scripts\\deploy_server.py --check
        살펴보기만 한다. 서버의 어떤 것도 바꾸지 않는다.

    .venv\\Scripts\\python.exe scripts\\deploy_server.py
        최신 코드를 받아 컨테이너를 다시 만들어 올린다. (몇 분 걸린다)

    .venv\\Scripts\\python.exe scripts\\deploy_server.py --branch dev
        main이 아닌 다른 가지를 반영하고 싶을 때. (기본값은 main)

처음 한 번은 접속 부품을 깔아야 한다:
    .venv\\Scripts\\python.exe -m pip install paramiko
"""

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PW_FILE = ROOT / "Rule_PW.txt"

# 윈도우 명령창의 기본 글자표(cp949)로는 찍히지 않는 기호가 있어 UTF-8로 맞춘다.
# 맞추지 않으면 화면에 글을 찍다가 프로그램이 그대로 멈춰 버린다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 운영서버 안에서 이 프로젝트가 놓인 자리와, 웹이 듣고 있는 문 번호
SRC_DIR = "/mnt/user/appdata/Web-PDFOCR/src"
CONTAINER = "Web-PDFOCR"
PORT = 2222
REPO_PATH = "github.com/hunmonkey/Web-PDFOCR.git"

# 컨테이너가 다 뜰 때까지 기다리는 한도
HEALTH_TRIES = 24
HEALTH_INTERVAL = 5


def die(message: str) -> None:
    print(f"\n[중단] {message}")
    sys.exit(1)


def read_secret(label: str) -> str:
    """Rule_PW.txt에서 값 하나를 읽는다. 읽기만 하고 화면에는 내보내지 않는다."""
    if not PW_FILE.is_file():
        die(f"{PW_FILE.name} 파일이 없습니다. 운영서버 접속 정보가 있어야 배포할 수 있습니다.")

    text = PW_FILE.read_text(encoding="utf-8")
    found = re.search(rf"^- {re.escape(label)}:\s*(.+)$", text, re.M)
    if not found:
        die(f"{PW_FILE.name}에서 '{label}' 값을 찾지 못했습니다.")
    return found.group(1).strip()


class Server:
    """운영서버와 이야기하는 통로."""

    def __init__(self) -> None:
        try:
            import paramiko
        except ImportError:
            die(
                "접속 부품이 없습니다. 아래 명령을 한 번 실행해 주세요.\n"
                "    .venv\\Scripts\\python.exe -m pip install paramiko"
            )

        self.host = read_secret("Unraid 서버 호스트(UNRAID_HOST)")
        self.port = int(read_secret("Unraid_SSH_PORT"))
        self.user = read_secret("Unraid_SSH_USER")
        self._password = read_secret("Unraid_SSH_PASSWORD")
        self._token = read_secret("Github_Access_Token(Classic)")
        self._github_user = read_secret("Github_User_ID")

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(
                hostname=self.host, port=self.port, username=self.user,
                password=self._password, timeout=20,
                allow_agent=False, look_for_keys=False,
            )
        except Exception as error:
            die(f"운영서버에 접속하지 못했습니다. ({type(error).__name__}: {error})")

    @property
    def fetch_url(self) -> str:
        """토큰을 담은 주소. 받아 오는 그 한 번의 명령에만 쓰고 서버에 남기지 않는다."""
        return f"https://{self._github_user}:{self._token}@{REPO_PATH}"

    def mask(self, text: str) -> str:
        """화면에 나갈 글에서 비밀번호와 토큰을 가린다."""
        for secret in (self._token, self._password):
            if secret:
                text = text.replace(secret, "***")
        return text

    def run(self, command: str, timeout: int = 900):
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err

    def show(self, label: str, command: str, timeout: int = 900) -> int:
        """명령 하나를 돌리고 결과를 보기 좋게 찍는다."""
        code, out, err = self.run(command, timeout)
        print(f"\n── {label}")
        for line in (out.strip(), err.strip()):
            if line:
                print("   " + self.mask(line).replace("\n", "\n   "))
        return code

    def close(self) -> None:
        self.client.close()


def inspect(server: Server) -> None:
    """서버를 살펴보기만 한다. 아무것도 바꾸지 않는다."""
    print("\n███ 서버 살펴보기 (아무것도 바꾸지 않습니다)")
    server.show("서버 이름과 시각", "hostname; date")
    server.show("지금 올라가 있는 코드", f"cd {SRC_DIR} && git log --oneline -1 && git branch --show-current")
    server.show("손대지 않은 상태인지", f"cd {SRC_DIR} && git status --short | head -10 || echo '   (깨끗함)'")
    server.show("컨테이너 상태", f"cd {SRC_DIR} && docker compose ps")
    server.show("웹이 응답하는지",
                f"curl -s -o /dev/null -w '응답코드 %{{http_code}}' http://127.0.0.1:{PORT}/login")
    server.show("디스크 여유", "df -h /mnt/user | tail -1")


def pull(server: Server, branch: str) -> None:
    """최신 코드를 받아 온다."""
    print(f"\n███ 1단계. 최신 코드 받기 ({branch} 가지)")

    code, out, _ = server.run(f"cd {SRC_DIR} && git status --porcelain")
    if out.strip():
        print("   " + out.strip().replace("\n", "\n   "))
        die("서버에서 직접 고친 파일이 있습니다. 덮어쓰지 않도록 멈췄습니다. 먼저 정리해 주세요.")

    if server.show("받아 오기",
                   f"cd {SRC_DIR} && git fetch '{server.fetch_url}' {branch} 2>&1 | tail -5") != 0:
        die("코드를 받아 오지 못했습니다.")

    # --ff-only: 서버 쪽에만 있는 딴 커밋이 있으면 억지로 합치지 않고 멈춘다.
    if server.show("받은 것으로 맞추기",
                   f"cd {SRC_DIR} && git merge --ff-only FETCH_HEAD 2>&1 | tail -5") != 0:
        die("받은 코드로 맞추지 못했습니다. 서버 쪽에만 있는 커밋이 있는지 확인해 주세요.")

    server.show("지금 올라간 코드", f"cd {SRC_DIR} && git log --oneline -1")
    server.show("토큰이 남지 않았는지 확인",
                f"cd {SRC_DIR} && git remote -v | head -1; "
                "test -f /root/.git-credentials && echo '주의: 자격증명 파일이 생겼습니다' "
                "|| echo '자격증명 파일 없음 (정상)'")


def rebuild(server: Server) -> None:
    """컨테이너를 다시 만들어 올린다."""
    print("\n███ 2단계. 컨테이너 다시 만들어 올리기 (몇 분 걸립니다)")
    if server.show("빌드와 실행",
                   f"cd {SRC_DIR} && docker compose up -d --build 2>&1 | tail -20",
                   timeout=2400) != 0:
        die("컨테이너를 올리지 못했습니다. 위 내용을 확인해 주세요.")


def verify(server: Server) -> None:
    """실제로 응답하는지 확인한다."""
    print("\n███ 3단계. 확인")

    for attempt in range(1, HEALTH_TRIES + 1):
        _, out, _ = server.run(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/login"
        )
        answer = out.strip()
        if answer == "200":
            print(f"\n── 웹 응답 확인: 200 ({attempt}번째 시도)")
            break
        print(f"   컨테이너가 다 뜨기를 기다리는 중… (응답 {answer or '없음'})")
        time.sleep(HEALTH_INTERVAL)
    else:
        die(f"{HEALTH_TRIES * HEALTH_INTERVAL}초를 기다렸지만 정상 응답을 받지 못했습니다. "
            f"'docker logs {CONTAINER}' 로 원인을 확인해 주세요.")

    server.show("컨테이너 상태", f"cd {SRC_DIR} && docker compose ps")
    server.show("시작 기록", f"docker logs --tail 10 {CONTAINER} 2>&1")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDFOCR 운영서버 배포 도우미")
    parser.add_argument("--check", action="store_true",
                        help="살펴보기만 하고 서버를 바꾸지 않는다")
    parser.add_argument("--branch", default="main",
                        help="반영할 가지 이름 (기본값: main)")
    args = parser.parse_args()

    server = Server()
    print(f"운영서버 {server.host} 에 접속했습니다.")
    try:
        if args.check:
            inspect(server)
            print("\n살펴보기를 마쳤습니다. 서버는 그대로입니다.")
            return

        inspect(server)
        pull(server, args.branch)
        rebuild(server)
        verify(server)
        print("\n반영을 마쳤습니다. https://pdf.jaehoon.org 에서 확인해 주세요.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
