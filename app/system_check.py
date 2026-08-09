# -*- coding: utf-8 -*-
"""
실행 환경 점검

이 PC가 어떤 그래픽카드를 갖고 있고, 윈도우 버전이 무엇이며, CPU가 몇 개인지 알아내서
"이 PC에는 어떤 방식으로 글자를 읽게 하는 것이 가장 좋은가"를 정한다.

세 가지 실행 방식이 있다.
  · CUDA      : NVIDIA 그래픽카드 전용. 가장 빠르다.
  · DirectML  : 윈도우가 제공하는 공용 방식. AMD·Intel 그래픽에서도 쓸 수 있다.
  · CPU       : 그래픽카드를 쓰지 않는다. 가장 느리지만 어디서나 된다.

주의: 이 세 가지를 담당하는 부품(onnxruntime-gpu / -directml / 기본)은 이름이 같아서
      한 PC에 하나만 설치할 수 있다. 그래서 설치할 때 골라야 한다.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

# 실행 방식 이름
BACKEND_CUDA = "CUDA"
BACKEND_DIRECTML = "DirectML"
BACKEND_CPU = "CPU"

# 설치할 부품 이름
PACKAGE_CUDA = "onnxruntime-gpu[cuda,cudnn]==1.26.0"
PACKAGE_DIRECTML = "onnxruntime-directml==1.24.4"
PACKAGE_CPU = "onnxruntime==1.28.0"

# DirectML은 윈도우 10의 이 빌드부터 쓸 수 있다.
# (rapidocr도 같은 기준으로 판단한다)
MIN_BUILD_FOR_DIRECTML = 18362

# CUDA 12를 쓰려면 필요한 최소 NVIDIA 드라이버 버전(윈도우 기준).
# NVIDIA가 공개한 CUDA 12 요구 사항이며, 이 값 자체를 직접 시험해 보지는 않았다.
MIN_NVIDIA_DRIVER_FOR_CUDA12 = 527.41

# 그래픽카드가 얼마나 새것인지 나타내는 번호(compute capability)의 최소 기준.
#
# 드라이버가 최신이어도 이 번호가 낮으면 CUDA 가속을 쓸 수 없다.
# onnxruntime을 만들 때 오래된 그래픽카드용 계산 코드를 빼고 만들기 때문이다.
#
# 실제로 겪은 일: 운영서버의 Quadro P620(번호 6.1)에서
# "no kernel image is available for execution on the device" 오류가 나면서 실패했다.
# 반면 RTX 5070 Ti(번호 12.0)는 정상 동작했다.
# 그래서 기준을 7.0으로 잡았다. (7.0~11.x 사이는 직접 확인하지 못했다)
MIN_COMPUTE_CAPABILITY = 7.0


@dataclass
class GpuInfo:
    name: str
    vendor: str            # NVIDIA / AMD / Intel / 기타
    driver_version: str = ""
    compute_capability: float = 0.0   # NVIDIA만 해당. 0이면 알아내지 못한 것

    @property
    def is_nvidia(self) -> bool:
        return self.vendor == "NVIDIA"

    @property
    def too_old_for_cuda(self) -> bool:
        """CUDA 가속을 쓰기에 너무 오래된 카드인지."""
        if not self.compute_capability:
            return False   # 알아내지 못했으면 일단 막지 않는다
        return self.compute_capability < MIN_COMPUTE_CAPABILITY


@dataclass
class SystemInfo:
    is_windows: bool = True
    windows_edition: str = ""    # 리눅스에서는 배포판 이름이 들어간다
    windows_build: int = 0       # 리눅스에서는 0
    cpu_name: str = ""
    cpu_cores: int = 0          # 진짜 코어 수
    cpu_threads: int = 0        # 동시에 처리할 수 있는 갈래 수
    ram_gb: float = 0.0
    gpus: list[GpuInfo] = field(default_factory=list)

    # 지금 설치되어 있는 것
    installed_package: str = ""      # onnxruntime-gpu 등. 없으면 빈 문자열
    available_providers: list[str] = field(default_factory=list)

    @property
    def nvidia_gpu(self) -> GpuInfo | None:
        for gpu in self.gpus:
            if gpu.is_nvidia:
                return gpu
        return None

    @property
    def supports_directml(self) -> bool:
        # DirectML은 윈도우가 제공하는 기능이므로 리눅스에서는 쓸 수 없다.
        # 윈도우 10 빌드 18362 이상이고, 화면을 그리는 장치가 하나라도 있어야 한다.
        if not self.is_windows:
            return False
        return self.windows_build >= MIN_BUILD_FOR_DIRECTML and bool(self.gpus)


# ─────────────────────────────────────────────────────────────────────────────
# 정보 모으기
# ─────────────────────────────────────────────────────────────────────────────
_PS_QUERY = r"""
$ErrorActionPreference = 'SilentlyContinue'
$gpus = Get-CimInstance Win32_VideoController |
        Select-Object Name, DriverVersion
$cpu  = Get-CimInstance Win32_Processor |
        Select-Object Name, NumberOfCores, NumberOfLogicalProcessors -First 1
$sys  = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
$os   = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, BuildNumber
@{ gpus = @($gpus); cpu = $cpu; ram = $sys.TotalPhysicalMemory
   osName = $os.Caption; osBuild = $os.BuildNumber } | ConvertTo-Json -Depth 4 -Compress
"""


def _vendor_of(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "rtx" in lowered:
        return "NVIDIA"
    if "amd" in lowered or "radeon" in lowered:
        return "AMD"
    if "intel" in lowered:
        return "Intel"
    return "기타"


def _query_linux() -> dict:
    """리눅스(도커 컨테이너 안)에서 하드웨어 정보를 모은다.

    윈도우와 달리 WMI가 없으므로 /proc 파일과 nvidia-smi로 알아낸다.
    그래픽카드는 nvidia-smi로만 확인한다. 도커 안에서는 nvidia 런타임으로
    넘겨받은 카드만 보이는 것이 맞기 때문이다.
    """
    result: dict = {"gpus": [], "osName": "", "osBuild": 0}

    try:
        with open("/etc/os-release", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("PRETTY_NAME="):
                    result["osName"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        result["osName"] = platform.system()

    model, cores = "", 0
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fp:
            for line in fp:
                if not model and line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                if line.startswith("cpu cores"):
                    cores = int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        pass

    threads = os.cpu_count() or 0
    result["cpu"] = {
        "Name": model or platform.processor(),
        "NumberOfCores": cores or threads,
        "NumberOfLogicalProcessors": threads,
    }

    try:
        with open("/proc/meminfo", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("MemTotal:"):
                    result["ram"] = int(line.split()[1]) * 1024  # kB -> 바이트
                    break
    except (OSError, ValueError, IndexError):
        pass

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            done = subprocess.run(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            )
            for line in (done.stdout or "").strip().splitlines():
                parts = [piece.strip() for piece in line.split(",")]
                if parts and parts[0]:
                    result["gpus"].append({
                        "Name": parts[0],
                        "DriverVersion": parts[1] if len(parts) > 1 else "",
                    })
        except (subprocess.SubprocessError, OSError):
            pass

    return result


def _query_windows() -> dict:
    """윈도우에게 하드웨어 정보를 물어본다. 실패하면 빈 값을 돌려준다."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return {}
    try:
        done = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", _PS_QUERY],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return json.loads(done.stdout) if done.stdout.strip() else {}
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return {}


def _nvidia_details() -> tuple[str, float]:
    """nvidia-smi로 드라이버 버전과 카드 세대 번호를 알아본다.

    NVIDIA가 없으면 빈 값과 0을 돌려준다.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return "", 0.0
    try:
        done = subprocess.run(
            [smi, "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        first = (done.stdout or "").strip().splitlines()[0]
        parts = [piece.strip() for piece in first.split(",")]
        driver = parts[0] if parts else ""
        capability = float(parts[1]) if len(parts) > 1 else 0.0
        return driver, capability
    except (subprocess.SubprocessError, OSError, IndexError, ValueError):
        return "", 0.0


def _installed_runtime() -> tuple[str, list[str]]:
    """지금 설치된 onnxruntime이 어떤 종류인지, 무엇을 쓸 수 있는지 알아본다."""
    from importlib.metadata import PackageNotFoundError, version

    installed = ""
    for name in ("onnxruntime-gpu", "onnxruntime-directml", "onnxruntime"):
        try:
            installed = f"{name}=={version(name)}"
            break
        except PackageNotFoundError:
            continue

    providers: list[str] = []
    if installed:
        try:
            import onnxruntime as ort
            providers = list(ort.get_available_providers())
        except Exception:
            providers = []
    return installed, providers


def collect() -> SystemInfo:
    """이 컴퓨터의 상태를 조사해서 돌려준다. (윈도우와 리눅스 모두)"""
    info = SystemInfo()
    info.is_windows = platform.system() == "Windows"
    raw = _query_windows() if info.is_windows else _query_linux()

    info.windows_edition = raw.get("osName") or platform.system()
    build = raw.get("osBuild") or (platform.version().split(".")[-1] if info.is_windows else 0)
    info.windows_build = int(build) if str(build).isdigit() else 0

    cpu = raw.get("cpu") or {}
    info.cpu_name = (cpu.get("Name") or platform.processor() or "").strip()
    info.cpu_cores = int(cpu.get("NumberOfCores") or 0)
    info.cpu_threads = int(cpu.get("NumberOfLogicalProcessors") or os.cpu_count() or 0)

    ram = raw.get("ram") or 0
    info.ram_gb = round(int(ram) / (1024 ** 3), 1) if ram else 0.0

    driver, capability = _nvidia_details()
    for entry in raw.get("gpus") or []:
        name = (entry.get("Name") or "").strip()
        if not name:
            continue
        vendor = _vendor_of(name)
        info.gpus.append(GpuInfo(
            name=name,
            vendor=vendor,
            # NVIDIA는 nvidia-smi 값이 더 정확하다. 나머지는 운영체제가 알려준 값을 쓴다.
            driver_version=driver if (vendor == "NVIDIA" and driver) else (entry.get("DriverVersion") or ""),
            compute_capability=capability if vendor == "NVIDIA" else 0.0,
        ))

    info.installed_package, info.available_providers = _installed_runtime()
    return info


# ─────────────────────────────────────────────────────────────────────────────
# 판단하기
# ─────────────────────────────────────────────────────────────────────────────
def recommend_package(info: SystemInfo) -> tuple[str, str]:
    """이 PC에 무엇을 설치하는 것이 좋은지 정한다. (설치할 것, 그 이유)"""
    nvidia = info.nvidia_gpu

    if nvidia is not None:
        # 카드가 너무 오래되면 드라이버가 최신이어도 CUDA 가속을 쓸 수 없다.
        if nvidia.too_old_for_cuda:
            reason = (
                f"{nvidia.name}은(는) 세대 번호가 {nvidia.compute_capability}로 낮아 "
                f"CUDA 가속을 쓸 수 없습니다. (최소 {MIN_COMPUTE_CAPABILITY} 필요) "
            )
            if info.supports_directml:
                return PACKAGE_DIRECTML, reason + "대신 DirectML로 설치합니다."
            return PACKAGE_CPU, reason + "CPU로 처리합니다."

        driver = _driver_as_number(nvidia.driver_version)
        if driver and driver < MIN_NVIDIA_DRIVER_FOR_CUDA12:
            if info.supports_directml:
                return PACKAGE_DIRECTML, (
                    f"{nvidia.name}이(가) 있지만 그래픽 드라이버가 {nvidia.driver_version}로 낮습니다. "
                    f"CUDA를 쓰려면 {MIN_NVIDIA_DRIVER_FOR_CUDA12} 이상이 필요합니다. "
                    "드라이버를 올리면 더 빨라집니다. 지금은 DirectML로 설치합니다."
                )
            return PACKAGE_CPU, (
                f"{nvidia.name}이(가) 있지만 드라이버가 낮고 DirectML도 쓸 수 없어 CPU로 설치합니다."
            )
        return PACKAGE_CUDA, f"{nvidia.name}에서 CUDA 가속을 쓸 수 있습니다. 가장 빠른 방식입니다."

    if info.supports_directml:
        names = ", ".join(gpu.name for gpu in info.gpus)
        return PACKAGE_DIRECTML, (
            f"NVIDIA 그래픽카드는 없지만 {names}에서 DirectML 가속을 쓸 수 있습니다."
        )

    if info.is_windows and info.windows_build and info.windows_build < MIN_BUILD_FOR_DIRECTML:
        return PACKAGE_CPU, (
            f"윈도우 빌드가 {info.windows_build}로 낮아 그래픽 가속을 쓸 수 없습니다. "
            f"(DirectML은 빌드 {MIN_BUILD_FOR_DIRECTML} 이상 필요)"
        )
    if not info.is_windows:
        return PACKAGE_CPU, (
            "NVIDIA 그래픽카드를 찾지 못했습니다. "
            "(리눅스에서는 DirectML을 쓸 수 없어 CPU로 처리합니다)"
        )
    return PACKAGE_CPU, "쓸 수 있는 그래픽 가속이 없어 CPU로 처리합니다."


def _driver_as_number(text: str) -> float:
    """드라이버 버전 문자열에서 앞쪽 숫자만 뽑아 비교할 수 있게 만든다."""
    parts = text.split(".")
    try:
        return float(f"{parts[0]}.{parts[1]}") if len(parts) >= 2 else float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def active_backend(providers: list[str] | None = None) -> str:
    """지금 설치된 것으로 실제로 어떤 방식을 쓰게 되는지 알려준다."""
    if providers is None:
        _, providers = _installed_runtime()
    if "CUDAExecutionProvider" in providers:
        return BACKEND_CUDA
    if "DmlExecutionProvider" in providers:
        return BACKEND_DIRECTML
    return BACKEND_CPU


def describe(info: SystemInfo) -> list[str]:
    """사람이 읽을 수 있는 요약을 줄 단위로 만든다."""
    system_text = info.windows_edition
    if info.windows_build:
        system_text += f" (빌드 {info.windows_build})"
    lines = [
        f"운영체제   : {system_text}",
        f"프로세서   : {info.cpu_name}",
        f"            코어 {info.cpu_cores}개 / 처리 갈래 {info.cpu_threads}개",
        f"메모리     : {info.ram_gb} GB",
    ]
    if info.gpus:
        for index, gpu in enumerate(info.gpus):
            label = "그래픽     :" if index == 0 else "            "
            driver = f" (드라이버 {gpu.driver_version})" if gpu.driver_version else ""
            age = f" 세대 {gpu.compute_capability}" if gpu.compute_capability else ""
            note = " - CUDA 쓰기엔 오래됨" if gpu.too_old_for_cuda else ""
            lines.append(f"{label} {gpu.name} [{gpu.vendor}]{driver}{age}{note}")
    else:
        lines.append("그래픽     : 찾지 못함")

    lines.append(f"설치된 부품: {info.installed_package or '없음'}")
    return lines
