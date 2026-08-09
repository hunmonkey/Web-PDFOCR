# =============================================================================
# PDFOCR 도커 이미지
#
# 운영서버(Unraid)에서 돌린다. 2222 포트로 서비스하고,
# Nginx Proxy Manager가 pdf.jaehoon.org 를 이 포트로 넘겨준다.
#
# 그래픽 가속:
#   Quadro P620(CUDA 12 계열 지원)을 쓸 수 있도록 onnxruntime-gpu를 넣는다.
#   CUDA 라이브러리는 pip으로 함께 받으므로 무거운 CUDA 기반 이미지를 쓰지 않는다.
#   그래픽카드 자체를 다루는 부분은 도커의 nvidia 런타임이 넘겨준다.
#   GPU가 느리거나 문제가 있으면 PDFOCR_FORCE_CPU=1 로 CPU만 쓰게 할 수 있다.
# =============================================================================

FROM python:3.12-slim

# 파이썬이 로그를 바로 내보내게 한다. (도커 로그에서 진행 상황을 보기 위해)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# opencv가 쓰는 그림 관련 라이브러리와, 그래픽카드 상태를 확인하는 도구가 필요하다.
# libgl1 / libglib2.0-0 : opencv가 불러오는 라이브러리
# curl                  : 컨테이너가 살아 있는지 확인할 때 사용
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 필요한 것을 먼저 설치한다. 소스만 바뀌었을 때 이 단계를 다시 하지 않아
# 다시 만들 때 훨씬 빠르다.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0"

# 소스 복사
COPY app ./app
COPY scripts ./scripts

# 글자 인식 모델을 미리 받아 둔다.
# 이 단계를 빼면 첫 변환 때 인터넷에서 받느라 한참 기다리게 되고,
# 인터넷이 막힌 곳에서는 아예 동작하지 않는다.
RUN python -c "\
import os; os.environ['PDFOCR_FORCE_CPU']='1';\
from app.ocr_engine import get_engine;\
get_engine(use_gpu=False);\
print('인식 모델 준비 완료')"

# 보관함과 설정이 담길 자리. 실제 폴더는 도커 실행 때 바깥과 연결한다.
ENV PDFOCR_STORAGE_DIR=/data \
    PDFOCR_CONFIG_DIR=/config
RUN mkdir -p /data /config

EXPOSE 2222

# 컨테이너가 살아 있는지 도커가 스스로 확인하게 한다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:2222/login || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2222"]
