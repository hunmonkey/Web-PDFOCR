# =============================================================================
# PDFOCR 도커 이미지
#
# 운영서버(Unraid)에서 돌린다. 2222 포트로 서비스하고,
# Nginx Proxy Manager가 pdf.jaehoon.org 를 이 포트로 넘겨준다.
#
# 왜 CPU용으로 만드는가:
#   이 서버의 그래픽카드는 Quadro P620이고 세대 번호(compute capability)가 6.1이다.
#   실제로 GPU용으로 만들어 돌려 보니 아래 오류가 나면서 실패했다.
#     "no kernel image is available for execution on the device"
#   CUDA 12 자체는 이 카드를 지원하지만, onnxruntime을 만들 때 오래된 카드용
#   계산 코드를 빼고 만들었기 때문이다.
#
#   그래서 CUDA 라이브러리(1.5GB 남짓)를 넣지 않는다. 이미지가 4.23GB에서 크게 줄고,
#   쓰지도 못할 것을 받느라 시간을 쓰지 않는다.
#   측정해 보니 이 서버 CPU(Xeon E-2236 12갈래)로 한 장에 0.5초 정도로 충분히 빠르다.
#
#   나중에 서버 그래픽카드를 새것(세대 번호 7.0 이상)으로 바꾸면
#   아래 pip install 줄을 onnxruntime-gpu[cuda,cudnn]==1.26.0 으로 바꾸고
#   docker-compose.yml 의 그래픽카드 항목을 되살리면 된다.
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
    && pip install "onnxruntime==1.28.0"

# 소스 복사
COPY app ./app
COPY scripts ./scripts

# 글자 인식 모델을 미리 받아 둔다.
# 이 단계를 빼면 첫 변환 때 인터넷에서 받느라 한참 기다리게 되고,
# 인터넷이 막힌 곳에서는 아예 동작하지 않는다.
RUN python -c "\
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
