# syntax=docker/dockerfile:1

# =====================
# Stage 1: Builder
# =====================
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip pipenv
COPY Pipfile Pipfile.lock ./

# Install dependencies system-wide
RUN pipenv install --deploy --system


# =====================
# Stage 2: Runtime
# =====================
FROM python:3.12-slim

# ---- System Libraries (IMPORTANT) ----
# Includes build-essential & cmake for PaddleX HPI (High Performance Inference)
# JIT-compilation of optimized inference kernels at startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    libreoffice \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

ENV FFMPEG_PATH="/usr/bin/ffmpeg"

WORKDIR /app

# ---- Copy Python deps from builder ----
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ---- Install HPI dependencies for CPU ----
# It's best to install HPI dependencies *before* initializing the pipeline, 
# so that the JIT compiler has everything it needs when the engine is instantiated.
RUN paddleocr install_hpi_deps cpu

# ---- Pre-download PaddleOCR models ----
# Initialize the pipeline once during build to download & cache all required
# models (~500 MB).  This avoids a slow first-request download at runtime.
ENV DISABLE_MODEL_SOURCE_CHECK=True
RUN python -c "\
from paddleocr import PaddleOCR; \
ocr = PaddleOCR( \
    device='cpu', \
    engine='onnxruntime', \
    text_detection_model_name='PP-OCRv6_small_det', \
    text_recognition_model_name='PP-OCRv6_small_rec', \
    use_doc_orientation_classify=True, \
    use_doc_unwarping=True, \
    use_textline_orientation=False, \
    enable_hpi=True, \
)"

# ---- Copy Application ----
COPY . .


# =====================
# Runtime Environment
# =====================
# ---- Disable GPU / CUDA ----
ENV CUDA_VISIBLE_DEVICES=-1

# ---- Paddle Safety ----
ENV FLAGS_use_cuda=False
ENV FLAGS_use_mkldnn=False

# ---- CPU Optimization (Safe Defaults) ----
ENV OMP_NUM_THREADS=2
ENV ONNX_NUM_THREADS=2

# EPYC NUMA awareness — pin threads to cores
ENV OMP_PROC_BIND=close
ENV OMP_PLACES=cores

ENV OPENBLAS_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# ONNX Runtime specific
ENV ONNXRUNTIME_DISABLE_CPU_AFFINITY=1

# ---- Silence Logs ----
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV GLOG_minloglevel=2
ENV FLAGS_log_level=2

# =====================
# App
# =====================
EXPOSE 8001

CMD ["bash", "-c", "DISABLE_MODEL_SOURCE_CHECK=True uvicorn main:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 300 --workers ${UVICORN_WORKERS:-1}"]
