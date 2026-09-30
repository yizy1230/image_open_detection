# image_open_detection worker — LLMDet (mmdet Grounding DINO)
# Build from workspace root:
#   docker build -f algos/image_open_detection/v1_0/Dockerfile -t algo-image_open_detection:v1_0 .
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    celery sqlalchemy psycopg2-binary minio pyyaml loguru requests \
    Pillow mmengine transformers timm nltk peft \
    "mmcv>=2.2.0" "mmdet>=3.3.0"

COPY common /app/common
COPY . /app/algo
COPY third_party/LLMDet /opt/LLMDet
WORKDIR /app/algo

ENV PYTHONPATH=/app:/app/algo:/opt/LLMDet
ENV WORKER_CONFIG_PATH=/app/algo/worker_config.yaml
ENV DEVICE=cuda
ENV LLMDET_ROOT=/opt/LLMDet
ENV LLMDET_MODEL_DIR=/checkpoints/LLMDet
ENV WORKSPACE_ROOT=/app/algo

CMD ["celery", "-A", "image_open_detection_worker.app", "worker", \
     "--pool=solo", "-Q", "image_open_detection", "-c", "1", "-n", "image_open_detection@%h"]
