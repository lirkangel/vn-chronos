FROM python:3.12-slim

WORKDIR /app

# Phase 1 deps only — uncomment torch/chronos lines for Phase 2
COPY requirements.txt .
RUN pip install --no-cache-dir \
    vnstock==4.0.2 \
    pandas==2.2.3 \
    tqdm==4.67.3 \
    requests==2.33.1

COPY vn_chronos/ ./vn_chronos/

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "vn_chronos.fetch_data"]
