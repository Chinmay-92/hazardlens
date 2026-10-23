# HazardLens container - runs on linux/arm64 (AWS Graviton / Fargate ARM64) and linux/amd64
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app

COPY hazardlens ./hazardlens
COPY data/samples ./data/samples

USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"
# single worker: the approval queue is in-memory (see docs/TECHNICAL_REPORT.md, Limitations)
CMD ["uvicorn", "hazardlens.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
