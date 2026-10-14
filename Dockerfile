FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MODEL_DIR=/app/models
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY api ./api
COPY db ./db
COPY alembic ./alembic
COPY alembic.ini xai_tasks.py ./
COPY scripts/run_migrations.sh ./scripts/run_migrations.sh
COPY models ./models
COPY web ./web
RUN chmod +x /app/scripts/run_migrations.sh && mkdir -p /app/mlruns && chown appuser:appuser /app/mlruns
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Fraud Detection API" \
      org.opencontainers.image.description="Fraud scoring API and asynchronous explanations; requires PostgreSQL and Redis. Demo model, not a validated financial control." \
      org.opencontainers.image.source="https://github.com/wtfashwin/fraud-detection" \
      org.opencontainers.image.documentation="https://github.com/wtfashwin/fraud-detection#readme" \
      org.opencontainers.image.revision=$VCS_REF
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=3)"
ENTRYPOINT ["/app/scripts/run_migrations.sh"]
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60"]

# Optional tools for training and serving an explicitly selected XGBoost/MLflow bundle.
FROM runtime AS training
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY training ./training
COPY scripts ./scripts
RUN mkdir -p /app/models/trained /app/data /app/plots && chown -R appuser:appuser /app/models/trained /app/data /app/plots
ENV MODEL_OUTPUT_DIR=/app/models/trained
USER 10001:10001

# Keep the default serving image small; training never overwrites its model bundle.
FROM runtime AS production
