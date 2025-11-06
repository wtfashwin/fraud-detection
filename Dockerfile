FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user and group for security
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libpq-dev \
	&& useradd --create-home --shell /bin/bash appuser \
	&& rm -rf /var/lib/apt/lists/*

# Install dependencies first to take advantage of layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and models, set ownership to non-root user
COPY . /app
RUN chown -R appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser run_migrations.sh /app/run_migrations.sh
RUN chmod +x /app/run_migrations.sh

ENTRYPOINT ["/app/run_migrations.sh"]
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.app:app", "--bind", "0.0.0.0:8000", "--workers", "2"]