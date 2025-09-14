# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
WORKDIR /app
COPY docker/requirements.txt ./requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
# install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
# create non-root user
RUN useradd --create-home appuser
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the minimal runtime code so ``app.py`` can import ``poker_ai``.
COPY src/poker_ai /app/poker_ai
COPY app.py .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl --fail http://localhost:8000/healthz || exit 1
USER appuser
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
