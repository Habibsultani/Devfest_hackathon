FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-lock.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/data/cache \
    && chown -R appuser:appuser /app
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser assets/ ./assets/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser requirements.txt ./
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
