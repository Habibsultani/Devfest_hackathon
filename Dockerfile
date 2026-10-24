FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser assets/ ./assets/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser scripts/doctor.py scripts/fetch_model.py ./scripts/
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
