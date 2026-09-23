FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn wheels build faster with these present
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-transformer.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    # CPU-only torch: this container has no GPU, and the full CUDA wheel is a
    # multi-GB download it doesn't need. Installing it explicitly first means
    # the requirements-transformer.txt install below sees the version
    # constraint already satisfied and won't pull the (much bigger) default.
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements-transformer.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/

# Train the baseline model at build time if no artifact was copied in
# (keeps the image self-contained for a first deploy / demo).
RUN python -c "from pathlib import Path; import subprocess, sys; \
    p = Path('models/baseline_classifier.joblib'); \
    subprocess.run([sys.executable, '-m', 'src.train'], check=True) if not p.exists() else None"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
