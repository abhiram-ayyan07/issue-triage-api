FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn wheels build faster with these present
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
