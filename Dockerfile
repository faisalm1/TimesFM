FROM python:3.11-slim

WORKDIR /app

# System deps for torch + lightgbm + pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install "setuptools>=68" wheel && \
    pip install -e ".[dev]" --no-build-isolation || true && \
    pip install numpy pandas pyarrow python-dotenv alpaca-py pytz \
        fastapi "uvicorn[standard]" lightgbm scikit-learn joblib apscheduler \
        timesfm torch --index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY src/ src/
COPY scripts/ scripts/

# Create data dirs (Parquet cache + ranking JSON + ML artifacts)
RUN mkdir -p data/cache data/ml_artifacts

# Expose API port
EXPOSE 8001

CMD ["python", "-m", "uvicorn", "gap_dashboard.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8001"]
