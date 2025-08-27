FROM python:3.12-slim

WORKDIR /app

# Install OS deps needed by some Python packages (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (cached if unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Pre-download the model into the image so container startup is fast.
# This caches the Keras weights at build time.
RUN python - <<'PY'
from tensorflow.keras.applications import ResNet50
_ = ResNet50(weights="imagenet")
print("ResNet50 cached in image")
PY

# Copy the modelserver code
COPY . .

# Minimal healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import sys; sys.exit(0)"

CMD ["python", "main.py"]
