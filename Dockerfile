FROM python:3.11-slim

# System dependencies for Kreuzberg
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-vie \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy project files
COPY pyproject.toml .
COPY app/ ./app/
COPY static/ ./static/

# Install Python deps with uv
RUN uv pip install --system --no-cache .

# Pre-download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Data directories
RUN mkdir -p /app/data/chroma /app/data/kuzu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
