# VESLINT Docker Image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r maritime && useradd -r -g maritime maritime

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/models plots logs && \
    chown -R maritime:maritime /app

# Switch to non-root user
USER maritime

# Create volume mount points
VOLUME ["/app/data", "/app/plots", "/app/logs"]

# Expose port for potential web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from src.config import config; print('OK')" || exit 1

# Default command
CMD ["python", "main.py", "--help"]

# Labels
LABEL maintainer="support@veslint.com"
LABEL version="1.0.0"
LABEL description="VEssel Surveillance & Logging INTelligence"
LABEL org.opencontainers.image.source="https://github.com/VESLINT/VESLINT"