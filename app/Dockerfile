FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add pip cache support
ARG PIP_CACHE_DIR=/tmp/pip-cache
RUN mkdir -p ${PIP_CACHE_DIR}

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with pip cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port for Gradio
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]