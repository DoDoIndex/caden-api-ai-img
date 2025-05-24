# Base image có Python và pip
FROM python:3.10-slim

# Set thư mục làm việc
WORKDIR /app

# Cài dependencies hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code và requirements
COPY . /app
COPY requirements.txt .

# Cài các thư viện Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Command chạy server FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
