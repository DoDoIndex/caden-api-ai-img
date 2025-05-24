FROM python:3.10-slim

# Cài đặt gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy và cài đặt requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY . .

# Chạy server với hypercorn (phiên bản cũ không hỗ trợ config nhiều tham số CLI)
CMD ["hypercorn", "main:app"]
