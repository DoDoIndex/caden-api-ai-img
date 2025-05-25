FROM python:3.10-slim

# Cài các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy requirements và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Mở cổng 8000
EXPOSE 8000

# Khởi chạy FastAPI server bằng Hypercorn
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000"]
