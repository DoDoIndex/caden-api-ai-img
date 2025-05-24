FROM python:3.10-slim

WORKDIR /app

# Cài đặt các dependencies hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Cài đặt Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 && \
    pip install --no-cache-dir opencv-python==4.8.1.78 Pillow==10.1.0 && \
    pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 