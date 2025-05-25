FROM python:3.12-slim

# 1. Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Tạo thư mục làm việc
WORKDIR /app

# 3. Copy toàn bộ project vào container
COPY . .

# 4. Cài đặt pip, wheel
RUN pip install --upgrade pip setuptools wheel

# 5. Cài đặt phụ thuộc Python
RUN pip install -r requirements.txt

# 6. Tải trọng số của mô hình Segment Anything (nếu cần)
# Bạn có thể tải tự động hoặc mount từ volume bên ngoài

# 7. Mở cổng cho FastAPI
EXPOSE 8000

# 8. Command để khởi chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
