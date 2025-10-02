# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /app

# Copy requirements và cài
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim
WORKDIR /app

# Copy từ builder (chỉ thư viện đã cài)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code (đã loại bỏ theo .dockerignore)
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
